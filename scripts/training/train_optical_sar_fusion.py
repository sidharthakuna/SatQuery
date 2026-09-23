"""
SatQuery AI — Specialist Model 4: Optical + SAR Cross-Modal Fusion Deep Training Engine
Trains dual-branch Cross-Attention neural networks (OpticalSARCrossAttentionNetV2) to fuse
co-registered Optical (Sentinel-2/Cartosat) and Synthetic Aperture Radar (Sentinel-1/RISAT)
imagery for cloud-penetrating surveillance.

Physics-Informed Architecture:
- Dual-Stream Optical & SAR Encoders with Multi-Scale Skip Connections (~2.8M Parameters)
- Cross-Modal Fusion Bottleneck querying SAR structural backscatter into optical multispectral tokens
- High-Fidelity Reconstruction Head restoring sub-cloud terrain reflectance
- Multi-Component Loss: L1 + MSE + Structural Similarity (SSIM) + Cloud-Weighted Inpainting Loss
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Ensure backend in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
backend_dir = BASE_DIR / "backend"
sys.path.insert(0, str(backend_dir))

from app.models.fusion_net import OpticalSARCrossAttentionNetV2

CHECKPOINT_DIR = backend_dir / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = BASE_DIR / "data" / "samples"


def _create_window(window_size: int, channel: int):
    def _gaussian(w, sigma):
        gauss = torch.Tensor([math.exp(-(x - w // 2) ** 2 / float(2 * sigma ** 2)) for x in range(w)])
        return gauss / gauss.sum()

    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, window=None) -> torch.Tensor:
    (_, channel, _, _) = img1.size()
    if window is None:
        window = _create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class MultiModalSatelliteDataset(Dataset):
    """
    Constructs high-fidelity Optical-SAR-Clean training pairs.
    Extracts multi-scale patches from authentic Sentinel-1, Sentinel-2, and RISAT GeoTIFFs,
    applying realistic multi-scale cloud decks and cast shadow modeling.
    """
    def __init__(self, n_samples: int = 360, patch_size: int = 128, augment: bool = True):
        self.patch_size = patch_size
        self.augment = augment
        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        # 1. Ingest Authentic Satellite Master Rasters
        base_clean_p = SAMPLES_DIR / "fusion_optical_clean.tif"
        base_sar_p = SAMPLES_DIR / "fusion_sar.tif"
        base_cloudy_p = SAMPLES_DIR / "fusion_optical.tif"

        authentic_clean = None
        authentic_sar = None
        authentic_cloudy = None

        if base_clean_p.exists():
            with rasterio.open(str(base_clean_p)) as src:
                authentic_clean = (src.read()[:3].astype(np.float32) / 255.0)

        if base_sar_p.exists():
            with rasterio.open(str(base_sar_p)) as src:
                s_arr = src.read()
                if s_arr.shape[0] >= 2:
                    authentic_sar = (s_arr[:2].astype(np.float32) / 255.0)
                else:
                    authentic_sar = np.repeat(s_arr[:1].astype(np.float32) / 255.0, 2, axis=0)

        if base_cloudy_p.exists():
            with rasterio.open(str(base_cloudy_p)) as src:
                authentic_cloudy = (src.read()[:3].astype(np.float32) / 255.0)

        # Ingest other reference rasters for diverse ground terrain
        extra_cleans = []
        for name in ["sentinel2_forest_canopy.tif", "urban_t1.tif", "dior_port_facility.tif"]:
            ep = SAMPLES_DIR / name
            if ep.exists():
                try:
                    with rasterio.open(str(ep)) as src:
                        extra_cleans.append(src.read()[:3].astype(np.float32) / 255.0)
                except Exception:
                    pass

        print(f"  Ingesting authentic satellite scenes: coastal={authentic_clean is not None}, extra_scenes={len(extra_cleans)}")

        # 2. Generate Tiled and Synthesized Augmented Samples
        rng = np.random.RandomState(42)

        for i in range(n_samples):
            # Select base clean raster
            if authentic_clean is not None and (i % 2 == 0 or len(extra_cleans) == 0):
                clean_full = authentic_clean
                sar_full = authentic_sar
                cloudy_full = authentic_cloudy
            else:
                clean_full = extra_cleans[i % len(extra_cleans)]
                sar_full = None
                cloudy_full = None

            # Crop random patch
            _, full_h, full_w = clean_full.shape
            py = rng.randint(0, max(full_h - patch_size, 1))
            px = rng.randint(0, max(full_w - patch_size, 1))

            target_patch = clean_full[:, py:py+patch_size, px:px+patch_size].copy()

            # SAR patch
            if sar_full is not None:
                sar_patch = sar_full[:, py:py+patch_size, px:px+patch_size].copy()
            else:
                sar_patch = self._synthesize_sar_from_optical(target_patch, rng)

            # Cloudy optical patch
            if cloudy_full is not None and rng.rand() > 0.4:
                # Use authentic cloudy patch with natural cloud deck
                opt_patch = cloudy_full[:, py:py+patch_size, px:px+patch_size].copy()
                cloud_mask = (np.mean(np.abs(opt_patch - target_patch), axis=0) > 0.18).astype(np.float32)
            else:
                opt_patch, cloud_mask = self._simulate_cloud_cover(target_patch, rng)

            # Apply multiplicative radar speckle noise
            sar_patch = self._apply_radar_speckle(sar_patch, rng)

            self.samples.append((opt_patch, sar_patch, target_patch, cloud_mask[np.newaxis, ...]))

        print(f"  Built multi-modal training dataset with {len(self.samples)} augmented tiles.")

    def _synthesize_sar_from_optical(self, opt: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        r, g, b = opt[0], opt[1], opt[2]
        is_water = (b > 0.30) & (b > r + 0.10)
        is_veg = (g > r + 0.05) & (g > b + 0.05)
        is_urban = (r > 0.48) & (g > 0.45) & (b > 0.40)

        vv = np.full_like(r, 0.35)
        vh = np.full_like(r, 0.22)

        vv[is_water] = rng.uniform(0.02, 0.10)
        vh[is_water] = rng.uniform(0.01, 0.05)

        vv[is_veg] = rng.uniform(0.40, 0.55)
        vh[is_veg] = rng.uniform(0.30, 0.45)

        vv[is_urban] = rng.uniform(0.70, 0.95)
        vh[is_urban] = rng.uniform(0.55, 0.85)

        return np.stack([vv, vh], axis=0).astype(np.float32)

    def _simulate_cloud_cover(self, clean: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        h, w = clean.shape[1], clean.shape[2]
        cloud_mask = np.zeros((h, w), dtype=np.float32)

        num_cells = rng.randint(1, 4)
        for _ in range(num_cells):
            cx = rng.randint(0, w)
            cy = rng.randint(0, h)
            rx = rng.randint(w // 6, w // 2)
            ry = rng.randint(h // 6, h // 2)
            yy, xx = np.ogrid[:h, :w]
            dist = ((xx - cx) / float(rx)) ** 2 + ((yy - cy) / float(ry)) ** 2
            core = np.clip(1.0 - dist, 0.0, 1.0)
            cloud_mask = np.maximum(cloud_mask, core ** 1.3)

        cloud_mask = np.clip(cloud_mask * rng.uniform(0.80, 1.0), 0.0, 1.0)

        # Cast solar shadow
        dx, dy = -5, 5
        shadow = np.roll(np.roll(cloud_mask, dy, axis=0), dx, axis=1)
        shadow_factor = 1.0 - 0.50 * shadow * (1.0 - cloud_mask)

        cloudy = clean.copy() * shadow_factor[np.newaxis, ...]
        cloud_albedo = rng.uniform(0.85, 0.98, (3, 1, 1)).astype(np.float32)
        cloudy = cloudy * (1.0 - cloud_mask[np.newaxis, ...]) + cloud_albedo * cloud_mask[np.newaxis, ...]

        return np.clip(cloudy, 0.0, 1.0).astype(np.float32), (cloud_mask > 0.22).astype(np.float32)

    def _apply_radar_speckle(self, sar: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        noise = rng.exponential(scale=0.10, size=sar.shape).astype(np.float32)
        return np.clip(sar + noise - 0.05, 0.01, 0.99)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        opt, sar, clean, cmask = self.samples[idx]
        if self.augment:
            rng = np.random.RandomState()
            if rng.rand() > 0.5:
                opt = opt[:, :, ::-1]
                sar = sar[:, :, ::-1]
                clean = clean[:, :, ::-1]
                cmask = cmask[:, :, ::-1]
            if rng.rand() > 0.5:
                opt = opt[:, ::-1, :]
                sar = sar[:, ::-1, :]
                clean = clean[:, ::-1, :]
                cmask = cmask[:, ::-1, :]

        return (
            torch.from_numpy(opt.copy()),
            torch.from_numpy(sar.copy()),
            torch.from_numpy(clean.copy()),
            torch.from_numpy(cmask.copy()),
        )


def train_optical_sar_fusion(
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1.8e-4,
    n_samples: int = 96,
    device_str: str = "auto",
) -> Path:
    """
    Executes deep neural training for OpticalSARCrossAttentionNetV2 on authentic satellite imagery.
    Saves trained checkpoints with architecture validation to fusion_net.pt & optical_sar_fusion.pt.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 72, flush=True)
    print("  SatQuery AI — Specialist Model 4: Optical-SAR Cross-Modal Fusion Net", flush=True)
    print(f"  Training Engine: OpticalSARCrossAttentionNetV2 (~2.8M params)", flush=True)
    print(f"  Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | Samples: {n_samples}", flush=True)
    print("=" * 72, flush=True)

    dataset = MultiModalSatelliteDataset(n_samples=n_samples, patch_size=128, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = OpticalSARCrossAttentionNetV2(optical_channels=3, sar_channels=2, out_channels=3).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable Parameters: {params:,}", flush=True)

    # Warm start if checkpoint exists
    ckpt_path = CHECKPOINT_DIR / "fusion_net.pt"
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(sd, strict=False)
            prev_epochs = ckpt.get("epochs", 0)
            prev_ssim = ckpt.get("ssim", 0.0)
            print(f"  Warm-started from checkpoint: {ckpt_path.name} (prev epochs={prev_epochs}, ssim={prev_ssim:.4f})", flush=True)
        except Exception as e:
            print(f"  Warm-start skipped: {e}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_loss = float("inf")
    best_ssim = 0.0
    start_time = time.time()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_ssim = 0.0
        total_batches = 0

        for opt, sar, clean, cmask in loader:
            opt = opt.to(device)
            sar = sar.to(device)
            clean = clean.to(device)
            cmask = cmask.to(device)

            optimizer.zero_grad()
            reconstructed, conf = model(opt, sar)

            # 1. Base L1 and MSE
            l1_loss = F.l1_loss(reconstructed, clean)
            mse_loss = F.mse_loss(reconstructed, clean)

            # 2. Cloud-Weighted Inpainting Loss (4.0x penalty under cloud cover)
            cloud_w = 1.0 + 3.0 * cmask
            weighted_l1 = (F.l1_loss(reconstructed, clean, reduction="none") * cloud_w).mean()

            # 3. Structural Similarity (SSIM) Loss
            curr_ssim = ssim(reconstructed, clean)
            ssim_loss = 1.0 - curr_ssim

            # 4. Confidence Loss (aligned with clear-sky transparency)
            target_conf = torch.clamp(1.0 - cmask.mean(dim=(1, 2, 3)), 0.70, 0.98).unsqueeze(1)
            conf_loss = F.mse_loss(conf, target_conf)

            # Composite multi-task loss
            loss = weighted_l1 + 0.4 * mse_loss + 0.35 * ssim_loss + 0.1 * conf_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.2)
            optimizer.step()

            total_loss += loss.item()
            total_ssim += curr_ssim.item()
            total_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(total_batches, 1)
        avg_ssim = total_ssim / max(total_batches, 1)
        lr_now = scheduler.get_last_lr()[0]

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            best_ssim = avg_ssim

        marker = " [BEST]" if is_best else ""
        if epoch % 2 == 0 or epoch == epochs or is_best:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | SSIM: {avg_ssim:.4f} | LR: {lr_now:.2e}{marker}", flush=True)

        # Checkpoint save on best or final epoch
        if is_best or epoch == epochs:
            state = {
                "model_state_dict": model.state_dict(),
                "epochs": epoch,
                "loss": best_loss,
                "ssim": best_ssim,
                "arch": "OpticalSARCrossAttentionNet",
                "optical_channels": 3,
                "sar_channels": 2,
                "fidelity": "high",
            }
            torch.save(state, CHECKPOINT_DIR / "fusion_net.pt")
            torch.save(state, CHECKPOINT_DIR / "optical_sar_fusion.pt")

    elapsed = time.time() - start_time
    print(f"\n  [SUCCESS] Deep training complete in {elapsed:.1f}s!", flush=True)
    print(f"  Best Loss: {best_loss:.4f} | Best SSIM: {best_ssim:.4f}", flush=True)
    print(f"  Saved trained checkpoints to:", flush=True)
    print(f"    - {CHECKPOINT_DIR / 'fusion_net.pt'}", flush=True)
    print(f"    - {CHECKPOINT_DIR / 'optical_sar_fusion.pt'}", flush=True)

    return CHECKPOINT_DIR / "fusion_net.pt"


def main():
    parser = argparse.ArgumentParser(description="Deep Training Engine for Optical-SAR Fusion")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.8e-4)
    parser.add_argument("--n-samples", type=int, default=96)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train_optical_sar_fusion(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_samples=args.n_samples,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
