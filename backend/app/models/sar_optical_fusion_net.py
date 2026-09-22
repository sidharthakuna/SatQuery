"""
SatQuery AI — SAR-Guided Optical Reconstruction Network (SAROpticalFusionNet)
Architecture based on state-of-the-art cloud removal benchmarks:
- Global-Local Fusion Network (GLF-CR)
- Feature Enhancement Network (FENet)
- Deep Sentinel-2 Cloud Removal (DSen2-CR)
- SOMA-1M / SEN12MS-CR paired dataset training formulation

Training Paradigm:
  CLEAR SENTINEL-2 (Ground Truth)
        │
        ├── Add realistic synthetic cloud mask + atmospheric haze
        │
        ▼
  CLOUDY SENTINEL-2 ───────┐
                           │
  SENTINEL-1 SAR ──────────┤
                           ▼
                 SAROpticalFusionNet (Dual-Branch Cross-Attention)
                           │
                           ▼
             ESTIMATED CLEAR SENTINEL-2
                           │
                           ▼
          Loss: L1 + Spectral Angle (SAM) + Gradient Edge + Perceptual
          Evaluated against ORIGINAL CLEAR SENTINEL-2
"""

import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Building Blocks: Residual Dense Block & Multi-Head Cross-Attention
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Standard Convolution + BatchNorm/InstanceNorm + LeakyReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualDenseBlock(nn.Module):
    """Residual dense feature extractor for fine high-frequency texture recovery."""
    def __init__(self, channels: int, growth_rate: int = 32):
        super().__init__()
        self.c1 = ConvBlock(channels, growth_rate)
        self.c2 = ConvBlock(channels + growth_rate, growth_rate)
        self.c3 = ConvBlock(channels + 2 * growth_rate, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.c1(x)
        x2 = self.c2(torch.cat([x, x1], dim=1))
        x3 = self.c3(torch.cat([x, x1, x2], dim=1))
        return x + x3 * 0.2


class CrossModalAttention(nn.Module):
    """
    Cross-Attention Module:
    Queries (Q) come from Cloudy Optical features.
    Keys (K) and Values (V) come from Sentinel-1 SAR microwave structural features.
    This injects radar dielectric structures directly into optical spatial tokens.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = (dim // num_heads) ** -0.5

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, opt_feat: torch.Tensor, sar_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = opt_feat.shape
        residual = opt_feat

        # Project with depthwise structural refinement
        q = self.q_proj(opt_feat).view(b, self.num_heads, c // self.num_heads, h * w)
        k = self.k_proj(sar_feat).view(b, self.num_heads, c // self.num_heads, h * w)
        v = self.v_proj(sar_feat).view(b, self.num_heads, c // self.num_heads, h * w)

        # Transposed Channel Cross-Attention: (C_head, HW) x (HW, C_head) -> (C_head, C_head)
        # O(HW) linear spatial complexity, memory-efficient for high-resolution satellite imagery
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.view(b, c, h, w)
        out = self.dwconv(out)
        out = self.out_proj(out)

        return self.norm(residual + out)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SAROpticalFusionNet: Dual-Branch Architecture
# ─────────────────────────────────────────────────────────────────────────────

class SAROpticalFusionNet(nn.Module):
    """
    Dual-branch deep architecture for SAR-guided optical cloud removal:
    - Optical Branch: Sentinel-2 (3 RGB channels + 1 Cloud Mask channel)
    - SAR Branch: Sentinel-1 (2 channels: VV, VH polarization)
    - Cross-Modal Fusion: Multi-scale attention fusing microwave backscatter into optical
    - Progressive Decoder: Multi-scale reconstruction with skip connections
    """
    def __init__(self, in_opt_ch: int = 4, in_sar_ch: int = 2, out_ch: int = 3, base_ch: int = 64):
        super().__init__()
        self.base_ch = base_ch

        # Optical Stream Encoders
        self.opt_conv0 = ConvBlock(in_opt_ch, base_ch)
        self.opt_down1 = nn.Sequential(ConvBlock(base_ch, base_ch * 2, stride=2), ResidualDenseBlock(base_ch * 2))
        self.opt_down2 = nn.Sequential(ConvBlock(base_ch * 2, base_ch * 4, stride=2), ResidualDenseBlock(base_ch * 4))

        # SAR Stream Encoders
        self.sar_conv0 = ConvBlock(in_sar_ch, base_ch)
        self.sar_down1 = nn.Sequential(ConvBlock(base_ch, base_ch * 2, stride=2), ResidualDenseBlock(base_ch * 2))
        self.sar_down2 = nn.Sequential(ConvBlock(base_ch * 2, base_ch * 4, stride=2), ResidualDenseBlock(base_ch * 4))

        # Cross-Modal Attention Fusion at Each Resolution Scale
        self.attn0 = CrossModalAttention(base_ch)
        self.attn1 = CrossModalAttention(base_ch * 2)
        self.attn2 = CrossModalAttention(base_ch * 4)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualDenseBlock(base_ch * 4),
            ResidualDenseBlock(base_ch * 4),
        )

        # Decoder with Skip Connections
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec_block2 = nn.Sequential(
            ConvBlock(base_ch * 4, base_ch * 2),
            ResidualDenseBlock(base_ch * 2),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec_block1 = nn.Sequential(
            ConvBlock(base_ch * 2, base_ch),
            ResidualDenseBlock(base_ch),
        )

        # Final Reconstruction Head
        self.final_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch // 2, out_ch, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Calibrated reflectance range [0.0, 1.0]
        )

    def forward(self, optical_with_mask: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            optical_with_mask: Tensor of shape (B, 4, H, W) -> [R, G, B, CloudMask]
            sar: Tensor of shape (B, 2, H, W) -> [VV, VH]
        Returns:
            estimated_clear: Tensor of shape (B, 3, H, W) -> estimated Sentinel-2 RGB
        """
        # Encoder Stage 0 (H, W)
        o0 = self.opt_conv0(optical_with_mask)
        s0 = self.sar_conv0(sar)
        f0 = self.attn0(o0, s0)

        # Encoder Stage 1 (H/2, W/2)
        o1 = self.opt_down1(o0)
        s1 = self.sar_down1(s0)
        f1 = self.attn1(o1, s1)

        # Encoder Stage 2 (H/4, W/4)
        o2 = self.opt_down2(o1)
        s2 = self.sar_down2(s1)
        f2 = self.attn2(o2, s2)

        # Bottleneck
        b = self.bottleneck(f2)

        # Decoder Stage 2
        d2 = self.up2(b)
        d2 = torch.cat([d2, f1], dim=1)
        d2 = self.dec_block2(d2)

        # Decoder Stage 1
        d1 = self.up1(d2)
        d1 = torch.cat([d1, f0], dim=1)
        d1 = self.dec_block1(d1)

        # Final Estimated Optical Surface
        recon = self.final_head(d1)
        return recon


# ─────────────────────────────────────────────────────────────────────────────
# 3. Scientific Loss Functions: Spectral Angle Mapper & Gradient Edge Loss
# ─────────────────────────────────────────────────────────────────────────────

class SpectralAngleMapperLoss(nn.Module):
    """
    Spectral Angle Mapper (SAM) Loss:
    Penalizes chromatic/spectral deviation across RGB bands to ensure
    realistic Sentinel-2 surface coloring rather than washed-out gray.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dot = torch.sum(pred * target, dim=1)
        norm_p = torch.norm(pred, p=2, dim=1)
        norm_t = torch.norm(target, p=2, dim=1)
        cos_sam = torch.clamp(dot / (norm_p * norm_t + self.eps), -1.0 + self.eps, 1.0 - self.eps)
        sam = torch.acos(cos_sam)
        return torch.mean(sam)


class GradientLoss(nn.Module):
    """
    Multi-Scale Gradient Edge Loss:
    Matches spatial discontinuities (roads, piers, building edges)
    delineated by Sentinel-1 SAR microwave backscatter.
    """
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kernel_x)
        self.register_buffer("ky", kernel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pred.shape
        kx = self.kx.repeat(c, 1, 1, 1)
        ky = self.ky.repeat(c, 1, 1, 1)

        grad_pred_x = F.conv2d(pred, kx, padding=1, groups=c)
        grad_pred_y = F.conv2d(pred, ky, padding=1, groups=c)
        grad_targ_x = F.conv2d(target, kx, padding=1, groups=c)
        grad_targ_y = F.conv2d(target, ky, padding=1, groups=c)

        loss_x = F.l1_loss(grad_pred_x, grad_targ_x)
        loss_y = F.l1_loss(grad_pred_y, grad_targ_y)
        return loss_x + loss_y


class SAROpticalCompositeLoss(nn.Module):
    """
    Composite Loss for SAR-Guided Cloud Removal:
    L = w_l1 * L1 + w_sam * SAM + w_grad * Edge + w_ssim * (1 - SSIM)
    """
    def __init__(self, w_l1: float = 1.0, w_sam: float = 0.25, w_grad: float = 0.20):
        super().__init__()
        self.w_l1 = w_l1
        self.w_sam = w_sam
        self.w_grad = w_grad

        self.l1 = nn.L1Loss()
        self.sam = SpectralAngleMapperLoss()
        self.grad = GradientLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1_loss = self.l1(pred, target)
        sam_loss = self.sam(pred, target)
        grad_loss = self.grad(pred, target)

        total = self.w_l1 * l1_loss + self.w_sam * sam_loss + self.w_grad * grad_loss
        return {
            "loss": total,
            "l1": l1_loss,
            "sam": sam_loss,
            "grad": grad_loss,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training Strategy: Realistic Synthetic Cloud Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticCloudGenerator:
    """
    Generates realistic multi-scale fractal cloud decks with atmospheric haze
    and variable opacity. Used for supervised training on clear Sentinel-2 pairs.
    """
    @staticmethod
    def generate_cloud_mask(h: int, w: int, coverage: float = 0.65) -> np.ndarray:
        """
        Creates smooth, organically clustered cloud alpha mask [0.0, 1.0].
        """
        from scipy.ndimage import gaussian_filter

        # Multi-frequency noise
        np.random.seed(None)
        noise1 = np.random.randn(h // 4, w // 4)
        noise2 = np.random.randn(h // 8, w // 8)

        from scipy.ndimage import zoom
        n1_up = zoom(noise1, (4, 4), order=1)[:h, :w]
        n2_up = zoom(noise2, (8, 8), order=1)[:h, :w]
        combined = n1_up * 0.7 + n2_up * 0.3

        # Thresholding for cloud coverage
        smooth = gaussian_filter(combined, sigma=4.0)
        norm = (smooth - smooth.min()) / max(np.ptp(smooth), 1e-6)
        
        # Sigmoidal cloud opacity with realistic feathered boundaries
        thresh = 1.0 - coverage
        mask = 1.0 / (1.0 + np.exp(-12.0 * (norm - thresh)))
        return np.clip(mask, 0.0, 1.0).astype(np.float32)


class PairedSAROpticalCloudDataset(torch.utils.data.Dataset):
    """
    Standard Benchmark Training Dataset:
    Takes real Clear Sentinel-2 imagery + Real Sentinel-1 SAR imagery,
    artificially synthesizes realistic clouds over the optical pass,
    and trains the network with the ground-truth clear optical as the target.
    """
    def __init__(self, clear_optical_paths: List[Path], sar_paths: List[Path], patch_size: int = 256):
        self.clear_paths = clear_optical_paths
        self.sar_paths = sar_paths
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.clear_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import rasterio
        ps = self.patch_size

        # 1. Load Clear Optical (Target Ground Truth)
        with rasterio.open(str(self.clear_paths[idx])) as src:
            clear_opt = src.read()[:3].astype(np.float32)
            if clear_opt.max() > 1.0:
                clear_opt = clear_opt / 255.0

        # 2. Load SAR (Microwave Guidance)
        with rasterio.open(str(self.sar_paths[idx % len(self.sar_paths)])) as src:
            sar = src.read()[:2].astype(np.float32)
            if sar.max() > 1.0:
                sar = sar / 255.0

        # Resize/Crop to patch_size
        h, w = min(clear_opt.shape[1], sar.shape[1]), min(clear_opt.shape[2], sar.shape[2])
        clear_opt = clear_opt[:, :ps, :ps]
        sar = sar[:, :ps, :ps]

        # 3. Synthesize Cloud Deck
        cmask = SyntheticCloudGenerator.generate_cloud_mask(ps, ps, coverage=np.random.uniform(0.4, 0.85))
        cloud_color = np.array([0.92, 0.95, 0.98], dtype=np.float32).reshape(3, 1, 1)

        # Cloudy Optical Input = (1 - mask) * Clear + mask * White Cloud Deck
        cloudy_opt = (1.0 - cmask) * clear_opt + cmask * cloud_color
        cloudy_opt = np.clip(cloudy_opt, 0.0, 1.0)

        # Stack [R, G, B, CloudMask]
        opt_input = np.concatenate([cloudy_opt, cmask[np.newaxis, ...]], axis=0)

        return (
            torch.from_numpy(opt_input).float(),
            torch.from_numpy(sar).float(),
            torch.from_numpy(clear_opt).float(),
        )
