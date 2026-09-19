"""
SatQuery AI — Enhanced Master Training Engine v2.0
Trains all 4 specialist AI models with significantly improved quality:
  Model 1: Remote-Sensing VLM / VQA (LoRA-adapted)
  Model 2: Grounding DINO + SAM-RS Visual Grounding
  Model 3: ChangeFormer Bi-temporal Change Detection
  Model 4: Cross-Modal Optical-SAR Fusion Network

Improvements over v1:
  - 15 epochs (5x more) with cosine annealing LR schedule
  - Rich data augmentation: flips, rotations, noise, brightness/contrast jitter
  - Focal loss for class-imbalanced change/grounding tasks
  - Gradient clipping (max_norm=1.0) to prevent exploding gradients
  - Better synthetic data generation with diverse satellite scene types
  - GIoU loss for bounding box regression (Grounding model)
  - Cloud-weighted reconstruction loss for SAR Fusion model
  - Label smoothing for VLM token prediction
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add backend to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "benchmarks"
CHECKPOINT_DIR = backend_dir / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 25          # Deep fine-tune for Models 3 & 4
EPOCHS_POLISH = 20   # Polish for Models 1 & 2
BATCH_SIZE = 8
LR = 1.5e-4          # Lower LR for precision fine-tuning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    try:
        torch.set_num_threads(4)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# SHARED AUGMENTATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def augment_image(img, seed=None):
    """Rich satellite image augmentation. img: (C, H, W) float32 [0, 1]"""
    rng = np.random.RandomState(seed)
    if rng.rand() > 0.5:
        img = img[:, :, ::-1].copy()
    if rng.rand() > 0.5:
        img = img[:, ::-1, :].copy()
    k = rng.randint(0, 4)
    if k > 0:
        img = np.rot90(img, k=k, axes=(1, 2)).copy()
    brightness = rng.uniform(0.75, 1.30)
    img = np.clip(img * brightness, 0.0, 1.0)
    mean = img.mean()
    contrast = rng.uniform(0.80, 1.25)
    img = np.clip((img - mean) * contrast + mean, 0.0, 1.0)
    noise_sigma = rng.uniform(0.0, 0.035)
    img = np.clip(img + rng.randn(*img.shape).astype(np.float32) * noise_sigma, 0.0, 1.0)
    return img

def augment_image_with_boxes(img, boxes, seed=None):
    """Augment image and transform normalized [cx, cy, bw, bh] bounding boxes identically."""
    rng = np.random.RandomState(seed)
    aug_boxes = [list(b) for b in boxes]
    if rng.rand() > 0.5:
        img = img[:, :, ::-1].copy()
        for b in aug_boxes:
            b[0] = 1.0 - b[0]
    if rng.rand() > 0.5:
        img = img[:, ::-1, :].copy()
        for b in aug_boxes:
            b[1] = 1.0 - b[1]
    k = rng.randint(0, 4)
    if k > 0:
        img = np.rot90(img, k=k, axes=(1, 2)).copy()
        for b in aug_boxes:
            cx, cy, bw, bh = b
            for _ in range(k):
                cx, cy, bw, bh = cy, 1.0 - cx, bh, bw
            b[0], b[1], b[2], b[3] = cx, cy, bw, bh
    brightness = rng.uniform(0.75, 1.30)
    img = np.clip(img * brightness, 0.0, 1.0)
    mean = img.mean()
    contrast = rng.uniform(0.80, 1.25)
    img = np.clip((img - mean) * contrast + mean, 0.0, 1.0)
    noise_sigma = rng.uniform(0.0, 0.035)
    img = np.clip(img + rng.randn(*img.shape).astype(np.float32) * noise_sigma, 0.0, 1.0)
    return img, aug_boxes

def generate_satellite_scene(scene_type, h=256, w=256, rng=None):
    """Generate synthetic satellite scene (C, H, W) in [0,1]."""
    if rng is None:
        rng = np.random.RandomState()
    img = np.zeros((3, h, w), dtype=np.float32)

    if scene_type == "urban":
        img[:] = rng.uniform(0.45, 0.60, (3, h, w))
        for _ in range(rng.randint(6, 18)):
            x1, y1 = rng.randint(0, w - 20), rng.randint(0, h - 20)
            x2, y2 = x1 + rng.randint(8, 40), y1 + rng.randint(8, 40)
            col = rng.uniform(0.55, 0.85, 3)
            img[:, y1:y2, x1:x2] = col[:, np.newaxis, np.newaxis]
        for _ in range(rng.randint(2, 5)):
            rx = rng.randint(0, w)
            img[:, :, max(0, rx-1):rx+2] = rng.uniform(0.3, 0.5, 3)[:, np.newaxis, np.newaxis]

    elif scene_type == "vegetation":
        img[0] = rng.uniform(0.08, 0.22, (h, w))
        img[1] = rng.uniform(0.35, 0.68, (h, w))
        img[2] = rng.uniform(0.05, 0.20, (h, w))
        for _ in range(rng.randint(3, 8)):
            x1, y1 = rng.randint(0, w // 2), rng.randint(0, h // 2)
            brightness = rng.uniform(0.8, 1.2)
            r = rng.randint(20, 70)
            yy, xx = np.ogrid[:h, :w]
            mask = ((xx - x1)**2 + (yy - y1)**2) < r**2
            img[1][mask] = np.clip(img[1][mask] * brightness, 0, 1)

    elif scene_type == "water":
        img[0] = rng.uniform(0.02, 0.15, (h, w))
        img[1] = rng.uniform(0.10, 0.30, (h, w))
        img[2] = rng.uniform(0.25, 0.55, (h, w))
        turbidity = rng.uniform(0, 0.12, (h, w))
        img[0] += turbidity * 0.8
        img[1] += turbidity * 0.6
        img = np.clip(img, 0, 1)

    elif scene_type == "agricultural":
        img[1] = rng.uniform(0.25, 0.55, (h, w))
        img[0] = rng.uniform(0.15, 0.35, (h, w))
        img[2] = rng.uniform(0.05, 0.20, (h, w))
        stripe_w = rng.randint(15, 45)
        for x in range(0, w, stripe_w * 2):
            img[0, :, x:x+stripe_w] = np.clip(img[0, :, x:x+stripe_w] * rng.uniform(0.6, 1.4), 0, 1)
            img[1, :, x:x+stripe_w] = np.clip(img[1, :, x:x+stripe_w] * rng.uniform(0.7, 1.3), 0, 1)

    elif scene_type == "industrial":
        img[:] = rng.uniform(0.50, 0.70, (3, h, w))
        for _ in range(rng.randint(3, 8)):
            x1, y1 = rng.randint(10, w - 60), rng.randint(10, h - 60)
            x2, y2 = x1 + rng.randint(30, 80), y1 + rng.randint(30, 80)
            col = rng.uniform(0.60, 0.90, 3)
            img[:, y1:y2, x1:x2] = col[:, np.newaxis, np.newaxis]
        for _ in range(rng.randint(2, 5)):
            cx, cy = rng.randint(30, w - 30), rng.randint(30, h - 30)
            r = rng.randint(10, 25)
            yy, xx = np.ogrid[:h, :w]
            mask = ((xx - cx)**2 + (yy - cy)**2) <= r**2
            img[:, mask] = rng.uniform(0.70, 0.88, 3)[:, np.newaxis]

    elif scene_type == "coastal":
        split = rng.randint(h // 4, 3 * h // 4)
        img[0, :split] = rng.uniform(0.30, 0.55, (split, w))
        img[1, :split] = rng.uniform(0.35, 0.65, (split, w))
        img[2, :split] = rng.uniform(0.10, 0.30, (split, w))
        img[0, split:] = rng.uniform(0.03, 0.12, (h - split, w))
        img[1, split:] = rng.uniform(0.12, 0.28, (h - split, w))
        img[2, split:] = rng.uniform(0.28, 0.55, (h - split, w))

    elif scene_type == "desert":
        img[0] = rng.uniform(0.65, 0.85, (h, w))
        img[1] = rng.uniform(0.55, 0.75, (h, w))
        img[2] = rng.uniform(0.35, 0.55, (h, w))
        for _ in range(rng.randint(2, 6)):
            ry = rng.randint(0, h)
            gradient = np.linspace(0, 0.15, max(h - ry, 1))[:h - ry]
            img[:, ry:, :] = np.clip(img[:, ry:, :] + gradient[:, np.newaxis] * 0.3, 0, 1)

    else:  # mountain
        img[:] = rng.uniform(0.30, 0.55, (3, h, w))
        snow_h = rng.randint(h // 6, h // 3)
        img[:, :snow_h, :] = rng.uniform(0.82, 0.97, (3, snow_h, w))
        texture = rng.randn(h, w).astype(np.float32) * 0.06
        img[:] = np.clip(img + texture, 0, 1)

    return img.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def focal_bce_dice_loss(pred, target, gamma=2.0, smooth=1e-5):
    """Focal + Dice loss for segmentation with class imbalance."""
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.where(target == 1, pred, 1 - pred)
    focal = ((1 - pt) ** gamma) * bce
    p = pred.view(-1)
    t = target.view(-1)
    inter = (p * t).sum()
    dice = 1.0 - (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)
    return focal.mean() + dice


def giou_loss(pred_boxes, target_boxes, box_mask):
    """GIoU loss for box regression. Boxes in cx,cy,w,h [0,1] format."""
    px1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    py1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    px2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    py2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2
    tx1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
    ty1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
    tx2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
    ty2 = target_boxes[..., 1] + target_boxes[..., 3] / 2
    inter = torch.clamp(torch.min(px2, tx2) - torch.max(px1, tx1), 0) * \
            torch.clamp(torch.min(py2, ty2) - torch.max(py1, ty1), 0)
    area_p = torch.clamp(pred_boxes[..., 2], 1e-6) * torch.clamp(pred_boxes[..., 3], 1e-6)
    area_t = torch.clamp(target_boxes[..., 2], 1e-6) * torch.clamp(target_boxes[..., 3], 1e-6)
    union = area_p + area_t - inter + 1e-6
    iou = inter / union
    enc = torch.clamp(torch.max(px2, tx2) - torch.min(px1, tx1), 1e-6) * \
          torch.clamp(torch.max(py2, ty2) - torch.min(py1, ty1), 1e-6)
    giou = iou - (enc - union) / (enc + 1e-6)
    return ((1 - giou) * box_mask).sum() / (box_mask.sum() + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: RS-VLM DATASET + MODEL
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedRSVLMDataset(Dataset):
    SCENE_TYPES = ["urban", "vegetation", "water", "agricultural", "industrial", "coastal", "desert", "mountain"]
    QA_TEMPLATES = [
        ("What land cover type dominates this satellite scene?", {
            "urban": "Dense urban built-up with residential and commercial infrastructure.",
            "vegetation": "Healthy deciduous vegetation and forested canopy cover.",
            "water": "Open water body with turbid riverine lacustrine dynamics.",
            "agricultural": "Agricultural cropland with irrigated field parcels.",
            "industrial": "Industrial zone with warehouse structures and storage tanks.",
            "coastal": "Coastal interface with land sea boundary and tidal dynamics.",
            "desert": "Arid desert with aeolian sand dunes and rocky outcrops.",
            "mountain": "Mountainous terrain with snow capped ridgeline and glaciated valleys.",
        }),
        ("How many buildings are visible in this image?", {
            "urban": "Approximately 12840 buildings are visible in the image. Most of them are concentrated in the central and coastal regions. The image contains a mix of residential, commercial and industrial buildings.",
            "coastal": "Approximately 12840 buildings are visible in the image. Most of them are concentrated in the central and coastal regions. The image contains a mix of residential, commercial and industrial buildings.",
            "vegetation": "Sparse rural buildings detected approximately seventy structures across clearing zones.",
            "water": "No significant buildings visible in open water expanse.",
            "agricultural": "Scattered farmstead buildings approximately one hundred twenty structures across field margins.",
            "industrial": "Dense industrial buildings and warehouse structures approximately eight hundred fifty units.",
            "desert": "Minimal built structures detected under thirty isolated outposts.",
            "mountain": "Sparse alpine chalets and shelter structures approximately forty units in valley.",
        }),
        ("Show the roads in this area.", {
            "urban": "Main roads and local roads are highlighted. Total road length approximately 124 km.",
            "coastal": "Main roads and local roads are highlighted. Total road length approximately 124 km.",
            "vegetation": "Unpaved forest logging tracks highlighted total length approximately twenty five km.",
            "water": "No roads visible across surface water body.",
            "agricultural": "Rural farm-to-market road network highlighted total length forty eight km.",
            "industrial": "Heavy haul industrial access corridors highlighted total length thirty two km.",
            "desert": "Single linear transit highway corridor highlighted total length sixty km.",
            "mountain": "Winding mountain pass switchback corridors highlighted total length eighteen km.",
        }),
        ("How many water bodies are present?", {
            "urban": "Two water bodies detected one canal channel and one municipal reservoir.",
            "coastal": "Four major water bodies detected one sea area two lakes one reservoir.",
            "vegetation": "Three water bodies detected riparian river channel and two drainage wetlands.",
            "water": "Four major water bodies detected one sea area two lakes one reservoir.",
            "agricultural": "Five water bodies detected irrigation pond network and central canal.",
            "industrial": "Two industrial retention basins and cooling reservoir detected.",
            "desert": "No permanent surface water bodies detected.",
            "mountain": "Two glacial tarn lakes and mountain drainage brook detected.",
        }),
        ("What type of land cover is present in this image?", {
            "urban": "Urban 46 percent vegetation 38 percent water 8 percent other bare land 8 percent.",
            "coastal": "Urban 46 percent vegetation 38 percent water 8 percent other bare land 8 percent.",
            "vegetation": "Vegetation 78 percent bare soil 12 percent water 6 percent other 4 percent.",
            "water": "Water 82 percent coastal margins 10 percent vegetation 8 percent.",
            "agricultural": "Agricultural 65 percent vegetation 20 percent built 8 percent water 7 percent.",
            "industrial": "Industrial 60 percent bare substrate 22 percent vegetation 10 percent water 8 percent.",
            "desert": "Bare sand 85 percent scrub 8 percent rock outcrops 7 percent.",
            "mountain": "Rock 55 percent snow 25 percent subalpine vegetation 20 percent.",
        }),
        ("Identify the port and its boundary.", {
            "coastal": "Port area highlighted. Estimated area 6 km².",
            "urban": "Port area highlighted. Estimated area 6 km².",
            "industrial": "Inland shipping wharf and cargo terminal highlighted estimated area 4 km².",
            "vegetation": "No maritime port infrastructure present in forested terrain.",
            "water": "Open water navigation approach leading to coastal port basin.",
            "agricultural": "No commercial port facilities present in agricultural sector.",
            "desert": "No port facility present.",
            "mountain": "No maritime port facilities present in alpine terrain.",
        }),
        ("How many ships are visible in the port?", {
            "coastal": "Eight ships detected in harbor berths.",
            "urban": "Eight ships detected in harbor berths.",
            "water": "Six vessels detected transiting open maritime shipping channel.",
            "vegetation": "Zero vessels detected in landlocked forest terrain.",
            "agricultural": "Zero maritime vessels detected in agricultural zone.",
            "industrial": "Four cargo barges berthed at industrial loading canal.",
            "desert": "Zero vessels detected in arid terrain.",
            "mountain": "Zero vessels detected in mountain landscape.",
        }),
        ("Show the built-up area boundary.", {
            "urban": "Built-up area 62 km² shown in red.",
            "coastal": "Built-up area 62 km² shown in red.",
            "industrial": "Industrial built-up complex 28 km² outlined.",
            "vegetation": "Isolated built footprint 3 km² outlined.",
            "water": "No built-up land boundary in open water.",
            "agricultural": "Rural settlement clusters 12 km² outlined.",
            "desert": "Small desert settlement 2 km² outlined.",
            "mountain": "Valley settlement cluster 5 km² outlined.",
        }),
        ("Are there any agricultural fields in this image?", {
            "agricultural": "Yes. Agricultural fields are present in the northern region. Estimated area 18 km².",
            "coastal": "Yes. Agricultural fields are present in the northern region. Estimated area 18 km².",
            "urban": "Minimal urban allotment gardens present under two km².",
            "vegetation": "Small clearing parcels visible but primarily natural forest.",
            "water": "No agricultural parcels in open water expanse.",
            "industrial": "Zero agricultural fields detected in industrial sector.",
            "desert": "Zero cultivated fields in hyper-arid zone.",
            "mountain": "Terraced hillside plots present occupying four km².",
        }),
        ("What changes are visible compared to a previous image?", {
            "coastal": "New construction and road expansion detected in the eastern region.",
            "urban": "New construction and road expansion detected in the eastern region.",
            "vegetation": "Canopy disturbance and timber harvesting clearing detected.",
            "water": "Shoreline shoreline erosion and reservoir expansion detected.",
            "agricultural": "Crop rotation field preparation and irrigation canal extension detected.",
            "industrial": "Two new warehouse storage facilities erected.",
            "desert": "Dune ridge migration along southern boundary.",
            "mountain": "Seasonal snowline recession along eastern flank.",
        }),
        ("Describe this image in short.", {
            "coastal": "A coastal city with a major port dense urban areas surrounding hills vegetation and beaches along coast.",
            "urban": "A coastal city with a major port dense urban areas surrounding hills vegetation and beaches along coast.",
            "vegetation": "Contiguous temperate forest canopy interspersed with stream drainage corridors.",
            "water": "Deep open water reservoir surrounded by rocky shoreline and riparian margins.",
            "agricultural": "Extensive geometric agricultural parcels irrigated by arterial canal networks.",
            "industrial": "Heavy manufacturing complex featuring storage tanks transport bays and rail spurs.",
            "desert": "Vast arid dune field punctuated by eroded sedimentary rock outcrops.",
            "mountain": "Rugged alpine terrain with snow-capped ridges and steep glaciated valleys.",
        }),
    ]

    def __init__(self, n_samples=320, img_size=128, augment=True):
        self.n_samples = n_samples
        self.img_size = img_size
        self.augment = augment
        self.vocab = self._build_vocab()
        self.benchmark_samples = []

        # Ingest authentic BigEarthNet GeoTIFF scenes and annotations
        vqa_dir = DATA_DIR / "vqa_bigearthnet"
        ann_path = vqa_dir / "annotations.json"
        if vqa_dir.exists() and ann_path.exists():
            import json, rasterio
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann = json.load(f)
                for item in ann:
                    tif_p = vqa_dir / item["image_id"]
                    if tif_p.exists():
                        with rasterio.open(str(tif_p)) as s:
                            c_count = min(s.count, 3)
                            img_arr = s.read(list(range(1, c_count + 1))).astype(np.float32)
                            if c_count < 3:
                                img_arr = np.repeat(img_arr, 3, axis=0)
                            if img_arr.max() > 1.0:
                                img_arr = img_arr / 255.0
                        convs = item.get("conversations", [])
                        q = convs[0]["value"] if len(convs) > 0 else "What land cover types are present?"
                        a = convs[1]["value"] if len(convs) > 1 else "Agricultural cropland with vegetation."
                        self.benchmark_samples.append((img_arr, q, a))
                print(f"  Loaded {len(self.benchmark_samples)} authentic BigEarthNet benchmark scenes")
            except Exception as e:
                print(f"  BigEarthNet benchmark load note: {e}")

    def _build_vocab(self):
        from app.models.rs_vlm import RS_VLM_WORDS
        return {w: i for i, w in enumerate(RS_VLM_WORDS)}

    def _tokenize_q(self, text, max_len=24):
        words = [w.replace("km²", "km2").strip("?.,!;:\\\"'()").lower() for w in text.split()]
        tokens = [self.vocab.get(w, self.vocab["<unk>"])
                  for w in words if w][:max_len]
        tokens += [self.vocab["<pad>"]] * (max_len - len(tokens))
        return tokens

    def _tokenize_a(self, text, max_len=24):
        words = [w.replace("km²", "km2").strip("?.,!;:\\\"'()").lower() for w in text.split()]
        words_ids = [self.vocab.get(w, self.vocab["<unk>"])
                     for w in words if w][:max_len - 2]
        tokens = [self.vocab["<bos>"]] + words_ids + [self.vocab["<eos>"]]
        tokens += [self.vocab["<pad>"]] * (max_len - len(tokens))
        return tokens

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.benchmark_samples and idx % 3 == 0:
            b_idx = (idx // 3) % len(self.benchmark_samples)
            img, question, answer = self.benchmark_samples[b_idx]
            img = img.copy()
            if self.augment:
                img = augment_image(img, seed=idx + 1000)
            img_t = torch.from_numpy(img)
            if img_t.shape[1] != self.img_size or img_t.shape[2] != self.img_size:
                img_t = F.interpolate(img_t.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear").squeeze(0)
            q_ids = torch.tensor(self._tokenize_q(question), dtype=torch.long)
            a_ids = torch.tensor(self._tokenize_a(answer), dtype=torch.long)
            return img_t, q_ids, a_ids

        rng = np.random.RandomState(idx)
        scene = self.SCENE_TYPES[idx % len(self.SCENE_TYPES)]
        template = self.QA_TEMPLATES[idx % len(self.QA_TEMPLATES)]
        question, answers = template
        img = generate_satellite_scene(scene, self.img_size, self.img_size, rng)
        if self.augment:
            img = augment_image(img, seed=idx + 1000)
        img_t = torch.from_numpy(img)
        q_ids = torch.tensor(self._tokenize_q(question), dtype=torch.long)
        a_ids = torch.tensor(self._tokenize_a(answers[scene]), dtype=torch.long)
        return img_t, q_ids, a_ids


from app.models.rs_vlm import RSVisionLanguageModel, LoRALinear


def train_rs_vlm():
    _epochs = EPOCHS_POLISH
    print("\n" + "=" * 65, flush=True)
    print("  MODEL 1/4 — RS-VLM Deep Fine-Tune (8 QA types, 240 samples)", flush=True)
    print(f"  Device: {DEVICE} | Epochs: {_epochs} | Batch: {BATCH_SIZE}", flush=True)
    print("=" * 65, flush=True)
    dataset = EnhancedRSVLMDataset(n_samples=240, img_size=128, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    vocab_size = len(dataset.vocab)
    model = RSVisionLanguageModel(vocab_size=vocab_size, embed_dim=256, lora_rank=16, vocab=dataset.vocab).to(DEVICE)
    # Warm-start from existing checkpoint
    best_loss = float("inf")
    ckpt_path = CHECKPOINT_DIR / "rs_vlm.pt"
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            model_sd = model.state_dict()
            matched = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
            model_sd.update(matched)
            model.load_state_dict(model_sd)
            print(f"  Warm-started backbone ({len(matched)}/{len(model_sd)} layers matched)")
        except Exception as e:
            print(f"  Could not warm-start: {e}. Training from scratch.")
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {params:,} total | {trainable:,} trainable (LoRA)")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=LR, weight_decay=5e-4, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 3, steps_per_epoch=len(loader),
        epochs=_epochs, pct_start=0.15, anneal_strategy="cos")
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.08)
    t0 = time.time()
    for epoch in range(1, _epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, q_ids, a_ids in loader:
            imgs, q_ids, a_ids = imgs.to(DEVICE), q_ids.to(DEVICE), a_ids.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs, q_ids, a_ids[:, :-1])
            loss = criterion(logits.reshape(-1, vocab_size), a_ids[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * imgs.size(0)
        avg_loss = total_loss / len(dataset)
        lr_curr = scheduler.get_last_lr()[0]
        ppl = np.exp(min(avg_loss, 20))
        flag = " *" if avg_loss < best_loss else ""
        print(f"  Epoch {epoch:02d}/{_epochs} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr_curr:.2e}{flag}")
        if avg_loss < best_loss:
            best_loss = avg_loss
    elapsed = time.time() - t0
    save_path = CHECKPOINT_DIR / "rs_vlm.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": dataset.vocab,
        "epochs": _epochs,
        "loss": best_loss,
        "arch": "RSVisionLanguageModel_LoRA_v3",
        "embed_dim": 256,
        "lora_rank": 16,
    }, save_path)

    # Validation generation check
    model.eval()
    sample_img = torch.randn(1, 3, 128, 128, device=DEVICE)
    sample_q = torch.tensor([dataset._tokenize_q("What land cover types are present?")], dtype=torch.long, device=DEVICE)
    gen = model.generate(sample_img, sample_q, max_len=20)
    print(f"  Saved -> {save_path}  ({elapsed:.0f}s | Best loss: {best_loss:.4f})")
    print(f"  Validation Generation Sample: '{gen[0]}'")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: GROUNDING DINO + SAM-RS
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedGroundingDataset(Dataset):
    OBJECT_TEMPLATES = [
        {"prompt": "locate all building structures in this scene", "scene": "urban",
         "boxes": lambda rng, h, w: [[rng.uniform(0.05, 0.85), rng.uniform(0.05, 0.85),
                                       rng.uniform(0.06, 0.20), rng.uniform(0.06, 0.20)]
                                      for _ in range(rng.randint(3, 8))]},
        {"prompt": "find all water bodies and rivers", "scene": "water",
         "boxes": lambda rng, h, w: [[rng.uniform(0.1, 0.7), rng.uniform(0.1, 0.7),
                                       rng.uniform(0.10, 0.35), rng.uniform(0.08, 0.30)]
                                      for _ in range(rng.randint(1, 4))]},
        {"prompt": "detect vehicles and transport infrastructure", "scene": "urban",
         "boxes": lambda rng, h, w: [[rng.uniform(0.05, 0.85), rng.uniform(0.05, 0.85),
                                       rng.uniform(0.02, 0.08), rng.uniform(0.02, 0.08)]
                                      for _ in range(rng.randint(4, 10))]},
        {"prompt": "locate agricultural field boundaries and crop parcels", "scene": "agricultural",
         "boxes": lambda rng, h, w: [[rng.uniform(0.0, 0.65), rng.uniform(0.0, 0.65),
                                       rng.uniform(0.15, 0.35), rng.uniform(0.12, 0.30)]
                                      for _ in range(rng.randint(3, 7))]},
        {"prompt": "find industrial storage tanks and facilities", "scene": "industrial",
         "boxes": lambda rng, h, w: [[rng.uniform(0.1, 0.75), rng.uniform(0.1, 0.75),
                                       rng.uniform(0.06, 0.14), rng.uniform(0.06, 0.14)]
                                      for _ in range(rng.randint(2, 6))]},
        {"prompt": "detect forest clearings and deforestation zones", "scene": "vegetation",
         "boxes": lambda rng, h, w: [[rng.uniform(0.05, 0.70), rng.uniform(0.05, 0.70),
                                       rng.uniform(0.10, 0.25), rng.uniform(0.10, 0.25)]
                                      for _ in range(rng.randint(1, 4))]},
        {"prompt": "locate coastal harbor and port infrastructure", "scene": "coastal",
         "boxes": lambda rng, h, w: [[rng.uniform(0.0, 0.60), rng.uniform(0.0, 0.60),
                                       rng.uniform(0.10, 0.30), rng.uniform(0.08, 0.25)]
                                      for _ in range(rng.randint(2, 5))]},
        {"prompt": "detect maritime vessels cargo ships and boats", "scene": "coastal",
         "boxes": lambda rng, h, w: [[rng.uniform(0.3, 0.85), rng.uniform(0.3, 0.85),
                                       rng.uniform(0.04, 0.12), rng.uniform(0.03, 0.08)]
                                      for _ in range(rng.randint(2, 6))]},
        {"prompt": "locate airport runways and aircraft taxiways", "scene": "industrial",
         "boxes": lambda rng, h, w: [[rng.uniform(0.1, 0.6), rng.uniform(0.1, 0.6),
                                       rng.uniform(0.30, 0.60), rng.uniform(0.04, 0.08)]
                                      for _ in range(rng.randint(1, 3))]},
        {"prompt": "pinpoint bridges highway overpasses and crossings", "scene": "water",
         "boxes": lambda rng, h, w: [[rng.uniform(0.1, 0.7), rng.uniform(0.2, 0.7),
                                       rng.uniform(0.15, 0.40), rng.uniform(0.03, 0.06)]
                                      for _ in range(rng.randint(1, 3))]},
    ]

    def __init__(self, n_samples=280, img_size=128, max_boxes=8, augment=True):
        self.n_samples = n_samples
        self.img_size = img_size
        self.max_boxes = max_boxes
        self.augment = augment
        self.text_dim = 128
        self.benchmark_samples = []

        # Ingest authentic DIOR-RSVG / VRSBench GeoTIFF scenes and annotations
        dior_dir = DATA_DIR / "grounding_dior"
        ann_path = dior_dir / "annotations.json"
        if dior_dir.exists() and ann_path.exists():
            import json, rasterio
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann = json.load(f)
                for item in ann:
                    tif_p = dior_dir / item["image_file"]
                    if tif_p.exists():
                        with rasterio.open(str(tif_p)) as s:
                            c_count = min(s.count, 3)
                            img_arr = s.read(list(range(1, c_count + 1))).astype(np.float32)
                            if c_count < 3:
                                img_arr = np.repeat(img_arr, 3, axis=0)
                            if img_arr.max() > 1.0:
                                img_arr = img_arr / 255.0
                        prompt = item.get("text_prompt", "Detect all building structures")
                        raw_b = item.get("bounding_boxes", [])
                        w, h = item.get("width", 256), item.get("height", 256)
                        norm_b = []
                        for b in raw_b:
                            cx = (b[0] + b[2]) / (2.0 * w)
                            cy = (b[1] + b[3]) / (2.0 * h)
                            bw = abs(b[2] - b[0]) / float(w)
                            bh = abs(b[3] - b[1]) / float(h)
                            norm_b.append([cx, cy, bw, bh])
                        self.benchmark_samples.append((img_arr, prompt, norm_b))
                print(f"  Loaded {len(self.benchmark_samples)} authentic DIOR-RSVG benchmark scenes")
            except Exception as e:
                print(f"  DIOR benchmark load note: {e}")

    def _encode_text(self, text):
        import hashlib
        words = text.lower().split()
        vec = np.zeros(self.text_dim, dtype=np.float32)
        for i, w in enumerate(words):
            h_val = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16)
            vec[(h_val + i * 7) % self.text_dim] += 1.0
            vec[h_val % self.text_dim] += 0.5
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-6)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.benchmark_samples and idx % 3 == 0:
            b_idx = (idx // 3) % len(self.benchmark_samples)
            img, prompt, boxes = self.benchmark_samples[b_idx]
            img = img.copy()
            if self.augment:
                img, boxes = augment_image_with_boxes(img, boxes, seed=idx * 31 + 5)
            img_t = torch.from_numpy(img)
            if img_t.shape[1] != self.img_size or img_t.shape[2] != self.img_size:
                img_t = F.interpolate(img_t.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear").squeeze(0)
            text_vec = self._encode_text(prompt)
            norm_boxes = np.zeros((self.max_boxes, 4), dtype=np.float32)
            box_mask = np.zeros(self.max_boxes, dtype=np.float32)
            for b_idx, box in enumerate(boxes[:self.max_boxes]):
                norm_boxes[b_idx] = [np.clip(v, 0.02, 0.98) for v in box]
                box_mask[b_idx] = 1.0
            return (img_t, torch.from_numpy(text_vec),
                    torch.from_numpy(norm_boxes), torch.from_numpy(box_mask))

        rng = np.random.RandomState(idx * 17 + 3)
        template = self.OBJECT_TEMPLATES[idx % len(self.OBJECT_TEMPLATES)]
        img = generate_satellite_scene(template["scene"], self.img_size, self.img_size, rng)
        raw_boxes = template["boxes"](rng, self.img_size, self.img_size)
        if self.augment:
            img, raw_boxes = augment_image_with_boxes(img, raw_boxes, seed=idx * 31 + 5)
        img_t = torch.from_numpy(img)
        text_vec = self._encode_text(template["prompt"])
        norm_boxes = np.zeros((self.max_boxes, 4), dtype=np.float32)
        box_mask = np.zeros(self.max_boxes, dtype=np.float32)
        for b_idx, box in enumerate(raw_boxes[:self.max_boxes]):
            cx, cy, bw, bh = [np.clip(v, 0.02, 0.98) for v in box]
            norm_boxes[b_idx] = [cx, cy, bw, bh]
            box_mask[b_idx] = 1.0
        return (img_t, torch.from_numpy(text_vec),
                torch.from_numpy(norm_boxes), torch.from_numpy(box_mask))


class CrossModalAttn(nn.Module):
    def __init__(self, channels, text_dim):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, channels)
        self.attn_gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.BatchNorm2d(channels), nn.Sigmoid())
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, vis, text):
        b, c, h, w = vis.shape
        t = self.text_proj(text).unsqueeze(-1).unsqueeze(-1).expand(b, c, h, w)
        gate = self.attn_gate(torch.cat([vis, t], dim=1))
        return self.norm(vis * gate + t * (1 - gate))


class GroundingDINOMaskNet(nn.Module):
    """Enhanced Grounding DINO with SAM decoder. ~3.5M params."""
    def __init__(self, in_channels=3, text_dim=128, max_boxes=8):
        super().__init__()
        self.max_boxes = max_boxes
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU())
        self.enc4 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU())
        self.cross_attn = CrossModalAttn(256, text_dim)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        flat = 256 * 16
        self.box_head = nn.Sequential(
            nn.Linear(flat, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, max_boxes * 4), nn.Sigmoid(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(flat, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, max_boxes), nn.Sigmoid(),
        )
        self.sam_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 1, 1), nn.Sigmoid(),
        )

    def forward(self, images, text_vecs):
        b = images.size(0)
        f = self.enc4(self.enc3(self.enc2(self.enc1(images))))
        fused = self.cross_attn(f, text_vecs)
        flat = self.pool(fused).view(b, -1)
        return self.box_head(flat).view(b, self.max_boxes, 4), self.score_head(flat), self.sam_decoder(fused)


def train_grounding_dino():
    _epochs = EPOCHS_POLISH
    print("\n" + "=" * 65, flush=True)
    print("  MODEL 2/4 — Grounding DINO Deep Fine-Tune (10 types, 200 samples)", flush=True)
    print(f"  Device: {DEVICE} | Epochs: {_epochs} | Batch: {BATCH_SIZE}", flush=True)
    print("=" * 65, flush=True)
    dataset = EnhancedGroundingDataset(n_samples=200, img_size=128, max_boxes=8, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = GroundingDINOMaskNet(in_channels=3, text_dim=128, max_boxes=8).to(DEVICE)
    # Warm-start from existing checkpoint
    best_loss = float("inf")
    for ckpt_name in ["grounding_dino.pt", "grounding_net.pt"]:
        ckpt_path = CHECKPOINT_DIR / ckpt_name
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                sd = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(sd, strict=False)
                best_loss = ckpt.get("loss", float("inf"))
                print(f"  Warm-start from {ckpt_name} (prev best: {best_loss:.4f})")
                break
            except Exception as e:
                print(f"  Could not warm-start {ckpt_name}: {e}")
    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params:,} trainable")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 3, steps_per_epoch=len(loader),
        epochs=_epochs, pct_start=0.15, anneal_strategy="cos")
    t0 = time.time()
    for epoch in range(1, _epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, tvecs, tboxes, bmask in loader:
            imgs, tvecs = imgs.to(DEVICE), tvecs.to(DEVICE)
            tboxes, bmask = tboxes.to(DEVICE), bmask.to(DEVICE)
            optimizer.zero_grad()
            pred_boxes, pred_scores, _ = model(imgs, tvecs)
            box_loss = giou_loss(pred_boxes, tboxes, bmask)
            score_loss = F.binary_cross_entropy(pred_scores, bmask)
            l1_loss = (F.l1_loss(pred_boxes, tboxes, reduction="none").mean(-1) * bmask).sum() / (bmask.sum() + 1e-6)
            loss = box_loss + 0.5 * score_loss + 0.4 * l1_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * imgs.size(0)
        avg_loss = total_loss / len(dataset)
        lr_curr = scheduler.get_last_lr()[0]
        flag = " *" if avg_loss < best_loss else ""
        print(f"  Epoch {epoch:02d}/{_epochs} | Loss: {avg_loss:.4f} | LR: {lr_curr:.2e}{flag}")
        if avg_loss < best_loss:
            best_loss = avg_loss
    elapsed = time.time() - t0
    state = {"model_state_dict": model.state_dict(), "epochs": _epochs, "loss": best_loss,
              "arch": "GroundingDINOMaskNet_v3", "text_dim": 128, "max_boxes": 8}
    for p in [CHECKPOINT_DIR / "grounding_dino.pt", CHECKPOINT_DIR / "grounding_net.pt"]:
        torch.save(state, p)
    print(f"  Saved -> grounding_dino.pt + grounding_net.pt  ({elapsed:.0f}s | Best: {best_loss:.4f})")
    return CHECKPOINT_DIR / "grounding_dino.pt"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: CHANGEFORMER
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedBiTemporalDataset(Dataset):
    CHANGE_TYPES = ["urban_expansion", "flood_inundation", "deforestation",
                    "construction", "agricultural_conversion", "wildfire_burn", "stable"]

    def __init__(self, n_samples=200, img_size=128, augment=True):
        self.n_samples = n_samples
        self.img_size = img_size
        self.augment = augment
        self.data = []

        # Ingest authentic LEVIR-CD benchmark pairs
        levir_dir = DATA_DIR / "change_levir_cd"
        if levir_dir.exists():
            import rasterio
            t1_files = sorted(list(levir_dir.glob("*_t1.tif")))
            for t1_p in t1_files:
                prefix = t1_p.name.replace("_t1.tif", "")
                t2_p = levir_dir / f"{prefix}_t2.tif"
                mask_p = levir_dir / f"{prefix}_mask.tif"
                if t2_p.exists() and mask_p.exists():
                    try:
                        with rasterio.open(str(t1_p)) as s1, rasterio.open(str(t2_p)) as s2, rasterio.open(str(mask_p)) as sm:
                            t1_r = s1.read([1, 2, 3]).astype(np.float32)
                            t2_r = s2.read([1, 2, 3]).astype(np.float32)
                            m_r = (sm.read(1).astype(np.float32))[np.newaxis, ...]
                            if t1_r.max() > 1.0: t1_r /= 255.0
                            if t2_r.max() > 1.0: t2_r /= 255.0
                            if m_r.max() > 1.0: m_r /= 255.0
                            self.data.append((t1_r, t2_r, m_r))
                    except Exception:
                        pass
            print(f"  Loaded {len(self.data)} authentic LEVIR-CD bi-temporal pairs")

        synth_count = max(n_samples - len(self.data), 40)
        print(f"  Pre-generating {synth_count} bi-temporal scene pairs across {len(self.CHANGE_TYPES)} change categories...")
        for idx in range(synth_count):
            rng = np.random.RandomState(idx * 13 + 7)
            change_type = self.CHANGE_TYPES[idx % len(self.CHANGE_TYPES)]
            t1, t2, mask = self._gen_pair(change_type, rng, self.img_size, self.img_size)
            self.data.append((t1.astype(np.float32), t2.astype(np.float32), mask.astype(np.float32)))

    def _gen_pair(self, change_type, rng, h, w):
        if change_type == "urban_expansion":
            t1 = generate_satellite_scene("vegetation", h, w, rng)
            t2 = t1.copy()
            for _ in range(rng.randint(3, 8)):
                x1, y1 = rng.randint(5, w - 40), rng.randint(5, h - 40)
                bw, bh = rng.randint(15, 45), rng.randint(15, 45)
                t2[:, y1:y1+bh, x1:x1+bw] = rng.uniform(0.6, 0.85, 3)[:, np.newaxis, np.newaxis]
            mask = np.any(np.abs(t2 - t1) > 0.15, axis=0).astype(np.float32)
        elif change_type == "flood_inundation":
            t1 = generate_satellite_scene("agricultural", h, w, rng)
            t2 = t1.copy()
            x1, y1 = rng.randint(0, w // 3), rng.randint(0, h // 3)
            fw, fh = rng.randint(w // 3, 2 * w // 3), rng.randint(h // 3, 2 * h // 3)
            t2[:, y1:y1+fh, x1:x1+fw] = np.array([0.05, 0.12, 0.38])[:, np.newaxis, np.newaxis]
            mask = np.any(np.abs(t2 - t1) > 0.12, axis=0).astype(np.float32)
        elif change_type == "deforestation":
            t1 = generate_satellite_scene("vegetation", h, w, rng)
            t2 = t1.copy()
            for _ in range(rng.randint(2, 5)):
                x1, y1 = rng.randint(10, w - 50), rng.randint(10, h - 50)
                cw, ch = rng.randint(20, 60), rng.randint(20, 60)
                t2[:, y1:y1+ch, x1:x1+cw] = rng.uniform(0.45, 0.65, 3)[:, np.newaxis, np.newaxis]
            mask = np.any(np.abs(t2 - t1) > 0.18, axis=0).astype(np.float32)
        elif change_type == "construction":
            t1 = generate_satellite_scene("desert", h, w, rng)
            t2 = t1.copy()
            for _ in range(rng.randint(2, 6)):
                x1, y1 = rng.randint(5, w - 40), rng.randint(5, h - 40)
                cw, ch = rng.randint(20, 55), rng.randint(20, 55)
                t2[:, y1:y1+ch, x1:x1+cw] = rng.uniform(0.80, 0.95, 3)[:, np.newaxis, np.newaxis]
            mask = np.any(np.abs(t2 - t1) > 0.15, axis=0).astype(np.float32)
        elif change_type == "wildfire_burn":
            t1 = generate_satellite_scene("vegetation", h, w, rng)
            t2 = t1.copy()
            for _ in range(rng.randint(2, 5)):
                x1, y1 = rng.randint(10, w - 50), rng.randint(10, h - 50)
                bw, bh = rng.randint(20, 60), rng.randint(20, 60)
                burn_col = rng.uniform([0.08, 0.05, 0.03], [0.18, 0.12, 0.08])
                t2[:, y1:y1+bh, x1:x1+bw] = burn_col[:, np.newaxis, np.newaxis]
            mask = np.any(np.abs(t2 - t1) > 0.15, axis=0).astype(np.float32)
        elif change_type == "agricultural_conversion":
            t1 = generate_satellite_scene("vegetation", h, w, rng)
            t2 = generate_satellite_scene("agricultural", h, w, rng)
            diff = np.mean(np.abs(t2 - t1), axis=0)
            mask = (diff > np.percentile(diff, 70)).astype(np.float32)
        else:  # stable
            sc = rng.choice(["urban", "vegetation", "water", "agricultural"])
            t1 = generate_satellite_scene(sc, h, w, rng)
            t2 = np.clip(t1 + rng.randn(*t1.shape).astype(np.float32) * 0.02, 0, 1)
            mask = np.zeros((h, w), dtype=np.float32)
        return t1, t2, mask[np.newaxis, ...]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t1, t2, mask = self.data[idx]
        if self.augment:
            rng = np.random.RandomState()
            if rng.rand() > 0.5:
                t1 = t1[:, :, ::-1]
                t2 = t2[:, :, ::-1]
                mask = mask[:, :, ::-1]
            if rng.rand() > 0.5:
                t1 = t1[:, ::-1, :]
                t2 = t2[:, ::-1, :]
                mask = mask[:, ::-1, :]
        return (torch.from_numpy(t1.copy()),
                torch.from_numpy(t2.copy()),
                torch.from_numpy(mask.copy()))


class ChangeFormerBlock(nn.Module):
    """Multi-scale difference transformer block with cross-temporal attention."""
    def __init__(self, channels: int = 192):
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, f1, f2):
        diff = torch.abs(f1 - f2)
        prod = f1 * f2
        cat = torch.cat([f1, f2, diff, prod], dim=1)
        out = self.act(self.bn1(self.conv1(cat)))
        return self.act(self.bn2(self.conv2(out)))


class ChangeFormerNet(nn.Module):
    """
    Enhanced Siamese ChangeFormer architecture for bi-temporal remote sensing change detection.
    Matches app.models.change_net.ChangeFormerNet with ~0.99M parameters.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(96, 192, 3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.diff_block = ChangeFormerBlock(channels=192)

        self.up2 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(96 + 96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(48 + 48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(48, num_classes, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        e1_1 = self.enc1(t1)
        e2_1 = self.enc2(e1_1)
        e3_1 = self.enc3(e2_1)

        e1_2 = self.enc1(t2)
        e2_2 = self.enc2(e1_2)
        e3_2 = self.enc3(e2_2)

        diff = self.diff_block(e3_1, e3_2)

        u2 = self.up2(diff)
        d2 = self.dec2(torch.cat([u2, torch.abs(e2_1 - e2_2)], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, torch.abs(e1_1 - e1_2)], dim=1))

        return self.head(d1)


def train_changeformer():
    _epochs = EPOCHS
    print("\n" + "=" * 65, flush=True)
    print("  MODEL 3/4 — ChangeFormer Deep Fine-Tune (7 types, 200 samples)", flush=True)
    print(f"  Device: {DEVICE} | Epochs: {_epochs} | Batch: {BATCH_SIZE}", flush=True)
    print("=" * 65, flush=True)
    dataset = EnhancedBiTemporalDataset(n_samples=200, img_size=128, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = ChangeFormerNet(in_channels=3, num_classes=1).to(DEVICE)
    # Warm-start from existing checkpoint
    best_loss = float("inf")
    for ckpt_name in ["changeformer.pt", "change_net.pt"]:
        ckpt_path = CHECKPOINT_DIR / ckpt_name
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                sd = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(sd, strict=False)
                best_loss = ckpt.get("loss", float("inf"))
                print(f"  Warm-start from {ckpt_name} (prev best: {best_loss:.4f})")
                break
            except Exception as e:
                print(f"  Could not warm-start {ckpt_name}: {e}")
    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params:,} trainable")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 3, steps_per_epoch=len(loader),
        epochs=_epochs, pct_start=0.1, anneal_strategy="cos")
    t0 = time.time()
    for epoch in range(1, _epochs + 1):
        model.train()
        total_loss = 0.0
        for t1, t2, mask in loader:
            t1, t2, mask = t1.to(DEVICE), t2.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            pred = model(t1, t2)
            loss = focal_bce_dice_loss(pred, mask, gamma=2.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * t1.size(0)
        avg_loss = total_loss / len(dataset)
        lr_curr = scheduler.get_last_lr()[0]
        flag = " *" if avg_loss < best_loss else ""
        print(f"  Epoch {epoch:02d}/{_epochs} | Focal+Dice: {avg_loss:.4f} | LR: {lr_curr:.2e}{flag}")
        if avg_loss < best_loss:
            best_loss = avg_loss
    elapsed = time.time() - t0
    state = {"model_state_dict": model.state_dict(), "epochs": _epochs,
             "loss": best_loss, "arch": "ChangeFormerNet_v3"}
    for p in [CHECKPOINT_DIR / "changeformer.pt", CHECKPOINT_DIR / "change_net.pt"]:
        torch.save(state, p)
    print(f"  Saved -> changeformer.pt + change_net.pt  ({elapsed:.0f}s | Best: {best_loss:.4f})")
    return CHECKPOINT_DIR / "changeformer.pt"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4: OPTICAL-SAR FUSION
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedOpticalSARDataset(Dataset):
    SCENE_TYPES = ["urban", "vegetation", "water", "agricultural", "coastal"]

    def __init__(self, n_samples=320, img_size=128, augment=True):
        self.n_samples = n_samples
        self.img_size = img_size
        self.augment = augment
        self.data = []

        # Ingest authentic SEN1-2 Optical-SAR triplets
        sen12_dir = DATA_DIR / "fusion_sen12"
        if sen12_dir.exists():
            import rasterio
            cloudy_files = sorted(list(sen12_dir.glob("*_optical_cloudy.tif")))
            for c_p in cloudy_files:
                prefix = c_p.name.replace("_optical_cloudy.tif", "")
                sar_p = sen12_dir / f"{prefix}_sar.tif"
                clean_p = sen12_dir / f"{prefix}_optical_clean.tif"
                if sar_p.exists() and clean_p.exists():
                    try:
                        with rasterio.open(str(c_p)) as sc, rasterio.open(str(sar_p)) as ss, rasterio.open(str(clean_p)) as st:
                            opt_c = sc.read([1, 2, 3]).astype(np.float32)
                            sar_c = ss.read([1, 2]).astype(np.float32)
                            target_c = st.read([1, 2, 3]).astype(np.float32)
                            if opt_c.max() > 1.0: opt_c /= 255.0
                            if sar_c.max() > 1.0: sar_c /= 255.0
                            if target_c.max() > 1.0: target_c /= 255.0
                            c_mask = (np.mean(np.abs(opt_c - target_c), axis=0, keepdims=True) > 0.2).astype(np.float32)
                            self.data.append((opt_c, sar_c, target_c, c_mask))
                    except Exception:
                        pass
            print(f"  Loaded {len(self.data)} authentic SEN1-2 / BigEarthNet Optical-SAR triplets")

        synth_count = max(n_samples - len(self.data), 50)
        print(f"  Pre-generating {synth_count} optical-SAR multi-modal pairs with cloud simulations...")
        for idx in range(synth_count):
            rng = np.random.RandomState(idx * 19 + 11)
            scene = self.SCENE_TYPES[idx % len(self.SCENE_TYPES)]
            h, w = self.img_size, self.img_size
            gt = generate_satellite_scene(scene, h, w, rng)
            sar = self._gen_sar(gt, rng)
            cloudy, cloud_mask = self._add_clouds(gt, rng)
            self.data.append((
                cloudy.astype(np.float32),
                sar.astype(np.float32),
                gt.astype(np.float32),
                cloud_mask[np.newaxis].astype(np.float32)
            ))

    def _gen_sar(self, optical, rng):
        # Physically-grounded SAR microwave radar simulation
        r, g, b = optical[0], optical[1], optical[2]
        h, w = r.shape

        is_water = (b > 0.35) & (b > r + 0.15) & (b > g + 0.05)
        is_veg = (g > r + 0.08) & (g > b + 0.08)
        is_urban = (r > 0.55) & (g > 0.50) & (b > 0.45)
        is_road = (np.abs(r - g) < 0.05) & (np.abs(g - b) < 0.05) & (r < 0.35)

        vv = np.full((h, w), 0.38, dtype=np.float32)
        vh = np.full((h, w), 0.24, dtype=np.float32)

        vv[is_water] = rng.uniform(0.04, 0.12)
        vh[is_water] = rng.uniform(0.02, 0.06)

        vv[is_road] = rng.uniform(0.16, 0.26)
        vh[is_road] = rng.uniform(0.08, 0.16)

        vv[is_veg] = rng.uniform(0.42, 0.58)
        vh[is_veg] = rng.uniform(0.32, 0.48)

        vv[is_urban] = rng.uniform(0.72, 0.95)
        vh[is_urban] = rng.uniform(0.55, 0.82)

        # Multiplicative Rayleigh radar speckle
        speckle_vv = rng.exponential(scale=0.15, size=(h, w)).astype(np.float32)
        speckle_vh = rng.exponential(scale=0.15, size=(h, w)).astype(np.float32)
        vv = np.clip(vv + speckle_vv - 0.07, 0.02, 0.98)
        vh = np.clip(vh + speckle_vh - 0.07, 0.01, 0.95)

        return np.stack([vv, vh], axis=0).astype(np.float32)

    def _add_clouds(self, optical, rng):
        h, w = optical.shape[1], optical.shape[2]
        cloud_mask = np.zeros((h, w), dtype=np.float32)

        num_clouds = rng.randint(1, 5)
        for _ in range(num_clouds):
            cx, cy = rng.randint(10, w - 10), rng.randint(10, h - 10)
            rx, ry = rng.randint(w // 8, w // 3), rng.randint(h // 8, h // 3)
            yy, xx = np.ogrid[:h, :w]
            dist_sq = ((xx - cx) / float(rx)) ** 2 + ((yy - cy) / float(ry)) ** 2
            core = np.clip(1.0 - dist_sq, 0.0, 1.0)
            cloud_mask = np.maximum(cloud_mask, core ** 1.5)

        cloud_mask = np.clip(cloud_mask * rng.uniform(0.85, 1.0), 0.0, 1.0)

        # Cast shadows shifted by solar azimuth (dx = -6, dy = +6)
        shadow_mask = np.roll(np.roll(cloud_mask, 6, axis=0), -6, axis=1)
        shadow_factor = 1.0 - 0.55 * shadow_mask * (1.0 - cloud_mask)

        cloudy = optical.copy() * shadow_factor[np.newaxis, ...]
        cloud_bright = rng.uniform(0.88, 0.98, (3, 1, 1)).astype(np.float32)
        cloudy = cloudy * (1.0 - cloud_mask[np.newaxis, ...]) + cloud_bright * cloud_mask[np.newaxis, ...]
        return np.clip(cloudy, 0.0, 1.0).astype(np.float32), (cloud_mask > 0.25).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cloudy, sar, target, cloud_mask = self.data[idx]
        if self.augment:
            rng = np.random.RandomState()
            if rng.rand() > 0.5:
                cloudy = cloudy[:, :, ::-1]
                sar = sar[:, :, ::-1]
                target = target[:, :, ::-1]
                cloud_mask = cloud_mask[:, :, ::-1]
            if rng.rand() > 0.5:
                cloudy = cloudy[:, ::-1, :]
                sar = sar[:, ::-1, :]
                target = target[:, ::-1, :]
                cloud_mask = cloud_mask[:, ::-1, :]
        return (torch.from_numpy(cloudy.copy()),
                torch.from_numpy(sar.copy()),
                torch.from_numpy(target.copy()),
                torch.from_numpy(cloud_mask.copy()))


class OpticalSARCrossAttentionNet(nn.Module):
    """Bidirectional cross-attention fusion. ~2.8M params."""
    def __init__(self, optical_channels=3, sar_channels=2, out_channels=3):
        super().__init__()
        self.opt_enc1 = nn.Sequential(nn.Conv2d(optical_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.opt_enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.opt_enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU())
        self.sar_enc1 = nn.Sequential(nn.Conv2d(sar_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.sar_enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.sar_enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU())
        self.opt_stream = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.GELU())
        self.sar_stream = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.GELU())
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.recon_head = nn.Sequential(nn.Conv2d(64, out_channels, 1), nn.Sigmoid())
        self.conf_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, optical, sar):
        o1 = self.opt_enc1(optical); o2 = self.opt_enc2(o1); o3 = self.opt_enc3(o2)
        s1 = self.sar_enc1(sar); s2 = self.sar_enc2(s1); s3 = self.sar_enc3(s2)
        fused = self.opt_stream(torch.cat([o3, s3], dim=1)) + self.sar_stream(torch.cat([s3, o3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(fused), o2 + s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), o1 + s1], dim=1))
        return self.recon_head(d1), self.conf_head(fused)


def train_optical_sar_fusion():
    _epochs = EPOCHS
    print("\n" + "=" * 65, flush=True)
    print("  MODEL 4/4 — Optical-SAR Fusion Deep Fine-Tune (200 samples, cycle loss)", flush=True)
    print(f"  Device: {DEVICE} | Epochs: {_epochs} | Batch: {BATCH_SIZE}", flush=True)
    print("=" * 65, flush=True)
    dataset = EnhancedOpticalSARDataset(n_samples=200, img_size=128, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = OpticalSARCrossAttentionNet(optical_channels=3, sar_channels=2, out_channels=3).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params:,} trainable")

    best_loss = float("inf")
    for ckpt_name in ["fusion_net.pt", "optical_sar_fusion.pt"]:
        ckpt_path = CHECKPOINT_DIR / ckpt_name
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                sd = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(sd, strict=False)
                best_loss = ckpt.get("loss", float("inf"))
                print(f"  Warm-start from {ckpt_name} (prev best: {best_loss:.4f})")
                break
            except Exception as e:
                print(f"  Could not warm-start {ckpt_name}: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 3, steps_per_epoch=len(loader),
        epochs=_epochs, pct_start=0.1, anneal_strategy="cos"
    )

    t0 = time.time()
    for epoch in range(1, _epochs + 1):
        model.train()
        total_loss = 0.0
        for opt, sar, target, cloud_mask in loader:
            opt, sar = opt.to(DEVICE), sar.to(DEVICE)
            target, cloud_mask = target.to(DEVICE), cloud_mask.to(DEVICE)
            optimizer.zero_grad()
            recon, conf = model(opt, sar)
            # Cloud-weighted reconstruction loss (3x weight on cloud pixels)
            w = 1.0 + 2.0 * cloud_mask
            recon_loss = (F.l1_loss(recon, target, reduction="none") * w).mean()
            mse_loss = F.mse_loss(recon, target)
            conf_loss = F.mse_loss(conf.mean(), torch.tensor(0.85, device=DEVICE))
            loss = recon_loss + 0.3 * mse_loss + 0.1 * conf_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * opt.size(0)
        avg_loss = total_loss / len(dataset)
        lr_curr = scheduler.get_last_lr()[0]
        marker = " *" if avg_loss < best_loss else ""
        print(f"  Epoch {epoch:02d}/{_epochs} | Recon+Conf: {avg_loss:.4f} | LR: {lr_curr:.2e}{marker}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            state = {"model_state_dict": model.state_dict(), "epochs": _epochs, "loss": best_loss,
                      "arch": "OpticalSARCrossAttentionNet", "optical_channels": 3, "sar_channels": 2}
            for p in [CHECKPOINT_DIR / "fusion_net.pt", CHECKPOINT_DIR / "optical_sar_fusion.pt"]:
                torch.save(state, p)
    elapsed = time.time() - t0
    print(f"  Saved -> fusion_net.pt + optical_sar_fusion.pt  ({elapsed:.0f}s | Best: {best_loss:.4f})")
    return CHECKPOINT_DIR / "fusion_net.pt"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SatQuery AI Enhanced Training v2.0")
    parser.add_argument("--model", type=str, default="all",
                        help="Models to train: 'all', '1,2', '3,4', 'vlm', 'grounding', 'changeformer', 'fusion'")
    parser.add_argument("--order", type=str, default="12_then_34",
                        choices=["12_then_34", "34_then_12"],
                        help="Execution sequence when training all models")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for Models 3 & 4 (ChangeFormer, Fusion)")
    parser.add_argument("--epochs-polish", type=int, default=12, help="Epochs for Models 1 & 2 (RS-VLM, Grounding)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    args = parser.parse_args()

    global EPOCHS, EPOCHS_POLISH, BATCH_SIZE, LR
    EPOCHS = args.epochs
    EPOCHS_POLISH = args.epochs_polish
    BATCH_SIZE = args.batch_size
    LR = args.lr

    print("\n" + "=" * 65)
    print("  SatQuery AI - Specialist Model Training Suite")
    print(f"  Device: {DEVICE} | Models: {args.model} | Order: {args.order}")
    print(f"  Epochs (M3/M4): {EPOCHS} | Epochs (M1/M2): {EPOCHS_POLISH} | Batch: {BATCH_SIZE}")
    print(f"  Checkpoints Output -> {CHECKPOINT_DIR}")
    print("=" * 65)

    start_all = time.time()
    saved = []

    m = args.model.lower()
    train_m1 = m in ("all", "1,2", "1", "vlm")
    train_m2 = m in ("all", "1,2", "2", "grounding")
    train_m3 = m in ("all", "3,4", "3", "changeformer")
    train_m4 = m in ("all", "3,4", "4", "fusion")
    train_m5 = m in ("all", "5", "intent", "controller")

    from training.train_agent_intent_v3 import train_agent_intent_v3

    if args.order == "12_then_34":
        if train_m1: saved.append(train_rs_vlm())
        if train_m2: saved.append(train_grounding_dino())
        if train_m3: saved.append(train_changeformer())
        if train_m4: saved.append(train_optical_sar_fusion())
        if train_m5: saved.append(train_agent_intent_v3(epochs=25, batch_size=64))
    else:
        if train_m3: saved.append(train_changeformer())
        if train_m4: saved.append(train_optical_sar_fusion())
        if train_m1: saved.append(train_rs_vlm())
        if train_m2: saved.append(train_grounding_dino())
        if train_m5: saved.append(train_agent_intent_v3(epochs=25, batch_size=64))

    total_elapsed = time.time() - start_all
    print("\n" + "=" * 65)
    print(f"  TRAINING COMPLETED SUCCESSFULLY in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print("  Generated Checkpoints:")
    for p in saved:
        size_mb = Path(p).stat().st_size / (1024 * 1024)
        print(f"    -> {Path(p).name}  ({size_mb:.1f} MB)")
    print("=" * 65)
    print("\n  Ready for verification: run python scripts/test_trained_models.py")
    print()


if __name__ == "__main__":
    main()
