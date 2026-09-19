# -*- coding: utf-8 -*-
"""
SatQuery AI - Fine-Tuning: Describe and Mark on Satellite Images

Jointly fine-tunes:
  1. RSVisionLanguageModel  - natural-language scene descriptions
  2. GroundingDINOMaskNet   - bounding boxes + segmentation masks

10 scene categories: buildings, ships, roads, vegetation, water,
flood zones, storage tanks, runways, bare soil, port infrastructure.

Loss: VLM cross-entropy + GIoU + L1 + mask BCE + score BCE.
Loads existing checkpoints; saves back to backend/data/checkpoints/.

Usage:
  python scripts/finetune_describe_and_mark.py
  python scripts/finetune_describe_and_mark.py --epochs 30 --batch-size 4
  python scripts/finetune_describe_and_mark.py --model vlm
  python scripts/finetune_describe_and_mark.py --model grounding
  python scripts/finetune_describe_and_mark.py --model both
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.models.rs_vlm import RSVisionLanguageModel, RS_VLM_VOCAB, tokenize_query
from app.models.grounding_net import GroundingDINOMaskNet

CHECKPOINT_DIR = BACKEND / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS      = 25
BATCH_SIZE  = 4
LR          = 8e-5
IMG_SIZE    = 512
NUM_BOXES   = 8
TEXT_DIM    = 128
VLM_ANS_LEN = 20

PAD_ID    = RS_VLM_VOCAB.get("<pad>", 0)
BOS_ID    = RS_VLM_VOCAB.get("<bos>", 2)
EOS_ID    = RS_VLM_VOCAB.get("<eos>", 3)
UNK_ID    = RS_VLM_VOCAB.get("<unk>", 1)
VOCAB_SIZE = len(RS_VLM_VOCAB)


def tokenize_answer(text, max_len=VLM_ANS_LEN):
    words = [w.strip("?.,!;:\"'()").lower() for w in text.split() if w]
    ids = [BOS_ID] + [RS_VLM_VOCAB.get(w, UNK_ID) for w in words] + [EOS_ID]
    ids = ids[:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


SCENE_TEMPLATES = [
    ("buildings",     "locate all buildings",
     ["buildings", "urban", "residential", "structures", "dense", "rooftop"],
     (4, 8), (0.04, 0.12)),
    ("ships",         "detect vessels and ships",
     ["vessels", "ships", "maritime", "berthed", "anchored", "harbor"],
     (2, 6), (0.06, 0.18)),
    ("roads",         "identify road network",
     ["roads", "linear", "arterial", "paved", "network", "transport"],
     (2, 5), (0.02, 0.08)),
    ("vegetation",    "map vegetation and forest canopy",
     ["vegetation", "forest", "canopy", "dense", "biomass", "agricultural"],
     (3, 7), (0.08, 0.22)),
    ("water",         "detect water bodies and rivers",
     ["water", "river", "lake", "flood", "inundation", "bodies"],
     (2, 5), (0.06, 0.20)),
    ("flood",         "identify flooded and submerged areas",
     ["flooded", "inundation", "submerged", "waterlogged", "zones", "affected"],
     (3, 6), (0.10, 0.28)),
    ("storage tanks", "locate storage tanks and industrial facilities",
     ["tanks", "storage", "industrial", "circular", "facilities", "farm"],
     (3, 8), (0.04, 0.10)),
    ("runways",       "detect airport runways",
     ["runway", "aircraft", "airport", "linear", "paved", "terminal"],
     (1, 3), (0.10, 0.35)),
    ("bare soil",     "map bare soil and land disturbance",
     ["bare", "soil", "disturbed", "land", "cleared", "erosion"],
     (2, 5), (0.10, 0.30)),
    ("port",          "map port infrastructure and berths",
     ["port", "harbor", "berths", "wharf", "docks", "maritime"],
     (3, 7), (0.06, 0.18)),
]

BG_COLORS = {
    "buildings":     (0.45, 0.42, 0.38), "ships":         (0.12, 0.22, 0.45),
    "roads":         (0.35, 0.32, 0.30), "vegetation":    (0.18, 0.38, 0.18),
    "water":         (0.10, 0.20, 0.50), "flood":         (0.20, 0.28, 0.55),
    "storage tanks": (0.50, 0.48, 0.45), "runways":       (0.40, 0.40, 0.42),
    "bare soil":     (0.60, 0.52, 0.38), "port":          (0.28, 0.35, 0.48),
}
OBJ_COLORS = {
    "buildings":     (0.72, 0.68, 0.65), "ships":         (0.85, 0.85, 0.80),
    "roads":         (0.55, 0.53, 0.50), "vegetation":    (0.22, 0.62, 0.22),
    "water":         (0.08, 0.18, 0.72), "flood":         (0.15, 0.30, 0.78),
    "storage tanks": (0.80, 0.78, 0.75), "runways":       (0.65, 0.65, 0.68),
    "bare soil":     (0.78, 0.65, 0.45), "port":          (0.62, 0.65, 0.72),
}


def make_image(cat, boxes):
    H = W = IMG_SIZE
    rng = np.random.RandomState()
    br, bg, bb = BG_COLORS.get(cat, (0.35, 0.35, 0.35))
    img = np.stack([
        np.full((H, W), br, np.float32) + rng.randn(H, W).astype(np.float32) * 0.04,
        np.full((H, W), bg, np.float32) + rng.randn(H, W).astype(np.float32) * 0.04,
        np.full((H, W), bb, np.float32) + rng.randn(H, W).astype(np.float32) * 0.04,
    ], axis=0)
    or_, og, ob = OBJ_COLORS.get(cat, (0.70, 0.70, 0.70))
    for b in boxes:
        x1, y1, x2, y2 = b
        px1, py1 = max(0, int(x1 * W)), max(0, int(y1 * H))
        px2, py2 = min(W, int(x2 * W)), min(H, int(y2 * H))
        if px2 > px1 and py2 > py1:
            n = rng.randn(3, py2 - py1, px2 - px1).astype(np.float32) * 0.05
            img[0, py1:py2, px1:px2] = np.clip(or_ + n[0], 0, 1)
            img[1, py1:py2, px1:px2] = np.clip(og  + n[1], 0, 1)
            img[2, py1:py2, px1:px2] = np.clip(ob  + n[2], 0, 1)
    return np.clip(img, 0, 1)


def rnd_boxes(n, sz, rng):
    boxes = []
    for _ in range(n):
        for _ in range(20):
            bw = rng.uniform(*sz); bh = rng.uniform(*sz)
            cx = rng.uniform(bw / 2, 1 - bw / 2)
            cy = rng.uniform(bh / 2, 1 - bh / 2)
            x1, y1, x2, y2 = cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2
            ok = not any(
                not (x2 < ob[0] or x1 > ob[2] or y2 < ob[1] or y1 > ob[3])
                for ob in boxes
            )
            if ok:
                boxes.append([x1, y1, x2, y2])
                break
    return boxes


def augment(img, boxes, rng):
    if rng.rand() > 0.5:
        img = img[:, :, ::-1].copy()
        boxes = [[1 - b[2], b[1], 1 - b[0], b[3]] for b in boxes]
    if rng.rand() > 0.5:
        img = img[:, ::-1, :].copy()
        boxes = [[b[0], 1 - b[3], b[2], 1 - b[1]] for b in boxes]
    br = rng.uniform(0.75, 1.30)
    img = np.clip(img * br, 0, 1)
    m = img.mean(); ct = rng.uniform(0.80, 1.25)
    img = np.clip((img - m) * ct + m, 0, 1)
    sigma = rng.uniform(0.0, 0.03)
    img = np.clip(img + rng.randn(*img.shape).astype(np.float32) * sigma, 0, 1)
    return img, boxes


def build_text_emb(q):
    v = np.zeros(TEXT_DIM, np.float32)
    for i, c in enumerate(q[:TEXT_DIM]):
        v[i % TEXT_DIM] += (ord(c) - 32) / 96.0
    n = np.linalg.norm(v)
    if n > 0:
        v /= n
    return torch.from_numpy(v)


class SatDescribeDataset(Dataset):
    def __init__(self, n=1200, aug=True):
        self.n = n
        self.aug = aug

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        cat, q, words, nr, sz = SCENE_TEMPLATES[idx % len(SCENE_TEMPLATES)]
        nw = rng.randint(4, min(7, len(words)) + 1)
        caption = " ".join(rng.choice(words, nw, replace=False).tolist())
        n_obj = int(rng.randint(nr[0], nr[1] + 1))
        boxes = rnd_boxes(n_obj, sz, rng)
        img = make_image(cat, boxes)
        if self.aug:
            img, boxes = augment(img, boxes, rng)

        q_ids = torch.tensor(tokenize_query(q),        dtype=torch.long)
        a_ids = torch.tensor(tokenize_answer(caption), dtype=torch.long)
        te    = build_text_emb(q)

        gb = np.zeros((NUM_BOXES, 4), np.float32)
        gs = np.zeros(NUM_BOXES,    np.float32)
        for i, b in enumerate(boxes[:NUM_BOXES]):
            gb[i] = b; gs[i] = 1.0
        gm = np.zeros((1, 32, 32), np.float32)
        for b in boxes:
            x1, y1 = int(b[0] * 32), int(b[1] * 32)
            x2, y2 = max(x1 + 1, int(b[2] * 32)), max(y1 + 1, int(b[3] * 32))
            gm[0, y1:y2, x1:x2] = 1.0

        return (torch.from_numpy(img), q_ids, a_ids, te,
                torch.from_numpy(gb), torch.from_numpy(gs), torch.from_numpy(gm))


def giou_loss(p, t, mask):
    eps = 1e-6
    px1, py1, px2, py2 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    gx1, gy1, gx2, gy2 = t[..., 0], t[..., 1], t[..., 2], t[..., 3]
    ix1 = torch.max(px1, gx1); iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2); iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pa    = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    ga    = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    union = pa + ga - inter + eps
    iou   = inter / union
    ex1 = torch.min(px1, gx1); ey1 = torch.min(py1, gy1)
    ex2 = torch.max(px2, gx2); ey2 = torch.max(py2, gy2)
    encl = (ex2 - ex1).clamp(0) * (ey2 - ey1).clamp(0) + eps
    giou = iou - (encl - union) / encl
    return ((1 - giou) * mask).sum() / (mask.sum() + eps)


def save_ckpt(model, fname, ep, loss):
    p = CHECKPOINT_DIR / fname
    torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "best_loss": loss}, p)
    mb = p.stat().st_size / 1024 / 1024
    print(f"  SAVED {fname}  ({mb:.1f} MB)")


def load_vlm():
    m = RSVisionLanguageModel()
    p = CHECKPOINT_DIR / "rs_vlm.pt"
    if p.exists():
        try:
            ck = torch.load(p, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ck.get("model_state_dict", ck), strict=False)
            print("  [VLM] Loaded existing checkpoint")
        except Exception as e:
            print(f"  [VLM] Fresh start ({e})")
    else:
        print("  [VLM] No checkpoint - training from scratch")
    return m.to(DEVICE)


def load_grounding():
    m = GroundingDINOMaskNet(in_channels=3, text_dim=TEXT_DIM, max_boxes=NUM_BOXES)
    p = CHECKPOINT_DIR / "grounding_dino.pt"
    if not p.exists():
        p = CHECKPOINT_DIR / "grounding_net.pt"
    if p.exists():
        try:
            ck = torch.load(p, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ck.get("model_state_dict", ck), strict=False)
            print("  [Grounding] Loaded existing checkpoint")
        except Exception as e:
            print(f"  [Grounding] Fresh start ({e})")
    else:
        print("  [Grounding] No checkpoint - training from scratch")
    return m.to(DEVICE)


def run_vlm(epochs, bs, ds):
    print("\n" + "=" * 60)
    print("  FINE-TUNING: RS-VLM (Describe satellite scenes)")
    print("=" * 60)
    model = load_vlm()
    model.train()
    for name, p in model.named_parameters():
        p.requires_grad = any(
            k in name for k in ["lora_A", "lora_B", "lm_head", "decoder", "pos_encoder"]
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
    crit   = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.10)
    opt    = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4
    )
    sch    = CosineAnnealingWarmRestarts(opt, T_0=max(1, epochs // 3))
    best   = float("inf")
    for ep in range(1, epochs + 1):
        t0 = time.time(); total = 0; nb = 0
        for imgs, q_ids, a_ids, _, _, _, _ in loader:
            imgs  = imgs.to(DEVICE)
            q_ids = q_ids.to(DEVICE)
            a_ids = a_ids.to(DEVICE)
            opt.zero_grad()
            logits = model(imgs, q_ids, a_ids[:, :-1])
            loss   = crit(logits.reshape(-1, VOCAB_SIZE), a_ids[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sch.step()
        avg = total / max(nb, 1)
        lr_now = opt.param_groups[0]["lr"]
        print(f"  Epoch {ep:>3}/{epochs}  loss={avg:.4f}  lr={lr_now:.2e}  {time.time()-t0:.1f}s")
        if avg < best:
            best = avg
            save_ckpt(model, "rs_vlm.pt", ep, best)
    print(f"  Best VLM loss: {best:.4f}")
    return model


def run_grounding(epochs, bs, ds):
    print("\n" + "=" * 60)
    print("  FINE-TUNING: GroundingDINOMaskNet (Mark & Segment)")
    print("=" * 60)
    model = load_grounding()
    model.train()
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_p:,}")
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch    = CosineAnnealingWarmRestarts(opt, T_0=max(1, epochs // 3))
    best   = float("inf")
    for ep in range(1, epochs + 1):
        t0 = time.time(); total = 0; nb = 0
        for imgs, _, _, te, gb, gs, gm in loader:
            imgs = imgs.to(DEVICE); te = te.to(DEVICE)
            gb   = gb.to(DEVICE);  gs = gs.to(DEVICE); gm = gm.to(DEVICE)
            opt.zero_grad()
            pb, ps, pm = model(imgs, te)
            mexp = gs.unsqueeze(-1)
            l1   = (F.l1_loss(pb, gb, reduction="none") * mexp).sum() / (mexp.sum() + 1e-6)
            gi   = giou_loss(pb, gb, gs)
            sc   = F.binary_cross_entropy(ps, gs)
            if pm.shape[-1] != 32:
                pm = F.interpolate(pm, (32, 32), mode="bilinear", align_corners=False)
            pm_logit = torch.log(pm.clamp(1e-6, 1-1e-6) / (1 - pm.clamp(1e-6, 1-1e-6)))
            mk   = F.binary_cross_entropy_with_logits(pm_logit, gm)
            loss = l1 + 2.0 * gi + sc + 0.5 * mk
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sch.step()
        avg = total / max(nb, 1)
        lr_now = opt.param_groups[0]["lr"]
        print(f"  Epoch {ep:>3}/{epochs}  loss={avg:.4f}  lr={lr_now:.2e}  {time.time()-t0:.1f}s")
        if avg < best:
            best = avg
            save_ckpt(model, "grounding_dino.pt", ep, best)
    print(f"  Best Grounding loss: {best:.4f}")
    return model


@torch.no_grad()
def smoke_test(vlm, grnd):
    print("\n" + "=" * 60)
    print("  SMOKE TEST - Describe & Mark Satellite Scenes")
    print("=" * 60)
    cases = [
        ("locate all buildings in the scene", "buildings"),
        ("detect vessels and ships",           "ships"),
        ("identify flooded areas",             "flood"),
    ]
    rng = np.random.RandomState(777)
    for q, cat in cases:
        boxes = rnd_boxes(4, (0.06, 0.14), rng)
        img   = make_image(cat, boxes)
        it    = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
        print(f"\n  Query  : {q}")
        if vlm is not None:
            vlm.eval()
            q_ids = torch.tensor([tokenize_query(q)], dtype=torch.long).to(DEVICE)
            desc  = vlm.generate(it, q_ids, max_len=12)
            print(f"  Desc   : {desc[0]}")
        if grnd is not None:
            grnd.eval()
            te = build_text_emb(q).unsqueeze(0).to(DEVICE)
            pb, ps, pm = grnd(it, te)
            score, idx = ps[0].max(0)
            b = pb[0, idx].cpu().numpy()
            n = (ps[0] > 0.5).sum().item()
            print(f"  Top Box: [{b[0]:.3f} {b[1]:.3f} {b[2]:.3f} {b[3]:.3f}]  conf={score:.3f}")
            print(f"  Detected {n} object(s) with confidence > 0.5")


def main():
    ap = argparse.ArgumentParser(
        description="SatQuery AI - Fine-tune: Describe & Mark satellite images"
    )
    ap.add_argument("--epochs",     type=int,   default=EPOCHS,     help="Training epochs")
    ap.add_argument("--batch-size", type=int,   default=BATCH_SIZE, help="Batch size")
    ap.add_argument("--samples",    type=int,   default=1200,       help="Synthetic dataset size")
    ap.add_argument("--model",      type=str,   default="both",
                    choices=["vlm", "grounding", "both"])
    ap.add_argument("--no-smoke",   action="store_true", help="Skip smoke test")
    args = ap.parse_args()

    print("=" * 60)
    print("  SatQuery AI - Fine-Tune: Describe & Mark")
    print(f"  Device     : {DEVICE}")
    print(f"  Model      : {args.model}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Samples    : {args.samples}")
    print(f"  Output     : {CHECKPOINT_DIR}")
    print("=" * 60)

    ds = SatDescribeDataset(n=args.samples, aug=True)
    print(f"  Dataset: {len(ds)} samples across {len(SCENE_TEMPLATES)} scene categories")

    t0 = time.time()
    vlm = grnd = None

    if args.model in ("vlm", "both"):
        vlm = run_vlm(args.epochs, args.batch_size, ds)

    if args.model in ("grounding", "both"):
        grnd = run_grounding(args.epochs, args.batch_size, ds)

    total = time.time() - t0
    print(f"\n  Total training time: {total:.1f}s ({total/60:.1f} min)")

    if not args.no_smoke:
        smoke_test(vlm, grnd)

    print("\n" + "=" * 60)
    print("  Fine-tuning complete!")
    print(f"  Checkpoints saved to: {CHECKPOINT_DIR}")
    print("  Restart the backend server to load the updated weights.")
    print("=" * 60)


if __name__ == "__main__":
    main()
