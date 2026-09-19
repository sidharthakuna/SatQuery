"""
SatQuery AI — Master Specialist Models Training Suite v3.0
Sequentially fine-tunes all 5 remote sensing specialist AI models:
  1. Agent Intent Controller v3.0 (6 classes, compound queries, multi-label)
  2. Remote-Sensing VLM / VQA (LoRA Fine-Tuning on Earth Observation queries)
  3. Grounding DINO + SAM-RS (Text-Guided Grounding on DIOR-RSVG)
  4. ChangeFormer (Siamese Bi-Temporal Change Analysis on LEVIR-CD)
  5. Optical + SAR Cross-Modal Fusion Net (Sentinel-1 & Sentinel-2 on SEN1-2)

Exports and verifies all 5 checkpoints in backend/data/checkpoints/
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add paths
root_dir = Path(__file__).resolve().parent.parent
training_dir = root_dir / "scripts" / "training"
backend_dir = root_dir / "backend"

sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(training_dir))
sys.path.insert(0, str(root_dir / "scripts"))

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_all_v3")

CHECKPOINT_DIR = backend_dir / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def verify_checkpoints() -> bool:
    """Verifies that all 5 specialist model checkpoints exist and can be loaded."""
    required = {
        "intent_net.pt": "AgentIntentNet v3.0 (Task Routing Controller)",
        "rs_vlm.pt": "RS-VLM (Vision-Language Q&A)",
        "grounding_net.pt": "Grounding DINO + SAM (Visual Target Localization)",
        "change_net.pt": "Siamese ChangeFormer (Bi-Temporal Change Detection)",
        "fusion_net.pt": "Optical-SAR Cross-Attention Fusion (Cloud Penetration)",
    }

    print("\n" + "=" * 65)
    print("  CHECKPOINT VERIFICATION SUITE")
    print("=" * 65)

    all_ok = True
    for filename, desc in required.items():
        ckpt_path = CHECKPOINT_DIR / filename
        if not ckpt_path.exists():
            print(f"  [MISSING] {filename:<18} - {desc}")
            all_ok = False
            continue

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            keys_info = list(ckpt.keys())[:4] if isinstance(ckpt, dict) else "raw tensor"
            print(f"  [VALID]   {filename:<18} ({size_mb:5.1f} MB) - {desc}")
        except Exception as err:
            print(f"  [CORRUPT] {filename:<18} - Error loading: {err}")
            all_ok = False

    print("=" * 65)
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI — Master Specialist Models Training Suite v3.0")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Models to train: 'all', 'intent', 'vlm', 'grounding', 'changeformer', 'fusion', or comma-separated list",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for ChangeFormer & Fusion")
    parser.add_argument("--epochs-polish", type=int, default=12, help="Number of epochs for RS-VLM & Grounding")
    parser.add_argument("--epochs-intent", type=int, default=25, help="Number of epochs for IntentNet v3.0")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device: 'cpu' or 'cuda'")
    args = parser.parse_args()

    # Configure enhanced module settings
    import train_all_models_enhanced as enhanced_module
    device_obj = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    enhanced_module.DEVICE = device_obj
    enhanced_module.EPOCHS = args.epochs
    enhanced_module.EPOCHS_POLISH = args.epochs_polish
    enhanced_module.BATCH_SIZE = args.batch_size
    enhanced_module.LR = args.lr

    print("\n" + "=" * 65)
    print("  SatQuery AI — Master Specialist Models Fine-Tuning v3.0")
    print(f"  Compute Device:     {device_obj}")
    print(f"  Selected Models:    {args.models}")
    print(f"  Epochs (M3/M4):     {args.epochs} | Epochs (M1/M2): {args.epochs_polish} | Epochs (Intent): {args.epochs_intent}")
    print(f"  Checkpoints Dir:    {CHECKPOINT_DIR}")
    print("=" * 65)

    start_all = time.time()
    saved_checkpoints = []

    m = args.models.lower().strip()
    train_intent = m in ("all", "intent", "controller", "1", "m1") or "intent" in m
    train_vlm = m in ("all", "vlm", "rs_vlm", "2", "m2") or "vlm" in m
    train_grounding = m in ("all", "grounding", "dino", "3", "m3") or "grounding" in m
    train_change = m in ("all", "change", "changeformer", "4", "m4") or "change" in m
    train_fusion = m in ("all", "fusion", "sar", "5", "m5") or "fusion" in m

    # 1. Model 1: Agent Controller IntentNet v3.0
    if train_intent:
        print("\n>>> [1/5] Fine-Tuning Agent Intent Controller v3.0 (6 Classes & Multi-Label) <<<")
        from training.train_agent_intent_v3 import train_agent_intent_v3
        p = train_agent_intent_v3(
            epochs=args.epochs_intent,
            batch_size=64,
            lr=5e-4,
            device_str=str(device_obj),
        )
        saved_checkpoints.append(p)

    # 2. Model 2: RS-VLM
    if train_vlm:
        print("\n>>> [2/5] Fine-Tuning Remote Sensing Vision-Language Model (RS-VLM) <<<")
        p = enhanced_module.train_rs_vlm()
        saved_checkpoints.append(p)

    # 3. Model 3: Grounding DINO + SAM
    if train_grounding:
        print("\n>>> [3/5] Fine-Tuning Grounding DINO + SAM Spatial Localization <<<")
        p = enhanced_module.train_grounding_dino()
        saved_checkpoints.append(p)

    # 4. Model 4: ChangeFormer Bi-Temporal Change Detection
    if train_change:
        print("\n>>> [4/5] Fine-Tuning Siamese ChangeFormer Bi-Temporal Delta Network <<<")
        p = enhanced_module.train_changeformer()
        saved_checkpoints.append(p)

    # 5. Model 5: Optical-SAR Cross-Modal Fusion
    if train_fusion:
        print("\n>>> [5/5] Fine-Tuning Optical-SAR Cross-Attention Cloud Penetration <<<")
        p = enhanced_module.train_optical_sar_fusion()
        saved_checkpoints.append(p)

    total_elapsed = time.time() - start_all
    print("\n" + "=" * 65)
    print(f"  TRAINING SUITE COMPLETED in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print("=" * 65)

    verify_checkpoints()


if __name__ == "__main__":
    main()
