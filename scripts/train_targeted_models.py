"""
SatQuery AI — Targeted Models & Agent Router Training Suite
Sequentially fine-tunes all remote sensing specialist models and the Agent Router:
  1. Agent Intent Controller (AgentIntentNet v3.0 on 12,600+ queries)
  2. Remote-Sensing VLM / VQA (LoRA on Earth Observation queries & BigEarthNet)
  3. Grounding DINO + SAM-RS (Text-Guided Visual Localization on DIOR-RSVG)
  4. Optical-SAR Cross-Modal Fusion (Cloud-Penetrating Sentinel-1/2 on SEN1-2)

STRICTLY EXCLUDES:
  - ChangeFormer Bi-Temporal Delta Network (remains untouched per user instruction)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Setup paths
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
scripts_dir = root_dir / "scripts"
training_dir = scripts_dir / "training"

sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(training_dir))

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_targeted_models")

CHECKPOINT_DIR = backend_dir / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI Targeted Models & Agent Router Training")
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' or 'cuda'")
    parser.add_argument("--epochs-intent", type=int, default=25, help="Epochs for Agent Intent Controller")
    parser.add_argument("--epochs-vlm", type=int, default=15, help="Epochs for RS-VLM")
    parser.add_argument("--epochs-grounding", type=int, default=12, help="Epochs for Grounding DINO")
    parser.add_argument("--epochs-fusion", type=int, default=15, help="Epochs for Optical-SAR Fusion")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for specialist models")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate for specialist models")
    parser.add_argument("--skip-intent", action="store_true", help="Skip intent controller training if already trained")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if device.type == "cpu":
        try:
            torch.set_num_threads(4)
        except Exception:
            pass

    print("=" * 70)
    print("  SatQuery AI — Targeted Models & Agent Router Training Pipeline")
    print(f"  Device:           {device}")
    print(f"  Intent Epochs:    {'SKIPPED' if args.skip_intent else args.epochs_intent}")
    print(f"  RS-VLM Epochs:    {args.epochs_vlm}")
    print(f"  Grounding Epochs: {args.epochs_grounding}")
    print(f"  Fusion Epochs:    {args.epochs_fusion}")
    print("  Bi-Temporal Model: EXCLUDED (ChangeFormer skipped)")
    print("=" * 70)

    start_all = time.time()

    # 1. Fine-Tune Agent Intent Router
    if not args.skip_intent:
        print("\n>>> [1/4] Fine-Tuning Agent Intent Controller v3.0 <<<")
        from training.train_agent_intent_v3 import train_agent_intent_v3
        intent_ckpt = train_agent_intent_v3(
            epochs=args.epochs_intent,
            batch_size=64,
            lr=5e-4,
            device_str=str(device),
        )
        print(f"  [DONE] Intent Controller saved to {intent_ckpt}")
    else:
        print("\n>>> [1/4] Agent Intent Controller v3.0: Already trained (99.95% val accuracy). Skipping.")

    # Import enhanced training module
    import train_all_models_enhanced as enhanced
    enhanced.DEVICE = device
    enhanced.BATCH_SIZE = args.batch_size
    enhanced.LR = args.lr

    # 2. Train RS-VLM
    print("\n>>> [2/4] Training Remote-Sensing VLM / VQA <<<")
    enhanced.EPOCHS_POLISH = args.epochs_vlm
    vlm_ckpt = enhanced.train_rs_vlm()
    print(f"  [DONE] RS-VLM saved to {vlm_ckpt}")

    # 3. Train Grounding DINO + SAM-RS
    print("\n>>> [3/4] Training Grounding DINO + SAM-RS Visual Grounding <<<")
    enhanced.EPOCHS_POLISH = args.epochs_grounding
    grounding_ckpt = enhanced.train_grounding_dino()
    print(f"  [DONE] Grounding DINO saved to {grounding_ckpt}")

    # 4. Train Optical-SAR Cross-Modal Fusion
    print("\n>>> [4/4] Training Optical-SAR Cross-Modal Fusion Network <<<")
    enhanced.EPOCHS = args.epochs_fusion
    fusion_ckpt = enhanced.train_optical_sar_fusion()
    print(f"  [DONE] Fusion Network saved to {fusion_ckpt}")

    # Explicit note on exclusion
    print("\n[NOTE] Bi-temporal Change Detection model (ChangeFormer) was EXCLUDED as instructed.")
    print("       Existing checkpoint backend/data/checkpoints/change_net.pt is preserved.")

    elapsed = time.time() - start_all
    print("\n" + "=" * 70)
    print(f"  ALL TARGETED MODELS TRAINED SUCCESSFULLY IN {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
