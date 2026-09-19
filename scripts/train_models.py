"""
SatQuery AI — Master Specialist Models Training Orchestrator
Sequentially trains all 4 remote sensing specialist AI models:
1. Remote-Sensing VLM / VQA (LoRA Fine-Tuning on BigEarthNet.txt)
2. Grounding DINO + SAM-RS (Text-Guided Grounding on DIOR-RSVG)
3. ChangeFormer & Change-VQA (Bi-Temporal Change Analysis on LEVIR-CD)
4. Optical + SAR Cross-Modal Fusion Net (Sentinel-1 & Sentinel-2 on SEN1-2)

Exports verified checkpoints to backend/data/checkpoints/
"""

import argparse
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

from train_all_models_enhanced import (
    train_rs_vlm,
    train_grounding_dino,
    train_changeformer,
    train_optical_sar_fusion,
    CHECKPOINT_DIR,
)
from train_agent_intent import train_agent_intent

BENCHMARKS_DIR = root_dir / "data" / "benchmarks"


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI — Master Specialist Models Training Suite")
    parser.add_argument("--models", type=str, default="all",
                        help="Models to train: 'all', '1,2', '3,4', '1', '2', '3', '4', '5', 'intent'")
    parser.add_argument("--order", type=str, default="12_then_34",
                        choices=["12_then_34", "34_then_12"],
                        help="Execution sequence when training all models")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for Models 3 & 4")
    parser.add_argument("--epochs-polish", type=int, default=12, help="Number of epochs for Models 1 & 2")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    args = parser.parse_args()

    import train_all_models_enhanced as enhanced_module
    enhanced_module.EPOCHS = args.epochs
    enhanced_module.EPOCHS_POLISH = args.epochs_polish
    enhanced_module.BATCH_SIZE = args.batch_size

    print("============================================================")
    print("  SatQuery AI — Autonomous Specialist Model Training Engine")
    print(f"  Target Device:    {enhanced_module.DEVICE}")
    print(f"  Selected Models:  {args.models}")
    print(f"  Execution Order:  {args.order}")
    print(f"  Output Directory: {CHECKPOINT_DIR}")
    print(f"  Epochs (M3 & M4): {args.epochs} | Epochs (M1 & M2): {args.epochs_polish}")
    print("============================================================")

    t0 = time.time()
    saved = []

    m = args.models.lower()
    train_m1 = m in ("all", "1,2", "1", "vlm")
    train_m2 = m in ("all", "1,2", "2", "grounding")
    train_m3 = m in ("all", "3,4", "3", "changeformer")
    train_m4 = m in ("all", "3,4", "4", "fusion")
    train_m5 = m in ("all", "5", "intent", "controller")

    if args.order == "12_then_34":
        if train_m1:
            print("\n>>> Stage 1: Fine-Tuning Model 1 (RS-VLM on BigEarthNet.txt)...")
            saved.append(train_rs_vlm())
        if train_m2:
            print("\n>>> Stage 2: Fine-Tuning Model 2 (Grounding DINO + SAM on DIOR-RSVG)...")
            saved.append(train_grounding_dino())
        if train_m3:
            print("\n>>> Stage 3: Deep Fine-Tuning Model 3 (ChangeFormer on LEVIR-CD & CDVQA)...")
            saved.append(train_changeformer())
        if train_m4:
            print("\n>>> Stage 4: Deep Fine-Tuning Model 4 (Optical-SAR Fusion on SEN1-2 & BigEarthNet-MM)...")
            saved.append(train_optical_sar_fusion())
        if train_m5:
            print("\n>>> Stage 5: Training Model 5 (Agent Intent & Controller Network)...")
            saved.append(train_agent_intent(epochs=25, batch_size=32))
    else:
        if train_m3:
            print("\n>>> Stage 1: Deep Fine-Tuning Model 3 (ChangeFormer on LEVIR-CD & CDVQA)...")
            saved.append(train_changeformer())
        if train_m4:
            print("\n>>> Stage 2: Deep Fine-Tuning Model 4 (Optical-SAR Fusion on SEN1-2 & BigEarthNet-MM)...")
            saved.append(train_optical_sar_fusion())
        if train_m1:
            print("\n>>> Stage 3: Fine-Tuning Model 1 (RS-VLM on BigEarthNet.txt)...")
            saved.append(train_rs_vlm())
        if train_m2:
            print("\n>>> Stage 4: Fine-Tuning Model 2 (Grounding DINO + SAM on DIOR-RSVG)...")
            saved.append(train_grounding_dino())
        if train_m5:
            print("\n>>> Stage 5: Training Model 5 (Agent Intent & Controller Network)...")
            saved.append(train_agent_intent(epochs=25, batch_size=32))

    total_time = time.time() - t0

    print("\n============================================================")
    print("  [TRAINING COMPLETE] Selected Specialist Models Successfully Trained!")
    print(f"  Total Duration: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")
    print("  Generated Checkpoints:")
    for p in saved:
        p_path = Path(p)
        print(f"    - {p_path.name}: {p_path.stat().st_size / (1024 * 1024):.1f} MB")
    print("============================================================")
    print("  Next: Verify models with `python scripts/test_trained_models.py`\n")


if __name__ == "__main__":
    main()
