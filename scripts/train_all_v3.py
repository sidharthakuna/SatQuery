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


DATA_DIR = root_dir / "data" / "benchmarks"


def verify_checkpoints() -> bool:
    """Verifies that all 6 specialist model checkpoints exist and can be loaded."""
    required = {
        "intent_net.pt": "AgentIntentNet v3.0 (Task Routing Controller)",
        "rs_vlm.pt": "RS-VLM (Vision-Language Q&A on BigEarthNet)",
        "grounding_dino.pt": "Grounding DINO + SAM (Visual Target Localization on DIOR-RSVG)",
        "grounding_net.pt": "Grounding Net Legacy Alias",
        "changeformer.pt": "Siamese ChangeFormer (Bi-Temporal Change on LEVIR-CD)",
        "change_net.pt": "Change Net Legacy Alias",
        "optical_sar_fusion.pt": "Optical-SAR Cross-Attention Fusion (SEN1-2 Cloud Removal)",
        "fusion_net.pt": "Fusion Net Legacy Alias",
        "knowledge_net.pt": "DomainKnowledgeNet (ISRO Missions & Science Retrieval)",
    }

    print("\n" + "=" * 65)
    print("  CHECKPOINT VERIFICATION SUITE")
    print("=" * 65)

    all_ok = True
    for filename, desc in required.items():
        ckpt_path = CHECKPOINT_DIR / filename
        if not ckpt_path.exists():
            print(f"  [MISSING] {filename:<22} - {desc}")
            all_ok = False
            continue

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            print(f"  [VALID]   {filename:<22} ({size_mb:5.1f} MB) - {desc}")
        except Exception as err:
            print(f"  [CORRUPT] {filename:<22} - Error loading: {err}")
            all_ok = False

    print("=" * 65)
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI — Master Specialist Models Training Suite v3.0")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Models to train: 'all', 'intent', 'vlm', 'grounding', 'changeformer', 'fusion', 'knowledge', or comma-separated list",
    )
    parser.add_argument("--epochs-change", type=int, default=10, help="Epochs for ChangeFormer")
    parser.add_argument("--epochs-fusion", type=int, default=10, help="Epochs for Optical-SAR Fusion")
    parser.add_argument("--epochs-grounding", type=int, default=6, help="Epochs for Grounding DINO")
    parser.add_argument("--epochs-vlm", type=int, default=8, help="Epochs for RS-VLM")
    parser.add_argument("--epochs-intent", type=int, default=15, help="Epochs for IntentNet v3.0")
    parser.add_argument("--epochs-knowledge", type=int, default=15, help="Epochs for DomainKnowledgeNet")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device: 'cpu' or 'cuda'")
    args = parser.parse_args()

    device_obj = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    print("\n" + "=" * 65)
    print("  SatQuery AI — Master Specialist Models Public Benchmark Training")
    print(f"  Compute Device:     {device_obj}")
    print(f"  Selected Models:    {args.models}")
    print(f"  Benchmarks Dir:     {DATA_DIR}")
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
    train_knowledge = m in ("all", "knowledge", "kb", "6", "m6") or "knowledge" in m

    # 1. Model 1: Agent Controller IntentNet v3.0
    if train_intent:
        print("\n>>> [1/6] Fine-Tuning Agent Intent Controller v3.0 (12,000+ Queries) <<<")
        from training.train_agent_intent_v3 import train_agent_intent_v3
        p = train_agent_intent_v3(
            epochs=args.epochs_intent,
            batch_size=64,
            lr=5e-4,
            device_str=str(device_obj),
        )
        saved_checkpoints.append(p)

    # 2. Model 2: RS-VLM on BigEarthNet & Conversational Domain Benchmark
    if train_vlm:
        print("\n>>> [2/6] Fine-Tuning RS-VLM on BigEarthNet & Real Satellite Benchmark <<<")
        from training.train_rs_vlm import train_rs_vlm
        p = train_rs_vlm(
            data_dir=DATA_DIR / "vqa_bigearthnet",
            epochs=args.epochs_vlm,
            batch_size=args.batch_size,
            lr=args.lr,
            device_str=str(device_obj),
        )
        saved_checkpoints.append(p)


    # 3. Model 3: Grounding DINO + SAM on DIOR-RSVG Benchmark
    if train_grounding:
        print("\n>>> [3/6] Fine-Tuning Grounding DINO + SAM on DIOR-RSVG Benchmark <<<")
        from training.train_grounding_dino import train_grounding_dino
        p = train_grounding_dino(
            data_dir=DATA_DIR / "grounding_dior",
            epochs=args.epochs_grounding,
            batch_size=4,
            lr=1e-3,
            device_str=str(device_obj),
        )
        saved_checkpoints.append(p)

    # 4. Model 4: ChangeFormer Bi-Temporal Delta on LEVIR-CD Benchmark
    if train_change:
        print("\n>>> [4/6] Fine-Tuning Siamese ChangeFormer on LEVIR-CD Benchmark <<<")
        from training.train_changeformer import train_changeformer
        p = train_changeformer(
            data_dir=DATA_DIR / "change_levir_cd",
            epochs=args.epochs_change,
            batch_size=4,
            lr=1e-3,
            device_str=str(device_obj),
        )
        saved_checkpoints.append(p)

    # 5. Model 5: Optical-SAR Cross-Modal Fusion on SEN1-2 Benchmark
    if train_fusion:
        print("\n>>> [5/6] Fine-Tuning Optical-SAR Cross-Attention on SEN1-2 Benchmark <<<")
        from training.train_optical_sar_fusion import train_optical_sar_fusion
        p = train_optical_sar_fusion(
            data_dir=DATA_DIR / "fusion_sen12",
            epochs=args.epochs_fusion,
            batch_size=args.batch_size,
            lr=1.5e-4,
            device_str=str(device_obj),
        )
        saved_checkpoints.append(p)

    # 6. Model 6: Domain Knowledge Net on ISRO & EO Science Corpus
    if train_knowledge:
        print("\n>>> [6/6] Fine-Tuning DomainKnowledgeNet on ISRO & EO Science Corpus <<<")
        from training.train_knowledge_net import train_knowledge_network
        train_knowledge_network(
            epochs=args.epochs_knowledge,
            lr=3e-4,
        )
        saved_checkpoints.append(CHECKPOINT_DIR / "knowledge_net.pt")

    total_elapsed = time.time() - start_all
    print("\n" + "=" * 65)
    print(f"  MASTER BENCHMARK TRAINING SUITE COMPLETED in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print("=" * 65)

    verify_checkpoints()


if __name__ == "__main__":
    main()

