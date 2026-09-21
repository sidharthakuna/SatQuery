"""
SatQuery AI — Local RS-VLM Conversational Fine-Tuning Engine (Zero Cloud APIs)
Fine-tunes the Remote-Sensing Vision-Language Model on authentic multi-turn
conversational dialogues, land-cover descriptions, and geospatial reasoning.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# Add backend and scripts to sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from app.models.rs_vlm import (
    RSVisionLanguageModel,
    RS_VLM_WORDS,
    RS_VLM_VOCAB,
    tokenize_query,
)
from vlm_dataset_builder import CONVERSATIONAL_SAMPLES, generate_training_batch

CHECKPOINT_DIR = BACKEND / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = CHECKPOINT_DIR / "rs_vlm.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_unified_vocabulary() -> Tuple[List[str], Dict[str, int]]:
    """
    Combines existing RS-VLM words with all vocabulary from conversational
    dialogues and remote-sensing VQA domain corpora.
    """
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
    word_set = set()

    for w in RS_VLM_WORDS:
        if w not in special_tokens:
            word_set.add(w)

    for s in CONVERSATIONAL_SAMPLES:
        texts = [s["query"], s["answer"]]
    # Ingest vocabulary from BigEarthNet public benchmark dataset
    ben_path = ROOT / "data" / "benchmarks" / "vqa_bigearthnet" / "annotations.json"
    if ben_path.exists():
        try:
            import json
            with open(ben_path, "r", encoding="utf-8") as f:
                ben_data = json.load(f)
                for item in ben_data:
                    for conv in item.get("conversations", []):
                        for word in conv.get("value", "").split():
                            clean = word.strip("?.,!;:\"'()/-").lower()
                            if clean and clean not in special_tokens:
                                word_set.add(clean)
        except Exception:
            pass


    # Additional conversational and remote sensing tokens
    extra_tokens = [
        "specifically", "furthermore", "regarding", "earlier", "previous", "discussion",
        "consequently", "demonstrating", "quadrant", "northwest", "northeast", "southwest",
        "southeast", "perimeter", "undisturbed", "permeable", "impervious", "containment",
        "cadastral", "reflectance", "chlorophyll", "biomass", "drainage", "tributary",
        "inundated", "submergence", "viaduct", "interchange", "cloverleaf", "fairway",
        "breakwater", "apron", "hangar", "taxiway", "pipeline", "hectares", "meters",
        "kilometers", "percentage", "density", "expansion", "settlement", "corridor",
        "safe", "evacuation", "hazard", "minimal", "moderate", "substantial", "elevated",
        "contrast", "absorption", "scattering", "albedo", "vegetative", "photosynthetic"
    ]
    for w in extra_tokens:
        word_set.add(w)

    sorted_words = special_tokens + sorted(list(word_set))
    vocab = {w: i for i, w in enumerate(sorted_words)}
    return sorted_words, vocab


def tokenize_sentence(text: str, vocab: Dict[str, int], max_len: int = 40) -> List[int]:
    """Tokenizes a sentence into token IDs with <bos> and <eos>."""
    bos_id = vocab.get("<bos>", 2)
    eos_id = vocab.get("<eos>", 3)
    unk_id = vocab.get("<unk>", 1)
    pad_id = vocab.get("<pad>", 0)

    words = [w.strip("?.,!;:\"'()").lower() for w in text.split() if w]
    ids = [bos_id] + [vocab.get(w, unk_id) for w in words] + [eos_id]

    if len(ids) > max_len:
        ids = ids[:max_len - 1] + [eos_id]
    else:
        ids = ids + [pad_id] * (max_len - len(ids))

    return ids


def run_fine_tuning(epochs: int = 15, batch_size: int = 8, lr: float = 2e-4):
    print("=" * 60)
    print("  SatQuery AI — Local RS-VLM Conversational Fine-Tuning")
    print("  Zero External APIs • 100% On-Device Neural Training")
    print(f"  Device: {DEVICE} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 60)

    words_list, vocab = build_unified_vocabulary()
    vocab_size = len(vocab)
    print(f"  Vocabulary Size: {vocab_size} tokens")

    # Initialize Model with unified vocabulary
    model = RSVisionLanguageModel(
        vocab_size=vocab_size,
        embed_dim=256,
        lora_rank=16,
        vocab=vocab,
    ).to(DEVICE)

    # Transfer compatible weights from existing checkpoint if available
    if CKPT_PATH.exists():
        try:
            old_ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
            old_state = old_ckpt.get("model_state_dict", old_ckpt)
            # Filter compatible layers (conv backbone and transformer blocks)
            cur_state = model.state_dict()
            transferred = 0
            for k, v in old_state.items():
                if k in cur_state and cur_state[k].shape == v.shape:
                    cur_state[k] = v
                    transferred += 1
            model.load_state_dict(cur_state)
            print(f"  Transferred {transferred} pre-trained weight tensors from {CKPT_PATH.name}")
        except Exception as e:
            print(f"  Checkpoint warm-start note: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"], label_smoothing=0.05)

    steps_per_epoch = 16
    start_time = time.time()
    best_loss = float("inf")

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            batch = generate_training_batch(batch_size=batch_size, img_size=128)

            images = torch.tensor(np.array([s["image"] for s in batch]), dtype=torch.float32, device=DEVICE)
            q_tokens = torch.tensor([tokenize_query(s["query"], max_len=24, vocab=vocab) for s in batch], dtype=torch.long, device=DEVICE)
            a_tokens = torch.tensor([tokenize_sentence(s["answer"], vocab=vocab, max_len=40) for s in batch], dtype=torch.long, device=DEVICE)

            # Input targets (without trailing token) and expected labels (shifted right)
            tgt_in = a_tokens[:, :-1]
            tgt_out = a_tokens[:, 1:]

            optimizer.zero_grad()
            logits = model(images, q_tokens, tgt_in)

            loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / steps_per_epoch

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save checkpoint with embedded vocabulary
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "words": words_list,
                "best_loss": best_loss,
                "model_type": "RSVisionLanguageModel_Conversational_v3",
            }, CKPT_PATH)

        if epoch % 3 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} (Best: {best_loss:.4f})")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"  Fine-Tuning Complete in {elapsed:.1f}s!")
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"  Checkpoint successfully exported to: {CKPT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune local RS-VLM model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    run_fine_tuning(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
