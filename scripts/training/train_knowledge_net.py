"""
SatQuery AI — Neural Domain Knowledge Network Training Engine
Trains DomainKnowledgeNet using Contrastive Multi-Task Learning to map
remote-sensing queries directly to verified scientific passages.
Zero Cloud APIs • 100% Local On-Device Neural Training.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add backend to sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from app.core.knowledge.knowledge_corpus import KNOWLEDGE_PASSAGES
from app.core.knowledge.neural_knowledge_net import (
    DomainKnowledgeNet,
    build_knowledge_token_vocab,
    tokenize_knowledge_text,
)

CHECKPOINT_DIR = BACKEND / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = CHECKPOINT_DIR / "knowledge_net.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Synthetic training query variations per passage
TRAINING_QUERIES = {
    "isro_cartosat": [
        "What is the spatial resolution of Cartosat-3?",
        "Tell me about Cartosat satellite capabilities and DEM generation.",
        "Which ISRO satellite provides sub-meter panchromatic imaging?",
        "How does Cartosat-2 compare to Cartosat-1?",
        "What are Cartosat satellites used for in urban mapping?",
    ],
    "isro_risat": [
        "What frequency does RISAT operate at?",
        "How does EOS-04 penetrate clouds during monsoon season?",
        "Tell me about ISRO radar satellites and C-band microwave.",
        "What polarizations are supported by RISAT-1?",
        "Can RISAT see through clouds and smoke at night?",
    ],
    "isro_resourcesat": [
        "What sensors are onboard Resourcesat-2A?",
        "What is the swath and resolution of AWiFS and LISS-4?",
        "How does LISS-3 monitor agricultural crop acreage?",
        "Explain Resourcesat multi-spectral bands.",
    ],
    "isro_oceansat": [
        "What is the purpose of Oceansat-3 OCM-3?",
        "How does the scatterometer measure ocean wind speed?",
        "Can Oceansat detect chlorophyll concentration in the sea?",
        "How are cyclones tracked by ISRO scatterometer?",
    ],
    "isro_nisar": [
        "What is NISAR and what frequencies does it use?",
        "Explain the difference between L-band and S-band in NISAR.",
        "How does NISAR measure ground deformation and earthquakes?",
        "Tell me about the NASA ISRO joint radar mission.",
    ],
    "physics_optical_vs_sar": [
        "What is the difference between optical and SAR imagery?",
        "Why can SAR penetrate clouds while optical cameras cannot?",
        "Explain active microwave radar versus passive optical sensors.",
        "Why do clouds block satellite photos?",
    ],
    "physics_sar_polarization": [
        "What does VV and VH polarization mean in SAR?",
        "Why does calm water appear dark on radar images?",
        "Explain specular reflection and double bounce scattering in SAR.",
        "How does volume scattering occur in forest canopies?",
    ],
    "formula_ndvi": [
        "What is the formula for NDVI and how does it work?",
        "How is vegetation health calculated from NIR and Red bands?",
        "What do NDVI values between 0.6 and 0.8 mean?",
        "Why do healthy leaves reflect near-infrared light?",
    ],
    "formula_ndwi": [
        "What is the formula for NDWI and MNDWI?",
        "How do you separate water from land in satellite images?",
        "Why is green and NIR used for water mapping?",
        "What is the difference between NDWI and MNDWI?",
    ],
    "formula_ndbi": [
        "How do you calculate built up index NDBI?",
        "What formula detects concrete buildings and impervious surfaces?",
        "Why do urban structures reflect shortwave infrared SWIR?",
    ],
    "disaster_ndma_flood": [
        "What are NDMA guidelines for flood safe zone buffers?",
        "How far should evacuation camps be from flood water?",
        "What is the CWC flood warning stage protocol?",
        "How does satellite imagery support flood disaster rescue?",
    ],
    "disaster_flood_spectral": [
        "How is flood inundation mapped from satellite imagery?",
        "What is the backscatter drop for flooded land in SAR?",
        "How to calculate flooded area in hectares from pixels?",
    ],
    "agri_phenology": [
        "What is the difference between Kharif and Rabi crop seasons?",
        "When is the peak NDVI for monsoon crops in India?",
        "How are cadastral parcel boundaries identified from space?",
    ],
}


def build_contrastive_dataset(vocab: Dict[str, int]) -> List[Tuple[List[int], List[int], List[int]]]:
    """
    Builds triplet training samples: (query_tokens, pos_passage_tokens, neg_passage_tokens).
    """
    passages_by_id = {p["id"]: p for p in KNOWLEDGE_PASSAGES}
    passage_ids = list(passages_by_id.keys())
    triplets = []

    for pid, queries in TRAINING_QUERIES.items():
        pos_p = passages_by_id.get(pid)
        if not pos_p:
            continue
        pos_text = f"{pos_p['title']} {pos_p['content']}"
        pos_tokens = tokenize_knowledge_text(pos_text, vocab=vocab, max_len=64)

        for q in queries:
            q_tokens = tokenize_knowledge_text(q, vocab=vocab, max_len=32)

            # Sample all other passages as distinct negative contrasts
            neg_candidates = [other_id for other_id in passage_ids if other_id != pid]
            for neg_id in neg_candidates:
                neg_p = passages_by_id[neg_id]
                neg_text = f"{neg_p['title']} {neg_p['content']}"
                neg_tokens = tokenize_knowledge_text(neg_text, vocab=vocab, max_len=64)
                triplets.append((q_tokens, pos_tokens, neg_tokens))

    return triplets


def train_knowledge_network(epochs: int = 20, lr: float = 3e-4):
    print("=" * 60)
    print("  SatQuery AI — Training DomainKnowledgeNet")
    print("  Our Own Neural Network & Knowledge-Base Engine")
    print(f"  Device: {DEVICE} | Target Checkpoint: {CKPT_PATH.name}")
    print("=" * 60)

    vocab, id2word = build_knowledge_token_vocab()
    print(f"  Knowledge Vocabulary: {len(vocab)} tokens")

    model = DomainKnowledgeNet(vocab_size=len(vocab), embed_dim=256).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    triplet_loss = nn.TripletMarginLoss(margin=0.4, p=2)

    triplets = build_contrastive_dataset(vocab)
    print(f"  Training Triplet Pairs: {len(triplets)} pairs")

    start_time = time.time()
    best_loss = float("inf")

    model.train()
    for epoch in range(1, epochs + 1):
        # Shuffle triplets
        np.random.shuffle(triplets)
        batch_size = 32
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i + batch_size]
            q_t = torch.tensor([item[0] for item in batch], dtype=torch.long, device=DEVICE)
            p_t = torch.tensor([item[1] for item in batch], dtype=torch.long, device=DEVICE)
            n_t = torch.tensor([item[2] for item in batch], dtype=torch.long, device=DEVICE)

            q_emb = model.encode(q_t)
            pos_emb = model.encode(p_t)
            neg_emb = model.encode(n_t)

            loss = triplet_loss(q_emb, pos_emb, neg_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)

        if avg_loss <= best_loss:
            best_loss = avg_loss
            # Precompute passage embeddings for instant sub-millisecond retrieval
            model.eval()
            with torch.no_grad():
                passage_embs = {}
                for p in KNOWLEDGE_PASSAGES:
                    p_text = f"{p['title']} {p['content']}"
                    tokens = torch.tensor([tokenize_knowledge_text(p_text, vocab, max_len=64)], dtype=torch.long, device=DEVICE)
                    emb = model.encode(tokens).cpu().numpy()[0]
                    passage_embs[p["id"]] = emb

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "best_loss": best_loss,
                "passage_embeddings": passage_embs,
                "passages": KNOWLEDGE_PASSAGES,
            }, CKPT_PATH)
            model.train()

        if epoch % 4 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] Triplet Loss: {avg_loss:.4f} (Best: {best_loss:.4f})")


    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"  DomainKnowledgeNet Training Complete in {elapsed:.1f}s!")
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"  Saved to: {CKPT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train_knowledge_network(epochs=20)
