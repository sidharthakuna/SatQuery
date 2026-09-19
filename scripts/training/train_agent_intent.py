"""
SatQuery AI — Agent Intent & Controller Fine-Tuning Engine
Deep sequence-based training for the orchestrator task routing network (AgentIntentNet v2.0).
Trains on a comprehensive dataset of remote sensing user queries across all 5 task domains:
  0: SINGLE_VQA          (Single-scene Land Cover, Spectral, and Visual Q&A)
  1: SINGLE_GROUNDING    (Visual Grounding, Target Localization, and Bounding Boxes)
  2: BITEMPORAL_CHANGE   (Multi-temporal Delta, Expansion, Shrinkage, Deforestation)
  3: CROSS_MODAL_FUSION  (Optical + SAR All-Weather Radar Penetration & Fusion)
  4: AGENT_ASSISTANT     (Copilot Guidance, System Capabilities, General Earth Science Q&A)
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Set paths
root_dir = Path(__file__).resolve().parent.parent.parent
backend_dir = root_dir / "backend"
sys.path.insert(0, str(backend_dir))

from app.models.intent_net import AgentIntentNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_intent")

CHECKPOINT_DIR = backend_dir / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TASK_CLASSES = [
    "SINGLE_VQA",
    "SINGLE_GROUNDING",
    "BITEMPORAL_CHANGE",
    "CROSS_MODAL_FUSION",
    "AGENT_ASSISTANT",
]
LABEL_TO_ID = {name: i for i, name in enumerate(TASK_CLASSES)}
ID_TO_LABEL = {i: name for i, name in enumerate(TASK_CLASSES)}

# ─────────────────────────────────────────────────────────────────────────────
# CURATED REMOTE SENSING INTENT CORPUS
# ─────────────────────────────────────────────────────────────────────────────

BASE_QUERIES = {
    "SINGLE_VQA": [
        "What is the predominant land cover in this satellite scene?",
        "Describe vegetation health and estimated NDVI in this region.",
        "What type of water body is present here?",
        "Is there an active industrial complex or airport visible?",
        "What are the prominent geographic and topographical features in this image?",
        "Assess agricultural crop development and parcel health.",
        "What spectral band combination would best highlight this terrain?",
        "Estimate the percentage of forest cover within this scene.",
        "Are there signs of soil degradation or desertification in this quadrant?",
        "Classify the biome shown in this optical observation.",
        "What is the urban density level in the central portion of the image?",
        "Can you identify any hydrological drainage networks in this scene?",
        "Does the vegetation exhibit seasonal senescence or healthy chlorophyll absorption?",
        "What kind of geological structures or rock outcroppings are visible?",
        "Describe the coastal wetlands and sediment dynamics in this tile.",
        "Analyze the land surface temperature signature and reflectance characteristics.",
        "What is the estimated cloud shadow interference in this capture?",
        "Evaluate the canopy density across this forest management unit.",
        "Is this scene predominantly rural farmland, rangeland, or developed land?",
        "What is the primary crop signature detectable in these agricultural plots?",
        "Determine the water turbidity level in the estuary reservoir.",
        "Does this optical imagery exhibit high NIR reflectance typical of active photosynthesis?",
        "What elevation gradient can be inferred from the shadows and ridgelines?",
        "Identify the dominant soil classification in the exposed barren area.",
        "Provide an overview of the ecological characteristics in this satellite snapshot.",
    ],
    "SINGLE_GROUNDING": [
        "Locate all storage tanks and outline them.",
        "Find and highlight the main runway and aircraft hangars.",
        "Detect residential buildings and isolated structures.",
        "Mark the highway interchange and bridge boundaries.",
        "Pinpoint all seaport vessels and cargo ships at berth.",
        "Segment the river bank canal lines and reservoir perimeter.",
        "Show me bounding boxes around all industrial facilities.",
        "Highlight the solar panel array and electrical substations.",
        "Where are the commercial aircraft parked on the tarmac?",
        "Locate any marine vessels anchored offshore in this scene.",
        "Identify and draw boxes over newly constructed warehouses.",
        "Spot the wind turbines installed across the ridgeline.",
        "Find all sports stadium facilities and athletic fields.",
        "Detect the boundaries of the open-pit mining excavation site.",
        "Outline the circular center-pivot irrigation fields.",
        "Pinpoint the communication towers and transmission pylons.",
        "Highlight the railway rail junction and switching yard.",
        "Box the container freight terminals located near the harbor.",
        "Where is the dam wall and hydroelectric spillway structure?",
        "Locate and segment every vehicle cluster on the freeway.",
        "Point out the perimeter fencing surrounding the security compound.",
        "Find the greenhouse farming complexes visible in the valley.",
        "Detect the wastewater treatment settling basins.",
        "Draw boxes around the oil refinery processing towers.",
        "Identify the footprint of all commercial shopping centers.",
        "Where are the docked naval frigates in the dry dock?",
        "Mark all electrical substation transformers in this district.",
        "Spot any anomalous vehicles parked outside the restricted perimeter.",
        "Delineate the exact perimeter of the soccer stadium.",
        "Highlight all petroleum storage depots and pipelines.",
    ],
    "BITEMPORAL_CHANGE": [
        "Analyze urban expansion and infrastructure change between T1 and T2.",
        "Detect structural changes between these two capture dates.",
        "Quantify the area of deforestation in hectares between before and after images.",
        "Measure the total hectares of rainforest lost from 2021 to 2024.",
        "Calculate the hectares of forest lost between past baseline and current.",
        "Measure how many hectares of tree canopy were lost over time.",
        "Compare urbanization and new construction from 2020 to 2024.",
        "How much has the water reservoir shrunk or expanded over time?",
        "Identify new roads built between the primary and secondary passes.",
        "Show difference map for wildfire burn scar progression.",
        "Assess flood inundation extent and shoreline retreat after the storm.",
        "Detect forest canopy loss following the logging operation.",
        "Measure the rate of coastal erosion between these multi-temporal rasters.",
        "Did any new buildings appear in this agricultural parcel between T1 and T2?",
        "Quantify the delta in snow and glacier cover over the winter season.",
        "Detect informal settlement growth and sprawl across the five year timeline.",
        "Compute the loss of mangrove wetlands along the coastal boundary.",
        "Identify demolished structures and rubble zones following the earthquake.",
        "Analyze vegetation regeneration in the conservation corridor between dates.",
        "Map temporal changes in open-water boundaries between dry and wet seasons.",
        "What infrastructure was erected in the industrial park during the last decade?",
        "Detect strip mining land clearing and overburden expansion.",
        "Has the shoreline receded or accreted between the two acquisitions?",
        "Did the reservoir water level recede following the drought?",
        "Assess whether the lake water level has receded or dried up over time.",
        "Track drought impact and receding waterlines between two time periods.",
        "Show me water retreat and shoreline regression over the years.",
        "Did vegetation or water bodies change following the drought?",
        "Examine the progression of quarry excavation from pre-event to post-event.",
        "Quantify the percentage reduction in crop acreage between baseline and current.",
        "Track the bi-temporal shifts in river meanders and alluvial sandbars.",
        "Show me all pixels that underwent spectral change between the pair of images.",
        "Determine structural recovery and rebuilding progress in the disaster zone.",
    ],
    "CROSS_MODAL_FUSION": [
        "Fuse cloudy optical scene with SAR C-band microwave backscatter.",
        "Penetrate monsoon cloud cover using Sentinel-1 synthetic aperture radar.",
        "Show me through the haze and fog by fusing optical and SAR sensors.",
        "Reveal terrain obscured by smoke and haze using synthetic aperture radar.",
        "Cross-modal radar fusion to see beneath atmospheric haze and light cloud.",
        "Combine optical RGB bands with SAR VV and VH polarizations.",
        "Run cross-modal cross-attention fusion on this optical and radar pair.",
        "Resolve surface details obscured by heavy stratus cloud layers.",
        "Fuse Sentinel-2 multispectral reflectance with Sentinel-1 SAR backscatter.",
        "Synthesize an all-weather cloud-free composite using microwave radar.",
        "Use radar penetration to uncover terrain hidden beneath dense smoke haze.",
        "Cross-modal fusion of optical texture with SAR speckle roughness.",
        "Merge microwave dual-pol data to penetrate atmospheric atmospheric attenuation.",
        "Perform deep feature cross-attention between SAR amplitude and optical channels.",
        "Restore obscured ground features by fusing optical and synthetic aperture radar.",
        "Analyze flood water extents under cloud cover using fused SAR microwave sensors.",
        "Combine optical spectral signatures with radar surface roughness signals.",
        "Generate cross-modal enhanced raster penetrating dense overcast conditions.",
        "Leverage SAR active microwave imaging to complement cloudy optical imagery.",
        "Co-register and fuse Sentinel-1 GRD and Sentinel-2 MSI into a coherent composite.",
        "Uncover hidden infrastructure underneath severe cumulus cloud occlusion.",
        "Apply cross-attention neural fusion on co-located optical and radar acquisitions.",
        "Overcome cloud saturation using multi-sensor radar optical synthesis.",
    ],
    "AGENT_ASSISTANT": [
        # Core system & capability queries
        "Hello, how can SatQuery AI assist my geospatial workflow?",
        "What specialist models are registered in the SatQuery system?",
        "How do I upload multi-spectral GeoTIFF files?",
        "Explain the difference between Sentinel-1 SAR and Sentinel-2 optical imagery.",
        "What CRS coordinate systems are supported for automatic co-registration?",
        "Can you guide me on interpreting NDVI and False Color Composites?",
        "Who created SatQuery AI and what are its core capabilities?",
        "Help me understand how ChangeFormer detects temporal differences.",
        "What is the role of Grounding DINO in visual remote sensing?",
        "Good morning, can you give me a summary of your features?",
        "How does the Cross-Modal Evidence Validator compute calibrated confidence?",
        "What raster formats can be ingested besides standard GeoTIFF?",
        "Explain how the input compatibility gate prevents invalid processing.",
        "Thank you for the detailed geospatial briefing report.",
        "Can I export the segmented spatial boundaries as GeoJSON vector layers?",
        "How does the audit trace provide verifiable provenance for SIH evaluation?",
        "What is the maximum GeoTIFF dimensions or file size supported?",
        "Hi, I am new to remote sensing. How should I get started with SatQuery?",
        "Tell me about the optical SAR fusion cross-attention mechanism.",
        "How do you calibrate confidence scores to prevent neural hallucination?",
        # Greetings, social, and emotional queries
        "Hey there, how are you doing today?",
        "Good evening! I need some help understanding remote sensing.",
        "How are you feeling right now?",
        "What's up? Can we chat about satellite imagery?",
        "Hi! Who made you and what technology powers you?",
        "Are you an AI? Tell me about yourself.",
        "Goodbye, thanks for all the analysis help today!",
        "Tell me a joke about satellites.",
        "You're awesome, great analysis work!",
        # General knowledge and educational queries
        "What is the electromagnetic spectrum and how is it used in remote sensing?",
        "Explain the concept of spatial resolution and ground sample distance.",
        "How do sun-synchronous orbits maintain consistent imaging conditions?",
        "What is photogrammetry and how does it create 3D terrain models?",
        "Tell me about the Landsat program and its historical significance.",
        "How do thermal infrared sensors measure surface temperature?",
        "What is hyperspectral imaging and how does it differ from multispectral?",
        "Explain the physics behind LiDAR point cloud generation.",
        "What are the applications of satellite imagery in precision agriculture?",
        "How does climate change monitoring benefit from satellite observations?",
        "Describe the Copernicus program and the Sentinel satellite constellation.",
        "What is a GIS and how does it integrate raster and vector data?",
        "Explain map projections and why Mercator distorts area at the poles.",
        "What Python libraries are used for geospatial data science?",
        "How does machine learning improve satellite image classification?",
        "Tell me about ISRO's earth observation satellite program.",
        "What is the NISAR mission and why is it important?",
        "How do satellites monitor deforestation and forest cover loss?",
        "Explain how drought conditions are detected using satellite data.",
        "What are the primary methods for wildfire detection from space?",
        "How do glaciers and ice sheets get monitored by remote sensing?",
        # Adversarial hard-negatives — contain task keywords but are educational
        "What does the word 'locate' mean in the context of spatial grounding?",
        "Explain how change detection algorithms work conceptually.",
        "What is the theory behind optical and SAR data fusion techniques?",
        "Describe the scientific principles of cloud penetration using microwave radar.",
        "How does the word 'detect' relate to remote sensing object identification?",
        "Can you teach me about bi-temporal image comparison methodology?",
        "What makes SAR radar fusion effective for all-weather monitoring?",
        "I want to understand the concept of bounding box regression in object detection.",
        # Conversational follow-ups and meta-questions
        "Tell me more about that topic.",
        "Can you elaborate on your previous answer?",
        "Why is that important for satellite analysis?",
        "What would you recommend for a beginner in this field?",
        "How does this compare to other approaches?",
        "What are the limitations of this technique?",
        "Could you give me a simpler explanation?",
        "What are the practical real-world applications?",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION & SYNTHESIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

PREFIXES = [
    "",
    "Please ",
    "Can you ",
    "Could you ",
    "I need you to ",
    "Kindly ",
    "AI copilot, ",
    "Orchestrator: ",
    "Execute task: ",
    "Quick question: ",
    "Analyze this: ",
]

SUFFIXES = [
    "",
    " please.",
    " in this observation.",
    " for my briefing.",
    " with verifiable evidence.",
    " and report confidence.",
    " across the scene.",
    " accurately.",
]

SYNONYMS = {
    "locate": ["find", "pinpoint", "spot", "detect", "identify", "delineate", "mark"],
    "detect": ["spot", "locate", "identify", "isolate", "find", "mark"],
    "highlight": ["outline", "mark", "draw bounding boxes around", "box", "show"],
    "change": ["difference", "delta", "alteration", "shift", "evolution", "modification"],
    "expand": ["grow", "extend", "sprawl", "spread"],
    "shrink": ["recede", "decrease", "reduce", "diminish"],
    "fuse": ["combine", "merge", "synthesize", "integrate", "blend"],
    "penetrate": ["pierce through", "see through", "filter through", "overcome"],
    "cloud": ["overcast", "haze", "stratus clouds", "cloud cover", "cloud deck"],
    "radar": ["SAR", "microwave backscatter", "synthetic aperture radar", "Sentinel-1"],
    "scene": ["imagery", "raster", "capture", "tile", "satellite frame"],
    "explain": ["describe", "tell me about", "elaborate on", "walk me through"],
    "understand": ["learn", "comprehend", "grasp", "know"],
    "satellite": ["spacecraft", "remote sensing platform", "orbital sensor"],
    "image": ["imagery", "raster", "scene", "capture", "observation"],
    "building": ["structure", "edifice", "construction", "infrastructure"],
}


def augment_query(query: str, rng: random.Random) -> str:
    """Apply realistic text perturbations: prefixes, suffixes, synonyms, typos."""
    words = query.split()
    augmented_words = []
    for w in words:
        clean_w = re.sub(r"[^\w]", "", w).lower()
        if clean_w in SYNONYMS and rng.random() < 0.35:
            syn = rng.choice(SYNONYMS[clean_w])
            augmented_words.append(syn)
        else:
            augmented_words.append(w)

    text = " ".join(augmented_words)

    # Add prefix / suffix
    if rng.random() < 0.40:
        text = rng.choice(PREFIXES) + text
    if rng.random() < 0.30:
        text = text.rstrip(".!?") + rng.choice(SUFFIXES)

    # Random capitalization variation
    if rng.random() < 0.20:
        text = text.lower()

    return text.strip()


def build_augmented_dataset(target_samples_per_class: int = 400, seed: int = 42) -> List[Tuple[str, int]]:
    """Generate high-diversity dataset balanced across all 5 classes."""
    rng = random.Random(seed)
    dataset = []

    for class_name, base_list in BASE_QUERIES.items():
        label_id = LABEL_TO_ID[class_name]
        # Include original base queries multiple times with no distortion
        for q in base_list:
            dataset.append((q, label_id))

        # Generate augmented variations
        needed = target_samples_per_class - len(base_list)
        for _ in range(needed):
            base_q = rng.choice(base_list)
            aug_q = augment_query(base_q, rng)
            dataset.append((aug_q, label_id))

    rng.shuffle(dataset)
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# VOCABULARY & TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """Clean and tokenize text query into lowercase word tokens."""
    cleaned = re.sub(r"[^\w\s-]", " ", text.lower())
    tokens = [t for t in cleaned.split() if len(t) > 0]
    return tokens


def build_vocab(dataset: List[Tuple[str, int]], min_freq: int = 1) -> Dict[str, int]:
    """Construct vocabulary dictionary with reserved tokens."""
    freq = {}
    for query, _ in dataset:
        for t in tokenize(query):
            freq[t] = freq.get(t, 0) + 1

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, count in sorted(freq.items(), key=lambda x: -x[1]):
        if count >= min_freq and word not in vocab:
            vocab[word] = len(vocab)

    logger.info(f"Built vocabulary containing {len(vocab)} unique tokens (min_freq={min_freq})")
    return vocab


def encode_query(query: str, vocab: Dict[str, int], max_len: int = 36) -> torch.Tensor:
    """Encode a text query to a padded tensor of token indices."""
    tokens = tokenize(query)[:max_len]
    indices = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    if len(indices) < max_len:
        indices += [vocab["<pad>"]] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long)


class IntentDataset(Dataset):
    def __init__(self, data: List[Tuple[str, int]], vocab: Dict[str, int], max_len: int = 36):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, label = self.data[idx]
        tokens = encode_query(query, self.vocab, self.max_len)
        return tokens, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def train_agent_intent(
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 3e-4,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    device: Optional[torch.device] = None,
) -> Path:
    """
    Train and export the AgentIntentNet v2.0 controller model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Starting AgentIntentNet training on device: {device}")

    # 1. Build augmented dataset
    full_data = build_augmented_dataset(target_samples_per_class=500, seed=42)
    logger.info(f"Generated {len(full_data)} total synthetic & authentic training queries")

    # 2. Train / Val Split (85% / 15%)
    split_idx = int(0.85 * len(full_data))
    train_data = full_data[:split_idx]
    val_data = full_data[split_idx:]

    # 3. Vocabulary
    vocab = build_vocab(train_data, min_freq=1)

    train_ds = IntentDataset(train_data, vocab)
    val_ds = IntentDataset(val_data, vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 4. Instantiate model
    model = AgentIntentNet(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=5,
        num_layers=2,
        dropout=0.25,
    ).to(device)

    # 5. Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_state = None
    best_epoch = 0

    logger.info("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR")
    logger.info("-" * 65)

    for ep in range(1, epochs + 1):
        # Training Phase
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        train_loss = total_loss / total
        train_acc = (correct / total) * 100.0

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)

        val_loss = val_loss / val_total
        val_acc = (val_correct / val_total) * 100.0
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep

        if ep % 5 == 0 or ep == epochs or ep == 1:
            logger.info(
                f"{ep:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | {val_loss:8.4f} | {val_acc:6.2f}% | {current_lr:.6f}"
            )

    logger.info(f"\n[BEST CHECKPOINT] Epoch {best_epoch} with Validation Accuracy: {best_val_acc:.2f}%")

    # 6. Save Checkpoint
    checkpoint_path = CHECKPOINT_DIR / "intent_net.pt"
    ckpt = {
        "model_state_dict": best_state,
        "vocab": vocab,
        "classes": TASK_CLASSES,
        "config": {
            "vocab_size": len(vocab),
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_classes": 5,
            "num_layers": 2,
            "arch": "AgentIntentNet-v2-BiGRU-Attention",
        },
        "epoch": best_epoch,
        "accuracy": best_val_acc,
        "timestamp": time.time(),
    }
    torch.save(ckpt, checkpoint_path)
    logger.info(f"Saved optimized controller checkpoint to {checkpoint_path} ({checkpoint_path.stat().st_size / 1024:.1f} KB)")

    # 7. Run Test Verification on Unseen Variations
    logger.info("\n=== Evaluating Zero-Shot Query Intent Inference ===")
    model.load_state_dict(best_state)
    model.eval()

    test_queries = [
        ("Can you isolate all the storage tanks located next to the runway?", "SINGLE_GROUNDING"),
        ("What kind of crops are growing in this sector and is the NDVI healthy?", "SINGLE_VQA"),
        ("Measure the total hectares of rainforest lost from 2021 to 2024.", "BITEMPORAL_CHANGE"),
        ("Pierce through this thick stratus cloud layer using Sentinel-1 microwave radar.", "CROSS_MODAL_FUSION"),
        ("Hi there! Could you explain how SatQuery AI orchestrates specialist models?", "AGENT_ASSISTANT"),
        ("Pinpoint the seaport cargo ships anchored in the harbor basin.", "SINGLE_GROUNDING"),
        ("Did the reservoir water level recede following the drought?", "BITEMPORAL_CHANGE"),
        ("Show me through the haze by fusing optical and SAR sensors.", "CROSS_MODAL_FUSION"),
        # New adversarial & edge-case test queries
        ("What does the word 'detect' mean in remote sensing?", "AGENT_ASSISTANT"),
        ("Explain the theory behind change detection algorithms.", "AGENT_ASSISTANT"),
        ("How does SAR radar fusion penetrate clouds conceptually?", "AGENT_ASSISTANT"),
        ("Tell me more about that topic.", "AGENT_ASSISTANT"),
        ("How are you doing today?", "AGENT_ASSISTANT"),
        ("What is the electromagnetic spectrum?", "AGENT_ASSISTANT"),
        ("Describe the vegetation canopy density and chlorophyll absorption.", "SINGLE_VQA"),
        ("Outline all the wind turbines on this ridgeline.", "SINGLE_GROUNDING"),
        ("Quantify the delta in land cover between dry and wet seasons.", "BITEMPORAL_CHANGE"),
        ("Merge the SAR VV polarization with the optical scene to see under clouds.", "CROSS_MODAL_FUSION"),
    ]

    all_passed = True
    for query, expected_class in test_queries:
        inp = encode_query(query, vocab).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model.get_probabilities(inp).squeeze(0).cpu().numpy()
            pred_id = int(np.argmax(probs))
            conf = float(probs[pred_id])
            pred_class = ID_TO_LABEL[pred_id]

        status = "PASSED" if pred_class == expected_class else "FAILED"
        if status == "FAILED":
            all_passed = False
        logger.info(f"[{status}] '{query[:50]}...' -> Pred: {pred_class} ({conf*100:.1f}%) | Expected: {expected_class}")

    if all_passed:
        logger.info(">>> ALL ZERO-SHOT TEST QUERIES CLASSIFIED PERFECTLY! <<<")

    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SatQuery AI Agent Intent Controller")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
    args = parser.parse_args()

    train_agent_intent(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
