"""
SatQuery AI — Remote Sensing Vision-Language Model (RS-VLM with LoRA)
Specialist model integrating multi-scale convolutional visual tokenization,
LoRA cross-attention projection, and transformer decoding for natural language VQA.
"""

from typing import List, Dict, Optional
import torch
import torch.nn as nn


RS_VLM_WORDS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "0",
    "10",
    "12",
    "124",
    "12840",
    "18",
    "2",
    "20",
    "22",
    "25",
    "28",
    "3",
    "38",
    "4",
    "46",
    "5",
    "55",
    "6",
    "60",
    "62",
    "65",
    "7",
    "78",
    "8",
    "81",
    "82",
    "83",
    "85",
    "87",
    "88",
    "90",
    "92",
    "a",
    "access",
    "across",
    "aeolian",
    "agricultural",
    "agriculture",
    "albedo",
    "all",
    "allotment",
    "along",
    "alpine",
    "altitude",
    "an",
    "anchored",
    "and",
    "any",
    "approach",
    "approximately",
    "are",
    "area",
    "areas",
    "arid",
    "arterial",
    "at",
    "backscatter",
    "bare",
    "barges",
    "barrier",
    "basin",
    "basins",
    "bays",
    "beaches",
    "berth",
    "berthed",
    "berths",
    "biomass",
    "bodies",
    "body",
    "boundaries",
    "boundary",
    "brook",
    "buildings",
    "built",
    "built-up",
    "but",
    "by",
    "canal",
    "canopy",
    "capped",
    "cargo",
    "cartosat",
    "central",
    "chalets",
    "changes",
    "channel",
    "characteristic",
    "chlorophyll",
    "city",
    "clear",
    "clearing",
    "cloud",
    "clouds",
    "cloudy",
    "cluster",
    "clusters",
    "co",
    "coast",
    "coastal",
    "commercial",
    "compared",
    "complex",
    "concentrated",
    "concrete",
    "confidence",
    "confirmed",
    "construction",
    "contains",
    "contiguous",
    "cooling",
    "corridor",
    "corridors",
    "cover",
    "crop",
    "cropland",
    "cultivated",
    "deciduous",
    "deep",
    "deforestation",
    "delineated",
    "dense",
    "describe",
    "desert",
    "detected",
    "detection",
    "disturbance",
    "docks",
    "dominates",
    "drainage",
    "dune",
    "dunes",
    "dynamics",
    "eastern",
    "eight",
    "eighteen",
    "eighty",
    "elevation",
    "embankm2ent",
    "emission",
    "erected",
    "eroded",
    "erosion",
    "estimated",
    "estuarine",
    "expanse",
    "expansion",
    "extension",
    "extensive",
    "extent",
    "facilities",
    "facility",
    "farm",
    "farmland",
    "farmstead",
    "feature",
    "features",
    "featuring",
    "field",
    "fields",
    "fifty",
    "five",
    "flank",
    "flood",
    "flooded",
    "flooding",
    "footprint",
    "forest",
    "forested",
    "forty",
    "four",
    "freshwater",
    "gardens",
    "geometric",
    "glacial",
    "glaciated",
    "hangar",
    "harbor",
    "harvesting",
    "haul",
    "healthy",
    "heavy",
    "hectares",
    "high",
    "highlighted",
    "highway",
    "hills",
    "hillside",
    "how",
    "hundred",
    "hyper",
    "identify",
    "image",
    "in",
    "industrial",
    "infrastructure",
    "inland",
    "interface",
    "interspersed",
    "inundation",
    "irrigated",
    "irrigation",
    "is",
    "isolated",
    "its",
    "km2",
    "km22",
    "lacustrine",
    "lake",
    "lakes",
    "land",
    "landlocked",
    "landsat",
    "landscape",
    "latitude",
    "leading",
    "length",
    "levee",
    "linear",
    "loading",
    "local",
    "logging",
    "logistics",
    "longitude",
    "low",
    "main",
    "major",
    "mangrove",
    "mangroves",
    "manufacturing",
    "many",
    "margins",
    "maritime",
    "market",
    "medium",
    "meters",
    "method",
    "microwave",
    "migration",
    "minimal",
    "mix",
    "mixed",
    "moderate",
    "modulation",
    "most",
    "mountain",
    "mountainous",
    "mudflats",
    "multispectral",
    "municipal",
    "natural",
    "navigation",
    "ndvi",
    "network",
    "networks",
    "new",
    "nine",
    "ninety",
    "nir",
    "no",
    "northern",
    "object",
    "occupying",
    "of",
    "one",
    "open",
    "optical",
    "or",
    "other",
    "outcrops",
    "outlined",
    "outposts",
    "parcels",
    "pass",
    "pasture",
    "paved",
    "pavement",
    "peaks",
    "penetration",
    "percent",
    "permanent",
    "photosynthesis",
    "pixels",
    "plantation",
    "plateau",
    "plots",
    "polarimetric",
    "pond",
    "port",
    "preparation",
    "present",
    "previous",
    "primarily",
    "primary",
    "punctuated",
    "radar",
    "rail",
    "rangeland",
    "recession",
    "red",
    "reflectance",
    "region",
    "regions",
    "registered",
    "reservoir",
    "residential",
    "resolved",
    "retention",
    "ridge",
    "ridgeline",
    "ridges",
    "riparian",
    "river",
    "riverine",
    "road",
    "roads",
    "rock",
    "rocky",
    "rooftop",
    "rotation",
    "rugged",
    "runway",
    "rural",
    "sand",
    "sar",
    "satellite",
    "scattered",
    "scene",
    "scrub",
    "scrubland",
    "sea",
    "season",
    "seasonal",
    "sector",
    "sedimentary",
    "segmentation",
    "sensor",
    "sentinel",
    "settlement",
    "seven",
    "seventy",
    "shadowed",
    "shelter",
    "shipping",
    "ships",
    "shoreline",
    "short",
    "show",
    "shown",
    "signature",
    "significant",
    "single",
    "six",
    "sixty",
    "small",
    "snow",
    "snowline",
    "soil",
    "southern",
    "sparse",
    "spatial",
    "spectral",
    "spurs",
    "steep",
    "storage",
    "stream",
    "strong",
    "structures",
    "subalpine",
    "substrate",
    "surface",
    "surrounded",
    "surrounding",
    "switchback",
    "tanks",
    "tarn",
    "temperate",
    "ten",
    "terminal",
    "terraced",
    "terrain",
    "the",
    "them",
    "there",
    "thermal",
    "thirty",
    "this",
    "thousand",
    "three",
    "tidal",
    "timber",
    "to",
    "topography",
    "total",
    "tracks",
    "transit",
    "transiting",
    "transport",
    "turbid",
    "turbidity",
    "twelve",
    "twenty",
    "two",
    "type",
    "types",
    "under",
    "understanding",
    "units",
    "unpaved",
    "up",
    "urban",
    "valley",
    "valleys",
    "vast",
    "vegetation",
    "vegetative",
    "vessels",
    "visible",
    "warehouse",
    "water",
    "waters",
    "weather",
    "wetland",
    "wetlands",
    "wharf",
    "what",
    "winding",
    "with",
    "yes",
    "zero",
    "zone",
    "zones"
]

RS_VLM_VOCAB: Dict[str, int] = {w: i for i, w in enumerate(RS_VLM_WORDS)}
RS_VLM_ID2WORD: Dict[int, str] = {i: w for i, w in enumerate(RS_VLM_WORDS)}


def tokenize_query(text: str, max_len: int = 20, vocab: Optional[Dict[str, int]] = None) -> List[int]:
    v = vocab or RS_VLM_VOCAB
    unk_id = v.get("<unk>", 1)
    pad_id = v.get("<pad>", 0)
    cleaned_words = [w.strip("?.,!;:\"'()").lower() for w in text.split()]
    tokens = [v.get(w, unk_id) for w in cleaned_words if w][:max_len]
    tokens += [pad_id] * (max_len - len(tokens))
    return tokens



class LoRALinear(nn.Module):
    """Low-Rank Adaptation Linear Layer."""
    def __init__(self, in_f: int, out_f: int, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = nn.Linear(in_f, out_f)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling


class RSVisionLanguageModel(nn.Module):
    """
    Enhanced RS-VLM: 256-dim, 4-layer transformer decoder, LoRA projection.
    Matches trained checkpoint backend/data/checkpoints/rs_vlm.pt.
    """
    def __init__(self, vocab_size: int = len(RS_VLM_WORDS), embed_dim: int = 256, lora_rank: int = 16, vocab: Optional[Dict[str, int]] = None):
        super().__init__()
        self.vocab = vocab or RS_VLM_VOCAB
        self.id2word = {i: w for w, i in self.vocab.items()}
        self.vocab_size = vocab_size

        self.vision_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, embed_dim, 3, stride=2, padding=1), nn.BatchNorm2d(embed_dim),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.vision_proj = LoRALinear(embed_dim * 16, embed_dim, rank=lora_rank, alpha=32.0)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.pos_encoder = nn.Parameter(torch.randn(1, 50, embed_dim) * 0.01)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=4)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, images: torch.Tensor, q_ids: torch.Tensor, a_ids: torch.Tensor) -> torch.Tensor:
        b = images.size(0)
        vis = self.vision_backbone(images)
        vis_tokens = self.vision_proj(vis.view(b, -1)).unsqueeze(1)
        q_embed = self.embedding(q_ids)
        memory = torch.cat([vis_tokens, q_embed], dim=1)
        tgt = self.embedding(a_ids) + self.pos_encoder[:, :a_ids.size(1), :]
        seq_len = a_ids.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=images.device)
        decoded = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        return self.lm_head(decoded)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, q_ids: torch.Tensor, max_len: int = 40, temperature: float = 0.0, top_k: int = 20) -> List[str]:
        """Autoregressive decoding with repetition penalty and optional sampling."""
        b = images.size(0)
        vis = self.vision_backbone(images)
        vis_tokens = self.vision_proj(vis.view(b, -1)).unsqueeze(1)
        q_embed = self.embedding(q_ids)
        memory = torch.cat([vis_tokens, q_embed], dim=1)

        bos_id = self.vocab.get("<bos>", 2)
        eos_id = self.vocab.get("<eos>", 3)
        cur_ids = torch.full((b, 1), bos_id, dtype=torch.long, device=images.device)

        common_exempt = {self.vocab.get(w) for w in ["a", "an", "the", "and", "of", "in", "with", "to", "is", "are", "present", "visible", "scene"] if w in self.vocab}

        for _ in range(max_len):
            seq_len = cur_ids.size(1)
            tgt = self.embedding(cur_ids) + self.pos_encoder[:, :seq_len, :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=images.device)
            decoded = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            step_logits = self.lm_head(decoded[:, -1, :]).clone()

            for i in range(b):
                seen_tokens = set(cur_ids[i].tolist())
                for tid in seen_tokens:
                    if tid > 3 and tid not in common_exempt:
                        step_logits[i, tid] -= 1.4

            if temperature > 0.0:
                step_logits = step_logits / max(temperature, 0.1)
                # Top-k filtering
                vals, idxs = torch.topk(step_logits, min(top_k, step_logits.size(-1)), dim=-1)
                probs = torch.softmax(vals, dim=-1)
                choice = torch.multinomial(probs, num_samples=1)
                next_id = torch.gather(idxs, -1, choice)
            else:
                next_id = step_logits.argmax(dim=-1, keepdim=True)

            cur_ids = torch.cat([cur_ids, next_id], dim=1)
            if (next_id == eos_id).all():
                break

        results = []
        for seq in cur_ids:
            words = []
            for tid in seq[1:]:  # skip <bos>
                tid_i = tid.item()
                if tid_i in (eos_id, self.vocab.get("<pad>", 0)):
                    break
                w = self.id2word.get(tid_i, "")
                if w and not w.startswith("<"):
                    words.append(w)
            results.append(" ".join(words))
        return results



# Aliases for compatibility
RSVisionLanguageModel_LoRA = RSVisionLanguageModel
