"""
SatQuery AI — Mock Inference Provider
Generates domain-realistic synthetic outputs for all 4 tasks
without requiring GPU hardware or model weights.
"""

import logging
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np

from config.constants import MOCK_LATENCY_RANGE_MS

logger = logging.getLogger(__name__)


class MockProvider:
    """
    Mock inference provider for CPU-only development and testing.
    Generates synthetic outputs that mimic real model responses:
    - Realistic domain vocabulary
    - Plausible confidence scores
    - Synthetic spatial outputs (masks, bounding boxes)
    - Simulated inference latency
    """

    def __init__(self):
        logger.info("MockProvider initialized (no GPU required)")
        self._initialized = True

    def predict_vqa(
        self,
        image: np.ndarray,
        query: str,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """Generate a synthetic VQA response with grounded overlay."""
        self._simulate_latency()
        from app.core.geospatial.grounded_analyzer import GroundedRSAnalyzer
        res = GroundedRSAnalyzer.analyze_single_scene(image, query=query)
        vqa_grounding = res.get("vqa_grounding")
        return {
            "text": res["answer"],
            "confidence": vqa_grounding.get("confidence", 0.88) if vqa_grounding else 0.88,
            "vqa_grounding": vqa_grounding,
            "mock": True,
        }

    def predict_grounding(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        """Generate synthetic bounding boxes."""
        self._simulate_latency()
        h, w = image.shape[-2:]
        count = random.randint(1, 5)
        boxes = []
        for _ in range(count):
            x1 = random.randint(0, w - 50)
            y1 = random.randint(0, h - 50)
            x2 = min(x1 + random.randint(30, 100), w)
            y2 = min(y1 + random.randint(30, 100), h)
            boxes.append([float(x1), float(y1), float(x2), float(y2)])

        return {
            "boxes": boxes,
            "labels": [text_prompt] * count,
            "scores": [round(random.uniform(0.7, 0.98), 2) for _ in range(count)],
            "mock": True,
        }

    def predict_change(
        self,
        image_t1: np.ndarray,
        image_t2: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Generate a synthetic change detection mask."""
        self._simulate_latency(300, 900)
        h, w = image_t1.shape[-2:]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Random change blobs
        for _ in range(random.randint(2, 4)):
            cx, cy = random.randint(50, w - 50), random.randint(50, h - 50)
            radius = random.randint(20, 60)
            yy, xx = np.ogrid[:h, :w]
            circle = (xx - cx) ** 2 + (yy - cy) ** 2
            mask[circle <= radius ** 2] = 1

        return {
            "mask": mask,
            "changed_pixels": int(np.sum(mask > 0)),
            "total_pixels": h * w,
            "mock": True,
        }

    def predict_fusion(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
        fusion_mode: str = "cross_attention",
    ) -> Dict[str, Any]:
        """Generate a synthetic fusion result."""
        self._simulate_latency(400, 1200)
        h, w = optical.shape[-2:]
        fusion_map = np.random.rand(h, w).astype(np.float32) * 0.3 + 0.5
        cloud_mask = np.zeros((h, w), dtype=np.uint8)

        # Simulate cloud regions resolved by SAR
        for _ in range(random.randint(1, 3)):
            cx, cy = random.randint(0, w), random.randint(0, h)
            rx, ry = random.randint(40, 100), random.randint(40, 100)
            yy, xx = np.ogrid[:h, :w]
            ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
            cloud_mask[ellipse <= 1.0] = 1

        return {
            "fusion_map": fusion_map,
            "cloud_mask": cloud_mask,
            "cloud_coverage_pct": round(np.sum(cloud_mask) / (h * w) * 100, 1),
            "mock": True,
        }

    def _simulate_latency(self, min_ms: int = None, max_ms: int = None):
        """Add realistic processing delay."""
        lo = min_ms or MOCK_LATENCY_RANGE_MS[0]
        hi = max_ms or MOCK_LATENCY_RANGE_MS[1]
        delay_s = random.randint(lo, hi) / 1000.0
        time.sleep(delay_s)

    def predict_intent(self, query: str) -> Dict[str, Any]:
        """Generate intent prediction using real IntentNet if checkpoint exists, or accurate multi-task fallback."""
        from config.settings import settings
        ckpt = settings.data_dir / "checkpoints" / "intent_net.pt"
        if ckpt.exists():
            try:
                from app.models.intent_net import load_intent_model
                model, vocab = load_intent_model(ckpt)
                if model:
                    return model.predict(query, vocab)
            except Exception:
                pass

        q = query.lower()
        has_grounding = any(w in q for w in ["find", "locate", "box", "tank", "runway", "building", "highlight", "segment", "outline"])
        has_change = any(w in q for w in ["change", "changed", "deforestation", "expand", "retreat", "recede", "compare", "temporal"])
        has_fusion = any(w in q for w in ["cloud", "fuse", "fusion", "radar", "sar", "penetrate"])
        has_chat = any(w in q for w in ["hi", "hello", "hey", "help", "who are you"])

        if (has_change and has_grounding) or (has_fusion and has_grounding) or ("and also" in q) or ("as well as" in q):
            task = "MULTI_MODEL"
        elif has_fusion:
            task = "CROSS_MODAL_FUSION"
        elif has_change:
            task = "BITEMPORAL_CHANGE"
        elif has_grounding:
            task = "SINGLE_GROUNDING"
        elif has_chat:
            task = "AGENT_ASSISTANT"
        else:
            task = "SINGLE_VQA"

        return {
            "task_type": task,
            "confidence": 0.95,
            "probabilities": {task: 0.95},
            "model": "MockIntentProvider",
        }

