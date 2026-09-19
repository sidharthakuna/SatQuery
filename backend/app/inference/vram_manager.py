"""
SatQuery AI — Dynamic GPU VRAM Manager
LRU model caching that automatically evicts inactive models to CPU RAM,
allowing all 4 specialist models to run on standard 8-12 GB laptop GPUs.
"""

import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class VRAMManager:
    """
    LRU-based GPU VRAM manager.

    Tracks loaded models and their approximate VRAM usage.
    When a new model needs loading and VRAM is insufficient,
    the least-recently-used model is evicted to CPU RAM.
    """

    # Approximate VRAM requirements per model (in GB)
    MODEL_VRAM_ESTIMATES: Dict[str, float] = {
        "rs_vlm": 3.5,          # GeoChat / LLaVA-1.5 7B (quantized)
        "grounding_dino": 1.8,  # Grounding DINO-B
        "sam_rs": 1.2,          # SAM-RS ViT-B
        "changeformer": 2.0,    # ChangeFormer-V6
        "cross_attention": 2.5, # Optical-SAR fusion network
    }

    def __init__(self, max_vram_gb: Optional[float] = None):
        self.max_vram_gb = max_vram_gb or settings.max_vram_gb
        self._loaded: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._total_used_gb: float = 0.0

    def request_model(self, model_id: str) -> bool:
        """
        Request VRAM allocation for a model.
        Evicts LRU models if needed to make room.

        Args:
            model_id: Identifier matching MODEL_VRAM_ESTIMATES keys

        Returns:
            True if model can be loaded (VRAM available or freed)
        """
        # Already loaded — move to end (most recently used)
        if model_id in self._loaded:
            self._loaded.move_to_end(model_id)
            self._loaded[model_id]["last_used"] = time.time()
            logger.debug(f"VRAM cache hit: {model_id}")
            return True

        required_gb = self.MODEL_VRAM_ESTIMATES.get(model_id, 2.0)

        # Evict LRU models until we have enough space
        while self._total_used_gb + required_gb > self.max_vram_gb and self._loaded:
            evicted_id, evicted_info = self._loaded.popitem(last=False)
            self._total_used_gb -= evicted_info["vram_gb"]
            self._evict_to_cpu(evicted_id)

        # Check if we can fit
        if self._total_used_gb + required_gb > self.max_vram_gb:
            logger.error(
                f"Cannot fit model '{model_id}' ({required_gb:.1f} GB) "
                f"in VRAM ({self.max_vram_gb:.1f} GB total, "
                f"{self._total_used_gb:.1f} GB used)"
            )
            return False

        # Allocate
        self._loaded[model_id] = {
            "vram_gb": required_gb,
            "loaded_at": time.time(),
            "last_used": time.time(),
        }
        self._total_used_gb += required_gb
        logger.info(
            f"VRAM allocated: {model_id} ({required_gb:.1f} GB). "
            f"Total used: {self._total_used_gb:.1f}/{self.max_vram_gb:.1f} GB"
        )
        return True

    def release_model(self, model_id: str):
        """Explicitly release a model's VRAM allocation."""
        if model_id in self._loaded:
            info = self._loaded.pop(model_id)
            self._total_used_gb -= info["vram_gb"]
            self._evict_to_cpu(model_id)

    def _evict_to_cpu(self, model_id: str):
        """Evict a model from GPU to CPU RAM."""
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info(f"Evicted model '{model_id}' from GPU to CPU RAM")

    def get_status(self) -> Dict[str, Any]:
        """Return current VRAM allocation status."""
        return {
            "max_vram_gb": self.max_vram_gb,
            "used_vram_gb": round(self._total_used_gb, 2),
            "free_vram_gb": round(self.max_vram_gb - self._total_used_gb, 2),
            "loaded_models": list(self._loaded.keys()),
            "model_count": len(self._loaded),
        }


# Singleton instance
vram_manager = VRAMManager()
