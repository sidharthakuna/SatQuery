"""
SatQuery AI — Inference Factory
Switches between Mock and CUDA inference providers based on INFERENCE_MODE.
"""

import logging
from typing import Optional

from config.settings import InferenceMode, settings

logger = logging.getLogger(__name__)

# Singleton instances
_mock_provider = None
_cuda_provider = None


class InferenceFactory:
    """
    Factory that returns the appropriate inference provider
    based on the INFERENCE_MODE environment variable.

    - MOCK: Instant synthetic outputs for CPU-only development
    - CUDA: Real GPU inference with loaded model weights
    """

    @staticmethod
    def get_provider():
        """
        Get or create the appropriate inference provider singleton.

        Returns:
            MockProvider or PyTorchProvider instance
        """
        global _mock_provider, _cuda_provider

        if settings.inference_mode == InferenceMode.MOCK:
            if _mock_provider is None:
                from app.inference.providers.mock_provider import MockProvider
                _mock_provider = MockProvider()
                logger.info("Initialized MOCK inference provider")
            return _mock_provider

        elif settings.inference_mode in (InferenceMode.CUDA, InferenceMode.CPU, InferenceMode.NEURAL):
            if _cuda_provider is None:
                from app.inference.providers.pytorch_provider import PyTorchProvider
                _cuda_provider = PyTorchProvider()
                logger.info(f"Initialized PyTorch inference provider ({settings.inference_mode.value})")
            return _cuda_provider

        else:
            raise ValueError(f"Unknown inference mode: {settings.inference_mode}")

    @staticmethod
    def get_mode() -> str:
        """Return the current inference mode string."""
        return settings.inference_mode.value

    @staticmethod
    def is_mock() -> bool:
        """Check if running in mock mode."""
        return settings.inference_mode == InferenceMode.MOCK

    @staticmethod
    def is_cuda() -> bool:
        """Check if running in neural/accelerated mode."""
        return settings.inference_mode in (InferenceMode.CUDA, InferenceMode.CPU, InferenceMode.NEURAL)
