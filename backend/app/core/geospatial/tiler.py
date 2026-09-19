"""
SatQuery AI — Sliding Window Chip Inferencer
Prevents OOM crashes on gigapixel GeoTIFFs by tiling into overlapping chips
with Gaussian-blended seams.
"""

import logging
from typing import Callable

import numpy as np

from config.constants import DEFAULT_CHIP_SIZE, DEFAULT_STRIDE

logger = logging.getLogger(__name__)


class SlidingWindowChipInferencer:
    """
    Processes arbitrarily large GeoTIFFs without CUDA OOM by:
    1. Tiling the raster into chip_size × chip_size patches
    2. Running model inference on each chip
    3. Blending overlapping predictions with a sinusoidal (Gaussian-like) window
    """

    def __init__(self, chip_size: int = DEFAULT_CHIP_SIZE, stride: int = DEFAULT_STRIDE):
        self.chip_size = chip_size
        self.stride = stride
        self._blend_window = self._build_blend_window()

    def _build_blend_window(self) -> np.ndarray:
        """Creates a 2D raised-cosine blending window to smooth chip boundaries without zero edges."""
        y_win = 0.2 + 0.8 * np.sin(np.linspace(0, np.pi, self.chip_size))
        x_win = 0.2 + 0.8 * np.sin(np.linspace(0, np.pi, self.chip_size))
        return np.outer(y_win, x_win).astype(np.float32)

    def run_tiled_inference(
        self,
        geotiff_path: str,
        model_fn: Callable[[np.ndarray], np.ndarray],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Runs sliding-window inference over an entire GeoTIFF.

        Args:
            geotiff_path: Path to input GeoTIFF raster
            model_fn: Callable that takes a (C, chip_size, chip_size) float32 array
                       and returns a (chip_size, chip_size) prediction map
            threshold: Binarization threshold for final mask

        Returns:
            Binary uint8 mask of shape (H, W) where 1 = positive prediction
        """
        import rasterio
        from rasterio.windows import Window

        with rasterio.open(geotiff_path) as src:
            H, W = src.height, src.width
            num_bands = src.count
            pred_accum = np.zeros((H, W), dtype=np.float32)
            weight_accum = np.zeros((H, W), dtype=np.float32)

            total_chips = 0

            for y in range(0, H, self.stride):
                for x in range(0, W, self.stride):
                    # Compute actual window dimensions (may be smaller at edges)
                    win_w = min(self.chip_size, W - x)
                    win_h = min(self.chip_size, H - y)

                    # Read chip from raster
                    chip = src.read(
                        window=Window(x, y, win_w, win_h)
                    ).astype(np.float32)

                    # Zero-pad to full chip size if at edge
                    padded = np.zeros(
                        (num_bands, self.chip_size, self.chip_size),
                        dtype=np.float32,
                    )
                    padded[:, :win_h, :win_w] = chip

                    # Run model inference
                    pred = np.squeeze(model_fn(padded))
                    if pred.ndim != 2:
                        continue

                    # Blend into accumulator
                    blend = self._blend_window[:win_h, :win_w]
                    pred_accum[y:y + win_h, x:x + win_w] += pred[:win_h, :win_w] * blend
                    weight_accum[y:y + win_h, x:x + win_w] += blend
                    total_chips += 1

            # Normalize and binarize
            safe_weights = np.maximum(weight_accum, 1e-6)
            blended = pred_accum / safe_weights
            binary_mask = (blended > threshold).astype(np.uint8)

            logger.info(
                f"Tiled inference complete: {total_chips} chips processed "
                f"({H}x{W} → {self.chip_size}×{self.chip_size} @ stride {self.stride})"
            )
            return binary_mask

    def run_tiled_inference_from_array(
        self,
        data: np.ndarray,
        model_fn: Callable[[np.ndarray], np.ndarray],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Same as run_tiled_inference but operates on an in-memory array
        instead of reading from a file.

        Args:
            data: Float32 array of shape (C, H, W)
            model_fn: Callable that takes (C, chip_size, chip_size) → (chip_size, chip_size)
            threshold: Binarization threshold

        Returns:
            Binary uint8 mask of shape (H, W)
        """
        C, H, W = data.shape
        pred_accum = np.zeros((H, W), dtype=np.float32)
        weight_accum = np.zeros((H, W), dtype=np.float32)

        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                win_w = min(self.chip_size, W - x)
                win_h = min(self.chip_size, H - y)

                padded = np.zeros((C, self.chip_size, self.chip_size), dtype=np.float32)
                padded[:, :win_h, :win_w] = data[:, y:y + win_h, x:x + win_w]

                pred = np.squeeze(model_fn(padded))
                if pred.ndim != 2:
                    continue
                blend = self._blend_window[:win_h, :win_w]
                pred_accum[y:y + win_h, x:x + win_w] += pred[:win_h, :win_w] * blend
                weight_accum[y:y + win_h, x:x + win_w] += blend

        safe_weights = np.maximum(weight_accum, 1e-6)
        return ((pred_accum / safe_weights) > threshold).astype(np.uint8)
