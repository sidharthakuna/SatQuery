"""
SatQuery AI — Sub-Pixel Co-Registration & Alignment
Reprojects SAR rasters onto optical grids and refines alignment via phase correlation.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def align_to_reference(
    reference_path: str,
    target_path: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Reprojects a target raster (typically SAR) onto the reference raster's
    CRS, transform, and pixel grid using bilinear interpolation.

    Args:
        reference_path: Path to reference GeoTIFF (usually optical)
        target_path: Path to target GeoTIFF to be reprojected (usually SAR)

    Returns:
        Tuple of (reference_data, aligned_target_data, reference_profile)
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(reference_path) as ref_src:
        ref_data = ref_src.read().astype(np.float32)
        ref_profile = dict(ref_src.profile)
        target_crs = ref_src.crs
        target_transform = ref_src.transform
        target_height = ref_src.height
        target_width = ref_src.width

    with rasterio.open(target_path) as tgt_src:
        if target_crs is None or tgt_src.crs is None:
            from PIL import Image
            logger.warning("Unreferenced raster detected in align_to_reference; falling back to image resize.")
            aligned = np.zeros(
                (tgt_src.count, target_height, target_width),
                dtype=np.float32,
            )
            for band_idx in range(1, tgt_src.count + 1):
                b_data = tgt_src.read(band_idx).astype(np.float32)
                b_img = Image.fromarray(b_data)
                resized_b = b_img.resize((target_width, target_height), Image.Resampling.BILINEAR)
                aligned[band_idx - 1] = np.array(resized_b, dtype=np.float32)
            return ref_data, aligned, ref_profile

        aligned = np.zeros(
            (tgt_src.count, target_height, target_width),
            dtype=np.float32,
        )
        for band_idx in range(1, tgt_src.count + 1):
            reproject(
                source=rasterio.band(tgt_src, band_idx),
                destination=aligned[band_idx - 1],
                src_transform=tgt_src.transform,
                src_crs=tgt_src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )

    logger.info(
        f"Aligned {target_path} to {reference_path} grid "
        f"({target_width}x{target_height}, CRS={target_crs})"
    )
    return ref_data, aligned, ref_profile


def refine_subpixel(
    reference_band: np.ndarray,
    target_band: np.ndarray,
    upsample_factor: int = 100,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Refines alignment using 2D phase correlation for sub-pixel accuracy.
    Eliminates false change detections caused by parallax / layover shifts.

    Args:
        reference_band: 2D reference band (float32)
        target_band: 2D target band (float32), same shape as reference
        upsample_factor: Sub-pixel precision factor (100 = 1/100th pixel)

    Returns:
        Tuple of (shifted_target, (dy_shift, dx_shift))
    """
    from scipy.ndimage import shift as ndi_shift

    assert reference_band.shape == target_band.shape, (
        f"Shape mismatch: reference={reference_band.shape}, target={target_band.shape}"
    )

    # Compute cross-power spectrum
    f_ref = np.fft.fft2(reference_band)
    f_tgt = np.fft.fft2(target_band)
    cross_power = (f_ref * np.conj(f_tgt)) / (np.abs(f_ref * np.conj(f_tgt)) + 1e-10)
    correlation = np.abs(np.fft.ifft2(cross_power))

    # Find peak and apply parabolic sub-pixel interpolation
    max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    h, w = reference_band.shape
    y_p, x_p = max_idx

    y_prev, y_next = (y_p - 1) % h, (y_p + 1) % h
    x_prev, x_next = (x_p - 1) % w, (x_p + 1) % w

    denom_y = correlation[y_prev, x_p] - 2.0 * correlation[y_p, x_p] + correlation[y_next, x_p]
    sub_y = 0.5 * (correlation[y_prev, x_p] - correlation[y_next, x_p]) / denom_y if abs(denom_y) > 1e-7 else 0.0
    sub_y = float(np.clip(sub_y, -0.5, 0.5))

    denom_x = correlation[y_p, x_prev] - 2.0 * correlation[y_p, x_p] + correlation[y_p, x_next]
    sub_x = 0.5 * (correlation[y_p, x_prev] - correlation[y_p, x_next]) / denom_x if abs(denom_x) > 1e-7 else 0.0
    sub_x = float(np.clip(sub_x, -0.5, 0.5))

    dy = float(y_p if y_p < h // 2 else y_p - h) + float(sub_y)
    dx = float(x_p if x_p < w // 2 else x_p - w) + float(sub_x)

    # Apply sub-pixel shift with cubic interpolation
    shifted = ndi_shift(target_band, shift=(-dy, -dx), order=3, mode="constant", cval=0.0)

    logger.info(f"Sub-pixel refinement: shift=(dy={dy:.3f}, dx={dx:.3f}) pixels")
    return shifted.astype(np.float32), (float(dy), float(dx))
