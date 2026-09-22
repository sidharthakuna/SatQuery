"""
SatQuery AI — Radiometric Calibration
Optical percentile stretch and SAR dB conversion per the master blueprint.
"""

import logging

import numpy as np

from config.constants import (
    OPTICAL_PERCENTILE_HIGH,
    OPTICAL_PERCENTILE_LOW,
    SAR_DB_MAX,
    SAR_DB_MIN,
    SAR_EPSILON,
)

logger = logging.getLogger(__name__)


def normalize_optical(band_data: np.ndarray) -> np.ndarray:
    """
    Normalize a single optical band using 2%–98% percentile linear stretch.

    Handles 12-bit/16-bit DN values from Cartosat-2S and Sentinel-2,
    as well as 8-bit masks and rasters with solid or zero backgrounds.
    Avoids cloud washout by ignoring extreme outliers.

    Args:
        band_data: 2D float32 array of DN values

    Returns:
        Normalized float32 array clipped to [0.0, 1.0]
    """
    finite_mask = np.isfinite(band_data)
    if not np.any(finite_mask):
        return np.zeros_like(band_data, dtype=np.float32)

    max_val = float(np.nanmax(band_data))
    min_val = float(np.nanmin(band_data))

    # If already normalized in [0.0, 1.0]
    if max_val <= 1.0 and min_val >= 0.0:
        return np.clip(band_data, 0.0, 1.0).astype(np.float32)

    # Exclude zeros/nodata if positive signal exists
    valid_mask = finite_mask & (band_data > 0)
    if not np.any(valid_mask):
        return np.zeros_like(band_data, dtype=np.float32)

    has_zeros = bool(np.any(band_data == 0))

    p_low, p_high = np.percentile(
        band_data[valid_mask],
        (OPTICAL_PERCENTILE_LOW, OPTICAL_PERCENTILE_HIGH),
    )

    # If the band has zero background or tight value distribution (e.g. discrete masks)
    if has_zeros or (p_high - p_low) < 5.0 or p_low < 0.0:
        p_low = 0.0

    if max_val <= 255.0 and max_val > 1.0:
        denom = max(float(p_high), 255.0)
    else:
        denom = max(float(p_high - p_low), 1.0)

    return np.clip((band_data - p_low) / denom, 0.0, 1.0).astype(np.float32)


def normalize_optical_toa(band_data: np.ndarray, scale: float = 10000.0) -> np.ndarray:
    """
    Normalize optical TOA reflectance by dividing by scale factor.
    Used for Level-1C Sentinel-2 products.

    Args:
        band_data: 2D float32 array of DN values
        scale: Division factor (default 10000 for Sentinel-2)

    Returns:
        Normalized float32 array clipped to [0.0, 1.0]
    """
    return np.clip(band_data / scale, 0.0, 1.0).astype(np.float32)


def calibrate_sar_to_db(sar_data: np.ndarray, k_cal: float = 0.0) -> np.ndarray:
    """
    Convert raw SAR DN/power values to calibrated backscatter σ⁰ in dB.

    Formula: σ⁰ (dB) = 10 · log₁₀(DN² + ε) - K_cal

    Args:
        sar_data: 2D float32 array of raw SAR amplitude/power
        k_cal: Calibration constant from sensor metadata (default 0.0)

    Returns:
        Float32 array of σ⁰ values in dB (typically -25 to 0 dB)
    """
    val = sar_data.astype(np.float32)
    if val.size > 0 and np.nanmax(val) > 1.01:
        val = val / max(float(np.nanmax(val)), 255.0)

    power = np.maximum(val ** 2, SAR_EPSILON)
    db = 10.0 * np.log10(power) - k_cal
    return db.astype(np.float32)


def normalize_sar(db_data: np.ndarray) -> np.ndarray:
    """
    Normalize SAR dB values to [0.0, 1.0] using standard terrestrial bounds.

    Range: [-25.0 dB, 0.0 dB]
    - Calm water ≈ -22 dB (very dark)
    - Urban double-bounce ≈ 0 to -5 dB (very bright)

    Args:
        db_data: 2D float32 array of σ⁰ values in dB

    Returns:
        Normalized float32 array clipped to [0.0, 1.0]
    """
    denom = SAR_DB_MAX - SAR_DB_MIN  # 0.0 - (-25.0) = 25.0
    return np.clip((db_data - SAR_DB_MIN) / denom, 0.0, 1.0).astype(np.float32)


def full_sar_pipeline(
    sar_data: np.ndarray,
    k_cal: float = 0.0,
) -> np.ndarray:
    """
    Complete SAR calibration pipeline: raw DN → dB → normalized [0,1].
    Handles raw DN [0, 255], normalized [0, 1], and pre-calculated negative dB values.

    Args:
        sar_data: 2D float32 array of raw SAR values
        k_cal: Calibration constant

    Returns:
        Normalized float32 array in [0.0, 1.0]
    """
    if sar_data.size > 0 and np.nanmin(sar_data) < -1.0:
        return normalize_sar(sar_data)
    db = calibrate_sar_to_db(sar_data, k_cal)
    return normalize_sar(db)
