"""
SatQuery AI — GeoTIFF Reader & Metadata Extractor
Handles safe windowed reads of multi-band, multi-gigabyte GeoTIFF rasters.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

from config.constants import (
    DEFAULT_CRS,
    OPTICAL_BAND_COUNTS,
    SAR_BAND_COUNTS,
    SUPPORTED_EXTENSIONS,
    THUMBNAIL_MAX_SIZE,
)
from config.settings import settings
from app.schemas.geospatial import GeoBoundsLatLon, GeoTIFFMetadata

logger = logging.getLogger(__name__)


def inspect_geotiff(file_path: str, file_id: Optional[str] = None) -> GeoTIFFMetadata:
    """
    Reads GeoTIFF headers without loading pixel data.
    Extracts CRS, dimensions, band count, bounds, and auto-detects modality.
    """
    import rasterio
    from pyproj import Transformer

    file_id = file_id or uuid4().hex[:12]
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {file_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {path.suffix}. Expected: {SUPPORTED_EXTENSIONS}")

    with rasterio.open(file_path) as src:
        crs_str = src.crs.to_string() if src.crs else "UNREFERENCED"

        # Transform bounds to WGS84 only if authentic CRS exists
        bounds_latlon = None
        if src.crs is not None:
            try:
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                min_lon, min_lat = transformer.transform(src.bounds.left, src.bounds.bottom)
                max_lon, max_lat = transformer.transform(src.bounds.right, src.bounds.top)
                if abs(min_lat) <= 90 and abs(max_lat) <= 90 and abs(min_lon) <= 180 and abs(max_lon) <= 180:
                    bounds_latlon = GeoBoundsLatLon(
                        min_lat=min_lat, min_lon=min_lon,
                        max_lat=max_lat, max_lon=max_lon,
                    )
            except Exception as e:
                logger.warning(f"CRS transform failed for {file_path}: {e}")

        modality = detect_modality(src.count, file_path)

        # Calculate authentic Ground Sampling Distance (GSD) in meters
        gsd_m = None
        if src.res:
            try:
                res_x = abs(float(src.res[0]))
                if (src.crs and src.crs.is_geographic) or res_x < 0.1:
                    gsd_m = round(res_x * 111320.0, 2)
                else:
                    gsd_m = round(res_x, 2)
            except Exception:
                pass

        if gsd_m is None or gsd_m <= 0:
            fn_lower = path.name.lower()
            if "cartosat" in fn_lower:
                gsd_m = 0.65
            elif "sentinel2" in fn_lower or "s2" in fn_lower:
                gsd_m = 10.0
            elif "landsat" in fn_lower:
                gsd_m = 30.0
            elif "sentinel1" in fn_lower or "sar" in fn_lower or "risat" in fn_lower:
                gsd_m = 10.0
            else:
                gsd_m = 10.0

        thumbnail_url = f"/api/v1/preview/{file_id or path.name}"

        return GeoTIFFMetadata(
            file_path=file_path,
            file_id=file_id,
            crs=crs_str,
            width=src.width,
            height=src.height,
            band_count=src.count,
            dtypes=list(src.dtypes),
            modality=modality,
            resolution=src.res,
            gsd_m=gsd_m,
            bounds_latlon=bounds_latlon,
            file_size_bytes=path.stat().st_size,
            thumbnail_url=thumbnail_url,
        )


def detect_modality(band_count: int, file_path: str = "") -> str:
    """
    Classifies a raster as OPTICAL or SAR based on band count heuristics.
    SAR images typically have 1-2 bands (VV, VH polarizations).
    Optical images have 3+ bands (RGB, NIR, SWIR, etc.).
    """
    fname = Path(file_path).stem.lower() if file_path else ""

    # Explicit filename hints take priority
    if any(k in fname for k in ["sar", "risat", "sentinel1", "s1", "radar", "vv", "vh", "hh", "hv", "grd", "slc"]):
        return "SAR"
    if any(k in fname for k in ["optical", "cartosat", "sentinel2", "s2", "rgb", "pan", "panchromatic", "mono", "gray", "b8"]):
        return "OPTICAL"

    # 2 bands is almost exclusively dual-pol SAR (VV+VH)
    if band_count == 2 and not any(k in fname for k in ["optical", "sentinel2", "s2", "rgb"]):
        return "SAR"
    elif band_count == 2:
        return "OPTICAL"
    if band_count in OPTICAL_BAND_COUNTS:
        return "OPTICAL"

    # 1 band without SAR keywords is typically panchromatic/grayscale optical
    if band_count == 1:
        return "OPTICAL"

    # Ambiguous — default to optical
    logger.warning(f"Ambiguous band count ({band_count}) for {file_path}, defaulting to OPTICAL")
    return "OPTICAL"


def read_bands_windowed(
    file_path: str,
    bands: Optional[List[int]] = None,
    window: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Memory-safe band reader with optional spatial windowing.

    Args:
        file_path: Path to GeoTIFF
        bands: 1-indexed band list, e.g. [1,2,3,4]. None = all bands.
        window: (col_off, row_off, width, height) pixel window. None = full extent.

    Returns:
        Float32 ndarray of shape (bands, height, width)
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(file_path) as src:
        rio_window = None
        if window is not None:
            col_off, row_off, w, h = window
            rio_window = Window(col_off, row_off, w, h)

        indexes = bands if bands else list(range(1, src.count + 1))
        data = src.read(indexes, window=rio_window).astype(np.float32)

    return data


def generate_rgb_thumbnail(file_path: str, output_path: str) -> str:
    """
    Creates a PNG preview thumbnail from the first 3 bands of a GeoTIFF.
    Applies 2%-98% percentile stretch for proper visualization.
    """
    import rasterio
    from PIL import Image

    from app.core.geospatial.calibration import normalize_optical

    with rasterio.open(file_path) as src:
        out_h = min(src.height, 512)
        out_w = min(src.width, 512)
        out_shape = (out_h, out_w)

        if src.count >= 3:
            rgb = src.read([1, 2, 3], out_shape=(3, *out_shape)).astype(np.float32)
        elif src.count == 2:
            # SAR: use VV, VH, VV as pseudo-RGB
            bands = src.read([1, 2], out_shape=(2, *out_shape)).astype(np.float32)
            rgb = np.stack([bands[0], bands[1], bands[0]], axis=0)
        else:
            band = src.read(1, out_shape=out_shape).astype(np.float32)
            rgb = np.stack([band, band, band], axis=0)

    # Normalize each band
    for i in range(3):
        rgb[i] = normalize_optical(rgb[i])

    # Convert to uint8 PIL image
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(np.transpose(rgb_uint8, (1, 2, 0)))
    if output_path.lower().endswith(".png"):
        output_path = output_path[:-4] + ".tif"
    pil_img.save(output_path, format="TIFF")

    return output_path
