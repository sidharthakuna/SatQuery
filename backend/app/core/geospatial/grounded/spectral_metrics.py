"""
SatQuery AI — Grounded Spectral & Spatial Metrics Engine
Calculates real radiometric, spectral, and structural metrics directly
from satellite pixel arrays. Zero hardcoded percentages or canned numbers.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_raster_spectral_indices(
    image: Any,
    image_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes authentic spectral proxies, edge frequencies, and land-cover shares
    directly from the raw pixel array.
    """
    from app.core.geospatial.grounded_analyzer import _to_float32_chw

    arr = _to_float32_chw(image)
    c, h, w = arr.shape
    total_pixels = max(h * w, 1)

    r = arr[0]
    g = arr[1] if c > 1 else arr[0]
    b = arr[2] if c > 2 else arr[0]

    # 1. NDWI Proxy: (Green - Red) / (Green + Red + eps) for optical RGB
    ndwi_proxy = (g - r) / (g + r + 1e-5)
    water_mask = (ndwi_proxy > 0.05) & (r < 0.28) & (g < 0.38)
    water_pixels = int(np.sum(water_mask))
    water_pct = round(float(water_pixels / total_pixels) * 100.0, 1)

    # 2. NDVI Proxy: Healthy green reflectance over red and blue
    veg_mask = (g > (r + 0.04)) & (g > (b + 0.04))
    veg_pixels = int(np.sum(veg_mask))
    veg_pct = round(float(veg_pixels / total_pixels) * 100.0, 1)

    # 3. Built-Up / Structural Impervious Proxy via High-Frequency Edge Gradients
    dy = np.abs(r[1:, :] - r[:-1, :])[:, :-1]
    dx = np.abs(r[:, 1:] - r[:, :-1])[:-1, :]
    edge_diff = dy + dx
    built_mask = edge_diff > 0.14
    built_pixels = int(np.sum(built_mask))
    built_pct = round(float(built_pixels / total_pixels) * 100.0, 1)

    # 4. Barren / Permeable Substrate
    other_pct = round(max(100.0 - (water_pct + veg_pct + built_pct), 0.0), 1)

    # 5. Spatial Quadrant Distribution (NW, NE, SW, SE, Central)
    mid_y, mid_x = h // 2, w // 2
    quadrants = {
        "North-West": {
            "veg_pct": round(float(np.mean(veg_mask[:mid_y, :mid_x])) * 100, 1),
            "water_pct": round(float(np.mean(water_mask[:mid_y, :mid_x])) * 100, 1),
            "built_pct": round(float(np.mean(built_mask[:mid_y-1, :mid_x-1])) * 100, 1),
        },
        "North-East": {
            "veg_pct": round(float(np.mean(veg_mask[:mid_y, mid_x:])) * 100, 1),
            "water_pct": round(float(np.mean(water_mask[:mid_y, mid_x:])) * 100, 1),
            "built_pct": round(float(np.mean(built_mask[:mid_y-1, mid_x-1:])) * 100, 1),
        },
        "South-West": {
            "veg_pct": round(float(np.mean(veg_mask[mid_y:, :mid_x])) * 100, 1),
            "water_pct": round(float(np.mean(water_mask[mid_y:, :mid_x])) * 100, 1),
            "built_pct": round(float(np.mean(built_mask[mid_y-1:, :mid_x-1])) * 100, 1),
        },
        "South-East": {
            "veg_pct": round(float(np.mean(veg_mask[mid_y:, mid_x:])) * 100, 1),
            "water_pct": round(float(np.mean(water_mask[mid_y:, mid_x:])) * 100, 1),
            "built_pct": round(float(np.mean(built_mask[mid_y-1:, mid_x-1:])) * 100, 1),
        },
    }

    # 6. Physical Area Estimation (Ground Sample Distance from image metadata)
    meta = image_meta or {}
    res_m = float(meta.get("spatial_resolution_m") or 10.0)
    pixel_area_m2 = res_m * res_m
    total_area_ha = round((total_pixels * pixel_area_m2) / 10000.0, 1)
    total_area_km2 = round(total_area_ha / 100.0, 2)

    return {
        "water_percent": water_pct,
        "vegetation_percent": veg_pct,
        "built_percent": built_pct,
        "other_percent": other_pct,
        "total_area_ha": total_area_ha,
        "total_area_km2": total_area_km2,
        "spatial_resolution_m": res_m,
        "quadrants": quadrants,
        "radiometry": {
            "mean_rgb": [round(float(arr[i].mean()), 3) for i in range(min(c, 3))],
            "std_rgb": [round(float(arr[i].std()), 3) for i in range(min(c, 3))],
        },
        "water_mask": water_mask,
        "veg_mask": veg_mask,
        "built_mask": built_mask,
    }
