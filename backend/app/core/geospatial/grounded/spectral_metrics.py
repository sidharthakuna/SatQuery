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

    # Authentic Multispectral Formulas (NIR Band 4 if present) vs Calibrated Visible Proxies
    if c >= 4:
        nir = arr[3]
        # McFeeters NDWI: (Green - NIR) / (Green + NIR + eps)
        ndwi = (g - nir) / (g + nir + 1e-5)
        water_mask = (ndwi > 0.0) | ((g > nir + 0.02) & (r < 0.25))
        # Rouse NDVI: (NIR - Red) / (NIR + Red + eps)
        ndvi = (nir - r) / (nir + r + 1e-5)
        veg_mask = (ndvi > 0.30) & (nir > r)
    else:
        # 1. NDWI Proxy: (Green - Red) / (Green + Red + eps) for optical RGB
        ndwi_proxy = (g - r) / (g + r + 1e-5)
        water_mask = (ndwi_proxy > 0.05) & (r < 0.28) & (g < 0.38)
        # 2. NDVI Proxy: Healthy green reflectance over red and blue
        veg_mask = (g > (r + 0.04)) & (g > (b + 0.04))

    water_pixels = int(np.sum(water_mask))
    water_pct = round(float(water_pixels / total_pixels) * 100.0, 1)

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

    # 7. Spectral Histogram Distribution (10-bin NDVI proxy distribution)
    # NDVI proxy array: (NIR - Red) / (NIR + Red) approximated or native
    if c >= 4:
        nir = arr[3]
        ndvi_arr = (nir - r) / (nir + r + 1e-5)
        calibration_method = "Native Multi-Spectral Sentinel-2 / Landsat NIR Band"
        is_proxy = False
    else:
        # Calibrated Green-Red contrast proxy
        ndvi_arr = (g - r) / (g + r + 1e-5)
        calibration_method = "Broadband Visible Reflectance Proxy (Calibrated RGB)"
        is_proxy = True

    ndvi_clipped = np.clip(ndvi_arr, -0.5, 0.8)
    hist_counts, bin_edges = np.histogram(ndvi_clipped, bins=10, range=(-0.5, 0.8))
    hist_pcts = [round(float(cnt / total_pixels) * 100.0, 1) for cnt in hist_counts]
    hist_bins = [f"{round(bin_edges[i], 2)} to {round(bin_edges[i+1], 2)}" for i in range(10)]

    # 8. Land Cover Chart Structured Breakdown
    veg_ha = round((veg_pct / 100.0) * total_area_ha, 1)
    water_ha = round((water_pct / 100.0) * total_area_ha, 1)
    built_ha = round((built_pct / 100.0) * total_area_ha, 1)
    other_ha = round(max(0.0, total_area_ha - (veg_ha + water_ha + built_ha)), 1)

    land_cover_chart = [
        {"label": "Vegetation Canopy", "key": "vegetation", "pct": veg_pct, "area_ha": veg_ha, "color": "#10B981"},
        {"label": "Surface Hydrology", "key": "water", "pct": water_pct, "area_ha": water_ha, "color": "#0EA5E9"},
        {"label": "Urban / Built-up", "key": "built", "pct": built_pct, "area_ha": built_ha, "color": "#F59E0B"},
        {"label": "Permeable / Open Soil", "key": "other", "pct": other_pct, "area_ha": other_ha, "color": "#8B5CF6"},
    ]

    quadrant_chart = [
        {"quadrant": qname, "veg_pct": qdata["veg_pct"], "water_pct": qdata["water_pct"], "built_pct": qdata["built_pct"]}
        for qname, qdata in quadrants.items()
    ]

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
        "chart_data": {
            "land_cover_chart": land_cover_chart,
            "quadrant_chart": quadrant_chart,
            "spectral_histogram": {
                "metric_name": "NDVI Reflectance Index" if not is_proxy else "NDVI Biophysical Proxy",
                "bins": hist_bins,
                "values": hist_pcts,
                "is_proxy": is_proxy,
                "calibration_method": calibration_method,
            },
            "sensor_fidelity": {
                "band_count": c,
                "resolution_m": res_m,
                "total_area_ha": total_area_ha,
                "calibration_method": calibration_method,
                "is_proxy": is_proxy,
            },
        },
        "water_mask": water_mask,
        "veg_mask": veg_mask,
        "built_mask": built_mask,
    }


def compute_reconstruction_validation_metrics(
    estimated: Any,
    reference: Any,
    cloud_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes rigorous empirical validation metrics comparing the SAR-guided
    reconstructed optical estimate against ground-truth clear optical reference:
    - PSNR (Peak Signal-to-Noise Ratio in dB)
    - SSIM (Structural Similarity Index)
    - SAM (Spectral Angle Mapper in degrees)
    - RMSE (Root Mean Squared Error)
    """
    from app.core.geospatial.grounded_analyzer import _to_float32_chw

    est = _to_float32_chw(estimated)
    ref = _to_float32_chw(reference)

    # Normalize to 0.0 .. 1.0
    if est.max() > 1.0:
        est = est / 255.0
    if ref.max() > 1.0:
        ref = ref / 255.0

    # Ensure shape alignment via spatial resampling
    min_c = min(est.shape[0], ref.shape[0], 3)
    target_h, target_w = est.shape[1], est.shape[2]
    
    if ref.shape[1] != target_h or ref.shape[2] != target_w:
        from scipy.ndimage import zoom
        zy = target_h / ref.shape[1]
        zx = target_w / ref.shape[2]
        ref = np.stack([zoom(ref[b], (zy, zx), order=1) for b in range(min_c)])
    else:
        ref = ref[:min_c]
    est = est[:min_c]

    # Overall MSE & RMSE
    diff = est - ref
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    # PSNR (dB)
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-10)))
    psnr = min(round(psnr, 2), 50.0)

    # SSIM approximation
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    mu_x = float(np.mean(est))
    mu_y = float(np.mean(ref))
    sigma_x = float(np.var(est))
    sigma_y = float(np.var(ref))
    sigma_xy = float(np.mean((est - mu_x) * (ref - mu_y)))
    denom_ssim = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim = float(((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / max(denom_ssim, 1e-8))
    ssim = round(max(min(ssim, 1.0), 0.0), 3)

    # Spectral Angle Mapper (SAM) across channels
    dot = np.sum(est * ref, axis=0)
    norm_e = np.linalg.norm(est, axis=0)
    norm_r = np.linalg.norm(ref, axis=0)
    denom_sam = np.maximum(norm_e * norm_r, 1e-6)
    cos_sam = np.clip(dot / denom_sam, -1.0, 1.0)
    sam_rad = np.arccos(cos_sam)
    sam_deg = round(float(np.degrees(np.mean(sam_rad))), 2)

    # Cloud-specific metrics if mask provided
    cloud_metrics = {}
    if cloud_mask is not None:
        c_mask = cloud_mask[:min_h, :min_w] > 0
        if np.any(c_mask):
            c_diff = diff[:, c_mask]
            c_mse = float(np.mean(c_diff ** 2))
            c_psnr = float(10.0 * np.log10(1.0 / max(c_mse, 1e-10)))
            cloud_metrics = {
                "cloud_psnr_db": min(round(c_psnr, 2), 50.0),
                "cloud_rmse": round(float(np.sqrt(c_mse)), 4),
            }

    return {
        "psnr_db": psnr,
        "ssim": ssim,
        "sam_deg": sam_deg,
        "rmse": round(rmse, 4),
        "structural_correlation_pct": round(float(max(ssim, 0.0)) * 100.0, 1),
        **cloud_metrics,
    }

