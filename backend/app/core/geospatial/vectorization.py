"""
SatQuery AI — Raster Mask Vectorization
Converts dense raster masks into lightweight GeoJSON polygons
for 60 FPS rendering in MapLibre/Leaflet.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from app.schemas.geospatial import (
    GeoJSONFeature,
    GeoJSONFeatureCollection,
    GeoJSONGeometry,
    GeoJSONProperties,
)

logger = logging.getLogger(__name__)


def mask_to_geojson(
    mask: np.ndarray,
    transform: Any,
    crs: str = "EPSG:4326",
    simplify_tolerance: float = 1.0,
    min_area_pixels: int = 50,
    class_name: str = "change",
) -> GeoJSONFeatureCollection:
    """
    Converts a binary raster mask into a GeoJSON FeatureCollection.
    Applies Douglas-Peucker simplification to keep output < 30KB.

    Args:
        mask: 2D uint8 array where 1 = positive region
        transform: Affine transform from rasterio (maps pixel → geo coords)
        crs: Coordinate reference system string
        simplify_tolerance: Douglas-Peucker tolerance (higher = more simplified)
        min_area_pixels: Minimum polygon area to keep (filters noise)
        class_name: Label for the detected features

    Returns:
        GeoJSONFeatureCollection with simplified polygons
    """
    import rasterio.features
    from shapely.geometry import mapping, shape

    features: List[GeoJSONFeature] = []
    mask_uint8 = mask.astype(np.uint8)

    # Extract shapes from raster
    shapes_gen = rasterio.features.shapes(
        mask_uint8,
        mask=mask_uint8 > 0,
        transform=transform,
    )

    feature_idx = 0
    for geom_dict, value in shapes_gen:
        if value == 0:
            continue

        # Convert to shapely for simplification
        polygon = shape(geom_dict)

        # Filter tiny noise polygons
        if polygon.area < min_area_pixels:
            continue

        # Douglas-Peucker simplification
        simplified = polygon.simplify(simplify_tolerance, preserve_topology=True)

        if simplified.is_empty:
            continue

        geojson_geom = mapping(simplified)

        features.append(GeoJSONFeature(
            geometry=GeoJSONGeometry(
                type=geojson_geom["type"],
                coordinates=geojson_geom["coordinates"],
            ),
            properties=GeoJSONProperties(
                label=f"{class_name}_{feature_idx}",
                class_name=class_name,
                area_hectares=_pixels_to_hectares(polygon.area, transform),
                confidence=1.0,
            ),
        ))
        feature_idx += 1

    logger.info(f"Vectorized {feature_idx} polygons from mask (class={class_name})")
    return GeoJSONFeatureCollection(features=features, crs=crs)


def _pixels_to_hectares(area_in_crs_units: float, transform: Any) -> float:
    """
    Approximate conversion from CRS-unit area to hectares.
    For EPSG:4326 (degrees), converts degrees² to m² using ~111,320m/deg.
    For projected CRS (meters), area is already in m².
    """
    try:
        pixel_width = abs(transform.a)
        if pixel_width < 1.0:  # Likely degrees
            m_per_deg = 111_320.0
            area_m2 = area_in_crs_units * (m_per_deg ** 2)
        else:
            area_m2 = area_in_crs_units
        return round(area_m2 / 10_000, 2)  # 1 hectare = 10,000 m²
    except Exception:
        return 0.0


def compute_change_statistics(mask: np.ndarray, transform: Any) -> Dict[str, float]:
    """
    Computes basic change statistics from a binary change mask.

    Returns:
        Dict with keys: total_pixels, changed_pixels, change_percent, change_hectares
    """
    total = mask.size
    changed = int(np.sum(mask > 0))
    pct = round((changed / max(total, 1)) * 100, 2)

    try:
        pixel_w = abs(transform.a)
        pixel_h = abs(transform.e)
        if pixel_w < 1.0:  # degrees
            m_per_deg = 111_320.0
            pixel_area_m2 = (pixel_w * m_per_deg) * (pixel_h * m_per_deg)
        else:
            pixel_area_m2 = pixel_w * pixel_h
        hectares = round((changed * pixel_area_m2) / 10_000, 2)
    except Exception:
        hectares = 0.0

    return {
        "total_pixels": total,
        "changed_pixels": changed,
        "change_percent": pct,
        "change_hectares": hectares,
    }
