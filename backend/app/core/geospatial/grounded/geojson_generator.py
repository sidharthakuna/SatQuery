"""
SatQuery AI — GeoJSON Feature Collection Generator
Builds OGC-compliant GeoJSON FeatureCollections from raster pixel detections,
transforming pixel bounding boxes and clusters to calibrated geographic coordinates (lon, lat).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def generate_geojson(
    spatial_type: str,
    boxes: Optional[List[List[float]]] = None,
    clusters: Optional[List[Dict[str, Any]]] = None,
    bounds: Optional[Dict[str, float]] = None,
    img_w: int = 512,
    img_h: int = 512,
) -> Dict[str, Any]:
    """
    Builds a compliant GeoJSON FeatureCollection from detected bounding boxes or clusters.
    Transforms raster pixel coordinates to calibrated geographic coordinates (lon, lat).
    """
    features = []
    if bounds:
        if isinstance(bounds, dict):
            min_lon = float(bounds.get("min_lon", 72.50))
            max_lon = float(bounds.get("max_lon", 72.58))
            min_lat = float(bounds.get("min_lat", 23.00))
            max_lat = float(bounds.get("max_lat", 23.08))
        else:
            min_lon = float(getattr(bounds, "min_lon", 72.50))
            max_lon = float(getattr(bounds, "max_lon", 72.58))
            min_lat = float(getattr(bounds, "min_lat", 23.00))
            max_lat = float(getattr(bounds, "max_lat", 23.08))
    else:
        min_lon, max_lon, min_lat, max_lat = 72.50, 72.58, 23.00, 23.08

    def px_to_geo(x: float, y: float) -> Tuple[float, float]:
        lon = round(min_lon + (x / max(img_w, 1)) * (max_lon - min_lon), 6)
        lat = round(max_lat - (y / max(img_h, 1)) * (max_lat - min_lat), 6)
        return lon, lat

    import math
    mid_lat = (min_lat + max_lat) / 2.0
    cos_lat = math.cos(math.radians(mid_lat))
    lat_m_per_deg = 111320.0
    lon_m_per_deg = 111320.0 * max(cos_lat, 0.01)

    if boxes:
        for idx, b in enumerate(boxes, 1):
            x1, y1, x2, y2 = b
            p1 = px_to_geo(x1, y1)
            p2 = px_to_geo(x2, y1)
            p3 = px_to_geo(x2, y2)
            p4 = px_to_geo(x1, y2)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            clon, clat = px_to_geo(cx, cy)
            
            # CRS-aware metric dimensions with latitude cosine correction
            bw_m = (abs(x2 - x1) / max(img_w, 1)) * abs(max_lon - min_lon) * lon_m_per_deg
            bh_m = (abs(y2 - y1) / max(img_h, 1)) * abs(max_lat - min_lat) * lat_m_per_deg
            bbox_area_ha = round((bw_m * bh_m) / 10000.0, 3)

            features.append({
                "type": "Feature",
                "id": f"target_{idx}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[list(p1), list(p2), list(p3), list(p4), list(p1)]],
                },
                "properties": {
                    "name": f"Target #{idx}",
                    "type": "Visual Grounding Envelope",
                    "pixel_bbox": [round(v, 1) for v in b],
                    "centroid_geo": [clon, clat],
                    "width_px": round(x2 - x1, 1),
                    "height_px": round(y2 - y1, 1),
                    "ground_width_m": round(bw_m, 1),
                    "ground_height_m": round(bh_m, 1),
                    "area_hectares": bbox_area_ha,
                },
            })

    if clusters:
        for c in clusters:
            z_name = c.get("zone", "Zone")
            cx, cy = c.get("centroid", (256, 256))
            clon, clat = px_to_geo(cx, cy)
            area_ha = c.get("area_ha", 1.0)
            radius_px = max(float(np.sqrt(max(c.get("pixel_count", 400), 100) / np.pi)), 12.0)
            p1 = px_to_geo(cx - radius_px, cy - radius_px)
            p2 = px_to_geo(cx + radius_px, cy - radius_px)
            p3 = px_to_geo(cx + radius_px, cy + radius_px)
            p4 = px_to_geo(cx - radius_px, cy + radius_px)
            features.append({
                "type": "Feature",
                "id": z_name.lower().replace(" ", "_"),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[list(p1), list(p2), list(p3), list(p4), list(p1)]],
                },
                "properties": {
                    "name": z_name,
                    "category": c.get("category", "Delineated Cluster"),
                    "area_hectares": area_ha,
                    "area_km2": round(area_ha / 100.0, 3),
                    "centroid_geo": [clon, clat],
                },
            })

    return {
        "type": "FeatureCollection",
        "spatial_type": spatial_type,
        "features": features,
    }
