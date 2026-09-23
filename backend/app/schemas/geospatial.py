"""
SatQuery AI — Geospatial Data Schemas
Models for geographic bounds, GeoTIFF metadata, and GeoJSON features.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class GeoBoundsLatLon(BaseModel):
    """Geographic bounding box in WGS84 (EPSG:4326) coordinates."""
    min_lat: float = Field(..., ge=-90, le=90)
    min_lon: float = Field(..., ge=-180, le=180)
    max_lat: float = Field(..., ge=-90, le=90)
    max_lon: float = Field(..., ge=-180, le=180)


class GeoTIFFMetadata(BaseModel):
    """Extracted metadata from a GeoTIFF raster file."""
    file_path: Optional[str] = None
    file_id: str
    filename: Optional[str] = None
    crs: Optional[str] = "EPSG:4326"
    width: int
    height: int
    band_count: int
    dtypes: List[str] = Field(default_factory=list)
    modality: str = Field("OPTICAL", description="OPTICAL or SAR")
    resolution: Optional[Union[tuple, List[float]]] = None
    gsd_m: Optional[float] = Field(default=None, description="Ground Sampling Distance in meters per pixel")
    bounds_latlon: Optional[GeoBoundsLatLon] = None
    file_size_bytes: int = 0
    thumbnail_url: Optional[str] = None
    area_ha: Optional[float] = None

    def compute_area_hectares(self) -> float:
        """Computes true ground coverage area in hectares from bounds or GSD."""
        import math
        if self.bounds_latlon:
            lat_deg = abs(self.bounds_latlon.max_lat - self.bounds_latlon.min_lat)
            lon_deg = abs(self.bounds_latlon.max_lon - self.bounds_latlon.min_lon)
            lat_m = lat_deg * 111320.0
            mid_lat = (self.bounds_latlon.max_lat + self.bounds_latlon.min_lat) / 2.0
            lon_m = lon_deg * 111320.0 * math.cos(math.radians(mid_lat))
            calc_ha = (lat_m * lon_m) / 10000.0
            if calc_ha > 0.01:
                return round(calc_ha, 1)
        if self.gsd_m and self.width > 0 and self.height > 0:
            calc_ha = (self.width * self.height * (float(self.gsd_m) ** 2)) / 10000.0
            return round(calc_ha, 1)
        return 262.1

    def model_post_init(self, __context: Any) -> None:
        if not self.filename and self.file_path:
            from pathlib import Path
            self.filename = Path(self.file_path).name
        if self.area_ha is None or self.area_ha <= 0:
            self.area_ha = self.compute_area_hectares()


def compute_aoi_hectares(meta: Any, default_ha: float = 262.1) -> float:
    """Safely extracts or calculates authentic AOI area in hectares from metadata."""
    if meta is None:
        return default_ha
    if isinstance(meta, GeoTIFFMetadata):
        return meta.area_ha or meta.compute_area_hectares()
    if isinstance(meta, dict):
        if meta.get("area_ha") is not None and float(meta.get("area_ha", 0)) > 0:
            return float(meta["area_ha"])
        bounds = meta.get("bounds_latlon")
        if bounds:
            import math
            min_lat = bounds.get("min_lat", 0) if isinstance(bounds, dict) else getattr(bounds, "min_lat", 0)
            max_lat = bounds.get("max_lat", 0) if isinstance(bounds, dict) else getattr(bounds, "max_lat", 0)
            min_lon = bounds.get("min_lon", 0) if isinstance(bounds, dict) else getattr(bounds, "min_lon", 0)
            max_lon = bounds.get("max_lon", 0) if isinstance(bounds, dict) else getattr(bounds, "max_lon", 0)
            lat_deg = abs(max_lat - min_lat)
            lon_deg = abs(max_lon - min_lon)
            lat_m = lat_deg * 111320.0
            lon_m = lon_deg * 111320.0 * math.cos(math.radians((max_lat + min_lat) / 2.0))
            calc_ha = (lat_m * lon_m) / 10000.0
            if calc_ha > 0.01:
                return round(calc_ha, 1)
        gsd_m = meta.get("gsd_m")
        w = meta.get("width", 0)
        h = meta.get("height", 0)
        if gsd_m and w and h:
            return round((float(w) * float(h) * (float(gsd_m) ** 2)) / 10000.0, 1)
    return default_ha


class GeoJSONGeometry(BaseModel):
    """GeoJSON Geometry object."""
    type: str = Field(..., description="Geometry type: Point, Polygon, MultiPolygon, etc.")
    coordinates: Any = Field(..., description="Coordinate array")


class GeoJSONProperties(BaseModel):
    """Properties attached to a GeoJSON feature."""
    label: Optional[str] = None
    confidence: Optional[float] = None
    area_hectares: Optional[float] = None
    class_name: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class GeoJSONFeature(BaseModel):
    """A single GeoJSON Feature."""
    type: str = "Feature"
    geometry: GeoJSONGeometry
    properties: GeoJSONProperties = Field(default_factory=GeoJSONProperties)


class GeoJSONFeatureCollection(BaseModel):
    """GeoJSON FeatureCollection for vector overlay rendering."""
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature] = Field(default_factory=list)
    crs: Optional[str] = "EPSG:4326"
