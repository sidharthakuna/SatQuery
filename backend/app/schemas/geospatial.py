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

    def model_post_init(self, __context: Any) -> None:
        if not self.filename and self.file_path:
            from pathlib import Path
            self.filename = Path(self.file_path).name


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
