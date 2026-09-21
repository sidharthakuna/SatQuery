"""
SatQuery AI — Grounded Geospatial Analytics Package
Clean, modular geospatial processing engines:
- spectral_metrics: True radiometric and spectral index computation
- vqa_card_generator: Dynamic VQA overlay and card asset generation
- geojson_generator: GeoJSON vector feature collection formatting
"""

from app.core.geospatial.grounded.spectral_metrics import compute_raster_spectral_indices
from app.core.geospatial.grounded.scene_telemetry import SceneTelemetry
from app.core.geospatial.grounded.vqa_card_generator import (
    generate_vqa_visual_overlay,
    generate_vqa_card_assets,
)
from app.core.geospatial.grounded.geojson_generator import generate_geojson
from app.core.geospatial.grounded.pattern_recognizer import PatternRecognizer

__all__ = [
    "compute_raster_spectral_indices",
    "SceneTelemetry",
    "generate_vqa_visual_overlay",
    "generate_vqa_card_assets",
    "generate_geojson",
    "PatternRecognizer",
]

