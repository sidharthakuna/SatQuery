"""
SatQuery AI — Query Request / Response Schemas
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .audit import (
    ConfidenceDecomposition,
    EvidenceGraph,
    EvidenceVerification,
    ExecutionTrace,
    InterpretedQuery,
)


class QueryRequest(BaseModel):
    """Incoming natural-language query with optional image references."""
    query: str = Field(..., min_length=1, max_length=2000, description="Natural-language question")
    image_ids: List[str] = Field(
        default_factory=list,
        description="0, 1, or 2 uploaded image file IDs",
    )
    history: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Prior conversation turns [{'role': 'user'|'assistant', 'content': '...'}]",
    )

    @field_validator("image_ids")
    @classmethod
    def validate_image_count(cls, v: List[str]) -> List[str]:
        """Enforce the 0-2 image limit at the schema level."""
        if len(v) > 2:
            raise ValueError(
                f"Too many image IDs supplied ({len(v)}). Maximum is 2."
            )
        return v



class VQAGrounding(BaseModel):
    """Grounded Visual Question Answering intelligence with visual overlay, method, and confidence."""
    overlay_url: str = Field(..., description="URL to the generated visual overlay image with highlighted features")
    legend_label: str = Field("Detected Features", description="Legend description, e.g. Detected Buildings")
    legend_color: str = Field("#ef4444", description="Hex color or CSS color of highlight overlay")
    method: str = Field("RS-VLM + Segmentation", description="Methodology used, e.g. RS-VLM + Segmentation")
    confidence: float = Field(0.87, description="Calibrated confidence score (0.0 to 1.0)")
    confidence_pct: Optional[int] = Field(None, description="Calibrated confidence percentage (0-100)")
    note: str = Field("Count is estimated and may vary for very small structures.", description="Operational note or caveat")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Quantitative measurements e.g. count, length_km, area_sqkm")

    def model_post_init(self, __context: Any) -> None:
        if self.confidence_pct is None:
            self.confidence_pct = int(round(self.confidence * 100)) if self.confidence <= 1.0 else int(self.confidence)


class SpatialEvidence(BaseModel):
    """Spatial output produced by a specialist tool."""
    type: str = Field(..., description="Evidence type: bounding_boxes | change_mask | fusion_map | multi_layer | vqa_grounding")
    bounding_boxes: Optional[List[List[float]]] = Field(None, description="List of [x1, y1, x2, y2] boxes")
    mask_url: Optional[str] = Field(None, description="URL to the generated mask image")
    geojson_url: Optional[str] = Field(None, description="URL to the GeoJSON output file")
    changed_area_hectares: Optional[float] = Field(None, description="Area of change in hectares")
    changed_area_percent: Optional[float] = Field(None, description="Percentage of area changed")
    clusters: Optional[List[Dict[str, Any]]] = Field(None, description="Marked spatial clusters/places with centroids and descriptions")
    layers: Optional[List[Dict[str, Any]]] = Field(None, description="Multiple spatial layers from compound/multi-tool execution")
    vqa_grounding: Optional[VQAGrounding] = Field(None, description="Grounded VQA visual overlay and telemetry")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional evidence metadata")


class SatQueryResult(BaseModel):
    """Complete response from a SatQuery AI query."""
    query: str
    text_response: str = Field(..., description="Natural-language answer to the query")
    spatial_evidence: Optional[SpatialEvidence] = None
    audit_trace: ExecutionTrace
    thumbnail_urls: List[str] = Field(default_factory=list, description="Preview image URLs")
    suggested_actions: List[str] = Field(default_factory=list, description="Contextual follow-up suggestions")
    confidence_decomposition: Optional[ConfidenceDecomposition] = None
    evidence_verification: Optional[EvidenceVerification] = None
    evidence_graph: Optional[EvidenceGraph] = None
    interpreted_query: Optional[InterpretedQuery] = None
    geojson_data: Optional[Dict[str, Any]] = Field(None, description="GeoJSON FeatureCollection payload for direct download/rendering")
    evidence_timeline: Optional[List[Dict[str, Any]]] = Field(None, description="Chronological acquisition and observed change timeline")
    tool_execution_plan: Optional[List[Dict[str, Any]]] = Field(None, description="Execution DAG showing specialist models invoked and dependencies")
    vqa_grounding: Optional[VQAGrounding] = Field(None, description="Structured VQA visual overlay, method, confidence, and note")


class ImageUploadResponse(BaseModel):
    """Response after successfully uploading a GeoTIFF."""
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    filename: str
    file_size_bytes: int
    modality: str = Field(..., description="Detected modality: OPTICAL or SAR")
    crs: Optional[str] = "UNREFERENCED"
    width: int
    height: int
    band_count: int
    bounds_latlon: Optional[Dict[str, float]] = None
    thumbnail_url: Optional[str] = None
