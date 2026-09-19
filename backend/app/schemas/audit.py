"""
SatQuery AI — Audit & Execution Trace Schemas
Structured, observable audit trail for every orchestrator run.
Includes Decomposed Confidence, Cross-Model Evidence Verification,
and Interactive Evidence Graph contracts.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Deterministic task classification labels."""
    SINGLE_VQA = "SINGLE_VQA"
    SINGLE_GROUNDING = "SINGLE_GROUNDING"
    BITEMPORAL_CHANGE = "BITEMPORAL_CHANGE"
    CROSS_MODAL_FUSION = "CROSS_MODAL_FUSION"
    AGENT_ASSISTANT = "AGENT_ASSISTANT"
    MULTI_MODEL = "MULTI_MODEL"
    COMPOUND_ANALYSIS = "COMPOUND_ANALYSIS"
    UNKNOWN = "UNKNOWN"


class ValidationReport(BaseModel):
    """Result of the InputCompatibilityGuard / InputIntelligenceGate validation step."""
    is_valid: bool = Field(..., description="Whether inputs passed all validation checks")
    image_count: int = Field(..., description="Number of images provided")
    modalities: List[str] = Field(default_factory=list, description="Detected modalities (OPTICAL, SAR)")
    crs_list: List[str] = Field(default_factory=list, description="CRS strings from each image")
    crs_compatible: bool = Field(True, description="Whether all CRS values are compatible")
    band_counts: List[int] = Field(default_factory=list, description="Band count per image")
    input_quality_score: float = Field(1.0, ge=0.0, le=1.0, description="Evaluated raster quality (NoData, cloud, saturation)")
    errors: List[str] = Field(default_factory=list, description="List of validation error messages")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    remediation_advice: List[str] = Field(default_factory=list, description="Actionable user instructions if gate detects anomalies")


class TraceStep(BaseModel):
    """A single step in the execution trace timeline."""
    step_index: int
    step_name: str
    status: str = "SUCCESS"
    duration_ms: float = 0.0
    details: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None  # Human-readable step description for streaming UI


class ConfidenceDecomposition(BaseModel):
    """
    Deconstructed multi-dimensional confidence metrics.
    Distinguishes model confidence from spatial consensus and sensor quality.
    """
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Weighted composite confidence score")
    model_confidence: float = Field(..., ge=0.0, le=1.0, description="Raw specialist neural prediction confidence")
    spatial_agreement: float = Field(..., ge=0.0, le=1.0, description="Spatial overlap consensus across tools/masks")
    input_quality: float = Field(..., ge=0.0, le=1.0, description="Sensor signal-to-noise, NoData and cloud penalty factor")
    cross_modal_agreement: float = Field(..., ge=0.0, le=1.0, description="Correlation between optical and SAR domains")
    calibration_method: str = Field("temperature_scaled_evidence_fusion", description="Confidence calibration technique")


class EvidenceNode(BaseModel):
    """A verifiable node in the observable Evidence Graph."""
    id: str
    title: str
    type: str = Field(..., description="query | input_optical | input_sar | specialist_model | change_mask | grounding_boxes | spatial_metrics | cross_verification | answer")
    status: str = Field("SUPPORTED", description="SUPPORTED | VERIFIED | CAVEAT | CONTRADICTED")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)


class EvidenceEdge(BaseModel):
    """Directed dependency link in the Evidence Graph."""
    source: str
    target: str
    label: Optional[str] = None


class EvidenceGraph(BaseModel):
    """Complete Directed Acyclic Graph connecting User Query to final Answer."""
    nodes: List[EvidenceNode] = Field(default_factory=list)
    edges: List[EvidenceEdge] = Field(default_factory=list)


class EvidenceVerification(BaseModel):
    """
    Cross-model evidence validation and counter-evidence findings.
    Prevents hallucination by checking independent evidence streams.
    """
    status: str = Field("SUPPORTED", description="SUPPORTED | UNCERTAIN | CONTRADICTED | INSUFFICIENT_EVIDENCE")
    supporting_sources: List[str] = Field(default_factory=list, description="Independent evidence streams confirming the answer")
    counter_evidence: List[str] = Field(default_factory=list, description="Anomalies, cloud obscuration or low-confidence caveats")
    hallucination_risk_score: float = Field(0.0, ge=0.0, le=1.0, description="Estimated risk of model confabulation")
    insufficient_evidence: bool = Field(False, description="Flag indicating system should refuse to guess (Don't Know mode)")
    refusal_reason: Optional[str] = Field(None, description="Explanation when evidence is insufficient for reliable answering")


class InterpretedQuery(BaseModel):
    """Structured query explainability object decomposed by the agent."""
    task_type: TaskType
    target_features: str = Field(..., description="Core entities or attributes targeted (e.g. built-up expansion)")
    temporal_relationship: Optional[str] = Field(None, description="Temporal relationship between inputs if multi-date")
    comparison_mode: Optional[str] = Field(None, description="e.g. baseline vs surveillance, or optical vs SAR")
    required_tools: List[str] = Field(default_factory=list, description="List of registered tools planned for execution")
    reasoning_summary: str = Field(..., description="Explainable breakdown of what the agent plans to verify")


class ExecutionTrace(BaseModel):
    """
    Complete, immutable execution audit trail.
    Contains every decision, tool call, and metric from a single query run.
    """
    trace_id: str = Field(..., description="Unique identifier for this trace")
    task_identified: TaskType
    input_validation: ValidationReport
    selected_tools: List[str] = Field(default_factory=list)
    parameters_applied: Dict[str, Any] = Field(default_factory=dict)
    execution_steps: List[TraceStep] = Field(default_factory=list)
    total_execution_time_ms: float = 0.0
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    status: str = "SUCCESS"
    error_message: Optional[str] = None
    confidence_decomposition: Optional[ConfidenceDecomposition] = None
    evidence_verification: Optional[EvidenceVerification] = None
    evidence_graph: Optional[EvidenceGraph] = None
    interpreted_query: Optional[InterpretedQuery] = None
