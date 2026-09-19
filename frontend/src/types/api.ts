export interface GeoBoundsLatLon {
  min_lat: number;
  min_lon: number;
  max_lat: number;
  max_lon: number;
}

export interface ImageUploadResponse {
  file_id: string;
  filename: string;
  file_size_bytes: number;
  modality: 'OPTICAL' | 'SAR' | string;
  crs: string;
  width: number;
  height: number;
  band_count: number;
  bounds_latlon: GeoBoundsLatLon | null;
  thumbnail_url: string | null;
}

export interface SpatialCluster {
  zone: string;
  area_ha?: number;
  category: string;
  centroid?: [number, number];
  pixel_count?: number;
  bbox?: [number, number, number, number];
}

export interface VQAGrounding {
  overlay_url: string;
  legend_label: string;
  legend_color: string;
  method: string;
  confidence: number;
  note: string;
  metrics?: Record<string, any>;
}

export interface SpatialEvidence {
  type: 'bounding_boxes' | 'change_mask' | 'fusion_map' | 'multi_layer' | 'vqa_grounding' | string;
  bounding_boxes?: [number, number, number, number][] | null;
  mask_url?: string | null;
  geojson_url?: string | null;
  changed_area_hectares?: number | null;
  changed_area_percent?: number | null;
  clusters?: SpatialCluster[];
  layers?: Record<string, any>[] | null;
  vqa_grounding?: VQAGrounding | null;
  extra?: Record<string, any>;
}

export interface ValidationReport {
  is_valid: boolean;
  image_count: number;
  modalities: string[];
  crs_list: string[];
  crs_compatible: boolean;
  band_counts: number[];
  input_quality_score?: number;
  errors: string[];
  warnings: string[];
  remediation_advice?: string[];
}

export interface ConfidenceDecomposition {
  overall_confidence: number;
  model_confidence: number;
  spatial_agreement: number;
  input_quality: number;
  cross_modal_agreement: number;
  calibration_method?: string;
}

export interface EvidenceNode {
  id: string;
  title: string;
  type: string;
  status: 'SUPPORTED' | 'VERIFIED' | 'CAVEAT' | 'CONTRADICTED' | string;
  confidence: number;
  details: Record<string, any>;
}

export interface EvidenceEdge {
  source: string;
  target: string;
  label?: string;
}

export interface EvidenceGraph {
  nodes: EvidenceNode[];
  edges: EvidenceEdge[];
}

export interface EvidenceVerification {
  status: 'SUPPORTED' | 'UNCERTAIN' | 'CONTRADICTED' | 'INSUFFICIENT_EVIDENCE' | string;
  supporting_sources: string[];
  counter_evidence: string[];
  hallucination_risk_score: number;
  insufficient_evidence: boolean;
  refusal_reason?: string | null;
}

export interface InterpretedQuery {
  task_type: TaskType;
  target_features: string;
  temporal_relationship?: string | null;
  comparison_mode?: string | null;
  required_tools: string[];
  reasoning_summary: string;
}

export interface EvidenceTimelineItem {
  date: string;
  label: string;
  event: string;
  metrics?: Record<string, any>;
}

export interface TraceStep {
  step_index: number;
  step_name: string;
  status: 'SUCCESS' | 'ERROR' | 'IN_PROGRESS' | string;
  duration_ms: number;
  details: Record<string, any>;
  message?: string;
}

export type TaskType =
  | 'SINGLE_VQA'
  | 'SINGLE_GROUNDING'
  | 'BITEMPORAL_CHANGE'
  | 'CROSS_MODAL_FUSION'
  | 'MULTI_MODEL'
  | 'COMPOUND_ANALYSIS'
  | 'AGENT_ASSISTANT'
  | 'UNKNOWN';


export interface ExecutionTrace {
  trace_id: string;
  task_identified: TaskType;
  input_validation: ValidationReport;
  selected_tools: string[];
  parameters_applied: Record<string, any>;
  execution_steps: TraceStep[];
  total_execution_time_ms: number;
  confidence_score: number;
  status: 'SUCCESS' | 'ERROR' | string;
  error_message?: string | null;
  confidence_decomposition?: ConfidenceDecomposition;
  evidence_verification?: EvidenceVerification;
  evidence_graph?: EvidenceGraph;
  interpreted_query?: InterpretedQuery;
}

export interface SatQueryResult {
  query: string;
  text_response: string;
  spatial_evidence?: SpatialEvidence | null;
  audit_trace: ExecutionTrace;
  thumbnail_urls: string[];
  suggested_actions?: string[];
  confidence_decomposition?: ConfidenceDecomposition;
  evidence_verification?: EvidenceVerification;
  evidence_graph?: EvidenceGraph;
  interpreted_query?: InterpretedQuery;
  geojson_data?: Record<string, any> | null;
  evidence_timeline?: EvidenceTimelineItem[];
  vqa_grounding?: VQAGrounding | null;
}

export interface ReportRequest {
  query: string;
  text_response: string;
  audit_trace: Record<string, any>;
  spatial_evidence?: Record<string, any> | null;
  mask_image_path?: string | null;
  image_metadata?: Record<string, any>[] | null;
  thumbnail_paths?: string[] | null;
  classification?: 'RESTRICTED' | 'CONFIDENTIAL' | 'SECRET' | 'OFFICIAL_USE_ONLY';
  layout_mode?: 'executive_summary' | 'rapid_assessment' | 'comprehensive';
  include_sensor_telemetry?: boolean;
  include_audit_trail?: boolean;
}

export interface ReportResponse {
  report_id: string;
  download_url: string;
  docx_download_url?: string;
  file_size_bytes: number;
}

export interface SystemHealth {
  status: string;
  service: string;
  version: string;
  inference_mode: string;
}
