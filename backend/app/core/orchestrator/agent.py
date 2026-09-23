"""
SatQuery AI — Query-to-Evidence Agentic Orchestrator
Coordinates the full 7-stage verifiable geospatial query pipeline:
  1. Query Understanding & Task Decomposition
  2. Input Intelligence & Compatibility Gate
  3. Registry-Driven Multi-Specialist Execution
  4. Cross-Model Evidence Validation & Counter-Evidence Detection
  5. Geo-Referenced Spatial Evidence & GeoJSON Generation
  6. Multi-Temporal Evidence Timeline Formulation
  7. Calibrated Cognitive Answer Synthesis
"""

import asyncio
import copy
import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

from app.core.geospatial.grounded_analyzer import GroundedRSAnalyzer
from app.core.orchestrator.agentic_synthesizer import AgenticCognitiveSynthesizer
from app.core.orchestrator.evidence_validator import CrossModelEvidenceValidator
from app.core.orchestrator.input_gate import InputIntelligenceGate
from app.core.orchestrator.router import QueryIntentClassifier
from app.core.orchestrator.tracer import AuditTraceBuilder
from app.schemas.audit import (
    ConfidenceDecomposition,
    EvidenceGraph,
    EvidenceVerification,
    InterpretedQuery,
    TaskType,
    TraceStep,
    ValidationReport,
)
from app.schemas.geospatial import GeoTIFFMetadata
from app.schemas.query import SatQueryResult, SpatialEvidence
from app.tools.base import ToolInput, ToolOutput
from app.tools.registry import ToolRegistry
from config.settings import settings

logger = logging.getLogger(__name__)


class SatQueryAgent:
    """
    Query-to-Evidence Geospatial Orchestration Agent.
    Transforms natural-language questions into multi-stage validated Earth
    observation intelligence with decomposed confidence and verifiable provenance.
    """

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.gate = InputIntelligenceGate()
        self.classifier = QueryIntentClassifier()
        self.registry = tool_registry or ToolRegistry()
        self.validator = CrossModelEvidenceValidator()
        self.synthesizer = AgenticCognitiveSynthesizer()

    async def process_query(
        self,
        query: str,
        image_metas: List[GeoTIFFMetadata],
        images: Optional[List[Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        on_step: Optional[Callable[[TraceStep], Any]] = None,
    ) -> SatQueryResult:
        """
        Process a user query through the complete 7-stage Query-to-Evidence pipeline.
        """
        tracer = AuditTraceBuilder()
        thumbnails = [getattr(m, "thumbnail_url", None) for m in image_metas if getattr(m, "thumbnail_url", None)]

        async def _log_step(
            step_name: str,
            status: str = "SUCCESS",
            details: Optional[Dict[str, Any]] = None,
            message: Optional[str] = None,
        ) -> TraceStep:
            st = tracer.record_step(step_name, status=status, details=details, message=message)
            if on_step:
                try:
                    res = on_step(st)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as cb_err:
                    logger.debug(f"on_step callback error: {cb_err}")
            return st

        try:
            # ═══════════════════════════════════════════════════
            #  Stage 1: Ingestion & Image Loading
            # ═══════════════════════════════════════════════════
            await _log_step("INGESTION", details={"query": query, "image_count": len(image_metas)}, message="Ingesting raster metadata and natural language query...")

            loaded_images = []
            if images:
                loaded_images = list(images)
            elif image_metas:
                # If CRSs differ between multi-temporal inputs, auto-reproject to reference grid
                if len(image_metas) >= 2 and image_metas[0].crs != image_metas[1].crs:
                    fp0 = getattr(image_metas[0], "file_path", None)
                    fp1 = getattr(image_metas[1], "file_path", None)
                    if fp0 and fp1 and Path(fp0).exists() and Path(fp1).exists():
                        try:
                            from app.core.geospatial.coregistration import align_to_reference
                            ref_arr, tgt_arr, _ = align_to_reference(fp0, fp1)
                            loaded_images = [ref_arr, tgt_arr]
                            await _log_step(
                                "AUTO_REPROJECTION",
                                details={"reference": fp0, "aligned_target": fp1},
                                message="Co-registering rasters: Reprojected Image B to reference grid",
                            )
                        except Exception as align_err:
                            logger.warning(f"Auto-alignment failed: {align_err}")

                if not loaded_images:
                    import rasterio
                    for meta in image_metas:
                        fp = getattr(meta, "file_path", None)
                        if fp and Path(fp).exists():
                            try:
                                with rasterio.open(fp) as src:
                                    loaded_images.append(src.read().astype(np.float32))
                            except Exception as e:
                                logger.warning(f"Could not load raster array from {fp}: {e}")

            # ═══════════════════════════════════════════════════
            #  Stage 2: Input Intelligence & Compatibility Gate
            # ═══════════════════════════════════════════════════
            validation = self.gate.evaluate(image_metas, loaded_images)
            tracer.set_validation(validation)
            await _log_step(
                "INPUT_INTELLIGENCE_GATE",
                status="SUCCESS" if validation.is_valid else "FAILED",
                details={
                    "errors": validation.errors,
                    "warnings": validation.warnings,
                    "quality_score": validation.input_quality_score,
                    "remediations": validation.remediation_advice,
                },
                message=f"Input Gate: Signal Quality {int(validation.input_quality_score*100)}% | Modalities: {', '.join(validation.modalities) or 'None'}",
            )

            if not validation.is_valid:
                tracer.set_error(f"Input validation failed: {'; '.join(validation.errors)}")
                remediation_text = "\n\n**Actionable Remediation:**\n" + "\n".join(f"- {r}" for r in validation.remediation_advice)
                return SatQueryResult(
                    query=query,
                    text_response=f"Cannot execute query. {'; '.join(validation.errors)}{remediation_text}",
                    audit_trace=tracer.build(),
                    thumbnail_urls=thumbnails,
                )

            # ═══════════════════════════════════════════════════
            #  Stage 3: Query Understanding & Task Decomposition
            # ═══════════════════════════════════════════════════
            task_type, tool_ids, params = self.classifier.classify(query, image_metas, history=history)
            tracer.set_task_type(task_type)
            tracer.set_selected_tools(tool_ids)
            tracer.set_parameters(params)

            is_conversational = (task_type == TaskType.AGENT_ASSISTANT) or (len(image_metas) == 0)

            # Build explainable InterpretedQuery decomposition (for spatial tasks)
            interpreted = None
            if not is_conversational:
                target_feat = self._extract_target_entity(query, task_type)
                temporal_rel = self._extract_temporal_relationship(image_metas)
                comp_mode = (
                    "Bi-Temporal Delta" if len(image_metas) >= 2 and task_type == TaskType.BITEMPORAL_CHANGE
                    else "Optical-SAR Cross-Attention Fusion" if task_type == TaskType.CROSS_MODAL_FUSION
                    else "Single-Scene Grounded Inspection" if len(image_metas) == 1
                    else "Geospatial Science Guidance"
                )

                interpreted = InterpretedQuery(
                    task_type=task_type,
                    target_features=target_feat,
                    temporal_relationship=temporal_rel,
                    comparison_mode=comp_mode,
                    required_tools=tool_ids,
                    reasoning_summary=(
                        f"Decomposed natural-language query into {task_type.value} targeting '{target_feat}'. "
                        f"Orchestrating {len(tool_ids)} specialist model(s) with observable evidence fusion."
                    ),
                )
                tracer.set_interpreted_query(interpreted)

                await _log_step(
                    "QUERY_UNDERSTANDING",
                    details=interpreted.model_dump(),
                    message=f"Task Planner: {task_type.value} | Target: '{target_feat}' | Tools: {', '.join(tool_ids)}",
                )
            else:
                await _log_step(
                    "CONVERSATIONAL_SYNTHESIS",
                    details={"task": "AGENT_ASSISTANT"},
                    message="Engaging SatQuery conversational copilot...",
                )

            if task_type == TaskType.UNKNOWN:
                tracer.set_error("Could not determine task type from query and inputs")
                return SatQueryResult(
                    query=query,
                    text_response="Unable to determine the appropriate analysis task. Please check your query or uploaded images.",
                    audit_trace=tracer.build(),
                    thumbnail_urls=thumbnails,
                )

            # ═══════════════════════════════════════════════════
            #  Stage 4: Multi-Specialist Tool Execution (Parallel & Chained)
            # ═══════════════════════════════════════════════════
            execution_plan = self._plan_execution_graph(task_type, tool_ids)
            image_meta_dicts = [m.model_dump() for m in image_metas]

            tool_outputs: List[ToolOutput] = []
            prior_outputs_map: Dict[str, Any] = {}

            # Partition into independent (wave 1) and dependent (wave 2) execution waves
            wave_1_tools = [s["tool_id"] for s in execution_plan if not s["dependencies"]]
            wave_2_tools = [s["tool_id"] for s in execution_plan if s["dependencies"]]

            async def _run_tool_coro(tid: str, current_input: ToolInput) -> Optional[ToolOutput]:
                if not is_conversational:
                    await _log_step(f"EXECUTING_{tid.upper()}", details={"tool_id": tid}, message=f"Executing specialist model: {tid}")
                try:
                    out = await asyncio.to_thread(self.registry.execute_tool, tid, current_input)
                    if not is_conversational:
                        await _log_step(
                            f"COMPLETED_{tid.upper()}",
                            details={
                                "confidence": out.confidence,
                                "has_mask": out.mask is not None,
                                "has_boxes": out.bounding_boxes is not None,
                            },
                            message=f"Specialist {tid} completed with {int(out.confidence*100)}% prediction confidence",
                        )
                    return out
                except Exception as err:
                    logger.error(f"Tool {tid} execution failed: {err}")
                    await _log_step(f"FAILED_{tid.upper()}", status="ERROR", details={"error": str(err)})
                    return None

            # Wave 1: Execute independent models in parallel
            w1_input = ToolInput(
                images=loaded_images,
                image_metas=image_meta_dicts,
                query=query,
                parameters=params,
                prior_outputs={},
            )
            w1_results = await asyncio.gather(*[_run_tool_coro(tid, w1_input) for tid in wave_1_tools])
            for tid, out in zip(wave_1_tools, w1_results):
                if out:
                    tool_outputs.append(out)
                    prior_outputs_map[tid] = out.model_dump()

            # Wave 2: Execute dependent tools chained with prior outputs
            for tid in wave_2_tools:
                w2_input = ToolInput(
                    images=loaded_images,
                    image_metas=image_meta_dicts,
                    query=query,
                    parameters=params,
                    prior_outputs=prior_outputs_map,
                )
                out = await _run_tool_coro(tid, w2_input)
                if out:
                    tool_outputs.append(out)
                    prior_outputs_map[tid] = out.model_dump()

            if not tool_outputs:
                tracer.set_error("All specialist executions failed")
                return SatQueryResult(
                    query=query,
                    text_response="All specialist models failed to execute. Please re-check the raster formats.",
                    audit_trace=tracer.build(),
                    thumbnail_urls=thumbnails,
                )

            # Extract unified spatial extra dict
            spatial_extra = {}
            for out in tool_outputs:
                if out.extra:
                    out_extra = copy.deepcopy(out.extra)
                    for k, v in out_extra.items():
                        if k not in spatial_extra:
                            spatial_extra[k] = v
                        elif isinstance(v, list) and isinstance(spatial_extra[k], list):
                            spatial_extra[k].extend(v)
                        elif isinstance(v, dict) and isinstance(spatial_extra[k], dict):
                            spatial_extra[k].update(v)

            # ═══════════════════════════════════════════════════
            #  Stage 5: Cross-Model Evidence Validation & Counter-Evidence
            # ═══════════════════════════════════════════════════
            verification = None
            decomposition = None
            graph = None

            if not is_conversational:
                await _log_step("EVIDENCE_VALIDATION", message="Cross-validating evidence streams and checking counter-evidence...")
                verification, decomposition, graph = self.validator.validate(
                    query=query,
                    task_type=task_type,
                    tool_outputs=tool_outputs,
                    validation_report=validation,
                    spatial_extra=spatial_extra,
                )
                tracer.set_evidence_verification(verification)
                tracer.set_confidence_decomposition(decomposition)
                tracer.set_evidence_graph(graph)
                tracer.set_confidence(decomposition.overall_confidence)

            # ═══════════════════════════════════════════════════
            #  Stage 6: Geo-Referenced Evidence & GeoJSON Generation
            # ═══════════════════════════════════════════════════
            if not is_conversational:
                await _log_step("FUSING_SPATIAL_EVIDENCE", message="Generating geospatial masks and vector GeoJSON features...")
            
            text_response, spatial_evidence, suggested_actions, vqa_grounding = self._fuse_outputs(
                outputs=tool_outputs,
                task_type=task_type,
                query=query,
                image_metas=image_metas,
                images=loaded_images,
                history=history,
            )

            geojson_data = None
            if not is_conversational:
                # Save mask to PNG if present
                if spatial_evidence and not spatial_evidence.mask_url:
                    if spatial_evidence.extra and spatial_evidence.extra.get("mask_url"):
                        spatial_evidence.mask_url = spatial_evidence.extra.get("mask_url")
                    elif any(o.mask is not None for o in tool_outputs):
                        mask_url = self._save_mask(tool_outputs, image_metas)
                        if mask_url:
                            spatial_evidence.mask_url = mask_url

                # Generate GeoJSON FeatureCollection
                bounds_dict = getattr(image_metas[0], "bounds_latlon", None) if image_metas else None
                img_w = getattr(image_metas[0], "width", 512) if image_metas else 512
                img_h = getattr(image_metas[0], "height", 512) if image_metas else 512
                
                geojson_data = GroundedRSAnalyzer.generate_geojson(
                    spatial_type=spatial_evidence.type if spatial_evidence else "features",
                    boxes=getattr(spatial_evidence, "bounding_boxes", None) if spatial_evidence else None,
                    clusters=getattr(spatial_evidence, "clusters", None) if spatial_evidence else None,
                    bounds=bounds_dict,
                    img_w=img_w,
                    img_h=img_h,
                )

                geojson_url = self._save_geojson(geojson_data)
                if spatial_evidence and geojson_url:
                    spatial_evidence.geojson_url = geojson_url
            else:
                spatial_evidence = None

            # ═══════════════════════════════════════════════════
            #  Stage 7: Evidence Timeline & Final Response Assembly
            # ═══════════════════════════════════════════════════
            timeline = None
            if not is_conversational and verification and decomposition:
                timeline = self._build_evidence_timeline(image_metas, task_type, spatial_extra, verification, decomposition)

            tracer.set_success()
            if not is_conversational and decomposition and verification:
                await _log_step(
                    "TRACE_COMPILED",
                    details={
                        "overall_confidence": decomposition.overall_confidence,
                        "verification_status": verification.status,
                    },
                    message=f"Query-to-Evidence complete: {verification.status} ({int(decomposition.overall_confidence*100)}% confidence)",
                )
            chart_data_payload = spatial_evidence.chart_data if spatial_evidence else None

            return SatQueryResult(
                query=query,
                text_response=text_response,
                spatial_evidence=spatial_evidence,
                audit_trace=tracer.build(),
                thumbnail_urls=thumbnails,
                suggested_actions=suggested_actions,
                confidence_decomposition=decomposition,
                evidence_verification=verification,
                evidence_graph=graph,
                interpreted_query=interpreted,
                geojson_data=geojson_data,
                evidence_timeline=timeline,
                tool_execution_plan=execution_plan,
                vqa_grounding=vqa_grounding,
                chart_data=chart_data_payload,
            )

        except Exception as e:
            logger.exception(f"Orchestrator pipeline error: {e}")
            tracer.set_error(str(e))
            return SatQueryResult(
                query=query,
                text_response=f"An internal analysis error occurred: {str(e)}",
                audit_trace=tracer.build(),
                thumbnail_urls=thumbnails,
            )

    # ──────────────────────────────────────────────────────────────────────────
    #  Helper Methods & Execution Planning
    # ──────────────────────────────────────────────────────────────────────────

    def _plan_execution_graph(
        self,
        task_type: TaskType,
        tool_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Builds a Directed Acyclic Graph (DAG) for multi-specialist tool execution.
        Independent tools execute concurrently, while dependent tools wait for parent outputs.
        """
        plan = []
        for tid in tool_ids:
            deps = []
            if tid == "tool_change_vqa":
                if "tool_change_detection" in tool_ids:
                    deps.append("tool_change_detection")
                elif "tool_optical_sar_fusion" in tool_ids:
                    deps.append("tool_optical_sar_fusion")
            elif tid == "tool_grounding" and "tool_optical_sar_fusion" in tool_ids:
                deps.append("tool_optical_sar_fusion")
            plan.append({
                "tool_id": tid,
                "dependencies": deps,
                "execution_mode": "sequential" if deps else "parallel",
            })
        return plan

    def _extract_target_entity(self, query: str, task_type: TaskType) -> str:
        q_lower = query.lower()
        if any(w in q_lower for w in ["building", "structure", "settlement"]):
            return "Built-up structures & buildings"
        elif any(w in q_lower for w in ["water", "flood", "lake", "river"]):
            return "Surface hydrology & water boundaries"
        elif any(w in q_lower for w in ["vegetation", "crop", "canopy", "forest"]):
            return "Vegetative canopy & crop parcels"
        elif any(w in q_lower for w in ["tank", "industrial"]):
            return "Industrial storage installations"
        elif any(w in q_lower for w in ["road", "runway", "corridor"]):
            return "Transportation corridors"
        elif task_type == TaskType.BITEMPORAL_CHANGE:
            return "Land-cover transformation delta"
        elif task_type == TaskType.CROSS_MODAL_FUSION:
            return "Cloud-penetrated ground features"
        return "Requested Earth observation features"

    def _extract_temporal_relationship(self, metas: List[GeoTIFFMetadata]) -> Optional[str]:
        if len(metas) < 2:
            return None
        t1 = getattr(metas[0], "filename", "T1 Baseline")
        t2 = getattr(metas[1], "filename", "T2 Surveillance")
        return f"{t1} (Baseline) → {t2} (Surveillance pass)"

    def _build_evidence_timeline(
        self,
        metas: List[GeoTIFFMetadata],
        task_type: TaskType,
        spatial_extra: Dict[str, Any],
        verification: EvidenceVerification,
        decomposition: ConfidenceDecomposition,
    ) -> Optional[List[Dict[str, Any]]]:
        if len(metas) >= 2 and task_type == TaskType.BITEMPORAL_CHANGE:
            t1_name = getattr(metas[0], "filename", "T1 Acquisition")
            t2_name = getattr(metas[1], "filename", "T2 Surveillance")
            ch_ha = spatial_extra.get("change_hectares", 0.0)
            ch_pct = spatial_extra.get("change_percent", 0.0)
            return [
                {"date": "T1 Baseline Pass", "label": t1_name, "event": "Undisturbed baseline state registered", "metrics": {"status": "Baseline"}},
                {"date": "T2 Surveillance Pass", "label": t2_name, "event": f"Observed ground transformation: {ch_ha:.1f} ha ({ch_pct:.1f}%)", "metrics": {"changed_ha": ch_ha, "changed_pct": ch_pct}},
                {"date": "Verification", "label": "Multi-Model Consensus", "event": f"Evidence status: {verification.status} (Agreement: {int(decomposition.spatial_agreement*100)}%)", "metrics": {"confidence": decomposition.overall_confidence}},
            ]
        elif len(metas) >= 2 and task_type == TaskType.CROSS_MODAL_FUSION:
            opt_name = getattr(metas[0], "filename", "Optical Multispectral Pass")
            sar_name = getattr(metas[1], "filename", "Sentinel-1 / RISAT SAR Pass")
            cloud_pct = spatial_extra.get("cloud_coverage_percent")
            recon_pct = spatial_extra.get("reconstructed_percent")
            cloud_str = f"with {cloud_pct:.1f}% cloud obstruction" if cloud_pct is not None else "under cloud cover"
            recon_str = f"Restored {recon_pct:.1f}% obscured surface features" if recon_pct is not None else "Synthesized cloud-penetrated fused multi-modal view"
            return [
                {"date": "Optical Acquisition", "label": opt_name, "event": f"Solar reflectance captured {cloud_str}", "metrics": {"cloud_pct": cloud_pct} if cloud_pct is not None else {}},
                {"date": "SAR Radar Acquisition", "label": sar_name, "event": "Active C-band microwave backscatter penetrated cloud deck", "metrics": {"radar_band": "C-band (5.4 GHz)"}},
                {"date": "Cross-Attention Fusion", "label": "Dual-Branch Fused View", "event": recon_str, "metrics": {"resolved_pct": recon_pct} if recon_pct is not None else {}},
            ]
        return None

    def _fuse_outputs(
        self,
        outputs: List[ToolOutput],
        task_type: TaskType,
        query: str = "",
        image_metas: Optional[List[GeoTIFFMetadata]] = None,
        images: Optional[List[Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[SpatialEvidence], List[str], Optional[Any]]:
        """
        Synthesizes text via AgenticCognitiveSynthesizer and assembles SpatialEvidence.
        """
        import re

        try:
            combined_text, suggested_actions = self.synthesizer.synthesize(
                query=query,
                task_type=task_type,
                tool_outputs=outputs,
                image_metas=image_metas,
                images=images,
                history=history,
            )
        except Exception as syn_err:
            logger.warning(f"Cognitive synthesis failed, fallback to raw output: {syn_err}")
            if len(outputs) == 1:
                combined_text = outputs[0].text_response or ""
            else:
                texts = [o.text_response for o in outputs if o.text_response]
                combined_text = "\n\n".join(texts)
            suggested_actions = ["Explain spatial evidence", "Export PDF Briefing"]

        # Strip any raw bracket artifacts from model passes
        combined_text = re.sub(r'\[(ChangeFormer[^\s\]]*|RS-VLM|Optical-SAR[^\s\]]*|Change-VQA)[^\]]*\]\s*', '', combined_text).strip()

        # Build spatial evidence combining all layers
        spatial = None
        all_boxes = []
        all_clusters = []
        merged_extra = {}
        has_boxes = False
        has_mask = False
        layers = []

        for output in outputs:
            out_extra = copy.deepcopy(output.extra) if output.extra else {}
            if output.extra:
                merged_extra.update(out_extra)
            if output.bounding_boxes:
                has_boxes = True
                all_boxes.extend(output.bounding_boxes)
                box_clusters = out_extra.get("clusters") or [
                    {
                        "zone": f"Target #{len(all_clusters)+i+1}",
                        "category": out_extra.get("dominant_class", "Detected Feature"),
                        "centroid": [round((b[0] + b[2]) / 2.0, 1), round((b[1] + b[3]) / 2.0, 1)],
                        "bbox": b,
                    }
                    for i, b in enumerate(output.bounding_boxes)
                ]
                all_clusters.extend(box_clusters)
                layers.append({
                    "layer_id": f"boxes_{output.tool_id}",
                    "type": "bounding_boxes",
                    "tool_id": output.tool_id,
                    "count": len(output.bounding_boxes),
                    "bounding_boxes": output.bounding_boxes,
                })
            if output.mask is not None:
                has_mask = True
                mask_clusters = out_extra.get("clusters", [])
                if mask_clusters:
                    all_clusters.extend(mask_clusters)
                mask_type = "change_mask" if task_type in [TaskType.BITEMPORAL_CHANGE, TaskType.MULTI_MODEL] else "fusion_map"
                layers.append({
                    "layer_id": f"mask_{output.tool_id}",
                    "type": mask_type,
                    "tool_id": output.tool_id,
                    "changed_hectares": output.extra.get("change_hectares"),
                    "changed_percent": output.extra.get("change_percent"),
                })

        # Deduplicate clusters by zone name preserving order and richness
        unique_clusters = []
        seen_zones = set()
        for c in all_clusters:
            z_key = c.get("zone", "").strip().lower()
            if z_key and z_key not in seen_zones:
                seen_zones.add(z_key)
                unique_clusters.append(c)
            elif not z_key:
                unique_clusters.append(c)
        all_clusters = unique_clusters

        # Deduplicate bounding boxes
        unique_boxes = []
        for b in all_boxes:
            is_dup = False
            for ub in unique_boxes:
                if len(b) == 4 and len(ub) == 4 and all(abs(b[k] - ub[k]) < 2.0 for k in range(4)):
                    is_dup = True
                    break
            if not is_dup:
                unique_boxes.append(b)
        all_boxes = unique_boxes

        if has_boxes and has_mask:
            spatial = SpatialEvidence(
                type="multi_layer",
                bounding_boxes=all_boxes,
                changed_area_hectares=merged_extra.get("change_hectares"),
                changed_area_percent=merged_extra.get("change_percent"),
                clusters=all_clusters,
                layers=layers,
                extra=merged_extra,
            )
        elif has_boxes:
            spatial = SpatialEvidence(
                type="bounding_boxes",
                bounding_boxes=all_boxes,
                clusters=all_clusters,
                layers=layers,
                extra=merged_extra,
            )
        elif has_mask:
            evidence_type = "change_mask" if task_type in [TaskType.BITEMPORAL_CHANGE, TaskType.MULTI_MODEL] else "fusion_map"
            spatial = SpatialEvidence(
                type=evidence_type,
                changed_area_hectares=merged_extra.get("change_hectares"),
                changed_area_percent=merged_extra.get("change_percent"),
                clusters=all_clusters,
                layers=layers,
                extra=merged_extra,
            )

        # Check for VQA Grounding
        vqa_grounding_obj = None
        for output in outputs:
            if output.extra and output.extra.get("vqa_grounding"):
                vg = output.extra["vqa_grounding"]
                from app.schemas.query import VQAGrounding
                if isinstance(vg, dict):
                    vqa_grounding_obj = VQAGrounding(**vg)
                elif isinstance(vg, VQAGrounding):
                    vqa_grounding_obj = vg
                if output.tool_id == "tool_rs_vqa" and output.text_response and not combined_text:
                    combined_text = output.text_response
                break

        # Multi-Model compound analysis card is STRICTLY opt-in for TaskType.MULTI_MODEL
        if task_type == TaskType.MULTI_MODEL:
            multi_model_card = GroundedRSAnalyzer.generate_multi_model_card_assets(
                images=images,
                image_metas=image_metas,
                query=query,
                tool_outputs=outputs,
            )
            merged_extra["multi_model_card"] = multi_model_card
            merged_extra["card_type"] = "multi_model"
            if spatial is None:
                spatial = SpatialEvidence(
                    type="multi_model",
                    mask_url=multi_model_card.get("unified_map_url"),
                    extra=merged_extra,
                )
            else:
                spatial.extra = merged_extra
                if not spatial.mask_url:
                    spatial.mask_url = multi_model_card.get("unified_map_url")
        elif task_type == TaskType.BITEMPORAL_CHANGE:
            if merged_extra.get("disaster_card"):
                merged_extra["card_type"] = "disaster"
            elif merged_extra.get("bitemporal_card"):
                merged_extra["card_type"] = "bitemporal"
        elif task_type == TaskType.CROSS_MODAL_FUSION:
            merged_extra["card_type"] = "optical_sar"
        elif task_type == TaskType.SINGLE_GROUNDING:
            merged_extra["card_type"] = "grounding"

        # Fallback card evidence
        if spatial is None and (merged_extra.get("grounding_card") or merged_extra.get("optical_sar_card")):
            card_mask = None
            if merged_extra.get("grounding_card"):
                card_mask = merged_extra["grounding_card"].get("detection_result_url")
            elif merged_extra.get("optical_sar_card"):
                card_mask = merged_extra["optical_sar_card"].get("fused_result_url")
            spatial = SpatialEvidence(
                type="card_evidence",
                mask_url=card_mask,
                extra=merged_extra,
            )

        if spatial is None and vqa_grounding_obj is not None:
            spatial = SpatialEvidence(
                type="vqa_grounding",
                mask_url=vqa_grounding_obj.overlay_url,
                vqa_grounding=vqa_grounding_obj,
                extra=merged_extra,
            )
        elif spatial is not None and vqa_grounding_obj is not None:
            spatial.vqa_grounding = vqa_grounding_obj
            if not spatial.mask_url:
                spatial.mask_url = vqa_grounding_obj.overlay_url

        # Extract and attach dynamic chart telemetry
        chart_data = merged_extra.get("chart_data")
        if not chart_data and images and len(images) > 0 and images[0] is not None:
            try:
                from app.core.geospatial.grounded.scene_telemetry import SceneTelemetry
                m_dict = image_metas[0].model_dump() if hasattr(image_metas[0], "model_dump") else (image_metas[0] if isinstance(image_metas[0], dict) else {})
                tel = SceneTelemetry.extract_telemetry(images[0], m_dict)
                chart_data = tel.get("chart_data")
                if chart_data:
                    merged_extra["chart_data"] = chart_data
            except Exception as e:
                logger.warning(f"Failed to generate dynamic chart_data: {e}")

        if spatial is not None and chart_data is not None:
            spatial.chart_data = chart_data
        elif spatial is None and chart_data is not None:
            spatial = SpatialEvidence(
                type="telemetry_chart",
                chart_data=chart_data,
                extra=merged_extra,
            )

        return combined_text, spatial, suggested_actions, vqa_grounding_obj

    def _save_mask(
        self,
        outputs: List[ToolOutput],
        image_metas: List[GeoTIFFMetadata],
    ) -> Optional[str]:
        """Save generated binary mask to uploads directory as contoured PNG."""
        from PIL import Image

        for output in outputs:
            if output.mask is not None and isinstance(output.mask, np.ndarray):
                mask_id = uuid4().hex[:12]
                mask_filename = f"mask_{mask_id}.png"
                mask_path = settings.upload_dir / mask_filename

                mask_arr = np.squeeze(output.mask)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[0]

                if mask_arr.ndim != 2:
                    continue

                h, w = mask_arr.shape
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                if np.issubdtype(mask_arr.dtype, np.floating):
                    is_obj = (mask_arr > 0.5)
                else:
                    is_obj = (mask_arr > 0)

                if np.any(is_obj):
                    from scipy.ndimage import binary_dilation, binary_erosion
                    boundary = binary_dilation(is_obj, iterations=2) & ~binary_erosion(is_obj, iterations=1)
                    fill_col = [37, 99, 235, 90]      # Ocean blue fill
                    border_col = [56, 189, 248, 255]  # Electric cyan contour (#38bdf8)
                    rgba[is_obj] = fill_col
                    rgba[boundary] = border_col

                img = Image.fromarray(rgba, mode="RGBA")
                img.save(str(mask_path), format="PNG")
                return f"/static/uploads/{mask_filename}"

        return None

    def _save_geojson(self, geojson_data: Dict[str, Any]) -> Optional[str]:
        """Save generated GeoJSON feature collection to uploads directory."""
        if not geojson_data or not geojson_data.get("features"):
            return None
        geojson_id = uuid4().hex[:12]
        geojson_filename = f"geojson_{geojson_id}.json"
        geojson_path = settings.upload_dir / geojson_filename
        try:
            with open(geojson_path, "w", encoding="utf-8") as f:
                json.dump(geojson_data, f, indent=2)
            return f"/static/uploads/{geojson_filename}"
        except Exception as e:
            logger.warning(f"Could not save GeoJSON file: {e}")
            return None


# ── Global Orchestrator Agent Singleton ─────────────────────────────
_shared_agent: Optional[SatQueryAgent] = None
_agent_lock = threading.Lock()


def get_orchestrator_agent() -> SatQueryAgent:
    """Shared singleton orchestrator agent across REST and WebSocket endpoints."""
    global _shared_agent
    if _shared_agent is None:
        with _agent_lock:
            if _shared_agent is None:
                _shared_agent = SatQueryAgent()
    return _shared_agent
