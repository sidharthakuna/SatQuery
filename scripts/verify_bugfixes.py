"""
Comprehensive Bugfix & Regression Verification Test
Tests all 8 bugfix locations:
1. Static files directory preparation
2. find_uploaded_file lookup with samples and stems
3. reader.py thumbnail resolution
4. pdf_generator candidate_dirs with samples_dir
5. Float mask thresholding in agent.py
6. GeoTIFFMetadata attribute access in tool_grounding and tool_optical_sar_fusion
7. GeoTIFFMetadata area_ha lookup in grounded_analyzer
8. End-to-end multi-task agent queries
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

import asyncio
import numpy as np
from config.settings import settings
from app.api.v1._file_utils import find_uploaded_file
from app.core.geospatial.reader import inspect_geotiff
from app.utils.pdf_generator import MissionBriefingGenerator
from app.core.orchestrator.agent import SatQueryAgent, get_orchestrator_agent
from app.core.geospatial.grounded_analyzer import GroundedRSAnalyzer
from app.tools.base import ToolInput
from app.tools.registry import ToolRegistry


async def run_all_checks():
    print("==================================================")
    print(" Running SatQuery Comprehensive Bugfix Verifications")
    print("==================================================")

    # ── Check 1: Static directories existence ────────────────
    print("\n[Check 1] Verifying static directories...")
    assert settings.upload_dir.exists(), f"Missing {settings.upload_dir}"
    assert settings.report_dir.exists(), f"Missing {settings.report_dir}"
    assert settings.samples_dir.exists(), f"Missing {settings.samples_dir}"
    print("  -> Passed: Static directories verified.")

    # ── Check 2: find_uploaded_file lookup ──────────────────
    print("\n[Check 2] Verifying find_uploaded_file with sample files...")
    fp1 = find_uploaded_file("cartosat_t1.tif")
    assert fp1 is not None and Path(fp1).exists(), f"Failed to find cartosat_t1.tif: {fp1}"
    fp2 = find_uploaded_file("cartosat_t1")
    assert fp2 is not None and Path(fp2).exists(), f"Failed to find cartosat_t1 by stem: {fp2}"
    fp3 = find_uploaded_file("risat_sar")
    assert fp3 is not None and Path(fp3).exists(), f"Failed to find risat_sar: {fp3}"
    print(f"  -> Passed: Found sample rasters: {Path(fp1).name}, {Path(fp3).name}")

    # ── Check 3: reader thumbnail resolution ────────────────
    print("\n[Check 3] Verifying thumbnail resolution in inspect_geotiff...")
    meta1 = inspect_geotiff(fp1, file_id="cartosat_t1")
    assert meta1.thumbnail_url is not None, f"Thumbnail not resolved for {fp1}"
    print(f"  -> Passed: Resolved thumbnail: {meta1.thumbnail_url}")

    # ── Check 4: pdf_generator candidate_dirs with samples ──
    print("\n[Check 4] Verifying PDF generator resolves sample rasters...")
    gen = MissionBriefingGenerator()
    resolved = gen._resolve_image_path("cartosat_t1.tif")
    assert resolved is not None and resolved.exists(), f"PDF generator could not resolve cartosat_t1.tif: {resolved}"
    print(f"  -> Passed: PDF generator resolved sample image: {resolved}")

    # ── Check 5 & 6: GeoTIFFMetadata access in tools ────────
    print("\n[Check 5 & 6] Verifying tools with GeoTIFFMetadata objects...")
    registry = ToolRegistry()

    # Grounding tool
    g_tool = registry.get_tool("tool_grounding")
    arr = np.random.uniform(0.1, 0.9, (3, 256, 256)).astype(np.float32)
    g_in = ToolInput(
        images=[arr],
        image_metas=[meta1],  # Pydantic GeoTIFFMetadata
        query="Locate all buildings",
        parameters={"box_threshold": 0.35},
    )
    g_out = g_tool.execute(g_in)
    assert g_out.bounding_boxes is not None, "Grounding tool failed to produce boxes"
    print(f"  -> Passed: Grounding tool executed with GeoTIFFMetadata. Boxes: {len(g_out.bounding_boxes)}")

    # Fusion tool
    f_tool = registry.get_tool("tool_optical_sar_fusion")
    meta_sar = inspect_geotiff(fp3, file_id="risat_sar")
    sar_arr = np.random.uniform(0.1, 0.9, (2, 256, 256)).astype(np.float32)
    f_in = ToolInput(
        images=[arr, sar_arr],
        image_metas=[meta1, meta_sar],
        query="Fuse optical and SAR through clouds",
        parameters={"cloud_threshold": 0.3},
    )
    f_out = f_tool.execute(f_in)
    assert f_out.mask is not None, "Fusion tool failed to produce mask"
    print(f"  -> Passed: Fusion tool executed with GeoTIFFMetadata. Mask shape: {f_out.mask.shape}")

    # ── Check 7: analyze_bitemporal with GeoTIFFMetadata ────
    print("\n[Check 7] Verifying GroundedRSAnalyzer.analyze_bitemporal with GeoTIFFMetadata...")
    t2_path = find_uploaded_file("cartosat_t2.tif")
    meta2 = inspect_geotiff(t2_path, file_id="cartosat_t2")
    bitemp = GroundedRSAnalyzer.analyze_bitemporal(
        images=[arr, arr],
        image_metas=[meta1, meta2],
        query="Detect changes between acquisitions",
    )
    assert "change_hectares" in bitemp, "bitemporal analysis missing change_hectares"
    print(f"  -> Passed: Bitemporal analysis with GeoTIFFMetadata. Extent: {bitemp['change_hectares']} ha")

    # ── Check 8: End-to-End Orchestrator Pipeline Execution ──
    print("\n[Check 8] Verifying End-to-End Orchestrator Agent across task types...")
    agent = get_orchestrator_agent()

    # Query A: Single Grounding
    res_ground = await agent.process_query(
        query="Locate and box all facilities",
        image_metas=[meta1],
    )
    assert res_ground.audit_trace.task_identified == "SINGLE_GROUNDING", f"Wrong task: {res_ground.audit_trace.task_identified}"
    assert res_ground.spatial_evidence is not None, "Missing spatial evidence"
    print(f"  -> Passed: Single Grounding (Confidence: {res_ground.audit_trace.confidence_score:.2f})")

    # Query B: Bi-Temporal Change
    res_change = await agent.process_query(
        query="What has changed between these two survey dates?",
        image_metas=[meta1, meta2],
    )
    assert res_change.audit_trace.task_identified == "BITEMPORAL_CHANGE", f"Wrong task: {res_change.audit_trace.task_identified}"
    print(f"  -> Passed: Bi-temporal Change (Confidence: {res_change.audit_trace.confidence_score:.2f})")

    # Query C: Optical-SAR Fusion
    res_fusion = await agent.process_query(
        query="Penetrate cloud layer and combine with radar",
        image_metas=[meta1, meta_sar],
    )
    assert res_fusion.audit_trace.task_identified == "CROSS_MODAL_FUSION", f"Wrong task: {res_fusion.audit_trace.task_identified}"
    print(f"  -> Passed: Cross-Modal Fusion (Confidence: {res_fusion.audit_trace.confidence_score:.2f})")

    # Query D: Compound Multi-Model Query (Explicit Multi-Model Intent)
    res_multi = await agent.process_query(
        query="Run multi-model integrated pipeline combining all models to analyze changes and highlight buildings",
        image_metas=[meta1, meta2],
    )
    assert res_multi.audit_trace.task_identified == "MULTI_MODEL", f"Wrong task: {res_multi.audit_trace.task_identified}"
    assert res_multi.tool_execution_plan is not None, "Missing tool execution plan"
    print(f"  -> Passed: Compound Multi-Model (Confidence: {res_multi.audit_trace.confidence_score:.2f})")

    # Query E: Conversational / General Copilot
    res_chat = await agent.process_query(
        query="Hello! How does SatQuery AI work?",
        image_metas=[],
    )
    assert res_chat.audit_trace.task_identified == "AGENT_ASSISTANT", f"Wrong task: {res_chat.audit_trace.task_identified}"
    print(f"  -> Passed: Agent Assistant Copilot (Confidence: {res_chat.audit_trace.confidence_score:.2f})")

    # ── Check 9: Slippy Map Tile Rendering ──────────────────
    print("\n[Check 9] Verifying tile rendering...")
    from app.api.v1.endpoints_tiles import _render_tile
    tile_bytes = _render_tile(fp1, z=0, x=0, y=0)
    assert tile_bytes is not None and len(tile_bytes) > 0, "Tile rendering returned empty bytes"
    print(f"  -> Passed: Rendered tile bytes: {len(tile_bytes)}")

    print("\n==================================================")
    print(" [SUCCESS] ALL 9 VERIFICATION CHECKS PASSED CLEANLY!")
    print("==================================================")


if __name__ == "__main__":
    asyncio.run(run_all_checks())
