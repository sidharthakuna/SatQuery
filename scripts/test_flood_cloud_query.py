"""
Test script to verify end-to-end multi-model execution for compound cloud-removal & flood-detection.
Query: 'remove clouds and tell the flooded area if we are giving the flooded area's sar and optical image view where clouds obstruct the optical image but we need flooded areas to detect'
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

import asyncio
from app.api.v1._file_utils import find_uploaded_file
from app.core.geospatial.reader import inspect_geotiff
from app.core.orchestrator.agent import get_orchestrator_agent
from app.schemas.audit import TaskType


async def main():
    print("================================================================")
    print(" Testing Multi-Model Optical-SAR Cloud Removal & Flood Detection")
    print("================================================================")

    query = (
        "remove clouds and tell the flooded area if we are giving the flooded area's sar and "
        "optical image view where clouds obstruct the optical image but we need flooded areas to detect"
    )

    opt_path = find_uploaded_file("cartosat_t1.tif")
    sar_path = find_uploaded_file("risat_sar.tif")

    assert opt_path is not None, "Missing cartosat_t1.tif"
    assert sar_path is not None, "Missing risat_sar.tif"

    meta_opt = inspect_geotiff(opt_path, file_id="cartosat_t1")
    meta_sar = inspect_geotiff(sar_path, file_id="risat_sar")

    print(f"\n[Inputs Loaded]")
    print(f"  Image 1 (Optical) : {meta_opt.file_path} | Modality: {meta_opt.modality}")
    print(f"  Image 2 (SAR)     : {meta_sar.file_path} | Modality: {meta_sar.modality}")
    print(f"  Query             : \"{query}\"")

    agent = get_orchestrator_agent()

    result = await agent.process_query(
        query=query,
        image_metas=[meta_opt, meta_sar],
    )

    trace = result.audit_trace
    print(f"\n[Execution Audit Trace]")
    print(f"  Task Identified   : {trace.task_identified}")
    print(f"  Selected Tools    : {trace.selected_tools}")
    print(f"  Confidence Score  : {trace.confidence_score:.2f}")
    print(f"  Execution Time    : {trace.total_execution_time_ms:.1f} ms")
    print(f"  DAG Plan Steps    : {result.tool_execution_plan}")

    # Assertions
    assert trace.task_identified == TaskType.MULTI_MODEL, f"Expected MULTI_MODEL, got {trace.task_identified}"
    assert "tool_optical_sar_fusion" in trace.selected_tools, "tool_optical_sar_fusion missing from tools"
    assert any(t in trace.selected_tools for t in ["tool_change_vqa", "tool_grounding"]), "Downstream flood tools missing"

    print(f"\n[Model Intelligence Synthesis Response]")
    print("-" * 60)
    print(result.text_response)
    print("-" * 60)

    if result.spatial_evidence:
        print(f"\n[Spatial Evidence]")
        print(f"  Type              : {result.spatial_evidence.type}")
        print(f"  Mask URL          : {result.spatial_evidence.mask_url}")
        print(f"  Clusters Count    : {len(result.spatial_evidence.clusters)}")
        print(f"  Layers Count      : {len(result.spatial_evidence.layers)}")
        for c in result.spatial_evidence.clusters[:3]:
            print(f"    - {c.get('zone')}: {c.get('area_ha')} ha ({c.get('category')})")

    print("\n================================================================")
    print(" [SUCCESS] Multi-Model Compound Cloud + Flood Query Executed Cleanly!")
    print("================================================================")


if __name__ == "__main__":
    asyncio.run(main())
