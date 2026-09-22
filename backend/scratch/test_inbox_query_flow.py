"""
Simulates natural user queries submitted from the inbox in the web interface.
Tests both spatial visual grounding (bounding boxes + masks) and VQA questions.
"""

import asyncio
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
backend_dir = root_dir / "backend"
sys.path.insert(0, str(backend_dir))

from app.core.geospatial.reader import inspect_geotiff
from app.core.orchestrator.agent import SatQueryAgent

samples_dir = root_dir / "data" / "samples"

async def test_inbox_queries():
    print("=" * 70)
    print("  SIMULATING USER INBOX QUERIES IN SATQUERY AI")
    print("=" * 70)

    agent = SatQueryAgent()

    # Query 1: Visual Grounding - Storage Tanks in Visakhapatnam Port
    port_file = samples_dir / "port_grounding.tif"
    meta_port = inspect_geotiff(str(port_file), file_id="port_grounding")
    query1 = "Identify and mark the storage tanks in this image"
    print(f"\n[INBOX QUERY 1] '{query1}'")
    print(f"Image: {port_file.name}")
    res1 = await agent.process_query(query=query1, image_metas=[meta_port])
    print(f"Task Classified: {res1.audit_trace.task_identified.value if res1.audit_trace else 'N/A'}")
    print(f"Spatial Evidence Type: {res1.spatial_evidence.type if res1.spatial_evidence else 'None'}")
    print(f"Bounding Boxes Count: {len(res1.spatial_evidence.bounding_boxes) if res1.spatial_evidence and res1.spatial_evidence.bounding_boxes else 0}")
    if res1.spatial_evidence and res1.spatial_evidence.clusters:
        print(f"Clusters: {[c.get('zone') for c in res1.spatial_evidence.clusters]}")
    print(f"Text Response Preview:\n{res1.text_response[:300]}...")

    # Query 2: Visual Grounding - Launchpads in Sriharikota Spaceport
    shar_file = samples_dir / "nasa_landsat_sriharikota.tif"
    meta_shar = inspect_geotiff(str(shar_file), file_id="nasa_landsat_sriharikota")
    query2 = "Locate all launchpad complexes and vehicle assembly structures"
    print(f"\n[INBOX QUERY 2] '{query2}'")
    print(f"Image: {shar_file.name}")
    res2 = await agent.process_query(query=query2, image_metas=[meta_shar])
    print(f"Task Classified: {res2.audit_trace.task_identified.value if res2.audit_trace else 'N/A'}")
    print(f"Spatial Evidence Type: {res2.spatial_evidence.type if res2.spatial_evidence else 'None'}")
    print(f"Bounding Boxes Count: {len(res2.spatial_evidence.bounding_boxes) if res2.spatial_evidence and res2.spatial_evidence.bounding_boxes else 0}")
    if res2.spatial_evidence and res2.spatial_evidence.clusters:
        print(f"Clusters: {[c.get('zone') for c in res2.spatial_evidence.clusters]}")
    print(f"Text Response Preview:\n{res2.text_response[:300]}...")

    # Query 3: Maritime Vessels
    query3 = "Detect and box all cargo vessels and ships berthed in the harbor"
    print(f"\n[INBOX QUERY 3] '{query3}'")
    print(f"Image: {port_file.name}")
    res3 = await agent.process_query(query=query3, image_metas=[meta_port])
    print(f"Task Classified: {res3.audit_trace.task_identified.value if res3.audit_trace else 'N/A'}")
    print(f"Bounding Boxes Count: {len(res3.spatial_evidence.bounding_boxes) if res3.spatial_evidence and res3.spatial_evidence.bounding_boxes else 0}")
    if res3.spatial_evidence and res3.spatial_evidence.clusters:
        print(f"Clusters: {[c.get('zone') for c in res3.spatial_evidence.clusters]}")

    print("\n" + "=" * 70)
    print("  INBOX QUERIES PROCESSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_inbox_queries())
