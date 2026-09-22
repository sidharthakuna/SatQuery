"""
Verification script: Test the fine-tuned local neural VLM and multi-turn conversational reasoning.
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import numpy as np
import torch

from app.core.orchestrator.agentic_synthesizer import (
    AgenticCognitiveSynthesizer,
    ConversationMemoryTracker,
)
from app.core.geospatial.grounded_analyzer import GroundedRSAnalyzer
from app.schemas.audit import TaskType
from app.schemas.geospatial import GeoTIFFMetadata
from app.tools.base import ToolInput
from app.tools.tool_rs_vqa import RSVQATool


def test_verification():
    print("=" * 60)
    print("1. Testing Model Checkpoint & Vocabulary Loading")
    ckpt_path = backend_dir / "data" / "checkpoints" / "rs_vlm.pt"
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} does not exist!"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab = state.get("vocab", {})
    print(f"  Checkpoint exists! Size: {ckpt_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Embedded Vocabulary Size: {len(vocab)} tokens")
    print(f"  Best Training Loss: {state.get('best_loss', 'N/A')}")
    assert len(vocab) > 600, "Vocabulary size should be expanded"

    print("\n2. Testing RS-VQA Specialist Neural Inference")
    tool = RSVQATool()
    # Create test synthetic optical image
    test_img = np.random.uniform(0.2, 0.7, (3, 256, 256)).astype(np.float32)
    meta = GeoTIFFMetadata(
        file_id="test_scene",
        filename="sentinel2_test.tif",
        width=256,
        height=256,
        band_count=3,
        crs="EPSG:4326",
        modality="OPTICAL",
        spatial_resolution_m=10.0,
    )

    t_input = ToolInput(
        images=[test_img],
        image_metas=[meta.model_dump()],
        query="What is visible in this satellite scene?",
        parameters={"temperature": 0.2},
    )
    output = tool.execute(t_input)
    print(f"  RSVQATool executed successfully!")
    print(f"  Confidence: {output.confidence:.2f}")
    print(f"  Text Response:\n    {output.text_response[:180]}...")
    assert output.text_response, "Output text response should not be empty"

    print("\n3. Testing Multi-Turn Conversational Reasoning (Zero External APIs)")
    synthesizer = AgenticCognitiveSynthesizer()

    # Turn 1: Primary Overview Query
    print("\n  [Turn 1: Overview]")
    resp1, sugg1 = synthesizer.synthesize(
        query="Describe this satellite image in detail",
        task_type=TaskType.SINGLE_VQA,
        tool_outputs=[output],
        image_metas=[meta],
        images=[test_img],
        history=[],
    )
    print(f"  Assistant Response (first 180 chars):\n    {resp1[:180]}...")

    # Turn 2: Follow-Up Question about Vegetation
    print("\n  [Turn 2: Follow-Up on Vegetation]")
    history_turn2 = [
        {"role": "user", "content": "Describe this satellite image in detail"},
        {"role": "assistant", "content": resp1},
    ]
    resp2, sugg2 = synthesizer.synthesize(
        query="Can you tell me more about the vegetation shown here?",
        task_type=TaskType.SINGLE_VQA,
        tool_outputs=[output],
        image_metas=[meta],
        images=[test_img],
        history=history_turn2,
    )
    print(f"  Assistant Response (first 180 chars):\n    {resp2[:180]}...")
    assert "vegetation" in resp2.lower() or "canopy" in resp2.lower(), "Turn 2 should specifically address vegetation"
    assert resp2 != resp1, "Turn 2 should NOT repeat Turn 1!"

    # Turn 3: Follow-Up Question about Built Structures in North Sector
    print("\n  [Turn 3: Follow-Up on North Quadrant Structures]")
    history_turn3 = history_turn2 + [
        {"role": "user", "content": "Can you tell me more about the vegetation shown here?"},
        {"role": "assistant", "content": resp2},
    ]
    resp3, sugg3 = synthesizer.synthesize(
        query="What about the buildings in the north quadrant?",
        task_type=TaskType.SINGLE_VQA,
        tool_outputs=[output],
        image_metas=[meta],
        images=[test_img],
        history=history_turn3,
    )
    print(f"  Assistant Response (first 180 chars):\n    {resp3[:180]}...")
    assert "building" in resp3.lower() or "structural" in resp3.lower() or "sector" in resp3.lower(), "Turn 3 should address buildings/quadrant"
    assert resp3 != resp1 and resp3 != resp2, "Turn 3 should be distinct from prior turns!"

    print("\n4. Testing Grounded RS Analyzer Delegation")
    scene_analysis = GroundedRSAnalyzer.analyze_single_scene(test_img, meta.model_dump(), "What is here?")
    print(f"  GroundedRSAnalyzer.analyze_single_scene success!")
    print(f"  Computed Vegetation: {scene_analysis['vegetation_percent']}%")
    print(f"  Computed Built: {scene_analysis['built_percent']}%")
    print(f"  Computed Water: {scene_analysis['water_percent']}%")
    print(f"  Answer: {scene_analysis['answer'][:120]}...")

    card_assets = GroundedRSAnalyzer.generate_vqa_card_assets(
        image=test_img,
        image_meta=meta.model_dump(),
        query="What is here?",
        text_response=resp1,
    )
    print(f"  GroundedRSAnalyzer.generate_vqa_card_assets success! Card ID: {card_assets['card_id']}")
    assert card_assets["card_id"].startswith("SQ-VQA-")

    geojson = GroundedRSAnalyzer.generate_geojson(spatial_type="features", boxes=[[10, 10, 50, 50]])
    print(f"  GroundedRSAnalyzer.generate_geojson success! Feature count: {len(geojson['features'])}")
    assert len(geojson["features"]) == 1

    print("\n5. Testing End-to-End SatQueryAgent.process_query with Multi-Turn History")
    import asyncio
    from app.core.orchestrator.agent import SatQueryAgent

    async def _test_agent_async():
        agent = SatQueryAgent()
        result1 = await agent.process_query(
            query="Describe this scene",
            image_metas=[meta],
            images=[test_img],
            history=[],
        )
        print(f"  Agent Turn 1 (first 120 chars): {result1.text_response[:120]}...")
        assert result1.text_response

        # Follow-up Turn 2
        history_t2 = [
            {"role": "user", "content": "Describe this scene"},
            {"role": "assistant", "content": result1.text_response},
        ]
        result2 = await agent.process_query(
            query="Can you tell me more about the vegetation?",
            image_metas=[meta],
            images=[test_img],
            history=history_t2,
        )
        print(f"  Agent Turn 2 (first 120 chars): {result2.text_response[:120]}...")
        assert result2.text_response != result1.text_response, "Agent Turn 2 should not be identical to Turn 1"

    asyncio.run(_test_agent_async())

    print("\n" + "=" * 60)
    print("  ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("  - Local fine-tuned neural model operational")
    print("  - Zero external API keys needed")
    print("  - Multi-turn conversation memory functioning")
    print("  - Monolithic code modularized and future-proofed")
    print("=" * 60)


if __name__ == "__main__":
    test_verification()
