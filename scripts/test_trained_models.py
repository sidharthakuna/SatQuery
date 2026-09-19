"""
SatQuery AI — Trained Specialist Models Verification Test
Directly tests loading and executing all 4 trained checkpoints:
1. rs_vlm.pt (RS-VLM VQA)
2. grounding_dino.pt / grounding_net.pt (Grounding DINO + SAM)
3. changeformer.pt / change_net.pt (ChangeFormer)
4. optical_sar_fusion.pt / fusion_net.pt (Optical-SAR Fusion)
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add backend to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

import numpy as np
from PIL import Image
from app.tools.base import ToolInput
from app.tools.tool_change_detection import ChangeDetectionTool
from app.tools.tool_grounding import GroundingTool
from app.tools.tool_optical_sar_fusion import OpticalSARFusionTool
from app.tools.tool_rs_vqa import RSVQATool

def main():
    print("============================================================")
    print("  SatQuery AI — Testing All 4 Trained Specialist Models")
    print("============================================================")

    # Dummy satellite test imagery (256x256 RGB)
    img1 = Image.fromarray(np.random.randint(40, 220, (256, 256, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(40, 220, (256, 256, 3), dtype=np.uint8))

    # 1. Test Model 1: RS-VLM VQA
    print("\n[1/4] Testing Model 1: Remote-Sensing VLM / VQA...")
    vqa_tool = RSVQATool()
    vqa_input = ToolInput(
        query="What is the primary land cover visible in this satellite scene?",
        images=[img1],
        image_metas=[{"width": 256, "height": 256}],
    )
    vqa_out = vqa_tool._cuda_execute(vqa_input)
    print(f"  -> Response: {vqa_out.text_response}")
    print(f"  -> Confidence: {vqa_out.confidence} | Latency: {vqa_out.extra.get('latency_ms')}ms")
    assert vqa_out.text_response, "VQA response empty"

    # 2. Test Model 2: Grounding DINO + SAM
    print("\n[2/4] Testing Model 2: Grounding DINO + SAM-RS...")
    ground_tool = GroundingTool()
    ground_input = ToolInput(
        query="Locate and segment all building facilities and structures",
        images=[img1],
        image_metas=[{"width": 256, "height": 256}],
        parameters={"box_threshold": 0.2},
    )
    ground_out = ground_tool._cuda_execute(ground_input)
    print(f"  -> Response: {ground_out.text_response}")
    print(f"  -> Bounding Boxes Detected: {len(ground_out.bounding_boxes)}")
    assert len(ground_out.bounding_boxes) > 0, "No bounding boxes predicted"

    # 3. Test Model 3: ChangeFormer
    print("\n[3/4] Testing Model 3: ChangeFormer Bi-Temporal Analysis...")
    change_tool = ChangeDetectionTool()
    change_input = ToolInput(
        query="Detect structural urban expansion between T1 and T2",
        images=[img1, img2],
        image_metas=[{"width": 256, "height": 256}, {"width": 256, "height": 256}],
    )
    change_out = change_tool._cuda_execute(change_input)
    print(f"  -> Response: {change_out.text_response}")
    print(f"  -> Change Mask Shape: {change_out.mask.shape if change_out.mask is not None else None}")
    assert change_out.mask is not None, "Change mask missing"

    # 4. Test Model 4: Optical + SAR Fusion
    print("\n[4/5] Testing Model 4: Optical + SAR Cross-Modal Fusion...")
    fusion_tool = OpticalSARFusionTool()
    fusion_input = ToolInput(
        query="Fuse cloudy optical scene with SAR C-band microwave backscatter",
        images=[img1, img2],
        image_metas=[{"width": 256, "height": 256}, {"width": 256, "height": 256}],
    )
    fusion_out = fusion_tool._cuda_execute(fusion_input)
    print(f"  -> Response: {fusion_out.text_response}")
    print(f"  -> Resolved Mask Shape: {fusion_out.mask.shape if fusion_out.mask is not None else None}")
    assert fusion_out.mask is not None, "Fusion mask missing"

    # 5. Test Model 5: Agent Intent & Controller Network
    print("\n[5/5] Testing Model 5: Agent Intent & Controller Network (AgentIntentNet v2.0)...")
    from app.inference.providers.pytorch_provider import PyTorchProvider
    provider = PyTorchProvider()
    assert hasattr(provider, "predict_intent"), "predict_intent method missing from provider"
    
    intent_tests = [
        ("Locate and box all the storage tanks near the runway", "SINGLE_GROUNDING"),
        ("Quantify the total hectares of rainforest lost from 2021 to 2024", "BITEMPORAL_CHANGE"),
        ("Pierce through cloud deck using Sentinel-1 microwave radar", "CROSS_MODAL_FUSION"),
        ("What is the predominant land cover and NDVI in this scene?", "SINGLE_VQA"),
        ("How can SatQuery AI assist my geospatial intelligence workflow?", "AGENT_ASSISTANT"),
        ("Detect changes between these dates and highlight all affected buildings", "MULTI_MODEL"),
    ]
    for q_text, expected_task in intent_tests:
        res = provider.predict_intent(q_text)
        print(f"  -> Query: '{q_text[:42]}...'")
        print(f"     Pred: {res['task_type']} (Confidence: {res['confidence']*100:.1f}%) | Model: {res['model']}")
        assert res['task_type'] == expected_task, f"Expected {expected_task}, got {res['task_type']}"

    print("\n============================================================")
    print("  [SUCCESS] All 5 Trained Specialist & Controller Models Verified!")
    print("============================================================")

if __name__ == "__main__":
    main()

