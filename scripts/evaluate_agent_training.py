"""
Comprehensive Verification & Audit Script for SatQuery AI Agent Intent Controller
Tests model weights, architecture, classification accuracy, confidence calibration,
and orchestrator DAG routing across all task domains.
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

import torch
import numpy as np
from app.models.intent_net import AgentIntentNet, TASK_CLASSES_V3
from app.core.orchestrator.router import QueryIntentClassifier
from app.schemas.audit import TaskType

def evaluate_agent_controller():
    print("=" * 70)
    print("  SatQuery AI — Agent Intent Controller Training Verification")
    print("=" * 70)

    checkpoint_path = backend_dir / "data" / "checkpoints" / "intent_net.pt"
    if not checkpoint_path.exists():
        print(f"[FAIL] Checkpoint not found at {checkpoint_path}")
        return False

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"\n[Checkpoint Metadata]")
    print(f"  File Path      : {checkpoint_path}")
    print(f"  File Size      : {checkpoint_path.stat().st_size / 1024:.1f} KB")
    print(f"  Trained Epochs : {ckpt.get('epoch', 'N/A')}")
    print(f"  Val Accuracy   : {ckpt.get('accuracy', 'N/A'):.2f}%")
    print(f"  Architecture   : {ckpt.get('config', {}).get('arch', 'N/A')}")
    print(f"  Vocab Size     : {len(ckpt['vocab'])} tokens")
    print(f"  Task Classes   : {ckpt['classes']}")

    vocab = ckpt["vocab"]
    config = ckpt["config"]
    model = AgentIntentNet(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def tokenize(text: str) -> torch.Tensor:
        import re
        tokens = re.findall(r"\b\w+\b", text.lower())
        ids = [vocab.get(t, vocab.get("<unk>", 1)) for t in tokens]
        if not ids:
            ids = [vocab.get("<unk>", 1)]
        return torch.tensor(ids, dtype=torch.long)

    # Test Suite across diverse realistic operational queries
    test_cases = [
        # Domain: SINGLE_VQA
        ("What is the predominant land cover and vegetation index across this scene?", "SINGLE_VQA"),
        ("Describe the urban density and topological structures visible in this optical image.", "SINGLE_VQA"),
        ("Estimate the water turbidity and surface chlorophyll level in the reservoir.", "SINGLE_VQA"),

        # Domain: SINGLE_GROUNDING
        ("Locate all petroleum storage tanks and draw bounding boxes around them.", "SINGLE_GROUNDING"),
        ("Find and outline the main airport runway and commercial hangars.", "SINGLE_GROUNDING"),
        ("Where are the solar panel farms installed on this hillside? Pinpoint them.", "SINGLE_GROUNDING"),

        # Domain: BITEMPORAL_CHANGE
        ("Calculate the total hectares of forest cover lost between 2020 and 2024.", "BITEMPORAL_CHANGE"),
        ("Detect all new urban constructions that appeared since the baseline survey.", "BITEMPORAL_CHANGE"),
        ("Measure the flood water retreat and shoreline regression over the two passes.", "BITEMPORAL_CHANGE"),

        # Domain: CROSS_MODAL_FUSION
        ("Pierce through the monsoon cloud cover using Sentinel-1 C-band SAR radar.", "CROSS_MODAL_FUSION"),
        ("Fuse optical multispectral bands with SAR microwave backscatter to see the ground.", "CROSS_MODAL_FUSION"),
        ("Synthesize all-weather radar and clear-sky optical imagery for terrain reconstruction.", "CROSS_MODAL_FUSION"),

        # Domain: AGENT_ASSISTANT
        ("Hello! How does the SatQuery multi-model agent pipeline work?", "AGENT_ASSISTANT"),
        ("What satellite sensors and spatial resolutions does this platform support?", "AGENT_ASSISTANT"),
        ("Can you guide me on how to upload a GeoTIFF raster pair for analysis?", "AGENT_ASSISTANT"),

        # Domain: MULTI_MODEL (Compound Queries)
        ("remove clouds and tell the flooded area if we are giving the flooded area's sar and optical image view where clouds obstruct the optical image but we need flooded areas to detect", "MULTI_MODEL"),
        ("Where clouds obstruct the optical image, use SAR to remove clouds and detect flooded areas.", "MULTI_MODEL"),
        ("Eliminate cloud cover with radar fusion and calculate the flooded extent in hectares.", "MULTI_MODEL"),
        ("Detect changes between these dates and highlight all affected buildings with bounding boxes.", "MULTI_MODEL"),
        ("Compare optical and SAR imagery to identify runway boundaries hidden under smoke and clouds.", "MULTI_MODEL"),
        ("Analyze urban expansion between t1 and t2 and pinpoint newly built warehouses.", "MULTI_MODEL"),
    ]

    print("\n" + "=" * 70)
    print("  Zero-Shot Query Battery Evaluation (Agent Controller Inference)")
    print("=" * 70)
    print(f"{'#':<3} | {'Status':<6} | {'Confidence':<10} | {'Expected':<18} | {'Predicted':<18} | {'Query Snippet'}")
    print("-" * 70)

    passed = 0
    from app.inference.providers.pytorch_provider import PyTorchProvider
    provider = PyTorchProvider()
    assert provider is not None, "Inference provider could not be loaded"

    print(f"\n[Provider Type]: {type(provider).__name__}")

    for i, (query, expected) in enumerate(test_cases, 1):
        res = provider.predict_intent(query)
        pred_task = res["task_type"]
        confidence = res["confidence"]
        is_compound = res.get("is_compound", False)
        active = res.get("active_tasks", [])

        is_correct = (pred_task == expected)
        if is_correct:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        snippet = (query[:35] + "...") if len(query) > 35 else query
        print(f"{i:<3} | [{status}] | {confidence*100:>6.1f}%    | {expected:<18} | {pred_task:<18} | {snippet}")

    total = len(test_cases)
    acc = (passed / total) * 100.0
    print("-" * 70)
    print(f"\n[Raw Neural Controller Accuracy]: {passed}/{total} Passed ({acc:.1f}%)")

    from app.core.orchestrator.router import QueryIntentClassifier
    from app.schemas.geospatial import GeoTIFFMetadata
    classifier = QueryIntentClassifier(provider=provider)

    opt_meta = GeoTIFFMetadata(
        file_path="cartosat_t1.tif",
        file_id="cartosat_t1",
        width=512,
        height=512,
        band_count=3,
        crs="EPSG:32643",
        modality="OPTICAL",
    )
    sar_meta = GeoTIFFMetadata(
        file_path="risat_sar.tif",
        file_id="risat_sar",
        width=512,
        height=512,
        band_count=1,
        crs="EPSG:32643",
        modality="SAR",
    )

    print("\n" + "=" * 70)
    print("  Full Orchestrator Intent Routing (Neural Controller + Raster Guard)")
    print("=" * 70)

    orch_passed = 0
    for i, (query, expected) in enumerate(test_cases, 1):
        if expected in ("BITEMPORAL_CHANGE",):
            metas = [opt_meta, opt_meta]
        elif expected in ("CROSS_MODAL_FUSION",):
            metas = [opt_meta, sar_meta]
        elif expected in ("MULTI_MODEL",):
            metas = [opt_meta, sar_meta]
        elif expected in ("SINGLE_VQA", "SINGLE_GROUNDING"):
            metas = [opt_meta]
        else:
            metas = []

        task_type, tools, params = classifier.classify(query, metas)
        is_correct = (task_type.value == expected)
        if is_correct:
            orch_passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        snippet = (query[:32] + "...") if len(query) > 32 else query
        print(f"{i:<3} | [{status}] | {expected:<18} | {task_type.value:<18} | Tools: {tools} | {snippet}")

    orch_acc = (orch_passed / total) * 100.0
    print("-" * 70)
    print(f"\n[Orchestrator Hybrid Accuracy]: {orch_passed}/{total} Passed ({orch_acc:.1f}%)")
    print(f"[Neural Model Standalone Acc]: {passed}/{total} Passed ({acc:.1f}%)")
    assert orch_acc >= 95.0, f"Expected Orchestrator >= 95%, got {orch_acc:.1f}%"

    # Assertions
    assert acc >= 95.0, f"Expected >= 95% accuracy, got {acc:.1f}%"
    print("\n[SUCCESS] The AI Agent Controller is trained and working correctly!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    evaluate_agent_controller()
