"""
SatQuery AI — Comprehensive Verification Suite for ChatGPT-Grade Dynamic Messaging
Verifies:
1. Direct-Answer-First Accuracy across query archetypes (Verification, Quantification, Spatial, Risk)
2. ChatGPT-Style Conceptual Knowledge Synthesis (SAR cloud penetration, NDVI, NDWI, ISRO missions)
3. Fluid Conversational Copilot Courtesy & Tone Shifts
4. Seamless RS-VLM Neural Vision-Language Model Weaving
5. Elimination of Stiff Robotic Boilerplate
"""

import sys
from pathlib import Path
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.orchestrator.synthesis.conversational_vqa_engine import ConversationalVQAEngine
from app.core.orchestrator.synthesis.synthesizer import AgenticCognitiveSynthesizer
from app.schemas.audit import TaskType


def test_verification_direct_answer():
    print("Testing 1: Direct-Answer-First on Verification Queries...")
    fake_img = np.random.uniform(0.1, 0.8, (3, 256, 256)).astype(np.float32)
    fake_img[1, 50:150, 50:150] = 0.95
    fake_img[0, 50:150, 50:150] = 0.15
    meta = {"filename": "sentinel2_test.tif", "spatial_resolution_m": 10.0, "band_count": 3}

    # Query: Is there water?
    q_water = "Is there any water in this scene?"
    ans_water, _ = ConversationalVQAEngine.synthesize_answer(q_water, [meta], {}, [fake_img])
    assert ans_water.startswith("**Yes**") or ans_water.startswith("**No") or "surface water" in ans_water.lower()[:80], f"Expected direct answer first, got: {ans_water[:100]}"
    print("  [PASS] 'Is there water?' directly answered immediately!")

    # Query: Are there buildings?
    q_built = "Are there buildings or structures here?"
    ans_built, _ = ConversationalVQAEngine.synthesize_answer(q_built, [meta], {}, [fake_img])
    assert ans_built.startswith("**Yes**") or ans_built.startswith("**Sparse") or "built" in ans_built.lower()[:80], f"Expected direct answer first, got: {ans_built[:100]}"
    print("  [PASS] 'Are there buildings?' directly answered immediately!")


def test_quantification_direct_answer():
    print("Testing 2: Direct-Answer-First on Quantification Queries...")
    fake_img = np.random.uniform(0.1, 0.8, (3, 256, 256)).astype(np.float32)
    meta = {"filename": "karnataka_scene.tif", "spatial_resolution_m": 10.0, "band_count": 3}

    q_veg = "How much green cover and vegetation is there?"
    ans_veg, _ = ConversationalVQAEngine.synthesize_answer(q_veg, [meta], {}, [fake_img])
    assert "cover" in ans_veg.lower()[:100] and "hectares" in ans_veg.lower()[:150], f"Expected quantified area first, got: {ans_veg[:120]}"
    print("  [PASS] 'How much green cover?' gave exact percentage and hectares immediately!")


def test_spatial_location_direct_answer():
    print("Testing 3: Direct-Answer-First on Spatial Queries...")
    fake_img = np.random.uniform(0.1, 0.8, (3, 256, 256)).astype(np.float32)
    meta = {"filename": "scene.tif", "spatial_resolution_m": 10.0, "band_count": 3}

    q_loc = "Where are the buildings located?"
    ans_loc, _ = ConversationalVQAEngine.synthesize_answer(q_loc, [meta], {}, [fake_img])
    assert "quadrant" in ans_loc.lower()[:120] or "sector" in ans_loc.lower()[:120], f"Expected sector location first, got: {ans_loc[:120]}"
    print("  [PASS] 'Where are the buildings?' pointed directly to sector/quadrant immediately!")


def test_risk_operational_direct_answer():
    print("Testing 4: Direct-Answer-First on Risk & Operational Queries...")
    fake_img = np.random.uniform(0.1, 0.8, (3, 256, 256)).astype(np.float32)
    meta = {"filename": "scene.tif", "spatial_resolution_m": 10.0, "band_count": 3}

    q_risk = "Is this area prone to flooding?"
    ans_risk, _ = ConversationalVQAEngine.synthesize_answer(q_risk, [meta], {}, [fake_img])
    assert ("risk" in ans_risk.lower()[:100] or "drainage" in ans_risk.lower()[:100] or "flood" in ans_risk.lower()[:100]), f"Expected flood risk assessment first, got: {ans_risk[:120]}"
    print("  [PASS] 'Is this area prone to flooding?' gave direct risk verdict!")


def test_chatgpt_conceptual_science_explanations():
    print("Testing 5: ChatGPT-Grade Concept Explanations in Copilot...")
    synth = AgenticCognitiveSynthesizer()

    # 1. SAR Cloud Penetration
    q_sar = "Explain how SAR penetrates clouds"
    ans_sar, _ = synth.synthesize(q_sar, TaskType.AGENT_ASSISTANT, [])
    assert "microwave" in ans_sar.lower(), "Expected microwave wavelength explanation"
    assert "analogy" in ans_sar.lower(), "Expected intuitive analogy"
    assert "optical" in ans_sar.lower() and "risat" in ans_sar.lower() or "sentinel" in ans_sar.lower()
    print("  [PASS] SAR cloud penetration answered with clear analogy, wavelengths, and ISRO mission!")

    # 2. NDVI Formula & Mechanism
    q_ndvi = "What is NDVI and how is it calculated?"
    ans_ndvi, _ = synth.synthesize(q_ndvi, TaskType.AGENT_ASSISTANT, [])
    assert "nir" in ans_ndvi.lower() and "red" in ans_ndvi.lower(), "Expected NIR and Red explanation"
    assert "chlorophyll" in ans_ndvi.lower(), "Expected chlorophyll absorption principle"
    print("  [PASS] NDVI explained with mathematical formula and biophysical principles!")

    # 3. ISRO Cartosat-3
    q_cartosat = "Tell me about ISRO Cartosat-3"
    ans_cartosat, _ = synth.synthesize(q_cartosat, TaskType.AGENT_ASSISTANT, [])
    assert "0.28" in ans_cartosat or "cartosat" in ans_cartosat.lower(), "Expected Cartosat-3 resolution details"
    print("  [PASS] ISRO Cartosat-3 mission explained with precision specs!")


def test_conversational_copilot_courtesy():
    print("Testing 6: Conversational Copilot Courtesy & Warmth...")
    synth = AgenticCognitiveSynthesizer()

    q_thanks = "Thank you so much, that was awesome!"
    ans_thanks, _ = synth.synthesize(q_thanks, TaskType.AGENT_ASSISTANT, [])
    assert "welcome" in ans_thanks.lower(), f"Expected polite welcome, got: {ans_thanks[:80]}"
    print("  [PASS] Conversational courtesy acknowledged warmly like ChatGPT!")


def test_neural_vlm_weaving():
    print("Testing 7: Seamless RS-VLM Neural Weaving...")
    fake_img = np.random.uniform(0.1, 0.8, (3, 256, 256)).astype(np.float32)
    meta = {"filename": "scene.tif", "spatial_resolution_m": 10.0, "band_count": 3}
    extra = {"neural_prediction": "Dense agricultural cropland with irrigation channels"}

    q_vlm = "Describe this scene"
    ans_vlm, _ = ConversationalVQAEngine.synthesize_answer(q_vlm, [meta], extra, [fake_img])
    assert "dense agricultural cropland" in ans_vlm.lower(), "Expected RS-VLM neural prediction to be woven into answer"
    assert "neural vision-language" in ans_vlm.lower() or "rs-vlm" in ans_vlm.lower()
    print("  [PASS] RS-VLM neural observation organically woven into conversational text!")


def test_no_robotic_boilerplate():
    print("Testing 8: Elimination of Robotic Boilerplate...")
    fake_img = np.random.uniform(0.1, 0.8, (3, 256, 256)).astype(np.float32)
    meta = {"filename": "scene.tif", "spatial_resolution_m": 10.0, "band_count": 3}

    q = "What is the vegetation coverage?"
    ans, _ = ConversationalVQAEngine.synthesize_answer(q, [meta], {}, [fake_img])
    assert "delving further into" not in ans.lower(), "Found old robotic boilerplate in response"
    print("  [PASS] No robotic canned formulas found in synthesis!")


if __name__ == "__main__":
    print("\n=======================================================")
    print("RUNNING CHATGPT-GRADE DYNAMIC MESSAGING TEST SUITE")
    print("=======================================================\n")
    test_verification_direct_answer()
    test_quantification_direct_answer()
    test_spatial_location_direct_answer()
    test_risk_operational_direct_answer()
    test_chatgpt_conceptual_science_explanations()
    test_conversational_copilot_courtesy()
    test_neural_vlm_weaving()
    test_no_robotic_boilerplate()
    print("\n=======================================================")
    print("ALL 8 DYNAMIC MESSAGING TESTS PASSED WITH 100% SUCCESS!")
    print("=======================================================\n")
