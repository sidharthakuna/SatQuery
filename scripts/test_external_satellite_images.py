"""
SatQuery AI — End-to-End Test Script with Public Satellite Images
Downloads small public satellite GeoTIFFs and tests all 4 specialist models
through the live API.

Usage:
    1. Start the backend:  cd backend && python -m uvicorn app.main:app --reload --port 8000
    2. Run this script:    python scripts/test_external_satellite_images.py

Requires: requests (pip install requests)
"""

import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package not found. Install with: pip install requests")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
BASE_URL = os.environ.get("SATQUERY_URL", "http://localhost:8000")
API = f"{BASE_URL}/api/v1"
DOWNLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "samples" / "test_external"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Small public satellite images (all < 5 MB)
TEST_IMAGES = {
    "sentinel2_rgb": {
        "url": "https://github.com/mommermi/geotiff_sample/raw/master/sample.tif",
        "filename": "sentinel2_rgb_sample.tif",
        "description": "Sentinel-2 RGB 3-band optical GeoTIFF (~1 MB)",
        "modality": "OPTICAL",
    },
    "landsat_rgb": {
        "url": "https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/RGB.byte.tif",
        "filename": "landsat_rgb_sample.tif",
        "description": "Landsat RGB 3-band optical GeoTIFF (~500 KB)",
        "modality": "OPTICAL",
    },
    "seaice_geotiff": {
        "url": "https://github.com/GeoTIFF/test-data/raw/main/files/nt_20201024_f18_nrt_s.tif",
        "filename": "seaice_sample.tif",
        "description": "Antarctic sea ice concentration GeoTIFF (~4 MB)",
        "modality": "OPTICAL",
    },
}

# ─────────────────────────────────────────────────────────────
# Pretty Print Helpers
# ─────────────────────────────────────────────────────────────
def header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")

def subheader(text):
    print(f"\n  --- {text} ---")

def success(text):
    print(f"  [PASS] {text}")

def fail(text):
    print(f"  [FAIL] {text}")

def info(text):
    print(f"  [INFO] {text}")


# ─────────────────────────────────────────────────────────────
# Step 1: Download Test Images
# ─────────────────────────────────────────────────────────────
def download_test_images():
    header("STEP 1: Downloading Public Satellite Test Images")
    
    for key, img in TEST_IMAGES.items():
        dest = DOWNLOAD_DIR / img["filename"]
        if dest.exists() and dest.stat().st_size > 1000:
            info(f"Already exists: {img['filename']} ({dest.stat().st_size:,} bytes)")
            continue
        
        info(f"Downloading {img['description']}...")
        try:
            resp = requests.get(img["url"], timeout=60, allow_redirects=True)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            success(f"Saved {img['filename']} ({len(resp.content):,} bytes)")
        except Exception as e:
            fail(f"Download failed for {key}: {e}")
            return False
    
    return True


# ─────────────────────────────────────────────────────────────
# Step 2: Check Backend Health
# ─────────────────────────────────────────────────────────────
def check_backend():
    header("STEP 2: Checking Backend Health")
    
    try:
        resp = requests.get(f"{BASE_URL}/docs", timeout=5)
        if resp.status_code == 200:
            success(f"Backend is running at {BASE_URL}")
            return True
    except requests.ConnectionError:
        pass
    
    fail(f"Backend not reachable at {BASE_URL}")
    print(f"\n  Start it with:")
    print(f"    cd backend && python -m uvicorn app.main:app --reload --port 8000\n")
    return False


# ─────────────────────────────────────────────────────────────
# Step 3: Upload Images
# ─────────────────────────────────────────────────────────────
def upload_images():
    header("STEP 3: Uploading Test Images to SatQuery AI")
    
    uploaded = {}
    
    for key, img in TEST_IMAGES.items():
        filepath = DOWNLOAD_DIR / img["filename"]
        if not filepath.exists():
            fail(f"File not found: {filepath}")
            continue
        
        subheader(f"Uploading: {img['filename']}")
        
        try:
            with open(filepath, "rb") as f:
                resp = requests.post(
                    f"{API}/upload",
                    files={"file": (img["filename"], f, "image/tiff")},
                    timeout=30,
                )
            
            if resp.status_code == 200:
                data = resp.json()
                file_id = data["file_id"]
                uploaded[key] = file_id
                
                success(f"Uploaded successfully!")
                info(f"  File ID:    {file_id}")
                info(f"  Modality:   {data.get('modality', 'N/A')}")
                info(f"  Dimensions: {data.get('width', '?')} x {data.get('height', '?')}")
                info(f"  Bands:      {data.get('band_count', '?')}")
                info(f"  CRS:        {data.get('crs', 'N/A')}")
                
                bounds = data.get("bounds_latlon")
                if bounds:
                    info(f"  Bounds:     ({bounds.get('min_lat', '?'):.4f}, {bounds.get('min_lon', '?'):.4f}) -> ({bounds.get('max_lat', '?'):.4f}, {bounds.get('max_lon', '?'):.4f})")
                
                info(f"  Thumbnail:  {BASE_URL}{data.get('thumbnail_url', '')}")
            else:
                fail(f"Upload failed (HTTP {resp.status_code}): {resp.text[:200]}")
                
        except Exception as e:
            fail(f"Upload error: {e}")
    
    return uploaded


# ─────────────────────────────────────────────────────────────
# Step 4: Run Test Queries Against Each Model
# ─────────────────────────────────────────────────────────────
def run_query(query, image_ids, test_name, timeout=60):
    """Execute a query and print results."""
    subheader(f"TEST: {test_name}")
    info(f"Query: \"{query}\"")
    info(f"Image IDs: {image_ids}")
    
    payload = {
        "query": query,
        "image_ids": image_ids,
        "history": [],
    }
    
    start = time.time()
    try:
        resp = requests.post(
            f"{API}/query",
            json=payload,
            timeout=timeout,
        )
        elapsed = time.time() - start
        
        if resp.status_code == 200:
            result = resp.json()
            success(f"Query completed in {elapsed:.2f}s")
            
            # Print text response
            text_resp = result.get("text_response", "")
            info(f"Response ({len(text_resp)} chars):")
            for line in text_resp[:500].split("\n"):
                print(f"    | {line}")
            if len(text_resp) > 500:
                print(f"    | ... (truncated, {len(text_resp)} total chars)")
            
            # Print confidence
            conf = result.get("confidence_decomposition")
            if conf:
                info(f"Confidence: {conf}")
            
            # Print audit trace summary
            trace = result.get("audit_trace", {})
            steps = trace.get("steps", [])
            if steps:
                info(f"Execution Trace ({len(steps)} steps):")
                for step in steps:
                    step_name = step.get("step_name", "?")
                    status = step.get("status", "?")
                    duration = step.get("duration_ms", 0)
                    print(f"    [{status}] {step_name} ({duration}ms)")
            
            # Print spatial evidence
            spatial = result.get("spatial_evidence")
            if spatial:
                info(f"Spatial Evidence Type: {spatial.get('type', 'N/A')}")
                if spatial.get("bounding_boxes"):
                    info(f"  Bounding Boxes: {len(spatial['bounding_boxes'])} detected")
                if spatial.get("mask_url"):
                    info(f"  Mask URL: {BASE_URL}{spatial['mask_url']}")
                if spatial.get("changed_area_percent"):
                    info(f"  Changed Area: {spatial['changed_area_percent']:.1f}%")
                if spatial.get("vqa_grounding"):
                    vqa = spatial["vqa_grounding"]
                    info(f"  VQA Grounding:")
                    info(f"    Method:     {vqa.get('method', 'N/A')}")
                    info(f"    Confidence: {vqa.get('confidence_pct', '?')}%")
                    info(f"    Metrics:    {vqa.get('metrics', {})}")
            
            # Print suggested actions
            actions = result.get("suggested_actions", [])
            if actions:
                info(f"Suggested Follow-ups:")
                for a in actions[:3]:
                    print(f"    -> {a}")
            
            return result
        else:
            fail(f"Query failed (HTTP {resp.status_code}) in {elapsed:.2f}s")
            try:
                error_detail = resp.json().get("detail", resp.text[:300])
            except Exception:
                error_detail = resp.text[:300]
            info(f"Error: {error_detail}")
            return None
            
    except requests.Timeout:
        fail(f"Query timed out after {timeout}s")
        return None
    except Exception as e:
        fail(f"Query error: {e}")
        return None


def test_all_models(uploaded_ids):
    header("STEP 4: Testing All Specialist Models")
    
    results = {}
    
    # ── Model 1: VQA & Scene Captioning (RS-VLM) ──────────
    if "sentinel2_rgb" in uploaded_ids:
        fid = uploaded_ids["sentinel2_rgb"]
        
        results["vqa_scene_description"] = run_query(
            query="Describe the land cover types visible in this satellite image. What terrain features can you identify?",
            image_ids=[fid],
            test_name="VQA - Scene Captioning (Sentinel-2 RGB)",
        )
        
        results["vqa_water_detection"] = run_query(
            query="Is there any water body visible in this image? If yes, estimate the approximate percentage of the image covered by water.",
            image_ids=[fid],
            test_name="VQA - Water Body Detection (Sentinel-2 RGB)",
        )
    
    if "landsat_rgb" in uploaded_ids:
        fid = uploaded_ids["landsat_rgb"]
        
        results["vqa_landsat_scene"] = run_query(
            query="What type of landscape is shown in this Landsat satellite image? Identify urban, agricultural, and natural features.",
            image_ids=[fid],
            test_name="VQA - Landsat Scene Analysis",
        )
    
    # ── Model 2: Text-Guided Visual Grounding ─────────────
    if "sentinel2_rgb" in uploaded_ids:
        fid = uploaded_ids["sentinel2_rgb"]
        
        results["grounding_vegetation"] = run_query(
            query="Find and highlight all vegetation patches in this image. Draw bounding boxes around each distinct vegetated area.",
            image_ids=[fid],
            test_name="Visual Grounding - Vegetation Detection",
        )
    
    if "landsat_rgb" in uploaded_ids:
        fid = uploaded_ids["landsat_rgb"]
        
        results["grounding_buildings"] = run_query(
            query="Locate all buildings and urban structures in this satellite image. Show their bounding boxes.",
            image_ids=[fid],
            test_name="Visual Grounding - Building Detection (Landsat)",
        )
    
    if "seaice_geotiff" in uploaded_ids:
        fid = uploaded_ids["seaice_geotiff"]
        
        results["grounding_ice"] = run_query(
            query="Identify and delineate the boundary between sea ice and open ocean in this Antarctic satellite image.",
            image_ids=[fid],
            test_name="Visual Grounding - Sea Ice Boundary (Antarctic)",
        )
    
    # ── Model 3: Bi-Temporal Change Detection ─────────────
    if "sentinel2_rgb" in uploaded_ids and "landsat_rgb" in uploaded_ids:
        fid1 = uploaded_ids["sentinel2_rgb"]
        fid2 = uploaded_ids["landsat_rgb"]
        
        results["change_detection"] = run_query(
            query="Compare these two satellite images and describe any changes between them. What areas have changed and what is the nature of the change?",
            image_ids=[fid1, fid2],
            test_name="Change Detection - Bi-Temporal Comparison",
        )
    
    # ── Model 4: Cross-Modal Fusion ───────────────────────
    if "sentinel2_rgb" in uploaded_ids:
        fid = uploaded_ids["sentinel2_rgb"]
        
        results["cross_modal"] = run_query(
            query="Analyze this optical satellite image and explain what additional information a SAR radar image of the same area would reveal.",
            image_ids=[fid],
            test_name="Cross-Modal Analysis - Optical vs SAR",
        )
    
    return results


# ─────────────────────────────────────────────────────────────
# Step 5: Test Multimodal Flood Endpoint
# ─────────────────────────────────────────────────────────────
def test_multimodal_flood(uploaded_ids):
    header("STEP 5: Testing Multimodal Flood Endpoint")
    
    subheader("TEST: Cloud-Penetrating Flood Assessment (default samples)")
    info("Using built-in Sentinel-1 SAR + cloudy Sentinel-2 optical samples")
    
    payload = {
        "query": "Fuse SAR and optical imagery to penetrate storm clouds, calculate total flooded area, and identify safe evacuation zones.",
    }
    
    start = time.time()
    try:
        resp = requests.post(
            f"{API}/multimodal-flood/analyze",
            json=payload,
            timeout=90,
        )
        elapsed = time.time() - start
        
        if resp.status_code == 200:
            result = resp.json()
            success(f"Flood analysis completed in {elapsed:.2f}s")
            
            text_resp = result.get("text_response", "")
            info(f"Response ({len(text_resp)} chars):")
            for line in text_resp[:500].split("\n"):
                print(f"    | {line}")
            
            spatial = result.get("spatial_evidence")
            if spatial:
                info(f"Spatial Evidence Type: {spatial.get('type', 'N/A')}")
                if spatial.get("changed_area_percent"):
                    info(f"  Flooded Area: {spatial['changed_area_percent']:.1f}%")
                if spatial.get("changed_area_hectares"):
                    info(f"  Flooded Area: {spatial['changed_area_hectares']:.1f} hectares")
                    
            return result
        else:
            fail(f"Flood analysis failed (HTTP {resp.status_code}) in {elapsed:.2f}s")
            info(f"Error: {resp.text[:300]}")
            return None
            
    except Exception as e:
        fail(f"Flood analysis error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Step 6: Test with Existing Project Samples
# ─────────────────────────────────────────────────────────────
def test_existing_samples():
    header("STEP 6: Testing Queries with Existing Project Samples")
    
    existing_samples = {
        "flood_pair": {
            "ids": ["flood_t1", "flood_t2"],
            "query": "What flood-related changes occurred between these two time periods? Estimate the area affected.",
            "test_name": "Change Detection - Existing Flood Pair (flood_t1 -> flood_t2)",
        },
        "urban_pair": {
            "ids": ["urban_t1", "urban_t2"],
            "query": "Compare these urban satellite images. Has there been any construction or demolition?",
            "test_name": "Change Detection - Existing Urban Pair (urban_t1 -> urban_t2)",
        },
        "forest_vqa": {
            "ids": ["forest_vqa"],
            "query": "Analyze this forest canopy image. What is the approximate forest density? Any signs of deforestation?",
            "test_name": "VQA - Existing Forest Canopy Analysis",
        },
        "coastal_vqa": {
            "ids": ["sentinel2_coastal"],
            "query": "Describe the coastal features in this Sentinel-2 image. Identify shoreline and water depth zones.",
            "test_name": "VQA - Existing Coastal Analysis",
        },
        "fusion_pair": {
            "ids": ["fusion_optical", "fusion_sar"],
            "query": "Fuse these optical and SAR images. What does the SAR reveal that the optical cannot?",
            "test_name": "Cross-Modal Fusion - Existing Optical + SAR Pair",
        },
    }
    
    results = {}
    for key, test in existing_samples.items():
        results[key] = run_query(
            query=test["query"],
            image_ids=test["ids"],
            test_name=test["test_name"],
        )
    
    return results


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
def print_summary(all_results):
    header("TEST SUMMARY")
    
    passed = 0
    failed = 0
    
    for name, result in all_results.items():
        if result is not None:
            success(f"{name}")
            passed += 1
        else:
            fail(f"{name}")
            failed += 1
    
    print(f"\n  {'='*50}")
    print(f"  Total: {passed + failed}  |  Passed: {passed}  |  Failed: {failed}")
    print(f"  {'='*50}")
    
    if failed == 0:
        print(f"\n  All tests passed! SatQuery AI is working correctly.")
    else:
        print(f"\n  {failed} test(s) failed. Check the output above for details.")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("""
    ================================================================
      SatQuery AI - External Satellite Image Tester
      Downloads public satellite GeoTIFFs and tests all 4
      specialist models through the live API.
    ================================================================
    """)
    
    # Step 1: Download images
    if not download_test_images():
        print("\nFailed to download test images. Exiting.")
        sys.exit(1)
    
    # Step 2: Check backend
    if not check_backend():
        sys.exit(1)
    
    # Step 3: Upload images
    uploaded_ids = upload_images()
    if not uploaded_ids:
        print("\nNo images uploaded successfully. Exiting.")
        sys.exit(1)
    
    all_results = {}
    
    # Step 4: Test all models with new external images
    new_results = test_all_models(uploaded_ids)
    all_results.update(new_results)
    
    # Step 5: Test multimodal flood endpoint
    flood_result = test_multimodal_flood(uploaded_ids)
    all_results["multimodal_flood"] = flood_result
    
    # Step 6: Test with existing project samples
    existing_results = test_existing_samples()
    all_results.update(existing_results)
    
    # Summary
    print_summary(all_results)
    
    # Save full results to JSON
    results_path = DOWNLOAD_DIR / "test_results.json"
    try:
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        info(f"\nFull results saved to: {results_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
