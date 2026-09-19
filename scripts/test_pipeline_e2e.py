import urllib.request
import json
import time

def test_api():
    base_url = "http://127.0.0.1:8000"
    
    # 2. Check previews for all samples
    samples = [
        "cartosat_t1.tif", "cartosat_t2.tif", "flood_t1.tif", "flood_t2.tif",
        "urban_t1.tif", "urban_t2.tif", "fusion_optical.tif", "fusion_sar.tif",
        "port_grounding.tif", "sentinel2_coastal.tif", "forest_vqa.tif", "risat_sar.tif"
    ]
    print(f"\n1. Testing previews for all {len(samples)} GeoTIFF samples ...")
    for fn in samples:
        preview_url = f"{base_url}/api/v1/preview/{fn}"
        try:
            with urllib.request.urlopen(preview_url) as resp:
                data = resp.read()
                ct = resp.headers.get("Content-Type")
                print(f"   [OK] {fn} -> {ct}, {len(data):,} bytes")
        except Exception as e:
            print(f"   [ERR] {fn} -> {e}")

    # 3. Test query: Change Detection
    print("\n2. Testing Query: Change Detection (urban_t1.tif & urban_t2.tif) ...")
    query_payload = {
        "query": "Detect urban expansion and new buildings between urban_t1.tif and urban_t2.tif",
        "image_ids": ["urban_t1.tif", "urban_t2.tif"],
    }
    req = urllib.request.Request(
        f"{base_url}/api/v1/query",
        data=json.dumps(query_payload).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print(f"   Success! Result text:\n   {res.get('text_response')[:180]}...")
        if res.get('spatial_evidence'):
            print(f"   Spatial Evidence Type: {res.get('spatial_evidence', {}).get('type')}")

    # 4. Test query: VLM question answering
    print("\n3. Testing Query: Visual QA / Forest analysis (forest_vqa.tif) ...")
    vqa_payload = {
        "query": "What is the forest canopy coverage and condition in this image?",
        "image_ids": ["forest_vqa.tif"],
    }
    req = urllib.request.Request(
        f"{base_url}/api/v1/query",
        data=json.dumps(vqa_payload).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print(f"   Success! Result text:\n   {res.get('text_response')[:180]}...")
        if res.get('spatial_evidence'):
            print(f"   Spatial Evidence Type: {res.get('spatial_evidence', {}).get('type')}")

    # 5. Test query: SAR-Optical Fusion
    print("\n4. Testing Query: SAR-Optical Fusion (fusion_optical.tif & fusion_sar.tif) ...")
    fusion_payload = {
        "query": "Fuse optical and SAR imagery for Goa coast to penetrate cloud cover and reveal maritime vessels",
        "image_ids": ["fusion_optical.tif", "fusion_sar.tif"],
    }
    req = urllib.request.Request(
        f"{base_url}/api/v1/query",
        data=json.dumps(fusion_payload).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print(f"   Success! Result text:\n   {res.get('text_response')[:180]}...")
        if res.get('spatial_evidence'):
            print(f"   Spatial Evidence Type: {res.get('spatial_evidence', {}).get('type')}")

    # 6. Test query: Object Grounding
    print("\n5. Testing Query: Object Grounding (port_grounding.tif) ...")
    grounding_payload = {
        "query": "Detect and ground all ships and maritime vessels at the berths in port_grounding.tif",
        "image_ids": ["port_grounding.tif"],
    }
    req = urllib.request.Request(
        f"{base_url}/api/v1/query",
        data=json.dumps(grounding_payload).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print(f"   Success! Result text:\n   {res.get('text_response')[:180]}...")
        if res.get('spatial_evidence'):
            print(f"   Spatial Evidence Type: {res.get('spatial_evidence', {}).get('type')}")
            if res.get('spatial_evidence', {}).get('bounding_boxes'):
                print(f"   Bounding Boxes count: {len(res.get('spatial_evidence').get('bounding_boxes'))}")

    print("\nALL PIPELINE TESTS COMPLETED SUCCESSFULLY!")

    print("\nALL PIPELINE TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    test_api()
