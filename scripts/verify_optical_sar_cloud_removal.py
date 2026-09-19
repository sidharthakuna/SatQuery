import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path
import numpy as np
from PIL import Image

def run_test():
    base_url = "http://127.0.0.1:8000"
    
    # 1. Health check
    req = urllib.request.urlopen(f"{base_url}/health")
    health = json.loads(req.read().decode())
    print("[1] Backend Health:", health.get("status"))

    # 2. Upload sample fusion files
    optical_path = Path("backend/data/samples/fusion_optical.tif")
    sar_path = Path("backend/data/samples/fusion_sar.tif")
    
    assert optical_path.exists(), f"Missing {optical_path}"
    assert sar_path.exists(), f"Missing {sar_path}"

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    def upload_file(p: Path, sensor: str):
        with open(p, "rb") as f:
            content = f.read()
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{p.name}"\r\n'
            f"Content-Type: image/tiff\r\n\r\n"
        ).encode("utf-8") + content + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="sensor_type"\r\n\r\n'
            f"{sensor}\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        
        req = urllib.request.Request(
            f"{base_url}/api/v1/upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())

    up_opt = upload_file(optical_path, "optical")
    print(f"[2] Uploaded Optical: id={up_opt['file_id']}, name={up_opt['filename']}")
    
    up_sar = upload_file(sar_path, "sar")
    print(f"[3] Uploaded SAR: id={up_sar['file_id']}, name={up_sar['filename']}")

    # 3. Post query
    query_payload = {
        "query": "Fuse the cloudy optical image with the SAR radar pass to remove clouds and generate a clear optical satellite image.",
        "image_ids": [up_opt["file_id"], up_sar["file_id"]]
    }
    
    q_req = urllib.request.Request(
        f"{base_url}/api/v1/query",
        data=json.dumps(query_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(q_req) as resp:
        query_res = json.loads(resp.read().decode())

    print("\n[4] Query Response:")
    print(" - Task Type:", query_res.get("task_type"))
    print(" - Confidence:", query_res.get("confidence"))
    
    spatial = query_res.get("spatial_evidence", {})
    mask_url = spatial.get("mask_url")
    print(" - Spatial Evidence Mask URL:", mask_url)
    print(" - Spatial Metrics:", spatial.get("metrics"))

    assert mask_url is not None, "Missing mask_url!"
    assert "card_fused" in mask_url, f"Expected card_fused in mask_url, got {mask_url}"

    # 4. Check generated file on disk
    mask_filename = Path(mask_url).name
    mask_disk_path = Path("backend/data/uploads") / mask_filename
    assert mask_disk_path.exists(), f"Expected {mask_disk_path} to exist!"
    
    img = Image.open(mask_disk_path).convert("RGB")
    arr = np.array(img)
    print(f"\n[5] Verified Reconstructed Image: {mask_disk_path.name}, Dimensions={arr.shape}")
    
    # Check that it's NOT a flood map with flat blue #1D4ED8 (29, 78, 216)
    blue_matches = (np.abs(arr[:, :, 0] - 29) < 5) & (np.abs(arr[:, :, 1] - 78) < 5) & (np.abs(arr[:, :, 2] - 216) < 5)
    pct_flood_blue = np.mean(blue_matches) * 100
    print(f" - Hardcoded flood blue pixel percentage: {pct_flood_blue:.2f}% (Expected: ~0%)")
    assert pct_flood_blue < 1.0, f"Too much flood blue found: {pct_flood_blue}%"

    # Check color variability in reconstructed land/water
    std_r = float(np.std(arr[:, :, 0]))
    std_g = float(np.std(arr[:, :, 1]))
    std_b = float(np.std(arr[:, :, 2]))
    print(f" - Color channel standard deviations: R={std_r:.1f}, G={std_g:.1f}, B={std_b:.1f}")
    assert std_r > 15 and std_g > 15 and std_b > 15, "Low image contrast!"

    # 5. Check audit trace / explanation text
    text_resp = query_res.get("text_response", "")
    print(f"\n[6] Text Response Excerpt:\n{text_resp[:350]}...")
    assert "cloud" in text_resp.lower(), "Expected cloud discussion in text response"
    assert "sar" in text_resp.lower(), "Expected SAR discussion in text response"

    print("\n[SUCCESS] ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_test()
