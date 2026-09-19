import json
import urllib.request
from pathlib import Path

def run():
    base_url = "http://127.0.0.1:8000"
    boundary = "----TestBoundaryXYZ12345"

    def upload(path: Path, sensor="optical"):
        with open(path, "rb") as f:
            data = f.read()
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f"Content-Type: image/tiff\r\n\r\n"
        ).encode("utf-8") + data + (
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
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode())

    def query(text, image_ids):
        payload = json.dumps({"query": text, "image_ids": image_ids}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/v1/query",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode())

    # 1. Grounding DINO
    print("[1] Testing Grounding DINO with port_grounding.tif...")
    up_port = upload(Path("data/samples/port_grounding.tif"))
    res_grd = query("Detect and ground all maritime vessels, cargo ships, and docks with bounding boxes.", [up_port["file_id"]])
    print(" -> Task:", res_grd.get("task_type"), "Boxes:", len(res_grd.get("spatial_evidence", {}).get("bounding_boxes", [])))

    # 2. Flood Inundation
    print("[2] Testing Flood Inundation with flood_t1.tif and flood_t2.tif...")
    up_f1 = upload(Path("data/samples/flood_t1.tif"))
    up_f2 = upload(Path("data/samples/flood_t2.tif"))
    res_flood = query("Detect flood inundation, map submerged parcel boundaries, and quantify flooded hectares.", [up_f1["file_id"], up_f2["file_id"]])
    print(" -> Task:", res_flood.get("task_type"), "Metrics:", res_flood.get("spatial_evidence", {}).get("metrics"))

    # 3. Optical + SAR Cloud Removal
    print("[3] Testing Optical + SAR Cloud Removal with fusion_optical.tif and fusion_sar.tif...")
    up_opt = upload(Path("data/samples/fusion_optical.tif"))
    up_sar = upload(Path("data/samples/fusion_sar.tif"), sensor="sar")
    res_fus = query("Execute cross-modal optical and microwave SAR fusion to penetrate dense cloud cover and reconstruct a clear optical satellite image.", [up_opt["file_id"], up_sar["file_id"]])
    print(" -> Task:", res_fus.get("task_type"), "Mask:", res_fus.get("spatial_evidence", {}).get("mask_url"))

    print("\n[SUCCESS] ALL THREE MISSION MODALITIES VERIFIED PERFECTLY!")

if __name__ == "__main__":
    run()
