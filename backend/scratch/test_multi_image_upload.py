import asyncio
import json
import websockets

async def test_multi():
    uri = "ws://127.0.0.1:8000/api/v1/ws/query"
    async with websockets.connect(uri) as ws:
        # Test: Query with 4 images uploaded simultaneously
        payload = {
            "query": "analyze multi-temporal change and optical sar coverage across all scenes",
            "image_ids": ["urban_t1.tif", "urban_t2.tif", "fusion_optical.tif", "fusion_sar.tif"]
        }
        await ws.send(json.dumps(payload))
        
        while True:
            resp_raw = await ws.recv()
            resp = json.loads(resp_raw)
            msg_type = resp.get("type")
            if msg_type == "ack":
                print("[ACK] Multi-image count accepted:", resp.get("data", {}).get("image_count"))
            elif msg_type == "step":
                step = resp.get("data", {})
                print(f"[STEP] {step.get('step_name')}: {step.get('message', '')}")
            elif msg_type == "result":
                data = resp.get("data", {})
                print("\n=== MULTI-IMAGE FINAL RESULT ===")
                print("Task identified:", data.get("audit_trace", {}).get("task_identified"))
                print("Confidence:", data.get("audit_trace", {}).get("confidence_score"))
                print("Thumbnails returned:", len(data.get("thumbnail_urls", [])))
                break
            elif msg_type in ("error", "warning"):
                print(f"[{msg_type.upper()}]", resp.get("data"))
                if msg_type == "error":
                    break

asyncio.run(test_multi())
