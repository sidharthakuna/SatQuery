import asyncio
import json
import websockets

async def test_cloud_and_spot():
    uri = "ws://127.0.0.1:8000/api/v1/ws/query"
    async with websockets.connect(uri) as ws:
        payload = {
            "query": "remove the clouds and spot the buildings",
            "image_ids": ["urban_t1.tif", "urban_t2.tif"]
        }
        await ws.send(json.dumps(payload))
        
        while True:
            resp_raw = await ws.recv()
            resp = json.loads(resp_raw)
            msg_type = resp.get("type")
            if msg_type == "ack":
                print("[ACK]", resp.get("data"))
            elif msg_type == "step":
                step = resp.get("data", {})
                print(f"[STEP] {step.get('step_name')}: {step.get('message', '')}")
            elif msg_type == "result":
                data = resp.get("data", {})
                print("\n=== FINAL RESULT ===")
                print("Task identified:", data.get("audit_trace", {}).get("task_identified"))
                print("Confidence:", data.get("audit_trace", {}).get("confidence_score"))
                print("\n--- TEXT RESPONSE ---")
                print(data.get("text_response"))
                extra = data.get("spatial_evidence", {}).get("extra", {})
                print("\n--- CARDS IN EXTRA ---")
                print("has optical_sar_card:", bool(extra.get("optical_sar_card")))
                print("has grounding_card:", bool(extra.get("grounding_card")))
                print("has bitemporal_card:", bool(extra.get("bitemporal_card")))
                print("has multi_model_card:", bool(extra.get("multi_model_card")))
                if extra.get("optical_sar_card"):
                    print("  optical_sar_card inputs:", extra["optical_sar_card"].get("inputs", []))
                    print("  optical_sar_card fused_url:", extra["optical_sar_card"].get("fused_url"))
                if extra.get("grounding_card"):
                    print("  grounding_card count:", extra["grounding_card"].get("count"))
                    print("  grounding_card boxes:", len(extra["grounding_card"].get("boxes", [])))
                break
            elif msg_type in ("error", "warning"):
                print(f"[{msg_type.upper()}]", resp.get("data"))
                if msg_type == "error":
                    break

asyncio.run(test_cloud_and_spot())
