import asyncio
import json
import websockets

async def test():
    uri = "ws://127.0.0.1:8000/api/v1/ws/query"
    async with websockets.connect(uri) as ws:
        # Test 1: Query with 2 urban images asking to remove clouds and explain urban expansion
        payload = {
            "query": "remove the clouds and explain the urban expansion",
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
                print("\n--- CLUSTERS ---")
                clusters = data.get("spatial_evidence", {}).get("clusters", [])
                for c in clusters:
                    print(f"  {c.get('zone')}: {c.get('category')} ({c.get('area_ha')} ha)")
                break
            elif msg_type in ("error", "warning"):
                print(f"[{msg_type.upper()}]", resp.get("data"))
                if msg_type == "error":
                    break

asyncio.run(test())
