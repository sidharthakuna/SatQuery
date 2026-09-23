import asyncio
import json
import websockets

async def test_ws():
    uri = 'ws://127.0.0.1:8000/api/v1/ws/query'
    async with websockets.connect(uri) as ws:
        payload = {
            'query': 'remove clouds',
            'image_ids': ['fusion_optical.tif', 'fusion_sar.tif']
        }
        await ws.send(json.dumps(payload))
        
        step_count = 0
        got_result = False
        error_msg = None
        
        while True:
            try:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                msg = json.loads(msg_raw)
                mtype = msg.get('type')
                if mtype == 'ack':
                    print('WS ACK received')
                elif mtype == 'step':
                    step_count += 1
                    sname = msg.get('data', {}).get('step_name')
                    print(f'WS STEP {step_count}: {sname}')
                elif mtype == 'result':
                    got_result = True
                    rdata = msg.get('data', {})
                    print('WS RESULT RECEIVED SUCCESSFULLY!')
                    print('TASK:', rdata.get('audit_trace', {}).get('task_identified'))
                    print('MASK URL:', rdata.get('spatial_evidence', {}).get('mask_url'))
                    print('TEXT LEN:', len(rdata.get('text_response', '')))
                    break
                elif mtype == 'error':
                    error_msg = msg.get('data', {}).get('message')
                    print('WS RETURNED ERROR:', error_msg)
                    break
            except asyncio.TimeoutError:
                print('WS TIMEOUT waiting for message')
                break

    assert got_result, f'Did not get result! Error: {error_msg}'
    print('ALL WS TESTS PASSED!')

if __name__ == '__main__':
    asyncio.run(test_ws())
