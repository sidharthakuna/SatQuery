import urllib.request
import json
import uuid

def run_tests():
    print("--- 1. Health Endpoint ---")
    h = json.loads(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())
    print("Backend Health:", h)

    print("--- 2. Frontend Dev Server ---")
    f_html = urllib.request.urlopen('http://127.0.0.1:5173/').read().decode()
    print("Frontend Serving HTML:", "SatQuery AI" in f_html)

    print("--- 3. Upload GeoTIFF ---")
    boundary = uuid.uuid4().hex
    with open('backend/data/samples/cartosat_t1.tif', 'rb') as f:
        tif_data = f.read()

    headers = {'Content-Type': f'multipart/form-data; boundary={boundary}'}
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="cartosat_t1.tif"\r\n'
        f'Content-Type: image/tiff\r\n\r\n'
    ).encode('latin1') + tif_data + f'\r\n--{boundary}--\r\n'.encode('latin1')

    req = urllib.request.Request('http://127.0.0.1:8000/api/v1/upload', data=body, headers=headers)
    upload_res = json.loads(urllib.request.urlopen(req).read().decode())
    print("Upload Result ID:", upload_res['file_id'])
    print("Modality:", upload_res['modality'], "| CRS:", upload_res['crs'])
    print("Thumbnail URL:", upload_res['thumbnail_url'])

    print("--- 4. Execute Query (Grounding) ---")
    query_payload = {
        'query': 'Locate all buildings in this scene',
        'image_ids': [upload_res['file_id']]
    }
    q_req = urllib.request.Request(
        'http://127.0.0.1:8000/api/v1/query',
        data=json.dumps(query_payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    q_res = json.loads(urllib.request.urlopen(q_req).read().decode())
    print("Task:", q_res['audit_trace']['task_identified'])
    print("Confidence:", q_res['audit_trace']['confidence_score'])
    print("Answer:", q_res['text_response'])
    print("Bounding Boxes Count:", len(q_res['spatial_evidence']['bounding_boxes']) if q_res['spatial_evidence'] else 0)

    print("--- 5. Map Tile Server ---")
    tile_url = f'http://127.0.0.1:8000/api/v1/tiles/{upload_res["file_id"]}/1/0/0.png'
    tile_data = urllib.request.urlopen(tile_url).read()
    print("Tile Fetched Bytes:", len(tile_data))

    print("--- 6. Generate PDF Briefing Report ---")
    rep_payload = {
        'query': q_res['query'],
        'text_response': q_res['text_response'],
        'audit_trace': q_res['audit_trace'],
        'spatial_evidence': q_res['spatial_evidence']
    }
    rep_req = urllib.request.Request(
        'http://127.0.0.1:8000/api/v1/report/generate',
        data=json.dumps(rep_payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    rep_res = json.loads(urllib.request.urlopen(rep_req).read().decode())
    print("PDF Report Generated ID:", rep_res['report_id'])
    print("Download URL:", rep_res['download_url'])

    pdf_data = urllib.request.urlopen(f'http://127.0.0.1:8000{rep_res["download_url"]}').read()
    print("Downloaded PDF Size Bytes:", len(pdf_data))
    print("\n>>> ALL INTEGRATION TESTS PASSED! <<<")

if __name__ == '__main__':
    run_tests()
