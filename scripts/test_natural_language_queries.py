"""
Test natural language query understanding on uploaded satellite images through the live backend API.
Verifies that natural English questions are understood as satellite image analysis queries (VQA, Grounding)
rather than generic educational definitions.
"""

import urllib.request
import json

def test_natural_queries():
    base_url = "http://127.0.0.1:8000/api/v1/query"

    queries_to_test = [
        {
            "query": "What is the predominant land cover in this scene?",
            "file_ids": ["cartosat_t1"],
            "expected_task": "SINGLE_VQA",
        },
        {
            "query": "Describe the urban density and visible structures in this optical image",
            "file_ids": ["cartosat_t1"],
            "expected_task": "SINGLE_VQA",
        },
        {
            "query": "Where are the buildings and runway? Locate them.",
            "file_ids": ["cartosat_t1"],
            "expected_task": "SINGLE_GROUNDING",
        },
        {
            "query": "remove clouds and tell the flooded area if we are giving the flooded area's sar and optical image view where clouds obstruct the optical image but we need flooded areas to detect",
            "file_ids": ["cartosat_t1", "risat_sar"],
            "expected_task": "MULTI_MODEL",
        },
    ]

    print("=" * 70)
    print("  Testing Natural Language Queries on Live SatQuery Backend")
    print("=" * 70)

    for i, item in enumerate(queries_to_test, 1):
        payload = {
            "query": item["query"],
            "image_ids": item["file_ids"],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(base_url, data=data, headers={"Content-Type": "application/json"})
        
        with urllib.request.urlopen(req) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            
        trace = res_json.get("audit_trace", {})
        task = trace.get("task_identified")
        conf = trace.get("confidence_score", 0.0)
        tools = trace.get("selected_tools", [])
        text_snippet = res_json.get("text_response", "")[:120].replace("\n", " ")
        
        status = "PASS" if task == item["expected_task"] else "FAIL"
        print(f"\n[Test #{i}] Status: {status}")
        print(f"  User Query     : \"{item['query'][:65]}...\"")
        print(f"  Images Given   : {item['file_ids']}")
        print(f"  Identified Task: {task} (Expected: {item['expected_task']}) | Conf: {conf:.2f}")
        print(f"  Tools Run      : {tools}")
        print(f"  Response Lead  : {text_snippet}...")
        
        assert task == item["expected_task"], f"Expected {item['expected_task']}, got {task}"

    print("\n" + "=" * 70)
    print("  [ALL TESTS PASSED] Natural English queries correctly analyze satellite images!")
    print("=" * 70)

if __name__ == "__main__":
    test_natural_queries()
