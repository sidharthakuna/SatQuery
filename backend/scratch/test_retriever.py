import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.knowledge import get_knowledge_retriever

kr = get_knowledge_retriever()
test_queries = [
    "What is the spatial resolution of Cartosat-3 and DEM?",
    "How does RISAT EOS-04 penetrate clouds using C-band SAR?",
    "What is the formula for NDVI and how does near-infrared reflect?",
    "Tell me about NDMA flood disaster protocols and evacuation camps",
    "What sensors does Resourcesat-2A carry for crop acreage mapping?",
]

for q in test_queries:
    print("=" * 60)
    print("QUERY:", q)
    results = kr.retrieve(q, top_k=2)
    for r in results:
        print(f"  -> [{r['score']:.3f}] {r['title']} ({r['id']})")
