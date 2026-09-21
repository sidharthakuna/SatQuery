"""
SatQuery AI — Neural Domain Knowledge Base Package
Provides on-device verified remote-sensing scientific intelligence:
- knowledge_corpus: Curated passages for ISRO missions, physics, and formulas
- neural_knowledge_net: DomainKnowledgeNet PyTorch embedding network
- knowledge_retriever: NeuralKnowledgeRetriever for sub-millisecond retrieval
"""

from app.core.knowledge.knowledge_corpus import KNOWLEDGE_PASSAGES, get_all_passages
from app.core.knowledge.neural_knowledge_net import (
    DomainKnowledgeNet,
    build_knowledge_token_vocab,
    tokenize_knowledge_text,
)
from app.core.knowledge.knowledge_retriever import (
    NeuralKnowledgeRetriever,
    get_knowledge_retriever,
)

__all__ = [
    "KNOWLEDGE_PASSAGES",
    "get_all_passages",
    "DomainKnowledgeNet",
    "build_knowledge_token_vocab",
    "tokenize_knowledge_text",
    "NeuralKnowledgeRetriever",
    "get_knowledge_retriever",
]

