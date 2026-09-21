"""
SatQuery AI — Neural Knowledge Retriever Engine
Retrieves authenticated Earth observation and remote-sensing science facts
using dense vector semantic similarity from our trained DomainKnowledgeNet.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from app.core.knowledge.knowledge_corpus import KNOWLEDGE_PASSAGES
from app.core.knowledge.neural_knowledge_net import (
    DomainKnowledgeNet,
    build_knowledge_token_vocab,
    tokenize_knowledge_text,
)
from config.settings import settings

logger = logging.getLogger(__name__)


class NeuralKnowledgeRetriever:
    """
    Sub-millisecond dense vector semantic retrieval over verified
    ISRO, sensor physics, and remote-sensing knowledge passages.
    """

    _instance: Optional["NeuralKnowledgeRetriever"] = None

    def __init__(self):
        self.device = torch.device("cpu")
        self.ckpt_path = settings.checkpoints_dir / "knowledge_net.pt"
        self.model: Optional[DomainKnowledgeNet] = None
        self.vocab: Dict[str, int] = {}
        self.passage_embs: Dict[str, np.ndarray] = {}
        self.passages: Dict[str, Dict[str, Any]] = {p["id"]: p for p in KNOWLEDGE_PASSAGES}
        self._load_network()

    @classmethod
    def get_instance(cls) -> "NeuralKnowledgeRetriever":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_network(self):
        """Loads trained weights and pre-computed passage vectors."""
        if self.ckpt_path.exists():
            try:
                ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
                self.vocab = ckpt.get("vocab", {})
                self.passage_embs = ckpt.get("passage_embeddings", {})
                self.model = DomainKnowledgeNet(vocab_size=len(self.vocab), embed_dim=256).to(self.device)
                self.model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                self.model.eval()
                logger.info(f"DomainKnowledgeNet loaded successfully from {self.ckpt_path.name}")
                return
            except Exception as e:
                logger.warning(f"DomainKnowledgeNet load failed, using keyword fallback: {e}")

        # Fallback vocab
        self.vocab, _ = build_knowledge_token_vocab()

    def retrieve(self, query: str, top_k: int = 2, threshold: float = 0.35) -> List[Dict[str, Any]]:
        """
        Retrieves top-k verified knowledge passages matching natural language query
        using Hybrid Dense Neural Semantic Retrieval + Domain Lexical Weighting.
        Returns:
            List of dicts: [{"id": ..., "title": ..., "content": ..., "score": ...}]
        """
        q_lower = query.lower().strip()

        # 1. Domain Lexical / Keyword Scoring
        kw_scores: Dict[str, float] = {}
        for pid, passage in self.passages.items():
            kw_score = 0.0
            for kw in passage.get("keywords", []):
                if kw in q_lower:
                    kw_score += 0.45
            title_words = [w.lower() for w in passage["title"].split() if len(w) > 3]
            for tw in title_words:
                if tw in q_lower:
                    kw_score += 0.25
            kw_scores[pid] = min(kw_score, 1.0)

        # 2. Neural Semantic Dense Retrieval
        neural_scores: Dict[str, float] = {}
        if self.model is not None and self.passage_embs:
            try:
                tokens = torch.tensor([tokenize_knowledge_text(query, self.vocab, max_len=48)], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    q_vec = self.model.encode(tokens).cpu().numpy()[0]

                for pid, p_vec in self.passage_embs.items():
                    sim = float(np.dot(q_vec, p_vec))
                    neural_scores[pid] = max(sim, 0.0)
            except Exception as e:
                logger.debug(f"Neural retrieval error: {e}")

        # 3. Hybrid Combined Scoring
        scored: List[Tuple[float, str]] = []
        for pid in self.passages.keys():
            n_score = neural_scores.get(pid, 0.0)
            k_score = kw_scores.get(pid, 0.0)
            if neural_scores:
                # If strong lexical keyword match, amplify; otherwise rely on neural semantics
                if k_score > 0.3:
                    combined = 0.45 * n_score + 0.55 * k_score
                else:
                    combined = 0.70 * n_score + 0.30 * k_score
            else:
                combined = k_score

            scored.append((combined, pid))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, pid in scored[:top_k]:
            if score >= threshold:
                passage = self.passages.get(pid, {})
                results.append({
                    "id": pid,
                    "title": passage.get("title", ""),
                    "content": passage.get("content", ""),
                    "score": round(score, 3),
                    "category": passage.get("category", ""),
                })
        return results


def get_knowledge_retriever() -> NeuralKnowledgeRetriever:
    return NeuralKnowledgeRetriever.get_instance()
