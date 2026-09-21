"""
SatQuery AI — Neural Domain Knowledge Network (DomainKnowledgeNet)
Our own dedicated PyTorch neural embedding network for encoding remote-sensing
scientific questions and domain knowledge passages into dense semantic vectors.
Enables fast, on-device, zero-cloud semantic retrieval with verified facts.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


KNOWLEDGE_VOCAB_SIZE = 1200
EMBED_DIM = 256


class DomainKnowledgeNet(nn.Module):
    """
    Dedicated Neural Embedding & Semantic Retrieval Network.
    Maps arbitrary remote-sensing queries and domain knowledge passages
    into a joint 256-dimensional metric space where cosine distance reflects
    scientific and semantic relevance.
    """

    def __init__(
        self,
        vocab_size: int = KNOWLEDGE_VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)

        # Bi-directional sequence encoder (captures forward and backward context)
        self.encoder = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        # Multi-head attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Semantic projection head
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def encode(self, tokens: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encodes tokenized text into a unit-normalized semantic embedding vector.
        Args:
            tokens: (batch_size, seq_len) token IDs
        Returns:
            (batch_size, embed_dim) normalized embeddings
        """
        x = self.embedding(tokens)  # (b, seq_len, 128)
        outputs, _ = self.encoder(x)  # (b, seq_len, hidden_dim)

        # Compute attention weights over sequence tokens
        attn_weights = self.attention(outputs)  # (b, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted pool over tokens
        pooled = torch.sum(outputs * attn_weights, dim=1)  # (b, hidden_dim)

        # Project and normalize to unit hypersphere
        emb = self.proj(pooled)  # (b, embed_dim)
        return F.normalize(emb, p=2, dim=-1)

    def forward(self, query_tokens: torch.Tensor, passage_tokens: torch.Tensor) -> torch.Tensor:
        """
        Computes cosine similarity between queries and candidate passages.
        Returns: (batch_size,) similarity scores in [-1.0, 1.0]
        """
        q_emb = self.encode(query_tokens)
        p_emb = self.encode(passage_tokens)
        return torch.sum(q_emb * p_emb, dim=-1)


def build_knowledge_token_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Builds vocabulary dictionary from remote-sensing and knowledge corpus tokens.
    """
    from app.core.knowledge.knowledge_corpus import KNOWLEDGE_PASSAGES

    special = ["<pad>", "<unk>", "<bos>", "<eos>"]
    words = set()

    for p in KNOWLEDGE_PASSAGES:
        text = f"{p['title']} {' '.join(p['keywords'])} {p['content']}"
        for w in text.split():
            clean = w.strip("?.,!;:\"'()/-").lower()
            if clean and len(clean) > 1:
                words.add(clean)

    sorted_words = special + sorted(list(words))
    vocab = {w: i for i, w in enumerate(sorted_words)}
    id2word = {i: w for i, w in enumerate(sorted_words)}
    return vocab, id2word


def tokenize_knowledge_text(text: str, vocab: Dict[str, int], max_len: int = 64) -> List[int]:
    """Tokenizes arbitrary text for DomainKnowledgeNet."""
    unk_id = vocab.get("<unk>", 1)
    pad_id = vocab.get("<pad>", 0)

    words = [w.strip("?.,!;:\"'()/-").lower() for w in text.split() if w]
    ids = [vocab.get(w, unk_id) for w in words][:max_len]
    ids += [pad_id] * (max_len - len(ids))
    return ids
