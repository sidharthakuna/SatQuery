"""
SatQuery AI — Agent Intent & Task Classification Network (AgentIntentNet v2.0)
Deep sequence-aware neural intent classifier for the SatQuery AI orchestrator.
Categorizes natural language geospatial questions into specialized execution tasks.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Learned attention-based sequence aggregation.
    Assigns higher importance weights to critical intent keywords
    (e.g., 'locate', 'penetrate', 'receded', 'growth', 'classify').
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, hidden_dim)
            mask: (B, seq_len) boolean mask (True for valid tokens, False for pad)
        Returns:
            pooled: (B, hidden_dim)
        """
        scores = self.attn(x).squeeze(-1)  # (B, seq_len)
        scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, seq_len, 1)
        pooled = torch.sum(x * weights, dim=1)  # (B, hidden_dim)
        return pooled


TASK_CLASSES_V3 = [
    "SINGLE_VQA",
    "SINGLE_GROUNDING",
    "BITEMPORAL_CHANGE",
    "CROSS_MODAL_FUSION",
    "AGENT_ASSISTANT",
    "MULTI_MODEL",
]


class AgentIntentNet(nn.Module):
    """
    Advanced Neural Query Classifier for SatQuery AI (v3.0).
    Combines learned word embeddings, Bidirectional GRU sequence modeling,
    context attention pooling, calibrated output heads, and multi-label intent support.
    """
    def __init__(
        self,
        vocab_size: int = 3500,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 6,
        num_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)

        # Bidirectional GRU captures syntactic and contextual dependencies
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention pooling emphasizes task-determining phrases
        self.pool = AttentionPooling(hidden_dim)

        # Classification MLP with LayerNorm & residual connection
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, num_classes)

        # Calibrated temperature parameter (learnable or fixed)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, seq_len) token indices
        Returns:
            logits: (B, num_classes)
        """
        mask = input_ids != 0  # (B, seq_len)
        embeds = self.drop(self.embedding(input_ids))  # (B, seq_len, embed_dim)

        gru_out, _ = self.gru(embeds)  # (B, seq_len, hidden_dim)
        pooled = self.pool(gru_out, mask)  # (B, hidden_dim)

        h = F.gelu(self.norm1(self.fc1(pooled) + pooled))  # Residual block
        h = self.drop(h)
        h = F.gelu(self.norm2(self.fc2(h)))
        logits = self.out(h) / torch.clamp(self.temperature, min=0.1, max=5.0)
        return logits

    def get_probabilities(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute softmax probability distribution over task classes (single-label)."""
        logits = self.forward(input_ids)
        return F.softmax(logits, dim=-1)

    def get_multilabel_probabilities(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute independent sigmoid probability distribution over all task classes (multi-label)."""
        logits = self.forward(input_ids)
        return torch.sigmoid(logits)

