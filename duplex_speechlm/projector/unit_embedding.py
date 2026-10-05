"""Unit embedding: maps discrete WavLM+k-means unit IDs to the speech
core's hidden size.

ARCHITECTURE CHANGE (previous version was Conv1D -> MLP -> RMSNorm,
projecting WavLM's continuous 768-dim features into Qwen's hidden size).
Per speechcore/discrete_tokenizer.py's docstring, the input to the speech
core is now a discrete unit ID (from the frozen WavLM+k-means pipeline),
not a continuous feature vector -- so "projecting" is now exactly what
every standard transformer LM does with its input tokens: an embedding
table lookup. No conv/MLP/norm stack is needed or appropriate here; adding
one would just be unjustified extra complexity on top of what a language
model already does natively.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UnitEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 1000, d_out: int = 1536) -> None:
        """vocab_size must match the discrete tokenizer's num_clusters
        (speechcore/discrete_tokenizer.py). d_out must match the speech
        core's hidden size."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_out)

    def forward(self, unit_ids: torch.Tensor) -> torch.Tensor:
        """unit_ids: (B, T) long tensor -> (B, T, d_out)."""
        return self.embedding(unit_ids)
