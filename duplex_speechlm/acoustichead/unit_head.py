"""Unit head: predicts a distribution over the next discrete unit ID from
the speech core's hidden state.

ARCHITECTURE CHANGE (previous version, mel_head.py, predicted continuous
mel-spectrogram frames via regression -- a single Linear, then a Conv1D +
residual-postnet stack after an earlier capacity upgrade). Per
speechcore/discrete_tokenizer.py's docstring, the training target is now a
discrete unit ID, not a continuous mel frame, so this is a standard LM
classification head: project to vocab-size logits, train with
cross-entropy. This is exactly what a transformer's output head normally
does, and is the specific mechanism that avoids the over-smoothing/blur
failure mode that plain L1/L2 mel regression is documented to have
(MELLE, arXiv:2407.08551) -- classification doesn't average over
plausible targets the way deterministic regression does.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UnitHead(nn.Module):
    def __init__(self, d_in: int = 1536, vocab_size: int = 1000) -> None:
        """d_in must match the speech core's hidden size.
        vocab_size must match the discrete tokenizer's num_clusters
        (speechcore/discrete_tokenizer.py)."""
        super().__init__()
        self.proj = nn.Linear(d_in, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (B, T, d_in) -> (B, T, vocab_size) logits."""
        return self.proj(hidden_states)
