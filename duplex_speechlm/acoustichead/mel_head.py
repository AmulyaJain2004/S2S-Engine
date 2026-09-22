"""Acoustic head: placeholder only. Predicts mel-spectrogram frames from the
speech core's hidden state. The final architecture is not decided yet --
this is a minimal, configurable stand-in so the rest of the pipeline has
something to call, per the fast-track plan's "clear placeholder/interface,
not a fake implementation" instruction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MelHead(nn.Module):
    def __init__(self, d_in: int = 1536, n_mels: int = 100) -> None:
        """d_in must match the speech core's hidden size (see
        speechcore/qwen_speech_core.py's verify_loaded() output).
        n_mels must match Vocos's feature extractor (100 for
        charactr/vocos-mel-24khz -- see vocoder/mel_utils.py)."""
        super().__init__()
        self.proj = nn.Linear(d_in, n_mels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (B, T, d_in) -> (B, T, n_mels). Transpose to
        (B, n_mels, T) before passing to Vocos's decode()."""
        return self.proj(hidden_states)
