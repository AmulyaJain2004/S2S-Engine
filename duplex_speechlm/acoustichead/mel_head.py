"""Acoustic head: predicts mel-spectrogram frames from the speech core's
hidden state.

CAPACITY UPGRADE (previous version was a single nn.Linear). That was never
actually stress-tested: the old duplex-fusion design let the model shortcut
training by echoing the ground-truth agent audio it was handed one frame
early (see speechcore/duplex_fusion.py's docstring), so a bare linear
readout was enough to look like it worked under teacher forcing. Now that
duplex fusion no longer hands the acoustic head's input a near-copy of the
target, genuine mel generation capacity actually matters. This adds:
- a small Conv1D stack for temporal context (mel frames are not
  independent of their neighbors; a per-frame-only Linear has no way to
  use that), and
- a lightweight residual "postnet" (the Tacotron2 pattern) that refines
  the coarse per-frame prediction using local context, instead of emitting
  the raw per-frame projection directly -- this specifically targets
  buzzy/blurry mel output, a known failure mode of frame-independent
  readouts.
Still deliberately small: this is a prototype's acoustic head, not a
production TTS decoder, and it should stay cheap enough to train on a
single A100/H100 session.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MelHead(nn.Module):
    def __init__(self, d_in: int = 1536, n_mels: int = 100, d_hidden: int = 512, conv_kernel: int = 5) -> None:
        """d_in must match the speech core's hidden size (see
        speechcore/qwen_speech_core.py's verify_loaded() output).
        n_mels must match Vocos's feature extractor (100 for
        charactr/vocos-mel-24khz -- see vocoder/mel_utils.py)."""
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU())
        self.temporal = nn.Sequential(
            nn.Conv1d(d_hidden, d_hidden, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.GELU(),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.GELU(),
        )
        self.coarse_mel = nn.Linear(d_hidden, n_mels)
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, d_hidden, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.GELU(),
            nn.Conv1d(d_hidden, n_mels, kernel_size=conv_kernel, padding=conv_kernel // 2),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (B, T, d_in) -> (B, T, n_mels). Transpose to
        (B, n_mels, T) before passing to Vocos's decode()."""
        h = self.in_proj(hidden_states)          # (B, T, d_hidden)
        h = self.temporal(h.transpose(1, 2))     # (B, d_hidden, T)
        coarse = self.coarse_mel(h.transpose(1, 2))  # (B, T, n_mels)
        coarse_t = coarse.transpose(1, 2)        # (B, n_mels, T)
        refined = coarse_t + self.postnet(coarse_t)  # residual refinement, Tacotron2-style
        return refined.transpose(1, 2)           # (B, T, n_mels)
