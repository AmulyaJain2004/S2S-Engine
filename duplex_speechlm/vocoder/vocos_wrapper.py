"""Frozen Vocos vocoder wrapper: mel-spectrogram in, waveform out.

This is the runtime vocoder per the project spec (not Kokoro -- Kokoro is a
full TTS system used only offline for Phase A synthetic data generation).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from vocos import Vocos

from vocoder.mel_utils import VOCOS_SAMPLE_RATE, waveform_to_mel


class VocosVocoder(nn.Module):
    def __init__(self, repo_id: str = "charactr/vocos-mel-24khz", device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.model = Vocos.from_pretrained(repo_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

    @torch.no_grad()
    def mel_from_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, n_samples) at VOCOS_SAMPLE_RATE -> (B, n_mels, T) mel."""
        return waveform_to_mel(self.model, waveform.to(self.device))

    @torch.no_grad()
    def waveform_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, T) -> (B, n_samples) reconstructed waveform."""
        return self.model.decode(mel.to(self.device))

    @torch.no_grad()
    def copy_synthesis(self, waveform: torch.Tensor) -> torch.Tensor:
        """Full round trip: waveform -> mel -> waveform. Used by the P0 check."""
        return self.model(waveform.to(self.device))
