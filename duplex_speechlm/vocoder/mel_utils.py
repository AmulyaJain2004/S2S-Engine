"""The single authoritative waveform -> Vocos-compatible mel function.

Verified directly against the installed `vocos` package source (not guessed):
the `charactr/vocos-mel-24khz` checkpoint's feature extractor is
`vocos.feature_extractors.MelSpectrogramFeatures(sample_rate=24000, n_fft=1024,
hop_length=256, n_mels=100, padding="center")`, built on
`torchaudio.transforms.MelSpectrogram(power=1)` followed by a safe-log.

Rather than re-declaring these numbers here (which could silently drift from
whatever checkpoint is actually loaded), this module always calls the loaded
Vocos model's own `.feature_extractor`, so correctness is guaranteed by
construction. Every other part of this project must call `waveform_to_mel`
from here rather than building its own mel pipeline.
"""
from __future__ import annotations

import torch

VOCOS_SAMPLE_RATE = 24000  # what charactr/vocos-mel-24khz expects on its input side


def waveform_to_mel(vocos_model, waveform: torch.Tensor) -> torch.Tensor:
    """
    vocos_model: a loaded `vocos.Vocos` instance (see vocoder/vocos_wrapper.py).
    waveform: (B, n_samples) float tensor, already at VOCOS_SAMPLE_RATE.
    Returns: (B, n_mels, T) log-mel features, exactly as Vocos's own backbone expects.
    """
    return vocos_model.feature_extractor(waveform)
