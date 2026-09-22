"""Frozen WavLM speech encoder wrapper.

This is P0-scope only: a clean, testable wrapper around the pretrained
microsoft/wavlm-base-plus model. No streaming/duplex buffering here yet --
that comes after P0 is verified end to end.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

WAVLM_SAMPLE_RATE = 16000  # WavLM's expected input sample rate; independent of Vocos's 24kHz output side


class WavLMEncoder(nn.Module):
    """Loads microsoft/wavlm-base-plus, freezes it, and exposes a simple
    waveform -> features interface at WavLM's native ~50Hz frame rate."""

    def __init__(self, model_name: str = "microsoft/wavlm-base-plus", device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        waveform: (B, n_samples) or (n_samples,) float tensor, any sample rate.
        Returns: (B, T, 768) hidden states at WavLM's native ~50Hz frame rate.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if sample_rate != WAVLM_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, WAVLM_SAMPLE_RATE)

        inputs = self.feature_extractor(
            [w.numpy() for w in waveform], sampling_rate=WAVLM_SAMPLE_RATE,
            return_tensors="pt", padding=True,
        )
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(input_values=input_values, attention_mask=attention_mask)
        return outputs.last_hidden_state  # (B, T, 768)


def _self_test(wav_path: str, device: torch.device) -> None:
    """Loads one WAV file and prints input/output shapes -- the Phase 3 smoke test."""
    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.mean(dim=0, keepdim=True)  # mono

    encoder = WavLMEncoder(device=device)
    features = encoder(waveform, sr)

    print("=== WavLM self-test ===")
    print(f"input shape:  {tuple(waveform.shape)} @ {sr}Hz")
    print(f"output shape: {tuple(features.shape)}")
    print(f"device:       {device}")
    approx_frame_rate = features.shape[1] / (waveform.shape[1] / sr)
    print(f"approx frame rate: {approx_frame_rate:.2f} Hz")


if __name__ == "__main__":
    import argparse
    from common import pick_device

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="Path to a short real WAV file")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    _self_test(args.wav, pick_device(args.device))
