"""Frozen WavLM speech encoder wrapper.

This is P0-scope only: a clean, testable wrapper around the pretrained
microsoft/wavlm-base-plus model. No streaming/duplex buffering here yet --
that comes after P0 is verified end to end.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

WAVLM_SAMPLE_RATE = 16000  # WavLM's expected input sample rate; independent of Vocos's 24kHz output side


class WavLMEncoder(nn.Module):
    """Loads microsoft/wavlm-base-plus, freezes it, and exposes a simple
    waveform -> features interface at WavLM's native ~50Hz frame rate.

    Accepts variable-length inputs (a list of 1-D tensors with different
    sample counts, e.g. a batch where the last chunk of a session is
    shorter than the rest) and returns a real frame-level attention mask
    alongside the hidden states -- not an approximation, but the mask HF's
    own `_get_feature_vector_attention_mask` computes from the exact conv
    stride/kernel math, so downstream code (fusion, the speech core,
    acoustic head, loss) can correctly ignore padded frames instead of
    guessing which frames are padding from the input duration ratio."""

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
    def forward(
        self, waveform: Union[torch.Tensor, Sequence[torch.Tensor]], sample_rate: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        waveform: (B, n_samples) tensor (equal-length batch), a 1-D tensor
            (single item), or a list/tuple of 1-D tensors with DIFFERENT
            lengths (variable-length batch) -- all at `sample_rate`.
        Returns: (hidden_states, frame_mask)
            hidden_states: (B, T, 768), T = max frame count in the batch,
                zero-padded past each item's real length.
            frame_mask: (B, T) long tensor, 1 for real frames, 0 for padding.
        """
        if torch.is_tensor(waveform):
            items: List[torch.Tensor] = list(waveform.unsqueeze(0)) if waveform.dim() == 1 else list(waveform)
        else:
            items = list(waveform)

        if sample_rate != WAVLM_SAMPLE_RATE:
            items = [torchaudio.functional.resample(w, sample_rate, WAVLM_SAMPLE_RATE) for w in items]

        inputs = self.feature_extractor(
            [w.numpy() for w in items], sampling_rate=WAVLM_SAMPLE_RATE,
            return_tensors="pt", padding=True,
        )
        input_values = inputs["input_values"].to(self.device)
        sample_attention_mask = inputs["attention_mask"].to(self.device)  # sample-level, not frame-level yet

        outputs = self.model(input_values=input_values, attention_mask=sample_attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, T, 768)

        frame_mask = self.model._get_feature_vector_attention_mask(hidden_states.shape[1], sample_attention_mask)
        return hidden_states, frame_mask


def _self_test(wav_path: str, device: torch.device) -> None:
    """Loads one WAV file and prints input/output shapes -- the Phase 3 smoke test."""
    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.mean(dim=0, keepdim=True)  # mono

    encoder = WavLMEncoder(device=device)
    features, frame_mask = encoder(waveform, sr)

    print("=== WavLM self-test ===")
    print(f"input shape:  {tuple(waveform.shape)} @ {sr}Hz")
    print(f"output shape: {tuple(features.shape)}")
    print(f"frame mask:   {tuple(frame_mask.shape)}, valid frames: {int(frame_mask.sum())}")
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
