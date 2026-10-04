"""P0: plumbing check only. NO training happens here.

Verifies, on one real WAV clip:
  1. WavLM extracts features successfully.
  2. Vocos produces a mel from the same audio, at its own exact config.
  3. Vocos reconstructs a waveform from that mel.

Per the project spec: if this round trip doesn't sound right, downstream
training must not start. This script reports numbers; a human still needs
to actually listen to outputs/p0_original.wav vs outputs/p0_reconstructed.wav
before calling P0 done.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/p0_roundtrip.py`

from common import ensure_dir, load_config, pick_device
from encoder.wavlm_encoder import WAVLM_SAMPLE_RATE, WavLMEncoder
from vocoder.mel_utils import VOCOS_SAMPLE_RATE
from vocoder.vocos_wrapper import VocosVocoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="Path to a short real WAV clip")
    parser.add_argument("--config", default="configs/p0.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device(cfg["runtime"]["device"])
    out_dir = ensure_dir(cfg["paths"]["output_dir"])

    print("Loading input audio...")
    waveform, sr = torchaudio.load(args.wav)
    waveform = waveform.mean(dim=0, keepdim=True)  # mono
    duration_s = waveform.shape[1] / sr

    torchaudio.save(str(out_dir / "p0_original.wav"), waveform, sr)

    print("Loading WavLM (frozen)...")
    wavlm = WavLMEncoder(cfg["models"]["wavlm_name"], device=device)
    wavlm_features, wavlm_frame_mask = wavlm(waveform, sr)
    frame_rate = wavlm_features.shape[1] / duration_s

    print("Loading Vocos (frozen)...")
    vocos = VocosVocoder(cfg["models"]["vocos_repo"], device=device)

    vocos_input = waveform
    if sr != VOCOS_SAMPLE_RATE:
        vocos_input = torchaudio.functional.resample(waveform, sr, VOCOS_SAMPLE_RATE)

    mel = vocos.mel_from_waveform(vocos_input)
    reconstructed = vocos.waveform_from_mel(mel).cpu()

    recon_path = out_dir / "p0_reconstructed.wav"
    torchaudio.save(str(recon_path), reconstructed, VOCOS_SAMPLE_RATE)

    print("\n=== P0 REPORT ===")
    print(f"Input sample rate:   {sr}")
    print(f"Input duration:      {duration_s:.3f}s")
    print(f"Input shape:         {tuple(waveform.shape)}")
    print()
    print("WavLM:")
    print(f"  feature shape:     {tuple(wavlm_features.shape)}")
    print(f"  frame rate est.:   {frame_rate:.2f} Hz  (expected ~50Hz)")
    print()
    print("Vocos:")
    print(f"  expected sample rate: {VOCOS_SAMPLE_RATE}")
    print(f"  mel shape:            {tuple(mel.shape)}")
    print(f"  reconstructed sample rate: {VOCOS_SAMPLE_RATE}")
    print(f"  reconstructed duration:    {reconstructed.shape[1] / VOCOS_SAMPLE_RATE:.3f}s")
    print()
    print(f"Saved: {out_dir / 'p0_original.wav'}")
    print(f"Saved: {recon_path}")
    print()
    print("Round-trip completed: YES (mechanically) -- but this only means the code ran.")
    print("ACTION REQUIRED: listen to both files yourself. Do not treat a successful")
    print("run of this script as proof the reconstruction is actually intelligible.")


if __name__ == "__main__":
    main()
