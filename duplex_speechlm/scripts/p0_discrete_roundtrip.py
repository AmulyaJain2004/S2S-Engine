"""P0 (discrete pipeline): plumbing check for the architecture Stage 2
actually depends on now. NO training happens here, and NO Qwen is
involved -- this isolates whether the pretrained WavLM-large + k-means(1000)
+ UnitHiFiGAN pipeline alone (speechcore/discrete_tokenizer.py) produces
intelligible audio on a real clip, before any training time is spent on
top of it.

This supersedes scripts/p0_roundtrip.py (which checked the OLD Vocos-mel
pipeline) as the actual relevant gate for Stage 2, per the architecture
change documented in speechcore/discrete_tokenizer.py's docstring. The old
script still works but no longer tests anything Stage 2's current
architecture depends on -- keep using THIS one for the discrete pipeline.

Per the same spirit as the original P0 gate: a successful run of this
script proves the code executed, not that the result is intelligible. A
human still needs to listen to outputs/p0_discrete_original.wav vs
outputs/p0_discrete_reconstructed.wav before trusting anything built on
top of this tokenizer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ensure_dir, pick_device
from speechcore.discrete_tokenizer import DiscreteSpeechTokenizer, VOCODER_OUTPUT_SAMPLE_RATE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="Path to a short real WAV clip (ideally real speech, not silence)")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = pick_device(args.device)
    out_dir = ensure_dir(args.output_dir)

    print("Loading input audio...")
    waveform, sr = torchaudio.load(args.wav)
    waveform = waveform.mean(dim=0)  # mono
    duration_s = waveform.shape[-1] / sr
    torchaudio.save(str(out_dir / "p0_discrete_original.wav"), waveform.unsqueeze(0), sr)

    print("Loading WavLM-large + k-means(1000) + UnitHiFiGAN (all frozen, all pretrained)...")
    tokenizer = DiscreteSpeechTokenizer(device=device)

    print("Encoding to discrete units...")
    units = tokenizer.encode(waveform, sr)
    frame_rate = units.shape[0] / duration_s

    print("Decoding units back to waveform...")
    reconstructed = tokenizer.decode(units)
    recon_path = out_dir / "p0_discrete_reconstructed.wav"
    torchaudio.save(str(recon_path), reconstructed.unsqueeze(0), VOCODER_OUTPUT_SAMPLE_RATE)

    print("\n=== P0 (DISCRETE) REPORT ===")
    print(f"Input sample rate:   {sr}")
    print(f"Input duration:      {duration_s:.3f}s")
    print(f"Unit sequence length: {units.shape[0]} (frame rate est. {frame_rate:.2f}Hz, expected ~50Hz)")
    print(f"Unit vocab size:      {tokenizer.vocab_size}")
    print(f"Reconstructed sample rate: {VOCODER_OUTPUT_SAMPLE_RATE}")
    print(f"Reconstructed duration:    {reconstructed.shape[-1] / VOCODER_OUTPUT_SAMPLE_RATE:.3f}s")
    print()
    print(f"Saved: {out_dir / 'p0_discrete_original.wav'}")
    print(f"Saved: {recon_path}")
    print()
    print("Round-trip completed: YES (mechanically) -- but this only means the code ran.")
    print("ACTION REQUIRED: listen to both files yourself. This tokenizer+vocoder pair is PRETRAINED")
    print("on LibriTTS/general speech, not on otoSpeech specifically -- if it sounds muffled or")
    print("unlike natural speech on YOUR data, that's a real domain-mismatch risk worth knowing about")
    print("before training on top of it, not just a training-time problem to fix later.")


if __name__ == "__main__":
    main()
