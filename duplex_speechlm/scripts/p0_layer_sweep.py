"""Diagnostic: is layer 7 (the default) actually the right WavLM-large layer
for THIS dataset's audio, or is content loss coming from somewhere else?

Why this exists: Section 4 (P0) on real otoSpeech audio showed the
tokenizer+vocoder round trip changing the actual SENTENCE CONTENT, not just
degrading audio quality -- e.g. original transcribes to one real sentence,
reconstruction transcribes to an unrelated generic one. That is a much more
serious finding than "sounds muffled": it means the discrete units aren't
preserving enough phonetic content for THIS audio, for some reason.

This script isolates two different possible causes on the SAME clip:
1. Our own resampling (saves the exact 16kHz input speechbrain actually
   sees -- if THIS already sounds wrong, the bug is in our preprocessing,
   not the pretrained tokenizer).
2. Layer choice: encodes+decodes the SAME clip through every layer
   speechbrain's WavLM-large k-means/vocoder set supports
   (1, 3, 7, 12, 18, 23), reusing the same loaded WavLM-large encoder
   instance across layers (only the k-means + vocoder differ per layer,
   so this doesn't re-download the 1.2GB encoder six times).

If EVERY layer loses content equally, that points to a domain mismatch
(this pretrained tokenizer/vocoder set was trained on LibriSpeech/LibriTTS --
clean, single-speaker, studio-quality audiobook speech -- not real
two-person conversational audio) rather than a wrong-layer problem, and the
discrete-unit approach as currently built may need a different pretrained
codec (e.g. the HuBERT or Wav2Vec2 variants speechbrain also ships, or a
codec actually suited to conversational speech) rather than another layer
swap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ensure_dir, pick_device
from speechcore.discrete_tokenizer import DISCRETE_SSL_SAMPLE_RATE, VOCODER_OUTPUT_SAMPLE_RATE

CANDIDATE_LAYERS = [1, 3, 7, 12, 18, 23]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="Path to a short real WAV clip (ideally real speech)")
    parser.add_argument("--output_dir", default="outputs/layer_sweep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kmeans_dataset", default="LibriSpeech")
    parser.add_argument("--vocoder_repo_id", default="speechbrain/hifigan-wavlm-k1000-LibriTTS")
    parser.add_argument("--num_clusters", type=int, default=1000)
    args = parser.parse_args()

    device = pick_device(args.device)
    out_dir = ensure_dir(args.output_dir)

    from speechbrain.integrations.audio_tokenizers.discrete_ssl import DiscreteSSL
    from speechbrain.integrations.huggingface.wavlm import WavLM

    print("Loading input audio...")
    waveform, sr = torchaudio.load(args.wav)
    waveform = waveform.mean(dim=0)  # mono
    torchaudio.save(str(out_dir / "sweep_original.wav"), waveform.unsqueeze(0), sr)

    # Save exactly what speechbrain sees, BEFORE any encoding -- isolates
    # whether our own resampling step is the problem.
    resampled = waveform
    if sr != DISCRETE_SSL_SAMPLE_RATE:
        resampled = torchaudio.functional.resample(waveform, sr, DISCRETE_SSL_SAMPLE_RATE)
    torchaudio.save(str(out_dir / "sweep_resampled_16k_input.wav"), resampled.unsqueeze(0), DISCRETE_SSL_SAMPLE_RATE)
    print(f"Saved sweep_resampled_16k_input.wav -- LISTEN TO THIS FIRST. If this already "
          f"sounds wrong/garbled, the bug is in our resampling, not the pretrained tokenizer.")

    print("Loading WavLM-large (frozen) ONCE, shared across all layers...")
    ssl_encoder = WavLM(
        source="microsoft/wavlm-large",
        save_path="pretrained_models/discrete_ssl",
        output_all_hiddens=True,
        freeze=True,
    )
    ssl_encoder.to(device)
    ssl_encoder.eval()

    wav_batch = resampled.unsqueeze(0).to(device)

    print(f"\n{'Layer':<8}{'#units':<10}{'unique':<10}{'min':<8}{'max':<8}saved file")
    for layer in CANDIDATE_LAYERS:
        try:
            model = DiscreteSSL(
                save_path="pretrained_models/discrete_ssl",
                ssl_model=ssl_encoder,
                kmeans_dataset=args.kmeans_dataset,
                vocoder_repo_id=args.vocoder_repo_id,
                num_clusters=args.num_clusters,
                layers_num=[layer],
                device=str(device),
            )
            model.to(device)
        except Exception as e:  # noqa: BLE001 -- not every layer is guaranteed available, report and continue
            print(f"{layer:<8}FAILED to load: {e}")
            continue

        with torch.no_grad():
            org_tokens, _emb, _proc = model.encode(wav_batch, SSL_layers=[layer])
            recon = model.decode(org_tokens, SSL_layers=[layer])

        units = org_tokens.squeeze(0).squeeze(-1)
        out_name = f"sweep_layer{layer}_reconstructed.wav"
        torchaudio.save(str(out_dir / out_name), recon.squeeze().cpu().unsqueeze(0), VOCODER_OUTPUT_SAMPLE_RATE)

        n_unique = units.unique().numel()
        print(f"{layer:<8}{units.shape[0]:<10}{n_unique:<10}{units.min().item():<8}{units.max().item():<8}{out_name}")
        # A very low "unique" count relative to the number of frames is a red flag:
        # it means most frames collapsed onto a handful of clusters -- a form of
        # mode collapse that would explain generic/unrelated reconstructed content
        # regardless of what was actually said.

    print(f"\nAll files saved to {out_dir}/. Listen to each sweep_layer<N>_reconstructed.wav and compare against "
          f"sweep_original.wav -- whichever layer actually preserves the real sentence content (if any) is the one "
          f"to use. If NONE of them do, that's strong evidence of a domain mismatch (this tokenizer/vocoder set was "
          f"pretrained on clean LibriSpeech/LibriTTS, not real conversational audio like otoSpeech) rather than a "
          f"wrong-layer problem, and a different pretrained codec may be needed.")


if __name__ == "__main__":
    main()
