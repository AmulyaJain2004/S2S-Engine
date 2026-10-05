"""Diagnostic: does the trained checkpoint produce clean mel under the SAME
teacher-forced conditions it was trained under -- i.e. with the real
ground-truth agent audio as the duplex fusion's "previous output" stream,
not the model's own self-generated audio?

Why this script exists: if eval/generate_stage2.py's autoregressive
self-feedback output sounds like noise, there are two very different
possible causes, and they need different fixes:
  1. The acoustic head (acoustichead/mel_head.py is currently a single
     nn.Linear, no temporal modeling) is too weak to produce clean mel
     frames at all -- needs an architecture change + retrain.
  2. Exposure bias: the model never saw its OWN imperfect past output
     during training (training always uses real ground-truth audio for
     that), so self-feedback generation can drift/compound errors fast --
     fixable without retraining (shorter generations, better windowing,
     scheduled sampling later, etc.).

This script runs the EXACT same forward pass as train/stage2_duplex.py
(one shot, teacher-forced with real agent audio, no window-by-window
self-feedback loop) on a held-out chunk, vocodes the predicted mel, and
saves it next to the ground-truth agent audio for that same chunk so you
can compare them directly. If THIS also sounds like noise, cause #1 is
the real problem. If this sounds reasonable but generate_stage2.py's
output doesn't, cause #2 is the real problem.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ensure_dir, load_config, pick_device
from encoder.wavlm_encoder import WAVLM_SAMPLE_RATE
from eval.generate_stage2 import load_trained_checkpoint
from train.stage2_duplex import build_models
from vocoder.mel_utils import VOCOS_SAMPLE_RATE


@torch.no_grad()
def check(cfg_path: str, session_dir: str, offset_s: float, duration_s: float, out_dir: str) -> None:
    cfg = load_config(cfg_path)
    device = pick_device(cfg["runtime"]["device"])
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg["runtime"]["dtype"]]
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    out_dir = ensure_dir(out_dir)

    wavlm, vocos, qwen_core, projector, fusion, acoustic_head = build_models(cfg, device, dtype)
    step = load_trained_checkpoint(ckpt_dir, qwen_core, projector, fusion, acoustic_head, device)
    for m in (projector, fusion, acoustic_head):
        m.eval()

    user_channel = cfg["audio"]["user_channel"]
    agent_channel = cfg["audio"]["agent_channel"]
    user_path = Path(session_dir) / f"speaker_{user_channel}.wav"
    agent_path = Path(session_dir) / f"speaker_{agent_channel}.wav"

    user_full, native_sr = torchaudio.load(str(user_path))
    agent_full, _ = torchaudio.load(str(agent_path))
    start = int(offset_s * native_sr)
    length = int(duration_s * native_sr)
    user_chunk = user_full.mean(dim=0)[start : start + length]
    agent_chunk = agent_full.mean(dim=0)[start : start + length]

    user_16k = torchaudio.functional.resample(user_chunk, native_sr, WAVLM_SAMPLE_RATE)
    agent_16k = torchaudio.functional.resample(agent_chunk, native_sr, WAVLM_SAMPLE_RATE)
    agent_24k = torchaudio.functional.resample(agent_chunk, native_sr, VOCOS_SAMPLE_RATE)

    # Exactly train/stage2_duplex.py's forward pass: one combined WavLM call,
    # teacher-forced fusion (real ground-truth agent embedding, shifted),
    # single non-autoregressive forward through Qwen, no self-feedback loop.
    combined_hidden, combined_mask = wavlm([user_16k, agent_16k], WAVLM_SAMPLE_RATE)
    user_feat, agent_feat = combined_hidden[0:1], combined_hidden[1:2]

    user_emb = projector(user_feat.to(dtype))
    agent_emb = projector(agent_feat.to(dtype))
    prev_output_emb = fusion.shift_with_start_token(agent_emb)
    fused = fusion(user_emb, prev_output_emb)

    hidden = qwen_core.forward(fused)  # non-cached, whole-chunk forward -- same as training
    pred_mel = acoustic_head(hidden).transpose(1, 2).float()

    target_mel = vocos.mel_from_waveform(agent_24k.unsqueeze(0))
    pred_mel = F.interpolate(pred_mel, size=target_mel.shape[-1], mode="linear", align_corners=False)

    reconstructed = vocos.waveform_from_mel(pred_mel).cpu().squeeze(0)

    gt_path = out_dir / "ground_truth_agent.wav"
    pred_path = out_dir / "teacher_forced_predicted_agent.wav"
    torchaudio.save(str(gt_path), agent_chunk.unsqueeze(0), native_sr)
    torchaudio.save(str(pred_path), reconstructed.unsqueeze(0), VOCOS_SAMPLE_RATE)

    mel_l1 = F.l1_loss(pred_mel, target_mel.float()).item()
    print(f"Checkpoint step: {step}")
    print(f"Teacher-forced mel L1 on this held-out chunk: {mel_l1:.4f}")
    print(f"Saved: {gt_path}  (real ground-truth agent audio for this chunk)")
    print(f"Saved: {pred_path}  (model's prediction, TEACHER-FORCED, no self-feedback)")
    print()
    print("Compare this to generate_stage2.py's output on the same session:")
    print("- If THIS sounds like noise too: the acoustic head (a single Linear layer) is")
    print("  too weak -- needs a real architecture upgrade (conv/transformer stack) + retrain.")
    print("- If THIS sounds reasonable but generate_stage2.py's self-feedback output doesn't:")
    print("  exposure bias in autoregressive generation is the real problem, not training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2.yaml")
    parser.add_argument("--session_dir", required=True, help="A processed/session_* directory (held-out ideally).")
    parser.add_argument("--offset_s", type=float, default=60.0, help="Where in the session to take the chunk from.")
    parser.add_argument("--duration_s", type=float, default=8.0, help="Should match configs/*.yaml's chunk_duration_s.")
    parser.add_argument("--out_dir", default="outputs/teacher_forced_check")
    args = parser.parse_args()
    check(args.config, args.session_dir, args.offset_s, args.duration_s, args.out_dir)
