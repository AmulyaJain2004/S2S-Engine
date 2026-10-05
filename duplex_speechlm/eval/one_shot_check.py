"""Diagnostic: does a ONE-SHOT, non-windowed forward pass (the whole chunk
through Qwen in a single call, exactly like training does it) sound
reasonable on a held-out chunk?

Why this still matters after the duplex-fusion redesign (see
speechcore/duplex_fusion.py's docstring): training now processes a whole
chunk in one non-cached forward call, while eval/generate_stage2.py
processes the same audio window-by-window with a KV cache. Those two
should be mathematically equivalent (a transformer's cached incremental
decoding is supposed to produce the same result as one big forward pass),
but it's cheap to verify directly rather than assume it: if this one-shot
path sounds fine but generate_stage2.py's windowed output doesn't, the bug
is in the windowing/KV-cache logic, not in training or model capacity.

(Earlier versions of this script compared "teacher-forced" against
"self-feedback" duplex fusion -- that distinction no longer exists, since
duplex fusion only ever takes the user stream as input now, in both
training and generation.)
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
    agent_24k = torchaudio.functional.resample(agent_chunk, native_sr, VOCOS_SAMPLE_RATE)

    # Exactly train/stage2_duplex.py's forward pass: one non-cached WavLM +
    # Qwen call over the whole chunk, user stream only.
    user_feat, _ = wavlm([user_16k], WAVLM_SAMPLE_RATE)
    user_emb = projector(user_feat.to(dtype))
    fused = fusion(user_emb)
    hidden = qwen_core.forward(fused)  # non-cached, whole-chunk forward -- same as training
    pred_mel = acoustic_head(hidden).transpose(1, 2).float()

    target_mel = vocos.mel_from_waveform(agent_24k.unsqueeze(0))
    pred_mel = F.interpolate(pred_mel, size=target_mel.shape[-1], mode="linear", align_corners=False)

    reconstructed = vocos.waveform_from_mel(pred_mel).cpu().squeeze(0).clamp(-1.0, 1.0)

    gt_path = out_dir / "ground_truth_agent.wav"
    pred_path = out_dir / "one_shot_predicted_agent.wav"
    torchaudio.save(str(gt_path), agent_chunk.unsqueeze(0), native_sr)
    torchaudio.save(str(pred_path), reconstructed.unsqueeze(0), VOCOS_SAMPLE_RATE)

    mel_l1 = F.l1_loss(pred_mel, target_mel.float()).item()
    print(f"Checkpoint step: {step}")
    print(f"One-shot (non-windowed) mel L1 on this held-out chunk: {mel_l1:.4f}")
    print(f"Saved: {gt_path}  (real ground-truth agent audio for this chunk)")
    print(f"Saved: {pred_path}  (model's one-shot prediction, non-windowed, no KV cache)")
    print()
    print("Compare this to generate_stage2.py's windowed output on the same session:")
    print("- If this sounds reasonable but generate_stage2.py's windowed output doesn't:")
    print("  the windowing/KV-cache logic has a bug -- the two should be equivalent.")
    print("- If both sound similar (good or bad): the model/training itself is the story,")
    print("  not the windowing mechanism.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2.yaml")
    parser.add_argument("--session_dir", required=True, help="A processed/session_* directory (held-out ideally).")
    parser.add_argument("--offset_s", type=float, default=60.0, help="Where in the session to take the chunk from.")
    parser.add_argument("--duration_s", type=float, default=8.0, help="Should match configs/*.yaml's chunk_duration_s.")
    parser.add_argument("--out_dir", default="outputs/teacher_forced_check")
    args = parser.parse_args()
    check(args.config, args.session_dir, args.offset_s, args.duration_s, args.out_dir)
