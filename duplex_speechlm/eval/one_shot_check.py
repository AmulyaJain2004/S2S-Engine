"""Diagnostic: does a ONE-SHOT, non-windowed forward pass (the whole chunk
through Qwen in a single call, exactly like training does it) produce
sensible unit predictions on a held-out chunk?

Why this still matters: training processes a whole chunk through Qwen in
one non-cached forward call, while eval/generate_stage2.py processes the
same kind of audio window-by-window with a KV cache. Those should be
mathematically equivalent (a transformer's cached incremental decoding is
supposed to produce the same result as one big forward pass), but it's
cheap to verify directly rather than assume it: if this one-shot path's
unit-accuracy/listening result is fine but generate_stage2.py's windowed
output isn't, the bug is in the windowing/KV-cache logic, not in training
or model capacity.
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
from eval.generate_stage2 import load_trained_checkpoint
from speechcore.discrete_tokenizer import VOCODER_OUTPUT_SAMPLE_RATE
from train.stage2_duplex import build_models


@torch.no_grad()
def check(cfg_path: str, session_dir: str, offset_s: float, duration_s: float, out_dir: str) -> None:
    cfg = load_config(cfg_path)
    device = pick_device(cfg["runtime"]["device"])
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg["runtime"]["dtype"]]
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    out_dir = ensure_dir(out_dir)

    tokenizer, qwen_core, embedding, unit_head = build_models(cfg, device, dtype)
    step = load_trained_checkpoint(ckpt_dir, qwen_core, embedding, unit_head, device)
    for m in (embedding, unit_head):
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

    # Exactly train/stage2_duplex.py's forward pass: one non-cached Qwen
    # call over the whole chunk, user stream only.
    user_units = tokenizer.encode(user_chunk, native_sr)
    target_units = tokenizer.encode(agent_chunk, native_sr)
    assert user_units.shape[0] == target_units.shape[0], "user/agent unit counts differ -- investigate the tokenizer"

    user_emb = embedding(user_units.unsqueeze(0))
    hidden = qwen_core.forward(user_emb)  # non-cached, whole-chunk forward -- same as training
    logits = unit_head(hidden).float().squeeze(0)  # (T, vocab)

    pred_units = logits.argmax(dim=-1)
    accuracy = (pred_units == target_units.to(device)).float().mean().item()
    loss = F.cross_entropy(logits, target_units.to(device)).item()

    reconstructed = tokenizer.decode(pred_units)
    ground_truth_decoded = tokenizer.decode(target_units)  # the tokenizer's OWN reconstruction of the real
                                                             # agent audio -- the ceiling this prediction is
                                                             # measured against, not the raw original file.

    gt_path = out_dir / "ground_truth_agent.wav"
    gt_reencoded_path = out_dir / "ground_truth_reencoded.wav"
    pred_path = out_dir / "one_shot_predicted_agent.wav"
    torchaudio.save(str(gt_path), agent_chunk.unsqueeze(0), native_sr)
    torchaudio.save(str(gt_reencoded_path), ground_truth_decoded.unsqueeze(0).clamp(-1.0, 1.0), VOCODER_OUTPUT_SAMPLE_RATE)
    torchaudio.save(str(pred_path), reconstructed.unsqueeze(0).clamp(-1.0, 1.0), VOCODER_OUTPUT_SAMPLE_RATE)

    print(f"Checkpoint step: {step}")
    print(f"One-shot (non-windowed) unit accuracy on this held-out chunk: {accuracy:.3f}")
    print(f"One-shot cross-entropy loss: {loss:.4f}")
    print(f"Saved: {gt_path}  (real, raw ground-truth agent audio)")
    print(f"Saved: {gt_reencoded_path}  (ground truth re-encoded+decoded through the SAME frozen tokenizer --")
    print("  this is the quality CEILING the prediction below can possibly reach, since the tokenizer itself")
    print("  is lossy. Compare the prediction against THIS, not the raw original.)")
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
    parser.add_argument("--out_dir", default="outputs/one_shot_check")
    args = parser.parse_args()
    check(args.config, args.session_dir, args.offset_s, args.duration_s, args.out_dir)
