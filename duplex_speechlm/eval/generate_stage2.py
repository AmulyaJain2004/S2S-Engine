"""Stage 2 inference / listening script: autoregressive duplex generation
from a trained checkpoint. This is the only way to actually hear what
Stage 2 learned -- a falling training loss alone does not tell you the
output is intelligible or that duplex timing makes sense.

SIMPLIFIED, TRAIN/INFERENCE-CONSISTENT DESIGN (previous version re-encoded
the model's own synthesized audio through WavLM every window as a
"previous output" signal for duplex fusion, mirroring how training used
to feed back ground-truth agent audio. That whole mechanism is gone --
see speechcore/duplex_fusion.py's docstring. It was also the actual
source of the noisy output in earlier runs: no amount of clamping,
loudness-matching, or crossfading the re-encoded audio fixed it, because
the problem was never signal corruption in that loop, it was that the
model had never learned genuine content generation in the first place
(the old design let it shortcut training by echoing real ground-truth
audio instead). Generation now uses the EXACT mechanism training uses:
only the user stream is ever encoded and fed in; the speech core's own
causal self-attention (carried across windows via the Qwen KV cache)
provides whatever "memory of what I've been saying" it needs, with no
re-encoded audio injected at all.

WINDOWED, NOT PER-FRAME (still true, and still a deliberate choice, now
for a much simpler reason): WavLM has a multi-frame receptive field and
Vocos vocodes in chunks, so true 20ms-frame-by-frame streaming still isn't
practical -- generation still proceeds in `--window_s` windows (default
0.4s, matching spec section 2's own accepted ~300-500ms chunk-latency
trade-off), encoding each window's user audio and running one KV-cached
Qwen step per window. There is no self-feedback loop left to need
windowing for; this windowing is purely about WavLM/Vocos chunk granularity.

OUTPUT-SIDE SAFETY (kept from before, still cheap and still worth doing):
Vocos's raw output is not guaranteed to stay within [-1, 1], so it's
clamped before saving. Each window is still vocoded independently with no
shared phase state, so a short crossfade is still applied at each stitch
point in the final saved waveform to mask boundary clicks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ensure_dir, load_config, pick_device
from encoder.wavlm_encoder import WAVLM_SAMPLE_RATE
from train.stage2_duplex import build_models  # reuse the EXACT same model construction as training
from vocoder.mel_utils import VOCOS_HOP_LENGTH, VOCOS_SAMPLE_RATE


def _crossfade_concat(chunks: List[torch.Tensor], fade_samples: int) -> torch.Tensor:
    """Concatenates waveform chunks with a short linear crossfade at each
    boundary instead of a hard cut, to mask the phase discontinuity each
    independently-vocoded window introduces. fade_samples is clamped to
    each chunk's own length so this is safe even for very short chunks."""
    if len(chunks) == 1:
        return chunks[0]
    out = chunks[0]
    for nxt in chunks[1:]:
        n = min(fade_samples, out.shape[0], nxt.shape[0])
        if n <= 0:
            out = torch.cat([out, nxt], dim=0)
            continue
        ramp = torch.linspace(0.0, 1.0, n)
        blended = out[-n:] * (1.0 - ramp) + nxt[:n] * ramp
        out = torch.cat([out[:-n], blended, nxt[n:]], dim=0)
    return out


def load_trained_checkpoint(ckpt_dir: Path, qwen_core, projector, fusion, acoustic_head, device) -> int:
    state_path = ckpt_dir / "train_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"No checkpoint at {state_path} -- train first (train/stage2_duplex.py).")
    state = torch.load(state_path, map_location=device)
    projector.load_state_dict(state["projector"])
    fusion.load_state_dict(state["fusion"])
    acoustic_head.load_state_dict(state["acoustic_head"])

    from peft import PeftModel

    qwen_core.model = PeftModel.from_pretrained(qwen_core.model.get_base_model(), str(ckpt_dir / "qwen_lora"))
    qwen_core.model.to(device)
    qwen_core.model.eval()
    print(f"Loaded checkpoint from step {state['step']} at {ckpt_dir}")
    return state["step"]


@torch.no_grad()
def generate(cfg_path: str, user_wav_path: str, window_s: float, max_duration_s: float, out_dir: str) -> None:
    cfg = load_config(cfg_path)
    device = pick_device(cfg["runtime"]["device"])
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg["runtime"]["dtype"]]
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    out_dir = ensure_dir(out_dir)

    wavlm, vocos, qwen_core, projector, fusion, acoustic_head = build_models(cfg, device, dtype)
    step = load_trained_checkpoint(ckpt_dir, qwen_core, projector, fusion, acoustic_head, device)
    for m in (projector, fusion, acoustic_head):
        m.eval()

    user_wave, native_sr = torchaudio.load(user_wav_path)
    user_wave = user_wave.mean(dim=0)  # mono
    if max_duration_s is not None:
        user_wave = user_wave[: int(max_duration_s * native_sr)]

    window_len = int(window_s * native_sr)
    n_windows = max(1, (user_wave.shape[0] + window_len - 1) // window_len)

    past_key_values = None
    synthesized_chunks = []

    print(f"Generating {n_windows} window(s) of {window_s}s each from checkpoint step {step} "
          f"(user-only input, KV-cached Qwen, no self-feedback loop)...")
    for i in range(n_windows):
        start = i * window_len
        window = user_wave[start : start + window_len]
        if window.shape[0] == 0:
            break

        # Kept on CPU before WavLM -- see train/stage2_duplex.py's identical note:
        # WavLMEncoder moves inputs to its own device internally and needs CPU
        # tensors for its feature extractor's .numpy() call.
        user_16k = torchaudio.functional.resample(window, native_sr, WAVLM_SAMPLE_RATE)
        user_feat, _ = wavlm([user_16k], WAVLM_SAMPLE_RATE)  # (1, T_u, 768); batch of 1, no real padding
        user_emb = projector(user_feat.to(dtype))  # (1, T_u, hidden)

        fused = fusion(user_emb)  # (1, T_u, hidden) -- no "previous output" input anymore
        hidden, past_key_values = qwen_core.forward_step(fused, past_key_values)
        pred_mel = acoustic_head(hidden).transpose(1, 2).float()  # (1, n_mels, T_u)

        # No ground-truth agent audio exists at real inference time, so the
        # target mel frame count for this window is estimated from Vocos's own
        # verified frame-rate formula (center=True STFT), not measured from a
        # target we don't have.
        window_samples_24k = round(window.shape[0] / native_sr * VOCOS_SAMPLE_RATE)
        target_mel_frames = max(1, window_samples_24k // VOCOS_HOP_LENGTH + 1)
        pred_mel = F.interpolate(pred_mel, size=target_mel_frames, mode="linear", align_corners=False)

        waveform_window = vocos.waveform_from_mel(pred_mel).cpu().squeeze(0).clamp(-1.0, 1.0)
        synthesized_chunks.append(waveform_window)

        if (i + 1) % 10 == 0 or i == n_windows - 1:
            print(f"  window {i + 1}/{n_windows} done")

    fade_samples = int(0.01 * VOCOS_SAMPLE_RATE)  # 10ms crossfade at each window stitch
    full_output = _crossfade_concat(synthesized_chunks, fade_samples).unsqueeze(0)
    out_path = out_dir / "generated_agent.wav"
    torchaudio.save(str(out_path), full_output, VOCOS_SAMPLE_RATE)

    user_out_path = out_dir / "input_user.wav"
    torchaudio.save(str(user_out_path), user_wave.unsqueeze(0), native_sr)

    print(f"\nSaved: {user_out_path}  (what the model heard)")
    print(f"Saved: {out_path}  (what the model generated)")
    print("Listen to both. This is the honest test of what Stage 2 actually learned --")
    print("a low training loss does not by itself mean this sounds like intelligible, well-timed speech.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2.yaml")
    parser.add_argument("--user_wav", required=True, help="A real user-channel WAV to feed the model.")
    parser.add_argument("--window_s", type=float, default=0.4,
                         help="Window size in seconds -- the spec's own accepted ~300-500ms chunk latency.")
    parser.add_argument("--max_duration_s", type=float, default=30.0,
                         help="Cap how much of --user_wav to process (generation is sequential, not batched).")
    parser.add_argument("--out_dir", default="outputs/generation")
    args = parser.parse_args()
    generate(args.config, args.user_wav, args.window_s, args.max_duration_s, args.out_dir)
