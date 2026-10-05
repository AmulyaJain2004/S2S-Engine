"""Stage 2 inference / listening script: autoregressive duplex generation
from a trained checkpoint. This is the only way to actually hear what
Stage 2 learned -- a falling training loss alone does not tell you the
output is intelligible or that duplex timing makes sense.

TRAIN/INFERENCE MISMATCH (exposure bias -- flagged, not hidden):
train/stage2_duplex.py teacher-forces the "previous output frame
embedding" half of duplex fusion using the REAL ground-truth agent audio,
shifted by one frame -- the model never has to rely on its own imperfect
past output during training. This script instead feeds back the model's
OWN synthesized audio from the previous window, which is the only option
at real inference time (there is no ground-truth agent audio to use).
This is the standard autoregressive exposure-bias gap: expect generation
quality to degrade more over a long run than the training loss alone
would suggest, especially after a short training run. Revisit with
scheduled sampling or more Stage 2 training if this bites.

WINDOWED, NOT PER-FRAME, DUPLEX FUSION (by design, not a shortcut): true
frame-by-frame (20ms) self-feedback would require vocoding every single
frame through Vocos and re-encoding it through WavLM before the NEXT
frame could even be computed -- Vocos isn't causal at that granularity
and WavLM has its own multi-frame receptive field, so that would be both
wrong and extremely slow. Per spec section 2's own accepted trade-off
("process audio in fixed chunks (roughly 300-500ms)"), this script
generates in windows of `--window_s` (default 0.4s): each window's duplex
fusion uses the PREVIOUS window's self-synthesized audio (re-encoded
through WavLM + projector, then time-aligned to the current window's
frame count) as the "previous output" stream -- not literally the one
immediately preceding frame. A Qwen KV cache carries real cross-window
context so the speech core still sees its full history, not just the
current window in isolation.

FIRST WINDOW: there is no self-generated history yet, so duplex fusion
for window 0 uses the learned start token (the same one position 0 uses
in training) broadcast across the whole window, not a single frame.

CLAMPING THE SELF-FED AUDIO (fixes a real, confirmed bug, not a cosmetic
safeguard): Vocos's raw output is NOT guaranteed to stay within [-1, 1] --
nothing in vocos_wrapper.py/mel_utils.py clips it. WavLM, however, expects
input in roughly that range (it was pretrained on normalized natural
speech); feeding it out-of-range values produces badly corrupted
features. The teacher-forced path (train/stage2_duplex.py,
eval/teacher_forced_check.py) NEVER re-encodes synthesized audio through
WavLM, so it was never exposed to this. This self-feedback path does, every
single window, which compounds across windows -- confirmed by
teacher_forced_check.py sounding clean while this script's output didn't.
Every self-fed chunk is clamped to [-1, 1] before being re-encoded (and
before being saved/concatenated) to close this off.
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
from train.stage2_duplex import build_models  # reuse the EXACT same model construction as training
from vocoder.mel_utils import VOCOS_HOP_LENGTH, VOCOS_SAMPLE_RATE


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
    prev_output_wave_native = None  # previous window's self-synthesized audio, at native_sr
    synthesized_chunks = []

    print(f"Generating {n_windows} window(s) of {window_s}s each from checkpoint step {step} "
          f"(self-feedback, KV-cached Qwen)...")
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
        t_u = user_emb.shape[1]

        if prev_output_wave_native is None:
            prev_output_emb = fusion.start_token.view(1, 1, -1).expand(1, t_u, -1).to(dtype)
        else:
            prev_16k = torchaudio.functional.resample(prev_output_wave_native, native_sr, WAVLM_SAMPLE_RATE)
            prev_feat, _ = wavlm([prev_16k], WAVLM_SAMPLE_RATE)
            prev_emb = projector(prev_feat.to(dtype))  # (1, T_prev, hidden)
            # Time-align the previous window's embedding to THIS window's frame
            # count -- the same legitimate rate-bridging interpolation used in
            # training, needed because the previously vocoded window's duration
            # rarely matches the input window's duration exactly.
            prev_output_emb = (
                F.interpolate(prev_emb.transpose(1, 2).float(), size=t_u, mode="linear", align_corners=False)
                .transpose(1, 2)
                .to(dtype)
            )

        fused = fusion(user_emb, prev_output_emb)  # (1, T_u, hidden)
        hidden, past_key_values = qwen_core.forward_step(fused, past_key_values)
        pred_mel = acoustic_head(hidden).transpose(1, 2).float()  # (1, n_mels, T_u)

        # No ground-truth agent audio exists at real inference time, so the
        # target mel frame count for this window is estimated from Vocos's own
        # verified frame-rate formula (center=True STFT), not measured from a
        # target we don't have.
        window_samples_24k = round(window.shape[0] / native_sr * VOCOS_SAMPLE_RATE)
        target_mel_frames = max(1, window_samples_24k // VOCOS_HOP_LENGTH + 1)
        pred_mel = F.interpolate(pred_mel, size=target_mel_frames, mode="linear", align_corners=False)

        waveform_window = vocos.waveform_from_mel(pred_mel).cpu().squeeze(0)  # (n_samples_24k,)
        raw_min, raw_max = waveform_window.min().item(), waveform_window.max().item()
        if raw_min < -1.0 or raw_max > 1.0:
            print(f"  [window {i + 1}] Vocos output out of [-1, 1] before clamping: "
                  f"min={raw_min:.3f} max={raw_max:.3f} -- this is exactly the corruption this clamp prevents.")
        waveform_window = waveform_window.clamp(-1.0, 1.0)
        synthesized_chunks.append(waveform_window)

        prev_output_wave_native = torchaudio.functional.resample(waveform_window, VOCOS_SAMPLE_RATE, native_sr)

        if (i + 1) % 10 == 0 or i == n_windows - 1:
            print(f"  window {i + 1}/{n_windows} done")

    full_output = torch.cat(synthesized_chunks, dim=0).unsqueeze(0)
    out_path = out_dir / "generated_agent.wav"
    torchaudio.save(str(out_path), full_output, VOCOS_SAMPLE_RATE)

    user_out_path = out_dir / "input_user.wav"
    torchaudio.save(str(user_out_path), user_wave.unsqueeze(0), native_sr)

    print(f"\nSaved: {user_out_path}  (what the model heard)")
    print(f"Saved: {out_path}  (what the model generated -- self-feedback autoregressive, NOT teacher-forced)")
    print("Listen to both. This is the honest test of what Stage 2 actually learned, exposure bias and all --")
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
