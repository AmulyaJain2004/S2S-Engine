"""Stage 2 inference / listening script: autoregressive duplex generation
from a trained checkpoint. This is the only way to actually hear what
Stage 2 learned -- a falling training loss alone does not tell you the
output is intelligible or that duplex timing makes sense.

DISCRETE-UNIT GENERATION (matches train/stage2_duplex.py's architecture --
see speechcore/discrete_tokenizer.py's docstring for the full reasoning):
the user audio is encoded into discrete unit IDs by the same frozen
WavLM-large + k-means(1000) pipeline used in training, embedded, and fed
through Qwen window-by-window with a KV cache. The classifier head picks
the most likely next agent unit ID at each frame (argmax -- greedy
decoding; see --sample for an alternative). Predicted unit IDs are
accumulated across ALL windows and decoded to waveform in ONE call at the
end, since the pretrained UnitHiFiGAN vocoder isn't chunk-limited the way
Vocos was -- this also means no crossfade-stitching hack is needed
anymore (removed, not left in as dead code).

STILL NO SELF-FEEDBACK AUDIO LOOP: only the user stream is ever encoded
and fed into the model, in both training and here -- the speech core's
own causal self-attention (carried across windows via the Qwen KV cache)
provides whatever "memory of what I've been saying" it needs.

WINDOWED, NOT PER-FRAME (still a deliberate choice, for the same reason as
before): WavLM has a multi-frame receptive field, so true 20ms-frame-by-
frame streaming still isn't practical for the ENCODE side -- generation
proceeds in `--window_s` windows (default 0.4s, matching spec section 2's
own accepted ~300-500ms chunk-latency trade-off) for encoding and KV-cached
Qwen stepping. The DECODE side (vocoding) is not windowed at all.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ensure_dir, load_config, pick_device
from speechcore.discrete_tokenizer import VOCODER_OUTPUT_SAMPLE_RATE
from train.stage2_duplex import build_models  # reuse the EXACT same model construction as training


def load_trained_checkpoint(ckpt_dir: Path, qwen_core, embedding, unit_head, device) -> int:
    state_path = ckpt_dir / "train_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"No checkpoint at {state_path} -- train first (train/stage2_duplex.py).")
    state = torch.load(state_path, map_location=device)
    embedding.load_state_dict(state["embedding"])
    unit_head.load_state_dict(state["unit_head"])

    from peft import PeftModel

    qwen_core.model = PeftModel.from_pretrained(qwen_core.model.get_base_model(), str(ckpt_dir / "qwen_lora"))
    qwen_core.model.to(device)
    qwen_core.model.eval()
    print(f"Loaded checkpoint from step {state['step']} at {ckpt_dir}")
    return state["step"]


@torch.no_grad()
def generate(
    cfg_path: str, user_wav_path: str, window_s: float, max_duration_s: float, out_dir: str, sample: bool = False
) -> None:
    cfg = load_config(cfg_path)
    device = pick_device(cfg["runtime"]["device"])
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg["runtime"]["dtype"]]
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    out_dir = ensure_dir(out_dir)

    tokenizer, qwen_core, embedding, unit_head = build_models(cfg, device, dtype)
    step = load_trained_checkpoint(ckpt_dir, qwen_core, embedding, unit_head, device)
    for m in (embedding, unit_head):
        m.eval()

    user_wave, native_sr = torchaudio.load(user_wav_path)
    user_wave = user_wave.mean(dim=0)  # mono
    if max_duration_s is not None:
        user_wave = user_wave[: int(max_duration_s * native_sr)]

    window_len = int(window_s * native_sr)
    n_windows = max(1, (user_wave.shape[0] + window_len - 1) // window_len)

    past_key_values = None
    predicted_units = []

    print(f"Generating {n_windows} window(s) of {window_s}s each from checkpoint step {step} "
          f"(discrete units, KV-cached Qwen, no self-feedback loop)...")
    for i in range(n_windows):
        start = i * window_len
        window = user_wave[start : start + window_len]
        if window.shape[0] == 0:
            break

        user_units = tokenizer.encode(window, native_sr).unsqueeze(0)  # (1, T_u)
        user_emb = embedding(user_units)  # (1, T_u, hidden)

        hidden, past_key_values = qwen_core.forward_step(user_emb, past_key_values)
        logits = unit_head(hidden).float()  # (1, T_u, vocab)

        if sample:
            probs = torch.softmax(logits, dim=-1)
            step_units = torch.multinomial(probs.squeeze(0), num_samples=1).squeeze(-1)
        else:
            step_units = logits.squeeze(0).argmax(dim=-1)  # (T_u,) greedy

        predicted_units.append(step_units.cpu())

        if (i + 1) % 10 == 0 or i == n_windows - 1:
            print(f"  window {i + 1}/{n_windows} done")

    full_units = torch.cat(predicted_units, dim=0)
    full_output = tokenizer.decode(full_units)  # one vocoder call over the whole sequence

    out_path = out_dir / "generated_agent.wav"
    torchaudio.save(str(out_path), full_output.unsqueeze(0).clamp(-1.0, 1.0), VOCODER_OUTPUT_SAMPLE_RATE)

    user_out_path = out_dir / "input_user.wav"
    torchaudio.save(str(user_out_path), user_wave.unsqueeze(0), native_sr)

    print(f"\nSaved: {user_out_path}  (what the model heard)")
    print(f"Saved: {out_path}  (what the model generated, {full_units.shape[0]} units decoded in one call)")
    print("Listen to both. This is the honest test of what Stage 2 actually learned --")
    print("a low training loss / high unit-accuracy does not by itself mean this sounds well-timed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2.yaml")
    parser.add_argument("--user_wav", required=True, help="A real user-channel WAV to feed the model.")
    parser.add_argument("--window_s", type=float, default=0.4,
                         help="Window size in seconds -- the spec's own accepted ~300-500ms chunk latency.")
    parser.add_argument("--max_duration_s", type=float, default=30.0,
                         help="Cap how much of --user_wav to process (generation is sequential, not batched).")
    parser.add_argument("--out_dir", default="outputs/generation")
    parser.add_argument("--sample", action="store_true",
                         help="Sample from the unit distribution instead of greedy argmax (more varied, less stable).")
    args = parser.parse_args()
    generate(args.config, args.user_wav, args.window_s, args.max_duration_s, args.out_dir, args.sample)
