# Live inference web UI — archived Stage 2 checkpoint

A standalone FastAPI + vanilla HTML/CSS/JS app for trying out the
**original** Stage 2 checkpoint (the step-20000 run, trained before the
duplex-fusion leakage fix and before the discrete-units pivot — see
`duplex_speechlm/README.md`'s history section).

**This folder is fully self-contained and deliberately decoupled from
`duplex_speechlm/`.** `backend/model_arch.py` is a frozen copy of the
model classes exactly as they existed when this checkpoint was trained
(git commit `7406dc4`). Changes to the main training pipeline — the
leakage fix, the discrete-units pivot, whatever comes next — never touch
this folder, and this folder never touches them. If you retrain the
discrete-units architecture later, that needs its own, different
inference app (the model shapes are completely different).

## What you need to download and where to put it

From your Google Drive (`S2S-Engine-runs/checkpoints/stage2/`, the run
that hit step 20000), download these two things and place them under
`inference_webapp/checkpoint/`:

```
inference_webapp/
  checkpoint/
    train_state.pt      <- contains step, projector, fusion, acoustic_head, optimizer state dicts
    qwen_lora/           <- the whole folder (PEFT LoRA adapter: adapter_config.json, adapter_model.safetensors, etc.)
```

Both come from the SAME training run — don't mix `train_state.pt` from
one checkpoint with `qwen_lora/` from another; the LoRA adapter and the
projector/fusion/acoustic_head weights were trained together and must be
loaded together.

If you'd rather put the checkpoint somewhere else (e.g. a mounted Drive
path), set `CHECKPOINT_DIR` to that path instead of moving files — see
below.

## Setup

```bash
cd inference_webapp/backend
pip install -r requirements.txt
```

You'll also need `huggingface-cli login` (or `HF_TOKEN` set) once, since
this downloads `microsoft/wavlm-base-plus`, `Qwen/Qwen2.5-1.5B-Instruct`,
and `charactr/vocos-mel-24khz` from the Hub the first time it runs (same
models the training pipeline used — no gated access needed for any of
these three, unlike otoSpeech).

## Running it

```bash
# from inference_webapp/backend/
export CHECKPOINT_DIR=../checkpoint   # default; only set this if you put it elsewhere
export DEVICE=cuda                     # falls back to cpu automatically if no GPU is available
uvicorn main:app --host 0.0.0.0 --port 8000
```

(On Windows PowerShell: `$env:CHECKPOINT_DIR = "../checkpoint"` instead of `export`.)

Then open **http://localhost:8000** in a browser. The page is served
directly by the same FastAPI process — nothing else to start.

The first startup loads WavLM, Qwen2.5-1.5B, Vocos, and your checkpoint,
which takes a few minutes depending on your connection and whether the
models are already cached locally (`~/.cache/huggingface`). Watch the
server console; `/api/health` on the page will show "Model ready" once done.

## Using it

1. Upload a WAV (any real speech clip — ideally something like the
   held-out otoSpeech user-channel clips, since that's the real domain
   this was trained on) or record one from your mic.
2. Adjust window size / max clip length if you want (defaults match what
   was used during the actual training run: 0.4s windows, up to 30s).
3. Click Generate. A progress bar tracks window-by-window generation
   server-side; when done, both the input and generated audio are
   playable side by side.

## What to actually expect

This checkpoint was trained with a duplex-fusion design later found to
let the model shortcut training by echoing ground-truth audio instead of
learning genuine independent generation (full story in
`duplex_speechlm/README.md`). Generation here still uses the
self-feedback loop with the clamp / loudness-matching / crossfade
mitigations that were the best fix available for this specific
architecture before it was superseded — real, but likely rough,
intelligibility. Don't take this as the project's ceiling; it's a
snapshot of one specific, since-corrected training run, kept runnable
for comparison.

## Troubleshooting

- **`/api/health` says "Checkpoint not found at ..."**: the path printed
  is exactly where the server looked. Confirm `train_state.pt` and
  `qwen_lora/` both exist directly inside it (not nested one level deeper
  from however your Drive download unpacked).
- **CUDA out of memory**: set `DEVICE=cpu`. It'll be much slower
  (WavLM + Qwen2.5-1.5B forward passes per window, on CPU) but will run.
- **Mic recording button does nothing**: browsers only grant microphone
  access over `https://` or `http://localhost` — if you're hitting this
  from another machine's IP, use the file-upload path instead, or set up
  TLS.
