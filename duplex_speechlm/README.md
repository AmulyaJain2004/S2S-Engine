# duplex_speechlm

End-to-end (no text bridge) duplex speech-to-speech prototype.
Architecture: WavLM (frozen) -> Projector (trainable) -> Qwen2.5-1.5B-Instruct
speech core (fine-tuned) -> Acoustic head (trainable) -> Vocos (frozen).

## Current status: Stage 2 training scaffolding in place. P0 must still be
## verified by ear per run/environment before trusting any training results.

- Repo scaffold (this layout)
- A real WavLM encoder wrapper (`encoder/wavlm_encoder.py`)
- A real Vocos vocoder wrapper with exact, verified mel matching (`vocoder/`)
- A real Projector architecture (`projector/projector.py`)
- Qwen2.5-1.5B-Instruct loader with the embedding-path override (accepts
  continuous `inputs_embeds`, never calls the pretrained `lm_head`) and
  optional LoRA wrapping (`speechcore/qwen_speech_core.py`)
- Duplex fusion: projects the user-frame embedding to the speech core's
  hidden size (`speechcore/duplex_fusion.py`) -- see "corrected design"
  below, this is NOT the original concat-with-previous-output design
- A real acoustic head: Conv1D temporal context + a residual postnet
  refinement (Tacotron2-style), not a bare linear readout
  (`acoustichead/mel_head.py`)
- An empty placeholder for personalization memory, explicitly Stage 4 work (`memory/`)
- otoSpeech dataset inspection + full-manifest (resumable, all-sessions)
  preprocessing (`dataprep/preprocess_otospeech.py`)
- Stage 2 training loop (`train/stage2_duplex.py`, `configs/stage2.yaml`),
  skipping Stage 1's synthetic content-bootstrap by direct instruction --
  trains directly on real otoSpeech duplex audio
- Stage 2 inference/listening script (`eval/generate_stage2.py`): windowed,
  KV-cached generation using the exact same mechanism as training -- the
  only way to actually hear what Stage 2 learned, since training loss alone
  doesn't tell you that
- A Colab A100 notebook (`notebooks/colab_stage2_a100.ipynb`) and an H100
  PBS-cluster job folder (`hpc/`), both driving the same training AND
  generation scripts

**Corrected design (a real bug, not a flagged gap -- read
`speechcore/duplex_fusion.py`'s docstring for the full reasoning):** duplex
fusion originally concatenated the user-frame embedding with the real
ground-truth agent audio's embedding, shifted by one 20ms frame. Adjacent
real-speech frames are highly autocorrelated, so this gave the model an
easy training shortcut -- echo the ground truth forward -- instead of
learning genuine content generation. It also meant training and real
inference used different mechanisms (inference has no ground-truth agent
audio), which is why generation-time self-feedback produced persistent
noise that no amount of inference-side signal processing (clamping,
loudness-matching, crossfading -- all tried, none fixed it) could resolve.
Confirmed by a direct test: a one-shot forward pass fed real ground-truth
audio produced actual intelligible words, proving the model/training
weren't the bottleneck -- the fusion mechanism was. Fixed by dropping the
explicit "previous output" input entirely; the speech core's own causal
self-attention over its own past hidden states now carries that memory,
the standard way any autoregressive transformer conditions on its own
history. **This was a breaking change -- any checkpoint trained before
this fix is incompatible and must be retrained from scratch.**

**Previously flagged gaps, resolved -- read the comments at the top of
`train/stage2_duplex.py` for the full reasoning:**
- WavLM's ~50Hz and Vocos's real ~93.75Hz mel rate have no integer ratio.
  Resolved by resampling the acoustic head's output along time, per item,
  to that item's own true mel length (computed from the real masks below,
  not a batch-wide approximation) -- this is a legitimate rate bridge
  between two fixed, externally-defined rates, not a quality shortcut.
- Training chunks are variable-length, not fixed-and-dropped: each
  session's trailing remainder is kept (down to `audio.min_chunk_s`), and
  `WavLMEncoder` now returns a real frame-level attention mask (via HF's
  own conv-stride-aware helper) that's threaded through the speech core's
  attention and the mel loss, so padded frames never leak into gradients.
- `user_channel`/`agent_channel` are no longer a silent guess:
  `train/stage2_duplex.py` now hard-refuses to start unless
  `audio.channel_roles_verified: true` is set, which should only happen
  after running `dataprep/print_channel_roles.py` against a real manifest
  and reading the printed speaker roles yourself.

**Still open:**
- The Stage 0 round-trip has not been verified by ear in this repo's
  history; it must be confirmed again whenever this runs on new infra
  (the Colab notebook's Section 4 is a hard gate for this).
- Still not here: Stage 1 (synthetic content bootstrap), personalization
  memory training, vocoder fine-tuning, compute gating, an ASR model of any
  kind.

## Requires network + GPU you don't have in a sandbox

This code needs Hugging Face Hub access (gated otoSpeech dataset + model
downloads) and ideally a GPU. Run it on Kaggle or Lightning.ai, not in an
isolated sandbox.

## Running P0

```bash
pip install -r requirements.txt
huggingface-cli login   # needed for the gated otoSpeech dataset access

# 1. Plumbing check: WavLM + Vocos round trip on a real WAV clip
python scripts/p0_roundtrip.py --wav /path/to/short_clip.wav
# Then ACTUALLY LISTEN to outputs/p0_original.wav vs outputs/p0_reconstructed.wav.

# 2. Inspect the real otoSpeech schema (one sample, streamed, no full download)
python dataprep/inspect_otospeech.py

# 3. Fill in the SCHEMA MAPPING section at the top of dataprep/preprocess_otospeech.py
#    using what step 2 actually printed. Then:
python dataprep/preprocess_otospeech.py
```

## Stop condition

Per the spec: do not proceed to Stage 1 (content-bootstrap training) until
P0's round trip is verified by ear, not just by the script exiting cleanly.
