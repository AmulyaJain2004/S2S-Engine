# duplex_speechlm

End-to-end (no text bridge) duplex speech-to-speech prototype.

**Architecture (current, Stage 2): discrete units, not continuous mel
regression.** User and agent audio are both encoded into discrete unit IDs
by a frozen, pretrained WavLM-large + k-means(1000) pipeline
(`speechcore/discrete_tokenizer.py`, via SpeechBrain's `DiscreteSSL`).
Qwen2.5-1.5B-Instruct (LoRA-adapted) is trained as an ordinary
autoregressive classifier -- embed the user's units, predict the agent's
units via cross-entropy -- and a pretrained, frozen UnitHiFiGAN vocoder
decodes predicted units back to waveform.

**Why discrete units, researched not assumed:** every full-duplex /
speech-LM system validated in the literature at scale -- dGSLM
(arXiv:2203.16502, HuBERT units + dual-tower LM + HiFi-GAN), Moshi (Kyutai,
Mimi RVQ codec + LM), SpeechGPT, VALL-E -- represents speech as discrete
tokens predicted via cross-entropy, not continuous vectors regressed via
L1/L2 loss. An earlier version of this project's Stage 2 used continuous
WavLM features + a mel-regression acoustic head, which the literature
(MELLE, arXiv:2407.08551; Ren et al., "Revisiting Over-Smoothness in TTS")
documents as prone to over-smoothed, blurry output -- deterministic
regression averages over plausible detail. Switching to discrete units
removes that failure mode structurally (classification doesn't average)
and matches the proven recipe instead of the harder, less-validated one.

**Why this stays feasible on a single Colab A100 session (the actual
constraint):** the k-means quantizer and the unit-to-waveform vocoder are
PRETRAINED, PUBLISHED checkpoints (`speechbrain/SSL_Quantization`,
`speechbrain/hifigan-wavlm-k1000-LibriTTS`) -- loaded frozen, never
trained here. Training a neural audio codec or GAN vocoder from scratch
would not fit this project's compute budget; reusing validated pretrained
ones (the same strategy dGSLM/GSLM use) is what makes the discrete-token
approach actually practical, not just theoretically better. Only the unit
embedding, LoRA adapters, and classification head are trained.

## Current status

- Repo scaffold (this layout)
- Discrete tokenizer/vocoder wrapper: WavLM-large -> k-means(1000) ->
  UnitHiFiGAN, all pretrained and frozen (`speechcore/discrete_tokenizer.py`)
- Unit embedding (`projector/unit_embedding.py`) and unit classification
  head (`acoustichead/unit_head.py`) -- the only new components trained
  for Stage 2, replacing the old continuous Projector/DuplexFusion/MelHead
- Qwen2.5-1.5B-Instruct loader with the embedding-path override (accepts
  continuous `inputs_embeds`, never calls the pretrained `lm_head`) and
  optional LoRA wrapping (`speechcore/qwen_speech_core.py`) -- unchanged
  by the discrete-unit pivot, since this part was already correct
- An empty placeholder for personalization memory, explicitly Stage 4 work (`memory/`)
- otoSpeech dataset inspection + full-manifest (resumable, all-sessions)
  preprocessing (`dataprep/preprocess_otospeech.py`)
- Stage 2 training loop (`train/stage2_duplex.py`, `configs/stage2.yaml`):
  cross-entropy over discrete units, skipping Stage 1's synthetic
  content-bootstrap by direct instruction -- trains directly on real
  otoSpeech duplex audio
- Stage 2 inference/listening script (`eval/generate_stage2.py`): windowed,
  KV-cached generation, decoding the full predicted unit sequence to
  waveform in one call -- the only way to actually hear what Stage 2
  learned, since training metrics alone don't tell you that
- `scripts/p0_discrete_roundtrip.py`: the Stage 0 plumbing check for the
  CURRENT architecture (supersedes `scripts/p0_roundtrip.py`, which only
  checked the old Vocos-mel pipeline)
- A Colab A100 notebook (`notebooks/colab_stage2_a100.ipynb`) and an H100
  PBS-cluster job folder (`hpc/`), both driving the same training AND
  generation scripts

## History: two real problems found and fixed, in order

1. **Duplex-fusion training/inference mismatch (fixed before the discrete
   pivot).** The original fusion design concatenated the user-frame
   embedding with the real ground-truth agent audio's embedding, shifted
   by one 20ms frame. Adjacent real-speech frames are highly
   autocorrelated, so this gave the model an easy training shortcut --
   echo the ground truth forward -- instead of learning genuine content
   generation. Confirmed directly: a one-shot forward pass fed real
   ground-truth audio produced actual intelligible words, proving the
   model/training weren't the bottleneck -- the fusion mechanism was.
   Fixed by dropping the explicit "previous output" input entirely; the
   speech core's own causal self-attention over its own past hidden
   states carries that memory instead, the standard way any
   autoregressive transformer conditions on its own history.
2. **Over-smoothing inherent to continuous mel regression (fixed by the
   discrete-unit pivot, not by patching the regression further).** Even
   after fix #1, plain L1 mel regression is documented in the literature
   to produce blurry/muffled output regardless of training quality. A
   spectral-flux loss term was tried as a cheap mitigation; the real fix,
   consistent with how validated systems (dGSLM, Moshi) are built, was to
   stop regressing continuous targets at all and switch to discrete units
   + cross-entropy, which is what this repo now does.

Every checkpoint from either earlier version is incompatible with the
current code and must be retrained from scratch.

**Still open:**
- The Stage 0 round-trip has not been verified by ear against the
  discrete pipeline in this repo's history; it must be confirmed whenever
  this runs on new infra (the Colab notebook's Section 4 is a hard gate
  for this).
- The pretrained vocoder was trained on LibriTTS/general speech, not
  otoSpeech specifically -- domain mismatch is a real, unverified risk
  Section 4's listening check is meant to catch early.
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

# 1. Plumbing check: WavLM-large -> k-means -> UnitHiFiGAN round trip on a real WAV clip
python scripts/p0_discrete_roundtrip.py --wav /path/to/short_clip.wav
# Then ACTUALLY LISTEN to outputs/p0_discrete_original.wav vs outputs/p0_discrete_reconstructed.wav.

# 2. Inspect the real otoSpeech schema (one sample, streamed, no full download)
python dataprep/inspect_otospeech.py

# 3. Fill in the SCHEMA MAPPING section at the top of dataprep/preprocess_otospeech.py
#    using what step 2 actually printed. Then:
python dataprep/preprocess_otospeech.py
```

(`scripts/p0_roundtrip.py`, the older Vocos-based check, still runs but no
longer tests anything the current architecture depends on.)

## Stop condition

Per the spec: do not proceed to Stage 1 (content-bootstrap training) until
P0's round trip is verified by ear, not just by the script exiting cleanly.
