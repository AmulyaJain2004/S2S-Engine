# duplex_speechlm

End-to-end (no text bridge) duplex speech-to-speech prototype.
Architecture: WavLM (frozen) -> Projector (trainable) -> Qwen2.5-1.5B-Instruct
speech core (fine-tuned) -> Acoustic head (trainable) -> Vocos (frozen).

## Current status: P0 (foundation + plumbing check). No training yet.

This repo currently implements the fast-track plan's **P0** stage only:

- Repo scaffold (this layout)
- A real WavLM encoder wrapper (`encoder/wavlm_encoder.py`)
- A real Vocos vocoder wrapper with exact, verified mel matching (`vocoder/`)
- A real Projector architecture, untrained (`projector/projector.py`)
- A Qwen2.5-1.5B-Instruct loader that verifies the pretrained model loads,
  with no training or embedding-path override yet (`speechcore/qwen_speech_core.py`)
- A placeholder acoustic head (`acoustichead/mel_head.py`)
- An empty placeholder for personalization memory, explicitly Stage 4 work (`memory/`)
- otoSpeech dataset inspection and one-session preprocessing scripts (`dataprep/`)

**What is deliberately NOT here yet:** Qwen/projector/acoustic-head training,
LoRA, duplex fusion, personalization, compute gating, an ASR model of any
kind, and full-dataset (20h) processing. See the project spec for why.

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
