# English duplex speech LM — MVP spec & build plan

2026-09-21 · @Someone

## 1. Scope and non-goals

**In scope for this MVP:** an English-only, end-to-end (no text bridge) speech-to-speech model, following the duplex pretrained-component architecture: speech encoder -> projector -> duplex speech core -> personalization memory conditioning -> acoustic head -> vocoder. Goal is a working proof of concept with short, personalized, looping exchanges, not production quality, not multilingual, not a paper-grade duplex benchmark score.

**Explicitly out of scope for now:** Hindi/multilingual support (revisit after English works), the cascaded ASR-LLM-TTS design from earlier discussion (superseded by this end-to-end decision), tool-calling/agentic actions, and the internal frame-level compute-skip gate (deferred until correctness is proven first, to avoid premature optimization).

**Definition of done for this phase:** the model holds a short (10-30 second) looped exchange, producing intelligible English speech, with measurably non-zero backchannel/overlap behavior learned from real conversational data (not just clean alternating turns), and with response/interruption latency in the same bounded \~200-300ms range accepted as the project's latency target in Section 2 -- this was the original motivating constraint and belongs in the definition of done, not just in the encoder's design notes.

## 2. Final architecture (with corrections)

This section states exactly what each block does and, where an earlier draft was ambiguous or wrong, what was corrected and why.

![End-to-end duplex speech LM architecture: speech encoder to projector to duplex speech core (conditioned on personalization memory) to acoustic head to vocoder](blob/2ded15f4-cc48)

**Speech encoder (frozen).** WavLM-base-plus, 16kHz mono input, outputs 50Hz continuous features (768-dim). Correction: a true end-to-end duplex model needs streaming-friendly input, but WavLM is not strictly causal (it has some internal lookahead). Resolution: process audio in fixed chunks (roughly 300-500ms) with a small overlap buffer rather than requiring a literally zero-lookahead model. A bounded \~200-300ms latency is an accepted engineering trade-off, not a violation of "full duplex" -- it is comparable to normal human response latency, and it is what lets us use an actually-available pretrained encoder instead of searching for an exotic fully-causal one.

**Projector (trainable).** Conv1D -> MLP -> RMSNorm, maps WavLM's 768-dim output to the speech core's hidden size.

**Personalization memory (trainable, small).** Not a text prefix (there is no text anywhere in this pipeline). It is a small set of learned conditioning vectors, updated after each session from a compact rolling state, injected into the speech core as a prefix (prefix-tuning style), not as natural language.

**Speech core (fine-tuned LLM, duplex).** Correction from the original from-scratch plan: do not reimplement GQA/RoPE/SwiGLU by hand and try to port pretrained weights into it. Instead, load the actual Hugging Face model class directly (Qwen2.5-1.5B-Instruct, Apache-2.0) and repurpose its input embedding path to accept continuous projector/memory embeddings instead of discrete token IDs, and replace its LM head with the acoustic head. This is far lower engineering risk than hand-porting weights into custom modules, and it is the correct way to "warm-start" from a pretrained LLM in practice.

Duplex fusion, concrete first pass: at every timestep, concatenate the incoming user-frame embedding with the speech core's own previously generated frame embedding, then pass through a single learned linear projection down to the model's hidden size before feeding it in. This is deliberately the simplest possible fusion function, not the final design -- it exists so Stage 2 has something concrete to run and iterate from, rather than blocking implementation on finding the "right" fusion mechanism first.

**Frame rate consistency (a concrete fix).** Everything runs at WavLM's native 50Hz (20ms hop) rather than inventing a separate custom frame rate. This avoids needing any up/downsampling logic between the encoder, the speech core, and the acoustic head's mel targets.

**Acoustic head (trainable).** Predicts mel-spectrogram frames from the speech core's hidden state, at 20ms hop to match the 50Hz rate above.

**Vocoder (frozen, then lightly fine-tuned) -- corrected pick.** The earlier draft named Kokoro here. That was a mismatch: Kokoro is a full text-to-speech system, not a standalone mel-to-waveform vocoder, so it cannot take our acoustic head's predicted mel frames as input. The correct component for this role is **Vocos** (`charactr/vocos-mel-24khz`, MIT license) -- a real standalone, fast, single-forward-pass mel-to-waveform vocoder built for exactly this purpose. Kokoro is still useful elsewhere (see Section 4, synthetic data generation), just not here.

**Critical constraint to enforce in code:** the mel-spectrogram extraction parameters used to compute training targets (n\_fft, hop length, n\_mels, fmin/fmax, sample rate) must exactly match Vocos's own expected mel configuration. Do not invent independent mel parameters -- copy Vocos's exact preprocessing config, or the vocoder will produce noise regardless of how well the acoustic head trains.

## 3. Pretrained models to download

| Model | Role | Source | License | Size | Frozen or fine-tuned |
| --- | --- | --- | --- | --- | --- |
| WavLM-base-plus | Speech encoder | `microsoft/wavlm-base-plus` on Hugging Face | MIT | \~95M params | Frozen |
| Qwen2.5-1.5B-Instruct | Speech core backbone (warm start) | `Qwen/Qwen2.5-1.5B-Instruct` on Hugging Face | Apache-2.0 | \~1.5B params | Fine-tuned (LoRA first, then partial unfreeze) |
| Vocos (mel, 24kHz) | Vocoder | `charactr/vocos-mel-24khz` on Hugging Face, or `pip install vocos` | MIT | \~14M params | Frozen initially, fine-tuned in stage 2 |
| Kokoro-82M | Synthetic data generation only (not part of the runtime model) | `hexgrad/Kokoro-82M` on Hugging Face | Apache-2.0 | 82M params | Not trained; used offline as a data-prep tool |

All four are small enough to load comfortably on a single T4/L4. Total combined footprint is well under what a 16GB GPU can hold for inference; training will need gradient/optimizer memory on top, budgeted for in Section 6.

## 4. Datasets to download

A real logical gap this section fixes: earlier discussion assumed synthetic TTS-bootstrapped dialogue would be enough training data. It is enough to teach content and turn-by-turn coherence, but it is **not** enough to teach genuine duplex behavior (backchannels, interruptions, overlaps), because synthetic TTS turns are generated one at a time with no natural overlap timing. This phase therefore needs two different kinds of data for two different purposes.

### Phase A data -- content/semantic bootstrap (synthetic, non-duplex)

- An existing open English text dialogue/instruction dataset (any reasonably sized open instruction-tuning set works here).
- Synthesize both sides of each exchange with **Kokoro-82M** (or Piper as a CPU-only fallback) to produce raw training audio.
- Purpose: teach the projector, speech core, and acoustic head to produce coherent, intelligible speech content before duplex dynamics are introduced at all.

### Phase B data -- real duplex fine-tuning (authentic overlap/backchannel dynamics)

This shortlist was checked again directly against the live Hugging Face pages (not just secondhand claims), which confirmed some numbers, corrected others, and left one important one unconfirmed. Ranked by what's actually verified and accessible right now:

- **OcularAI American English Full-Duplex Two-Speaker Dataset -- independently verified.** Fetched directly from its Hugging Face card: 954 conversations, \~230 hours of conversation audio (\~455 hours of isolated per-speaker audio), CC-BY-NC-4.0 (research/non-commercial), gated by a simple access-request form, and only \~5.56GB total file size (Opus-compressed) -- comfortably small for Lightning's free storage. This is now the clear best first real-data source: request the \~5 hour free preview sample immediately, and request full access in parallel since it's free either way.
- **otoSpeech-full-duplex-task-oriented-20h** -- confirmed real and accessible the same gated (free, contact-info) way as before: 20 hours, English, two-speaker, 48kHz channel-separated, with synchronized interaction-event annotations, seven task types. License terms should be double-checked on the live card at request time.
- **otoSpeech-full-duplex-turn-104h -- real but only partially verifiable right now.** Directly confirmed: this dataset exists, is gated the same way, and totals 290GB. Not yet confirmable: its dataset card is currently empty on Hugging Face, so the specific claims of 104.94 hours, a 17-label turn/backchannel/interruption annotation taxonomy, and non-commercial license terms with derived-data restrictions could not be independently verified at the time of this check. Treat those specific numbers as plausible but unconfirmed until the card is populated or access is granted -- worth requesting access now, verifying on arrival, before planning Stage 2's scale-up run around it.
- **Multi-stream Spontaneous Conversation Training Dataset (English)** -- MagicHub, free, real, small. Its hour count is reported inconsistently across secondhand sources (roughly 15h total across English+Chinese by one account versus a 5h English-only sample by another) -- confirm the actual figure on the live MagicHub page before relying on it, the same direct-verification standard applied to OcularAI and otoSpeech above.
- **HumDial-FDBench** -- evaluation/benchmark use only, not training data (the underlying 100+ hour training set is challenge-participant-only).
- **Hume-DaiKon** -- large (743.4h, 5 languages including English) newly surfaced lead; access/license unverified, worth checking.
- **AMI Meeting Corpus** -- now clearly deprioritized below the options above; keep only as a fallback if the OcularAI/otoSpeech path stalls.
- **Fisher English corpus** -- unchanged: literature-standard, LDC-licensed, optional stretch.
- **Fallback: record a small custom corpus** -- unchanged, still a legitimate last resort using the HumDial-style protocol described in the previous revision.

**Required normalization across sources.** OcularAI, otoSpeech, MagicHub, and any custom recording will each ship different metadata formats. Before Stage 2 can use them together, every source needs converting to one internal schema: a per-20ms-frame label of `user_active` / `agent_active` / `overlap` / `pause`, derived from each source's own channel/timestamp data. This is also exactly the signal the duplex fusion mechanism (Section 2) needs to learn from, so it is not extra work invented for its own sake -- it is the actual training target shape.

**Staged data volume targets** (engineering judgment, not a literature-defined threshold -- no paper claims a fixed number of hours is "enough"):

| Amount | Purpose |
| --- | --- |
| \~5h | Pipeline/prototype correctness check |
| \~20-30h | First reasonable MVP experiment |
| \~50-100h | A meaningfully healthier duplex fine-tune |
| 100h+ | Better diversity/coverage, if compute allows later |

Given the speech core warm-starts from a 1.5B pretrained backbone rather than learning language from zero, the 20-30h tier is a reasonable first serious target -- the model mainly needs to adapt existing knowledge to audio-in/audio-out and to interaction dynamics, not learn language itself from this data.

**One caution carried over directly:** don't manufacture overlap by shifting and mixing independently recorded tracks as the primary Stage 2 signal. That teaches the acoustic pattern of overlap without the natural human timing that makes it meaningful -- fine as a minor augmentation on top of real overlap, not as a substitute for it.

**Concrete instantiation of the table above, given what's actually verified accessible:**

- Run 1 (\~5h): OcularAI free preview sample alone.
- Run 2 (\~25h): OcularAI sample + otoSpeech-20h.
- Run 3 (\~100h+): add otoSpeech-104h once its access and card contents are verified (see the unconfirmed-numbers note above), or the OcularAI full 230h dataset as an alternative/additional scale-up path -- both are free, so the choice between them can wait until both are actually in hand.

Comparing model behavior (backchannel rate, overlap handling, interruption response, latency) across these three runs turns this from "did I build a voice bot" into a small, genuine empirical question worth reporting on.

### What NOT to download

No ASR dataset or ASR model is needed anywhere in this plan. That was specific to the cascaded design we moved away from. This end-to-end architecture never transcribes anything at inference time.

## 5. Compute and account management

**Primary: Lightning.ai, multiple free accounts, one per pipeline stage.**

- Account A -- data prep: stream/download datasets, run WavLM feature extraction and Vocos-mel target extraction, run the Kokoro synthetic-data generation for Phase A. Push processed caches (never raw audio) to a private Hugging Face Hub dataset repo.
- Account B -- training: pull processed data from the Hub, run the staged training in Section 7 on a long-running T4/L4 studio (unlimited session time on the free tier is the reason this works at all without constant restarts).
- Account C -- vocoder fine-tuning and evaluation, or a parallel experiment (for example, comparing two personalization-memory update rules) while B is busy.

**Secondary: Google Colab.** Used only for interactive debugging and the rare case an A100 burst is needed for a few hours (a short LoRA fine-tune) that Lightning's 4-hour A100/H100/H200 cap won't comfortably cover.

**Storage discipline (this will bite if ignored):** Lightning's free tier gives 10GB free / 50GB persistent storage per account. Never keep raw corpus audio on disk longer than it takes to extract features from it. Keep only: extracted features, mel targets, model checkpoints, and manifests. Cross-account handoff happens through a private Hugging Face Hub repo, not manual file copying between Gmail accounts.

## 6. Software and environment requirements

- Python 3.11
- `torch` (2.x, matching the CUDA build available on the Lightning/Colab GPU image)
- `transformers` (for WavLM and Qwen2.5-1.5B-Instruct)
- `peft` (LoRA fine-tuning of the speech core)
- `vocos` (`pip install vocos`)
- `datasets` and `huggingface_hub` (streaming datasets, pushing/pulling checkpoints and caches)
- `torchaudio`, `soundfile`, `librosa` (audio I/O and mel extraction -- mel parameters must match Vocos's config exactly, see Section 2)
- `kokoro` (Apache-2.0 package, for offline Phase A synthetic data generation only)
- `accelerate` (mixed precision, simple multi-GPU if ever available)
- A lightweight experiment logger (Weights & Biases free tier, or plain CSV/JSON logging if avoiding another account to manage)

No ASR library (faster-whisper, etc.) is needed -- that was specific to the cascaded design this project moved away from. No TTS-serving library is needed either; Kokoro is invoked only in offline data-prep scripts, never at inference time.

## 7. Training phases (data and models mapped per stage)

**Stage 0 -- plumbing check.** No training. Verify WavLM extraction, mel extraction (matching Vocos's config exactly), and Vocos reconstruction round-trip on a handful of real audio clips. If copy-synthesis through Vocos alone doesn't sound right, nothing downstream will either -- fix this before writing a single line of model training code.

**Stage 1 -- content bootstrap (Phase A data).** Train projector + speech core (LoRA on Qwen2.5-1.5B-Instruct) + acoustic head, teacher-forced against Vocos-matching mel targets from the synthetic Kokoro-generated pairs. Vocoder frozen. Goal: coherent, intelligible speech content, no duplex behavior expected yet.

**Stage 2 -- duplex fine-tune (Phase B data).** Continue training on real conversational data, following the run ladder in Section 4: Run 1 on the \~5h OcularAI free sample, Run 2 adding otoSpeech-20h (\~25h total), Run 3 scaling to otoSpeech-104h or the OcularAI full 230h set once verified in hand. At each run, feed both the user stream and the model's own prior output stream into the speech core via the duplex fusion in Section 2. This is the stage that actually teaches backchanneling/interruption/overlap -- it cannot be skipped or replaced with more synthetic data.

**Stage 3 -- vocoder fine-tune.** Fine-tune Vocos on the model's own predicted mels (not just ground truth) to close the train/inference mismatch, using the standard TTS-pipeline trick discussed earlier.

**Stage 4 -- personalization memory training.** Once the base duplex model is stable, train the memory module's update rule on repeated multi-session interactions (can reuse Stage 2 data reorganized into multi-session sequences, or newly collected data if available).

**Stage 5 (optional, later) -- internal frame-level compute gate.** Add the silence-skipping optimization from Section 1's non-goals once everything above works end to end.

**Fast-track option.** If the near-term goal is a convincing prototype rather than the full research-grade build, Stages 0-2 above (renamed P0-P2 for this purpose) are sufficient to demonstrate the core claim: a warm-started LLM mapping streaming speech to speech, with real backchannel/interruption behavior. Stages 3-5 (vocoder fine-tune, personalization, compute gate) can be deferred as explicit future work without weakening that claim -- they improve quality and efficiency, but P0-P2 is where the architecture is actually proven or disproven.

## 8. Repository structure

```
duplex_speechlm/
  encoder/        WavLM loading + chunked streaming wrapper
  projector/      Conv1D -> MLP -> RMSNorm adapter
  memory/         personalization conditioning module + rolling-state update rule
  speechcore/     Qwen2.5-1.5B-Instruct loading, embedding-path override, duplex fusion
  acoustichead/   mel-frame prediction head
  vocoder/        Vocos wrapper, mel-config matching, fine-tuning script
  dataprep/       Kokoro synthetic generation (Phase A), OcularAI/otoSpeech normalization (Phase B)
  train/          one script per stage (0-4), config-driven
  eval/           intelligibility check, duplex behavior diagnostics (backchannel rate, overlap handling, interruption response, stop/response latency, in the spirit of FullDuplexBench)
  configs/        one YAML per experiment/account
```

Config-driven from the start, since models/thresholds/data paths will change constantly across the Lightning.ai accounts described in Section 5.

## 9. Logical cross-verification: corrections made and open risks

This is a direct answer to "cross-verify it logically" -- these are the specific contradictions or gaps found while writing this plan, and how each was resolved.

1. **Vocoder mismatch (corrected).** Kokoro was wrongly proposed earlier as the vocoder. Kokoro is a full text-to-speech system, not a generic mel-to-waveform vocoder, and cannot consume our acoustic head's predicted mel frames. Fixed by switching the actual runtime vocoder to Vocos, and repositioning Kokoro as an offline synthetic-data-generation tool only.
2. **Strict causality vs. real availability of pretrained encoders (resolved as a trade-off, not an error).** No freely available pretrained SSL speech encoder is strictly causal. Rather than search for one that doesn't practically exist at this scale, the plan accepts a bounded \~200-300ms lookahead buffer, matching roughly human response latency, and keeps WavLM as the encoder.
3. **Frame-rate consistency (resolved).** Earlier diagrams implied a Mimi-style 12.5Hz rate left over from the cascaded-design discussion. Since Mimi is no longer part of this design, the plan standardizes everything on WavLM's native 50Hz to avoid unnecessary up/downsampling logic between encoder, speech core, and acoustic head.
4. **Mel-parameter consistency (flagged as a build-time risk, not yet resolved by design alone).** The acoustic head's training targets must use the exact same mel extraction config as Vocos expects. This is not automatic -- it has to be enforced in code by importing Vocos's own preprocessing function rather than writing an independent one. Flagged explicitly in Sections 2 and 7 (Stage 0) so it gets checked before any model training starts, not after.
5. **Synthetic data cannot teach duplex behavior (corrected).** The original assumption -- that TTS-bootstrapped synthetic dialogue would be sufficient training data -- was wrong for this specific end-to-end duplex architecture (it may have been adequate for the earlier cascaded design, where turn-taking was handled by an external mechanism). Real two-track conversational data with genuine overlaps is required for Stage 2, hence Section 4's Phase B sources.
6. **Warm-starting mechanism (corrected).** The earlier custom from-scratch transformer modules (RMSNorm/GQA/SwiGLU written by hand) are not the right vehicle for warm-starting from a pretrained LLM's actual weights -- porting weights into a hand-written reimplementation is high-risk and unnecessary. The corrected approach loads the real Hugging Face model class directly and only modifies its embedding input path and output head.
7. **Data source shortlist was incomplete (corrected).** The original Phase B shortlist (MagicHub, AMI, Easy Turn, Fisher) missed several more accessible free options: OcularAI's free preview samples, otoSpeech's request-gated releases, and a large newly surfaced lead (Hume-DaiKon). It also didn't catch that HumDial's public Hugging Face release is a benchmark/test split, not the training data. Section 4 now reflects the corrected, ranked shortlist and flags the MagicHub hour count as needing direct re-verification rather than trusting either secondhand figure.
8. **Duplex fusion mechanism (partially resolved).** Section 2 now specifies a concrete first-pass fusion (concatenate user-frame and prior-output embeddings, then a single linear projection) rather than leaving it fully open. This is explicitly a starting point to iterate from during Stage 2, not a final design.
9. **Secondhand dataset numbers need direct verification, not just a second secondhand source (corrected practice).** Checking a dataset's own live page directly caught real differences: OcularAI's actual dataset statistics (954 conversations, \~230h, CC-BY-NC-4.0, \~5.56GB) matched what was claimed and are now trustworthy. otoSpeech-full-duplex-turn-104h, by contrast, was confirmed to exist and be gated with a 290GB footprint, but its specific hour count, 17-label annotation taxonomy, and license terms could not be confirmed because its dataset card is currently empty -- those numbers are carried in this plan as unconfirmed, not verified, until the card is populated or access is granted.

**Open risks not fully resolved yet, to revisit once Stage 0/1 produce real results:**

- otoSpeech-full-duplex-turn-104h's exact hours, annotation taxonomy, and license need confirming once its card is populated or access granted, before Stage 2's scale-up run is planned around it in detail.
- The concrete first-pass duplex fusion (concatenate + linear) is a reasonable starting point but unproven; whether it's expressive enough to actually learn turn-taking, or needs something richer, is an open empirical question for Stage 2.
- With OcularAI's full 230h and otoSpeech's real-annotation datasets now confirmed accessible, the earlier concern about total real duplex data volume being too modest is substantially reduced -- but actual model behavior at each run in the ladder (Section 4) is still the thing that will confirm or deny this, not the hour counts alone.

## 10. Immediate next action items

1. Request access to all three verified/likely datasets now, since all are free and gated only by an access form: OcularAI's free preview sample, OcularAI's full 230h dataset, otoSpeech-full-duplex-task-oriented-20h, and otoSpeech-full-duplex-turn-104h (to check its actual card contents on arrival).
2. Create the Lightning.ai accounts (A/B/C) and a private Hugging Face Hub account/token for cross-account artifact sharing.
3. On Account A: install requirements from Section 6, download WavLM-base-plus and Vocos, run the Stage 0 (P0) plumbing check (encode -> mel-extract -> Vocos-reconstruct round trip on a few real clips) before anything else.
4. Once the OcularAI free sample arrives, write the per-source normalization script converting it into the unified per-20ms-frame `user_active`/`agent_active`/`overlap`/`pause` schema from Section 4 -- this becomes the template for normalizing the other sources as they arrive.
5. Set up the Kokoro-based Phase A synthetic data generation script.
6. Scaffold the repo structure from Section 8 with config-driven stage scripts, even before Stage 0 code is finished.
7. Proceed through P0 -> P1 -> P2 (Section 7's fast-track note) before deciding whether Stages 3-5 are worth building next, or whether to stop at a working prototype.
