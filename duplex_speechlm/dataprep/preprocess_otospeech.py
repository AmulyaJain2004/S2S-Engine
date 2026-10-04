"""Processes ONE otoSpeech session into:

    processed/
      session_<id>/
        speaker_0.wav
        speaker_1.wav
        metadata.json

IMPORTANT -- read before running: the JSON field names below (SESSION_ID_KEY,
EVENTS_KEY, etc.) are placeholders. Run dataprep/inspect_otospeech.py FIRST,
look at the real printed JSON structure, and fill in the actual key names in
the SCHEMA MAPPING section below. This script deliberately raises a clear
error rather than guessing field names, per the instruction not to invent
metadata that hasn't been observed.

Does not modify the raw dataset shards. Does not process more than one
session. Requires network access to huggingface.co, same as the inspection
script.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import soundfile as sf
import numpy as np
from datasets import Audio, load_dataset

REPO_ID = "otoearth/otoSpeech-full-duplex-task-oriented-20h"

# ============ SCHEMA MAPPING ============
# All confirmed directly against the dataset's own JSON Schema files
# (session-v1.schema.json, event-v1.schema.json) -- not guessed.
EVENTS_KEY = "events"
TASK_INFO_KEY = "task"
SPEAKERS_KEY = "speakers"              # exactly 2 entries, each with its own "channel" field (0 or 1)
REDACTION_SEGMENTS_KEY = "redacted_segments"   # session-level: [{"start_sec": ..., "end_sec": ...}, ...]
EVENT_REDACTED_FLAG = "redacted"       # per-event boolean, separate from the session-level segments above
EVENT_SUMMARY_KEY = "event_summary"    # {"total", "primary", "telemetry", "excluded_without_server_t"}

# NOT enumerable from the schema -- event "type" is deliberately an open string
# (minLength 1, no enum) in event-v1.schema.json. The actual backchannel/
# interruption/overlap vocabulary must be discovered empirically (see
# inspect_otospeech.py's vocabulary collector) or cross-checked against the
# task walkthrough page linked in the dataset README.
# ==========================================


class SchemaNotConfigured(RuntimeError):
    pass


def _preprocess_sample(sample: dict, repo_id: str, output_dir: str) -> dict:
    """Shared logic for turning one streamed WebDataset sample into a
    processed/session_<id>/ directory. Returns the processed_meta dict
    (also what gets written to metadata.json and folded into the manifest)."""
    raw_bytes = sample["flac"]["bytes"]
    raw_json = sample["json"]
    meta = json.loads(raw_json) if isinstance(raw_json, (bytes, str)) else raw_json

    # session_id is a required top-level field in the JSON itself -- confirmed
    # against session-v1.schema.json, more reliable than the WebDataset "__key__"
    # convention this script assumed before that schema was available.
    session_id = meta["session_id"]

    array_sc, sr = sf.read(io.BytesIO(raw_bytes), always_2d=True)  # (num_samples, num_channels)
    array = array_sc.T  # -> (num_channels, num_samples), matching the rest of this function

    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError(f"Expected 2-channel stereo audio, got shape {array.shape}.")

    # speakers: exactly 2 entries per session-v1.schema.json, each carrying its
    # own "channel" field. Match on that rather than assuming array position.
    speakers = meta[SPEAKERS_KEY]
    channel_to_speaker = {s["channel"]: s for s in speakers}
    if set(channel_to_speaker.keys()) != {0, 1}:
        raise ValueError(
            f"Expected speaker channels {{0, 1}}, found {set(channel_to_speaker.keys())} -- "
            f"this session's speaker data doesn't match the confirmed schema, investigate before proceeding."
        )

    # Mute (zero out) session-level redacted spans in BOTH channels, in place,
    # before writing the WAVs. This preserves timeline/duration alignment
    # while actually removing the content, per the dataset's usage terms
    # ("respect redacted segments... avoid reconstructing or inferring
    # removed content") rather than just noting the spans in metadata and
    # leaving the audio untouched.
    redacted_spans = meta.get(REDACTION_SEGMENTS_KEY, [])
    muted_array = array.copy()
    for span in redacted_spans:
        start_idx = int(span["start_sec"] * sr)
        end_idx = int(span["end_sec"] * sr)
        muted_array[:, start_idx:end_idx] = 0.0
    if redacted_spans:
        print(f"Muted {len(redacted_spans)} redacted segment(s) in both channels.")

    session_dir = Path(output_dir) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    speaker_0_path = session_dir / "speaker_0.wav"
    speaker_1_path = session_dir / "speaker_1.wav"
    sf.write(str(speaker_0_path), muted_array[0], sr)
    sf.write(str(speaker_1_path), muted_array[1], sr)

    # Drop events that are individually redacted (per-event "redacted" flag,
    # distinct from the session-level muted spans above), and separate
    # interaction-relevant events from bookkeeping/telemetry using the real
    # "kind" enum, rather than treating every event in the list as equally
    # relevant to duplex/turn-taking training.
    all_events = meta.get(EVENTS_KEY, [])
    usable_events = [e for e in all_events if not e.get(EVENT_REDACTED_FLAG, False)]
    interaction_events = [e for e in usable_events if e.get("kind") == "interaction" and e.get("is_primary")]

    event_summary = meta.get(EVENT_SUMMARY_KEY, {})
    if event_summary and event_summary.get("total") != len(all_events):
        print(f"WARNING: event_summary.total ({event_summary.get('total')}) does not match "
              f"len(events) ({len(all_events)}) -- investigate before trusting this session's event data.")

    duration_s = array.shape[1] / sr
    processed_meta = {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "source_repo": repo_id,
        "sample_rate": sr,
        "duration_seconds": duration_s,
        "speaker_0_wav": speaker_0_path.name,
        "speaker_1_wav": speaker_1_path.name,
        "task_info": meta.get(TASK_INFO_KEY),
        "speakers_by_channel": {str(k): v for k, v in channel_to_speaker.items()},
        "redacted_segments_muted": redacted_spans,
        "event_summary": event_summary,
        "num_events_total": len(all_events),
        "num_events_usable": len(usable_events),
        "num_events_interaction_primary": len(interaction_events),
        "interaction_events": interaction_events,  # payload_json inside each is still a raw string -- see note below
    }
    metadata_path = session_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(processed_meta, f, indent=2, default=str)

    # Basic validation
    assert speaker_0_path.exists() and speaker_1_path.exists(), "one or both speaker WAVs missing after write"
    dur0 = sf.info(str(speaker_0_path)).duration
    dur1 = sf.info(str(speaker_1_path)).duration
    assert abs(dur0 - dur1) < 0.01, f"speaker track durations misaligned: {dur0}s vs {dur1}s"

    print(f"session {session_id}: {duration_s:.1f}s, events {len(all_events)}/{len(usable_events)}/"
          f"{len(interaction_events)} (total/usable/interaction-primary), redacted spans muted: {len(redacted_spans)}")

    return processed_meta


def preprocess_one_session(repo_id: str, output_dir: str) -> Path:
    """Back-compat single-sample entry point (used by the original P0-era
    smoke test). Prefer preprocess_all_sessions for real training runs."""
    data_files = {"train": f"hf://datasets/{repo_id}/data/train/shard-*.tar"}
    dataset = load_dataset("webdataset", data_files=data_files, split="train", streaming=True)
    dataset = dataset.cast_column("flac", Audio(decode=False))
    sample = next(iter(dataset))
    processed_meta = _preprocess_sample(sample, repo_id, output_dir)
    print("=== PREPROCESSING REPORT (single session) ===")
    for k, v in processed_meta.items():
        if k != "interaction_events":
            print(f"  {k}: {v}")
    return Path(output_dir) / f"session_{processed_meta['session_id']}"


def preprocess_all_sessions(
    repo_id: str,
    output_dir: str,
    max_sessions: int | None = None,
    skip_existing: bool = True,
) -> Path:
    """Streams through every shard in the dataset (no full download to a
    single blob first) and preprocesses each session into
    processed/session_<id>/, same as preprocess_one_session but for the
    whole (or a capped subset of the) 20h corpus.

    Writes processed/manifest.json at the end: a flat list of every
    session's processed_meta, which train/stage2_duplex.py reads to build
    its dataset index. Safe to re-run/resume: skip_existing=True (default)
    skips any session_<id> directory that already has metadata.json,
    so a Colab disconnect or PBS wall-time cutoff doesn't force a full
    restart from session 1.
    """
    data_files = {"train": f"hf://datasets/{repo_id}/data/train/shard-*.tar"}
    dataset = load_dataset("webdataset", data_files=data_files, split="train", streaming=True)
    dataset = dataset.cast_column("flac", Audio(decode=False))

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.json"
    manifest: list[dict] = []
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    seen_ids = {m["session_id"] for m in manifest}

    n_done, n_skipped, n_failed = 0, 0, 0
    for i, sample in enumerate(dataset):
        if max_sessions is not None and n_done >= max_sessions:
            break

        raw_json = sample["json"]
        peek_meta = json.loads(raw_json) if isinstance(raw_json, (bytes, str)) else raw_json
        session_id = peek_meta.get("session_id", f"unknown_{i}")

        if skip_existing and session_id in seen_ids:
            n_skipped += 1
            continue

        try:
            processed_meta = _preprocess_sample(sample, repo_id, str(out_root))
        except Exception as exc:  # noqa: BLE001 -- one bad session must not kill a multi-hour run
            print(f"[WARN] session {session_id} (index {i}) failed to preprocess: {exc!r} -- skipping.")
            n_failed += 1
            continue

        manifest.append(processed_meta)
        seen_ids.add(session_id)
        n_done += 1

        if n_done % 10 == 0:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2, default=str)
            print(f"[checkpoint] {n_done} sessions processed so far, manifest saved.")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    total_hours = sum(m["duration_seconds"] for m in manifest) / 3600.0
    print("\n=== FULL PREPROCESSING REPORT ===")
    print(f"sessions newly processed: {n_done}")
    print(f"sessions skipped (already done): {n_skipped}")
    print(f"sessions failed:          {n_failed}")
    print(f"total sessions in manifest: {len(manifest)}")
    print(f"total audio in manifest:    {total_hours:.2f}h")
    print(f"manifest written to:        {manifest_path}")

    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default=REPO_ID)
    parser.add_argument("--output_dir", default="processed")
    parser.add_argument("--one_session_only", action="store_true",
                         help="Old P0-era behavior: process exactly one sample and stop.")
    parser.add_argument("--max_sessions", type=int, default=None,
                         help="Cap the number of NEW sessions processed this run (omit for the full dataset).")
    args = parser.parse_args()

    if args.one_session_only:
        preprocess_one_session(args.repo_id, args.output_dir)
    else:
        preprocess_all_sessions(args.repo_id, args.output_dir, max_sessions=args.max_sessions)
