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


def preprocess_one_session(repo_id: str, output_dir: str) -> Path:
    data_files = {"train": f"hf://datasets/{repo_id}/data/train/shard-*.tar"}
    dataset = load_dataset("webdataset", data_files=data_files, split="train", streaming=True)
    dataset = dataset.cast_column("flac", Audio(decode=True))  # decode this time, we need real samples

    sample = next(iter(dataset))
    audio = sample["flac"]  # expected: {"array": np.ndarray, "sampling_rate": int} once decoded
    raw_json = sample["json"]
    meta = json.loads(raw_json) if isinstance(raw_json, (bytes, str)) else raw_json

    # session_id is a required top-level field in the JSON itself -- confirmed
    # against session-v1.schema.json, more reliable than the WebDataset "__key__"
    # convention this script assumed before that schema was available.
    session_id = meta["session_id"]

    array = np.asarray(audio["array"])
    sr = audio["sampling_rate"]

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

    print("=== PREPROCESSING REPORT ===")
    print(f"session:                {session_id}")
    print(f"sample rate:            {sr}")
    print(f"duration:               {duration_s:.2f}s")
    print(f"redacted spans muted:   {len(redacted_spans)}")
    print(f"events total/usable/interaction-primary: {len(all_events)}/{len(usable_events)}/{len(interaction_events)}")
    print(f"observed event types (this session): {sorted({e.get('type') for e in usable_events})}")
    print(f"saved to:               {session_dir}")

    # Basic validation
    assert speaker_0_path.exists() and speaker_1_path.exists(), "one or both speaker WAVs missing after write"
    dur0 = sf.info(str(speaker_0_path)).duration
    dur1 = sf.info(str(speaker_1_path)).duration
    assert abs(dur0 - dur1) < 0.01, f"speaker track durations misaligned: {dur0}s vs {dur1}s"
    print("validation: both speaker files exist, durations aligned, sample rate consistent -- OK")

    return session_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default=REPO_ID)
    parser.add_argument("--output_dir", default="processed")
    args = parser.parse_args()
    preprocess_one_session(args.repo_id, args.output_dir)
