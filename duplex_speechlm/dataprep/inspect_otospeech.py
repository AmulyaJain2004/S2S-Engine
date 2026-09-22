"""Streams exactly ONE sample from otoSpeech-full-duplex-task-oriented-20h and
prints its actual structure. Does NOT download the full dataset.

Run this BEFORE writing any preprocessing logic that assumes a schema --
the field names below (channels, events, task info) are what to verify, not
what to trust blindly. If the real sample's keys differ from what's printed
here, preprocess_otospeech.py must be updated to match, not the other way
around.

Requires network access to huggingface.co and a Hugging Face token with
approved access to the gated otoSpeech dataset (huggingface-cli login, or
set the HF_TOKEN environment variable) -- this will NOT run in a sandboxed
environment with no internet access to the Hub.
"""
from __future__ import annotations

import argparse
import json

from datasets import Audio, load_dataset

REPO_ID = "otoearth/otoSpeech-full-duplex-task-oriented-20h"


def inspect_one_sample(repo_id: str = REPO_ID) -> dict:
    print(f"Streaming one sample from {repo_id} (no full download)...")
    data_files = {"train": f"hf://datasets/{repo_id}/data/train/shard-*.tar"}
    dataset = load_dataset(
        "webdataset",
        data_files=data_files,
        split="train",
        streaming=True,
    )
    # Do not decode the audio column yet -- just inspect structure first.
    dataset = dataset.cast_column("flac", Audio(decode=False))

    sample = next(iter(dataset))

    print("\n=== SAMPLE KEYS ===")
    print(list(sample.keys()))

    print("\n=== AUDIO METADATA (undecoded) ===")
    audio_info = sample.get("flac")
    if isinstance(audio_info, dict):
        for k, v in audio_info.items():
            if k == "bytes":
                print(f"  bytes: <{len(v)} bytes, not printed>")
            else:
                print(f"  {k}: {v}")
    else:
        print(f"  (unexpected type: {type(audio_info)})")

    print("\n=== JSON STRUCTURE ===")
    json_field = sample.get("json")
    if isinstance(json_field, (bytes, str)):
        parsed = json.loads(json_field)
    else:
        parsed = json_field  # datasets may already parse it depending on version
    print(json.dumps(parsed, indent=2, default=str)[:4000])  # cap output, this is an inspection not a dump

    print("\n=== EVENT TYPE / KIND VOCABULARY (this one sample only) ===")
    events = parsed.get("events", []) if isinstance(parsed, dict) else []
    observed_types = sorted({e.get("type") for e in events})
    observed_kinds = sorted({e.get("kind") for e in events})
    print(f"kinds observed:  {observed_kinds}  (schema enum: lifecycle/exposure/interaction/state/outcome/telemetry)")
    print(f"types observed:  {observed_types}")
    print("NOTE: event 'type' is an open string in the schema (no enum) -- a single session's "
          "vocabulary is not the full picture. Run this across multiple/all sessions, or check "
          "the task walkthrough page linked in the README, before assuming these are all the types "
          "that exist. Cross-reference against event_summary.total/primary/telemetry in the session JSON.")

    print("\n=== FIELDS TO LOOK FOR MANUALLY IN THE ABOVE ===")
    print("- task information (task type / task id)")
    print("- speaker/channel information (which channel = which speaker)")
    print("- event list: structure, timestamps, event type/label vocabulary")
    print("- anything resembling turn-taking, overlap, interruption, backchannel, pause")
    print("Do not assume any of these exist under a guessed key name -- read the printed JSON above.")

    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default=REPO_ID)
    args = parser.parse_args()
    inspect_one_sample(args.repo_id)
