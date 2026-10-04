"""Prints each session's speakers_by_channel from an already-written
manifest.json (see preprocess_otospeech.py), plus a best-effort role
guess and a cross-session consistency check, so a human can confirm which
physical channel (0 or 1) is actually the user/customer side BEFORE
training trusts configs/*.yaml's user_channel/agent_channel.

This does NOT auto-decide the mapping. otoSpeech's exact speaker-role field
names were not independently verified when preprocess_otospeech.py was
written (see its SCHEMA MAPPING note) -- this script prints what's
actually there and makes a labelled guess for a human to confirm, not to
trust blindly. train/stage2_duplex.py will refuse to run until you've set
audio.channel_roles_verified: true in your config, which is meant to only
happen after you've actually looked at this output.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

USER_ROLE_HINTS = ("user", "customer", "caller", "client", "patient")
AGENT_ROLE_HINTS = ("agent", "assistant", "operator", "support", "bot", "system")


def _guess_role(speaker: dict) -> str:
    blob = json.dumps(speaker, default=str).lower()
    if any(h in blob for h in AGENT_ROLE_HINTS):
        return "agent-like"
    if any(h in blob for h in USER_ROLE_HINTS):
        return "user-like"
    return "UNKNOWN"


def main(manifest_path: str, n_sessions: int) -> None:
    with open(manifest_path) as f:
        manifest = json.load(f)
    if not manifest:
        raise RuntimeError(f"{manifest_path} is empty -- run dataprep/preprocess_otospeech.py first.")

    print(f"Inspecting {min(n_sessions, len(manifest))} of {len(manifest)} sessions in {manifest_path}\n")
    guesses_by_channel: dict[str, Counter] = {"0": Counter(), "1": Counter()}

    for sess in manifest[:n_sessions]:
        by_channel = sess.get("speakers_by_channel", {})
        print(f"--- session {sess['session_id']} ---")
        print(json.dumps(by_channel, indent=2, default=str))
        for ch_str, speaker in by_channel.items():
            guess = _guess_role(speaker)
            print(f"  channel {ch_str}: best-effort guess = {guess}")
            guesses_by_channel.setdefault(ch_str, Counter())[guess] += 1
        print()

    print("=== SUMMARY (still just a heuristic guess -- read the raw JSON above yourself) ===")
    for ch_str, counter in guesses_by_channel.items():
        print(f"channel {ch_str} guesses across inspected sessions: {dict(counter)}")

    inconsistent = any(len(counter) > 1 for counter in guesses_by_channel.values() if counter)
    if inconsistent:
        print(
            "\nWARNING: channel-to-role mapping was NOT consistent across the sessions inspected. "
            "If roles swap channel per session, a single global user_channel/agent_channel config "
            "value is wrong for this dataset -- training would need per-session channel assignment, "
            "not a fixed global one. Do not set channel_roles_verified: true until this is resolved."
        )

    print(
        "\nNext step: once you've confirmed which channel is actually the user/customer side "
        "(and confirmed it's the SAME channel across sessions, or handled the per-session case above), "
        "set audio.user_channel / audio.agent_channel in your config accordingly, AND set "
        "audio.channel_roles_verified: true -- train/stage2_duplex.py refuses to run otherwise."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="processed/manifest.json")
    parser.add_argument("--n_sessions", type=int, default=5)
    args = parser.parse_args()
    main(args.manifest, args.n_sessions)
