"""Downloads a single file from Google Drive given a share link or raw
file ID, using gdown (no OAuth/Drive API credentials needed) -- only
works for files shared as "Anyone with the link", which is the tradeoff
for not needing you to set up a Google Cloud project + OAuth client.
"""
from __future__ import annotations

import re
from pathlib import Path

import gdown

# Matches the file ID out of the two common Drive share-link shapes:
#   https://drive.google.com/file/d/<ID>/view?usp=sharing
#   https://drive.google.com/open?id=<ID>
#   https://drive.google.com/uc?id=<ID>&export=download
_DRIVE_ID_PATTERNS = [
    re.compile(r"/file/d/([a-zA-Z0-9_-]{10,})"),
    re.compile(r"[?&]id=([a-zA-Z0-9_-]{10,})"),
]


def extract_drive_file_id(link_or_id: str) -> str:
    """Accepts a full Drive share link OR a bare file ID, returns the ID."""
    link_or_id = link_or_id.strip()
    if "drive.google.com" not in link_or_id and "/" not in link_or_id:
        return link_or_id  # already looks like a bare ID
    for pattern in _DRIVE_ID_PATTERNS:
        match = pattern.search(link_or_id)
        if match:
            return match.group(1)
    raise ValueError(
        f"Couldn't find a Drive file ID in '{link_or_id}' -- paste the full share link "
        f"(File > Share > copy link) or just the file ID from it."
    )


def download_drive_file(link_or_id: str, dest_dir: Path) -> Path:
    """Downloads the file to dest_dir, returns the local path.
    Raises a clear error if the file isn't shared as "Anyone with the link" --
    gdown gets an HTML permission-denied page in that case, not the file."""
    file_id = extract_drive_file_id(link_or_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"{file_id}.wav"

    result = gdown.download(id=file_id, output=str(out_path), quiet=True, fuzzy=True)
    if result is None or not out_path.exists() or out_path.stat().st_size < 1000:
        raise RuntimeError(
            f"Download failed for file ID {file_id} (got nothing or a near-empty file). "
            f"Most likely cause: the file isn't shared as 'Anyone with the link' -- "
            f"right-click it in Drive -> Share -> change to 'Anyone with the link' -> Viewer."
        )
    return out_path
