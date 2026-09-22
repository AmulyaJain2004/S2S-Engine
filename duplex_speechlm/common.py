"""Small shared utilities: config loading and device selection. No model code here."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import yaml


def load_config(path: str = "configs/p0.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pick_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
