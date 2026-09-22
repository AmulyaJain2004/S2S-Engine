"""Speech core: loads the REAL Qwen2.5-1.5B-Instruct via Hugging Face.

Per the spec: do not reimplement the transformer. Warm-start from the actual
pretrained weights and later modify the embedding input path / output head.

P0/foundation scope only: this file verifies the model loads and exposes its
hidden size (needed by projector/projector.py's d_out and by
acoustichead/mel_head.py's input width). The embedding-path override and the
duplex fusion (concatenate + linear, per the spec) are Stage 2 work, not
implemented here yet -- adding them now would be exactly the kind of
premature complexity the fast-track plan is trying to avoid.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenSpeechCore(nn.Module):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.hidden_size = self.model.config.hidden_size

    def verify_loaded(self) -> dict:
        """Smoke test: confirms the model loaded correctly and reports the
        hidden size the projector/acoustic head must match."""
        return {
            "model_name": self.model.config._name_or_path,
            "hidden_size": self.hidden_size,
            "num_layers": self.model.config.num_hidden_layers,
            "device": str(self.device),
        }

    # NOTE: forward() intentionally not implemented yet. The embedding-path
    # override (accepting continuous projector/memory embeddings instead of
    # token IDs) and the duplex fusion are Stage 2 work.


if __name__ == "__main__":
    from common import pick_device

    core = QwenSpeechCore(device=pick_device("cuda"))
    info = core.verify_loaded()
    print("=== Qwen speech core load check ===")
    for k, v in info.items():
        print(f"{k}: {v}")
