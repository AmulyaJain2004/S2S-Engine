"""Speech core: loads the REAL Qwen2.5-1.5B-Instruct via Hugging Face.

Per the spec: do not reimplement the transformer. Warm-start from the actual
pretrained weights and only modify the embedding input path / output head.

Stage 2 scope (this file): the embedding-path override and LoRA wrapping are
now implemented.

- "Embedding-path override": HF's AutoModelForCausalLM already accepts
  `inputs_embeds` in place of `input_ids` -- no architecture surgery is
  needed to feed it the unit embeddings from projector/unit_embedding.py.
  "Replacing the LM head with the unit head" means: never call the
  model's own `lm_head`; instead take `output_hidden_states=True` and read
  the last hidden state, which the caller (train/stage2_duplex.py) feeds
  into acoustichead/unit_head.py's classification head. The pretrained
  lm_head is left untouched but unused.
- LoRA: applied via peft, targeting Qwen2's attention + MLP projections.
  The base weights stay frozen; only LoRA adapters train.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN2_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class QwenSpeechCore(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[torch.device] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.hidden_size = self.model.config.hidden_size
        self.use_lora = use_lora

        if use_lora:
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=QWEN2_LORA_TARGET_MODULES,
                bias="none",
                task_type="FEATURE_EXTRACTION",  # we never use .generate()/the LM head here
            )
            self.model = get_peft_model(self.model, lora_cfg)
        else:
            for p in self.model.parameters():
                p.requires_grad_(False)

        self.model.to(self.device)

    def verify_loaded(self) -> dict:
        """Smoke test: confirms the model loaded correctly and reports the
        hidden size the projector/acoustic head must match."""
        base = self.model.get_base_model() if self.use_lora else self.model
        return {
            "model_name": base.config._name_or_path,
            "hidden_size": self.hidden_size,
            "num_layers": base.config.num_hidden_layers,
            "device": str(self.device),
            "use_lora": self.use_lora,
        }

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """inputs_embeds: (B, T, hidden_size) continuous embeddings -- the
        duplex-fused speech frames, NOT token embeddings.
        attention_mask: (B, T), 1 for real frames, 0 for padding.
        Returns: (B, T, hidden_size) last hidden state, i.e. the input to
        the acoustic head. The model's own lm_head is never called."""
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return outputs.hidden_states[-1]

    def forward_step(self, inputs_embeds: torch.Tensor, past_key_values=None):
        """Inference-time counterpart to forward(): KV-cached, so each call
        only needs to pass the NEW frames' embeddings (one window's worth),
        not the whole history -- the cache carries everything before that.
        Used by eval/generate_stage2.py for autoregressive duplex generation;
        training never calls this (training sees the whole chunk at once).

        inputs_embeds: (B, T_new, hidden_size), continuous frames for just
            this step (e.g. one ~300-500ms window).
        past_key_values: whatever the previous call returned, or None for
            the first step.
        Returns: (hidden_states, new_past_key_values).
            hidden_states: (B, T_new, hidden_size) -- only for the NEW
            frames just passed in, matching how KV-cached generation works."""
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        return outputs.hidden_states[-1], outputs.past_key_values

    def trainable_parameters(self):
        return (p for p in self.model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from common import pick_device

    core = QwenSpeechCore(device=pick_device("cuda"), use_lora=True)
    info = core.verify_loaded()
    print("=== Qwen speech core load check ===")
    for k, v in info.items():
        print(f"{k}: {v}")

    n_trainable = sum(p.numel() for p in core.trainable_parameters())
    print(f"trainable params (LoRA only): {n_trainable:,}")
