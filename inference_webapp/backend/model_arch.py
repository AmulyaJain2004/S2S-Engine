"""Model architecture for the ORIGINAL (pre-fix) Stage 2 checkpoint --
continuous WavLM features, the leaky concat-based DuplexFusion, a
single-Linear MelHead, WavLM-base-plus + Vocos.

DELIBERATELY FROZEN / DECOUPLED FROM duplex_speechlm/: the main repo has
since moved on (duplex-fusion leakage fix, then a full pivot to discrete
units -- see duplex_speechlm/README.md). This file is a self-contained
copy of the classes as they existed when the checkpoint being served here
was actually trained (git commit 7406dc4), so this app keeps working
regardless of what happens to the main training pipeline next, and so
nothing in the main pipeline can accidentally break this archived
inference path. Do not "helpfully" sync this with duplex_speechlm/'s
current code -- that would defeat the point.

KNOWN LIMITATION, shipped deliberately: this checkpoint was trained with a
duplex-fusion design later found to let the model shortcut training by
echoing ground-truth audio instead of learning genuine generation (see
the main repo's history). Generation here uses the self-feedback loop
with the clamp / loudness-match / crossfade mitigations that were the
best fix available for THIS architecture before it was superseded --
expect real, but not perfect, intelligibility.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer, Wav2Vec2FeatureExtractor, WavLMModel
from vocos import Vocos

WAVLM_SAMPLE_RATE = 16000
VOCOS_SAMPLE_RATE = 24000
VOCOS_HOP_LENGTH = 256  # verified against charactr/vocos-mel-24khz's own feature extractor config
QWEN2_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class WavLMEncoder(nn.Module):
    """Frozen microsoft/wavlm-base-plus: waveform -> ~50Hz continuous features."""

    def __init__(self, model_name: str = "microsoft/wavlm-base-plus", device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

    @torch.no_grad()
    def forward(
        self, waveform: Union[torch.Tensor, Sequence[torch.Tensor]], sample_rate: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(waveform):
            items: List[torch.Tensor] = list(waveform.unsqueeze(0)) if waveform.dim() == 1 else list(waveform)
        else:
            items = list(waveform)
        if sample_rate != WAVLM_SAMPLE_RATE:
            items = [torchaudio.functional.resample(w, sample_rate, WAVLM_SAMPLE_RATE) for w in items]

        inputs = self.feature_extractor(
            [w.numpy() for w in items], sampling_rate=WAVLM_SAMPLE_RATE, return_tensors="pt", padding=True
        )
        input_values = inputs["input_values"].to(self.device)
        sample_attention_mask = inputs["attention_mask"].to(self.device)
        outputs = self.model(input_values=input_values, attention_mask=sample_attention_mask)
        hidden_states = outputs.last_hidden_state
        frame_mask = self.model._get_feature_vector_attention_mask(hidden_states.shape[1], sample_attention_mask)
        return hidden_states, frame_mask


def waveform_to_mel(vocos_model, waveform: torch.Tensor) -> torch.Tensor:
    return vocos_model.feature_extractor(waveform)


class VocosVocoder(nn.Module):
    """Frozen charactr/vocos-mel-24khz: mel-spectrogram <-> waveform."""

    def __init__(self, repo_id: str = "charactr/vocos-mel-24khz", device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.model = Vocos.from_pretrained(repo_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

    @torch.no_grad()
    def mel_from_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform_to_mel(self.model, waveform.to(self.device))

    @torch.no_grad()
    def waveform_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        return self.model.decode(mel.to(self.device))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(dtype)


class Projector(nn.Module):
    """WavLM features -> Conv1D -> MLP -> RMSNorm -> Qwen-compatible width."""

    def __init__(self, d_in: int = 768, d_out: int = 1536, conv_kernel: int = 3, mlp_mult: float = 4.0) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_in, kernel_size=conv_kernel, padding=conv_kernel // 2)
        d_hidden = int(d_out * mlp_mult)
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )
        self.norm = RMSNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h = self.mlp(h)
        return self.norm(h)


class DuplexFusion(nn.Module):
    """Concat(user_frame, previous-output_frame) -> Linear -> hidden size.
    The "leaky" design: previous-output is ground truth at train time,
    self-generated audio at inference time (see module docstring)."""

    def __init__(self, d_in: int, d_hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * d_in, d_hidden)
        self.start_token = nn.Parameter(torch.zeros(d_in))

    def forward(self, user_emb: torch.Tensor, prev_output_emb: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([user_emb, prev_output_emb], dim=-1)
        return self.proj(fused)

    def shift_with_start_token(self, agent_emb: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing helper: shifts the real agent-channel embedding
        right by one frame, filling position 0 with the learned start
        token. Used only by the teacher-forced diagnostic path, which
        feeds the model the real answer instead of its own guess."""
        b, t, d = agent_emb.shape
        start = self.start_token.view(1, 1, d).expand(b, 1, d).to(agent_emb.dtype)
        return torch.cat([start, agent_emb[:, :-1, :]], dim=1)


class MelHead(nn.Module):
    """Single-Linear readout: speech core hidden state -> mel frame."""

    def __init__(self, d_in: int = 1536, n_mels: int = 100) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, n_mels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class QwenSpeechCore(nn.Module):
    """Qwen2.5-1.5B-Instruct with the embedding-path override (accepts
    continuous inputs_embeds, never calls the pretrained lm_head) and LoRA."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[torch.device] = None,
        use_lora: bool = True,
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
                task_type="FEATURE_EXTRACTION",
            )
            self.model = get_peft_model(self.model, lora_cfg)
        else:
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.model.to(self.device)

    def forward_step(self, inputs_embeds: torch.Tensor, past_key_values=None):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        return outputs.hidden_states[-1], outputs.past_key_values
