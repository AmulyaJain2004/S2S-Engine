"""Loads the trained checkpoint once and exposes a simple generate()
call. Generation logic ported verbatim (same math, same fixes) from
duplex_speechlm's eval/generate_stage2.py as it stood at the commit that
matches this checkpoint's architecture (see model_arch.py's docstring) --
not re-derived, to avoid introducing any subtle behavioral drift from
what was actually verified working at that point.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torchaudio

from model_arch import (
    VOCOS_HOP_LENGTH,
    VOCOS_SAMPLE_RATE,
    WAVLM_SAMPLE_RATE,
    DuplexFusion,
    MelHead,
    Projector,
    QwenSpeechCore,
    VocosVocoder,
    WavLMEncoder,
)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().clamp_min(1e-8).sqrt()


def _crossfade_concat(chunks: list, fade_samples: int) -> torch.Tensor:
    if len(chunks) == 1:
        return chunks[0]
    out = chunks[0]
    for nxt in chunks[1:]:
        n = min(fade_samples, out.shape[0], nxt.shape[0])
        if n <= 0:
            out = torch.cat([out, nxt], dim=0)
            continue
        ramp = torch.linspace(0.0, 1.0, n)
        blended = out[-n:] * (1.0 - ramp) + nxt[:n] * ramp
        out = torch.cat([out[:-n], blended, nxt[n:]], dim=0)
    return out


class DuplexInferencePipeline:
    def __init__(self) -> None:
        self.checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoint"))
        self.wavlm_name = os.environ.get("WAVLM_NAME", "microsoft/wavlm-base-plus")
        self.qwen_name = os.environ.get("QWEN_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
        self.vocos_repo = os.environ.get("VOCOS_REPO", "charactr/vocos-mel-24khz")
        self.lora_r = int(os.environ.get("LORA_R", 16))
        self.lora_alpha = int(os.environ.get("LORA_ALPHA", 32))
        self.lora_dropout = float(os.environ.get("LORA_DROPOUT", 0.05))

        requested_device = os.environ.get("DEVICE", "cuda")
        self.device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
        dtype_name = os.environ.get("DTYPE", "bf16" if self.device.type == "cuda" else "fp32")
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]

        self.step: Optional[int] = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        state_path = self.checkpoint_dir / "train_state.pt"
        lora_path = self.checkpoint_dir / "qwen_lora"
        if not state_path.exists() or not lora_path.exists():
            raise FileNotFoundError(
                f"Expected '{state_path.name}' and '{lora_path.name}/' inside {self.checkpoint_dir} "
                f"-- place your downloaded checkpoint files there (see the README) before starting the server."
            )

        print(f"[pipeline] device={self.device} dtype={self.dtype}")
        print(f"[pipeline] loading WavLM ({self.wavlm_name})...")
        self.wavlm = WavLMEncoder(self.wavlm_name, device=self.device)
        print(f"[pipeline] loading Vocos ({self.vocos_repo})...")
        self.vocos = VocosVocoder(self.vocos_repo, device=self.device)
        print(f"[pipeline] loading Qwen speech core ({self.qwen_name}) + LoRA...")
        self.qwen_core = QwenSpeechCore(
            self.qwen_name,
            device=self.device,
            use_lora=True,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            dtype=self.dtype,
        )
        hidden_size = self.qwen_core.hidden_size
        self.projector = Projector(d_in=768, d_out=hidden_size).to(device=self.device, dtype=self.dtype)
        self.fusion = DuplexFusion(d_in=hidden_size, d_hidden=hidden_size).to(device=self.device, dtype=self.dtype)
        self.acoustic_head = MelHead(d_in=hidden_size, n_mels=100).to(device=self.device, dtype=self.dtype)

        print(f"[pipeline] loading checkpoint from {self.checkpoint_dir}...")
        state = torch.load(state_path, map_location=self.device)
        self.projector.load_state_dict(state["projector"])
        self.fusion.load_state_dict(state["fusion"])
        self.acoustic_head.load_state_dict(state["acoustic_head"])

        from peft import PeftModel

        self.qwen_core.model = PeftModel.from_pretrained(self.qwen_core.model.get_base_model(), str(lora_path))
        self.qwen_core.model.to(self.device)
        self.qwen_core.model.eval()
        for m in (self.projector, self.fusion, self.acoustic_head):
            m.eval()

        self.step = state["step"]
        self._loaded = True
        print(f"[pipeline] ready, checkpoint step {self.step}")

    @torch.no_grad()
    def generate(
        self,
        user_wave: torch.Tensor,
        native_sr: int,
        window_s: float = 0.4,
        max_duration_s: float = 30.0,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> torch.Tensor:
        """user_wave: mono (n_samples,) float tensor at native_sr.
        Returns: mono waveform tensor at VOCOS_SAMPLE_RATE (24000Hz)."""
        if not self._loaded:
            self.load()

        if max_duration_s is not None:
            user_wave = user_wave[: int(max_duration_s * native_sr)]

        window_len = int(window_s * native_sr)
        n_windows = max(1, (user_wave.shape[0] + window_len - 1) // window_len)

        past_key_values = None
        prev_output_wave_native = None
        synthesized_chunks = []

        for i in range(n_windows):
            start = i * window_len
            window = user_wave[start : start + window_len]
            if window.shape[0] == 0:
                break

            user_16k = torchaudio.functional.resample(window, native_sr, WAVLM_SAMPLE_RATE)
            user_feat, _ = self.wavlm([user_16k], WAVLM_SAMPLE_RATE)
            user_emb = self.projector(user_feat.to(self.dtype))
            t_u = user_emb.shape[1]

            if prev_output_wave_native is None:
                prev_output_emb = self.fusion.start_token.view(1, 1, -1).expand(1, t_u, -1).to(self.dtype)
            else:
                prev_16k = torchaudio.functional.resample(prev_output_wave_native, native_sr, WAVLM_SAMPLE_RATE)
                prev_feat, _ = self.wavlm([prev_16k], WAVLM_SAMPLE_RATE)
                prev_emb = self.projector(prev_feat.to(self.dtype))
                prev_output_emb = (
                    F.interpolate(prev_emb.transpose(1, 2).float(), size=t_u, mode="linear", align_corners=False)
                    .transpose(1, 2)
                    .to(self.dtype)
                )

            fused = self.fusion(user_emb, prev_output_emb)
            hidden, past_key_values = self.qwen_core.forward_step(fused, past_key_values)
            pred_mel = self.acoustic_head(hidden).transpose(1, 2).float()

            window_samples_24k = round(window.shape[0] / native_sr * VOCOS_SAMPLE_RATE)
            target_mel_frames = max(1, window_samples_24k // VOCOS_HOP_LENGTH + 1)
            pred_mel = F.interpolate(pred_mel, size=target_mel_frames, mode="linear", align_corners=False)

            waveform_window = self.vocos.waveform_from_mel(pred_mel).cpu().squeeze(0)
            waveform_window = waveform_window.clamp(-1.0, 1.0)
            synthesized_chunks.append(waveform_window)

            waveform_24k = torchaudio.functional.resample(window, native_sr, VOCOS_SAMPLE_RATE)
            loudness_scale = (_rms(waveform_24k) / _rms(waveform_window)).clamp(0.2, 5.0)
            feedback_window = (waveform_window * loudness_scale).clamp(-1.0, 1.0)
            prev_output_wave_native = torchaudio.functional.resample(feedback_window, VOCOS_SAMPLE_RATE, native_sr)

            if progress_cb is not None:
                progress_cb((i + 1) / n_windows)

        fade_samples = int(0.01 * VOCOS_SAMPLE_RATE)
        return _crossfade_concat(synthesized_chunks, fade_samples)

    @torch.no_grad()
    def generate_teacher_forced(
        self, user_wave: torch.Tensor, agent_wave: torch.Tensor, native_sr: int
    ) -> torch.Tensor:
        """Teacher-forced diagnostic path: feeds the model the REAL agent
        audio (shifted by one frame) as duplex fusion's "previous output"
        signal, instead of self-generated audio. One non-cached forward
        pass over the whole clip, no windowing, no self-feedback loop --
        ported verbatim from duplex_speechlm's eval/teacher_forced_check.py
        as it stood at the commit matching this checkpoint (git 4578a81).

        This is NOT a real inference mode -- it requires the real answer
        as input, which doesn't exist at actual deployment time. It exists
        because this checkpoint's self-feedback output was confirmed noisy
        while this teacher-forced path was confirmed to produce actual
        intelligible words, isolating the problem to the duplex-fusion
        design (since fixed in later work) rather than model capacity.

        user_wave, agent_wave: mono (n_samples,) tensors at native_sr, the
        SAME length (same time window of a real two-channel recording).
        Returns: predicted agent waveform at VOCOS_SAMPLE_RATE (24000Hz).
        """
        if not self._loaded:
            self.load()

        user_16k = torchaudio.functional.resample(user_wave, native_sr, WAVLM_SAMPLE_RATE)
        agent_16k = torchaudio.functional.resample(agent_wave, native_sr, WAVLM_SAMPLE_RATE)

        combined_hidden, _ = self.wavlm([user_16k, agent_16k], WAVLM_SAMPLE_RATE)
        user_feat, agent_feat = combined_hidden[0:1], combined_hidden[1:2]

        user_emb = self.projector(user_feat.to(self.dtype))
        agent_emb = self.projector(agent_feat.to(self.dtype))
        prev_output_emb = self.fusion.shift_with_start_token(agent_emb)
        fused = self.fusion(user_emb, prev_output_emb)

        hidden = self.qwen_core.forward_step(fused, None)[0]  # non-cached, whole-clip forward
        pred_mel = self.acoustic_head(hidden).transpose(1, 2).float()

        agent_24k = torchaudio.functional.resample(agent_wave, native_sr, VOCOS_SAMPLE_RATE)
        target_mel = self.vocos.mel_from_waveform(agent_24k.unsqueeze(0))
        pred_mel = F.interpolate(pred_mel, size=target_mel.shape[-1], mode="linear", align_corners=False)

        return self.vocos.waveform_from_mel(pred_mel).cpu().squeeze(0).clamp(-1.0, 1.0)


pipeline = DuplexInferencePipeline()
