"""Discrete speech tokenizer/vocoder: wraps SpeechBrain's DiscreteSSL to
turn raw waveform into discrete unit IDs and back, using PRETRAINED,
FROZEN components only -- no tokenizer/vocoder training happens anywhere
in this codebase.

ARCHITECTURE CHANGE, not a patch (replaces vocoder/vocos_wrapper.py +
projector/projector.py's old role for Stage 2): the previous pipeline fed
continuous WavLM features into Qwen and trained a mel-regression acoustic
head against Vocos targets with an L1 (+ spectral-flux) loss. That design
is documented in the literature (MELLE, arXiv:2407.08551; Ren et al.,
"Revisiting Over-Smoothness in TTS") to be prone to over-smoothed,
blurry output -- deterministic regression against a continuous target
averages over plausible detail. Every validated full-duplex / speech-LM
system that actually works at scale instead represents speech as DISCRETE
tokens predicted via ordinary cross-entropy: dGSLM (arXiv:2203.16502,
HuBERT units + dual-tower LM + HiFi-GAN), Moshi (Kyutai, Mimi RVQ codec +
LM), SpeechGPT, VALL-E. This switches to that proven recipe, using WavLM
(the encoder this project already uses) instead of HuBERT, so the
"speech core" (Qwen, LoRA-adapted) is now trained the way LLMs are
actually good at: next-unit classification, not continuous regression.

WHY THIS IS BUDGET-FEASIBLE on a single Colab A100 session (the actual
constraint driving this choice): the k-means quantizer (k=1000, WavLM-large
layer 7) and the unit-to-waveform vocoder are PUBLISHED, PRETRAINED
checkpoints (speechbrain/SSL_Quantization, speechbrain/hifigan-wavlm-k1000-LibriTTS)
-- loaded frozen, never trained here. Training a neural audio codec or a
GAN vocoder from scratch would not fit this project's compute budget;
reusing validated pretrained ones (the same strategy dGSLM/GSLM use) is
what makes the discrete-token approach actually practical here, not just
theoretically better.

SIDE EFFECT: WavLM's native ~50Hz frame rate now applies to BOTH the input
(user) and target (agent) streams identically (both are unit sequences
derived from the exact same WavLM+k-means pipeline), so the WavLM/Vocos
rate-mismatch interpolation code in the old train/stage2_duplex.py is no
longer needed at all -- removed, not papered over.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchaudio

DISCRETE_SSL_SAMPLE_RATE = 16000  # WavLM's native input rate, same as before
VOCODER_OUTPUT_SAMPLE_RATE = 16000  # verified against speechbrain/hifigan-wavlm-k1000-LibriTTS's own
                                    # hyperparams.yaml: upsample_factors [5,4,4,2,2] -> product 320,
                                    # and 320 samples/frame * 50 frames/sec (WavLM's native rate) = 16000Hz.
                                    # Not guessed -- computed from the checkpoint's own published config.


class DiscreteSpeechTokenizer(nn.Module):
    """Frozen wrapper around speechbrain's DiscreteSSL: waveform -> unit
    IDs (encode) and unit IDs -> waveform (decode), via a pretrained
    WavLM-large + k-means(1000) + UnitHiFiGAN pipeline. Nothing in this
    class is trained; it exists purely to give the rest of this codebase
    a clean, verified interface instead of depending on speechbrain's API
    directly in multiple places."""

    def __init__(
        self,
        ssl_model: str = "wavlm-large",
        layer_num: int = 7,
        num_clusters: int = 1000,
        kmeans_dataset: str = "LJSpeech",
        kmeans_repo_id: str = "speechbrain/SSL_Quantization",
        vocoder_repo_id: str = "speechbrain/hifigan-wavlm-k1000-LibriTTS",
        save_path: str = "pretrained_models/discrete_ssl",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        from speechbrain.lobes.models.huggingface_transformers.discrete_ssl import DiscreteSSL

        self.device = device or torch.device("cpu")
        self.layer_num = layer_num
        self.vocab_size = num_clusters
        self.model = DiscreteSSL(
            save_path=save_path,
            ssl_model=ssl_model,
            kmeans_dataset=kmeans_dataset,
            kmeans_repo_id=kmeans_repo_id,
            vocoder_repo_id=vocoder_repo_id,
            num_clusters=num_clusters,
            layer_num=layer_num,
        )
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """waveform: (n_samples,) or (1, n_samples) at any sample rate.
        Returns: (T,) long tensor of unit IDs at WavLM's native ~50Hz."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if sample_rate != DISCRETE_SSL_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, DISCRETE_SSL_SAMPLE_RATE)
        waveform = waveform.to(self.device)
        tokens = self.model.encode(waveform, SSL_layers=[self.layer_num])
        return tokens.squeeze(0).squeeze(-1).long()  # (T,)

    @torch.no_grad()
    def decode(self, unit_ids: torch.Tensor) -> torch.Tensor:
        """unit_ids: (T,) long tensor of unit IDs.
        Returns: (n_samples,) waveform at the vocoder's native sample rate."""
        tokens = unit_ids.view(1, -1, 1).to(self.device)
        waveform = self.model.decode(tokens, SSL_layers=[self.layer_num])
        return waveform.squeeze().cpu()


if __name__ == "__main__":
    import argparse

    from common import pick_device

    parser = argparse.ArgumentParser(description="Smoke test: encode then decode one real WAV clip.")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = pick_device(args.device)
    tok = DiscreteSpeechTokenizer(device=device)
    wav, sr = torchaudio.load(args.wav)
    units = tok.encode(wav.mean(dim=0), sr)
    print(f"Encoded {wav.shape[-1] / sr:.2f}s of audio -> {units.shape[0]} units "
          f"({units.shape[0] / (wav.shape[-1] / sr):.1f}Hz)")
    recon = tok.decode(units)
    print(f"Decoded back to {recon.shape[-1]} samples.")
