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
checkpoints (both inside speechbrain/hifigan-wavlm-k1000-LibriTTS on HF --
see the VERIFIED API note below) -- loaded frozen, never trained here.
Training a neural audio codec or a GAN vocoder from scratch would not fit
this project's compute budget; reusing validated pretrained ones (the
same strategy dGSLM/GSLM use) is what makes the discrete-token approach
actually practical here, not just theoretically better.

VERIFIED AGAINST THE ACTUAL INSTALLED speechbrain==1.1.1 API, NOT JUST
DOCUMENTATION (an earlier version of this file was written from web
research alone and had real, confirmed bugs -- see git history. Fixed by
actually installing speechbrain locally, reading the installed source
directly, and running a real encode/decode round trip on a synthetic
waveform before trusting any of this again):
- `DiscreteSSL`'s `ssl_model` argument must be an already-constructed
  SpeechBrain SSL wrapper object (e.g. `WavLM(...)`), NOT a string.
- There is no separate `kmeans_repo_id` -- the k-means checkpoints live in
  the SAME HF repo as the vocoder (`vocoder_repo_id`), confirmed by
  actually downloading kmeans/LibriSpeech_wavlmmodel_k1000_L7.pt from
  speechbrain/hifigan-wavlm-k1000-LibriTTS.
- The constructor argument is `layers_num` (a LIST), not `layer_num`.
- `encode()` returns a 3-tuple `(org_tokens, org_embedding,
  processed_tokens)`, not a single tensor -- `org_tokens` is what
  SpeechBrain's own `forward()` feeds to `decode()`, confirmed by reading
  their `forward()` implementation, so that's what this wrapper uses too.
- `kmeans_dataset` must be `"LibriSpeech"` -- `"LJSpeech"` (an earlier,
  unverified guess in this file) is not a valid value for this repo and
  silently fails to locate the k-means checkpoint file.
- `ssl_model` (the HF source string) must be the FULL repo id
  `"microsoft/wavlm-large"`, not a short name like `"wavlm-large"`.
- Empirically confirmed shapes on a real round trip: a 2s clip yields 99
  tokens (~49.5Hz, matching WavLM's native ~50Hz) and decodes back to
  31680 samples = 99 * 320 -- confirming the 16kHz / 320-samples-per-frame
  / 50Hz chain end to end, not just from reading hyperparams.yaml.

EXPECTED, BENIGN WARNING: loading the pretrained k-means checkpoint prints
an sklearn `InconsistentVersionWarning` (it was pickled with scikit-learn
1.5.0; newer scikit-learn versions still load it fine, just warn). Verified
the actual round trip still works correctly despite this -- not something
to debug if you see it.

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
VOCODER_OUTPUT_SAMPLE_RATE = 16000  # empirically confirmed: 99 frames -> 31680 samples = 320 samples/frame
                                    # (320 * 50 frames/sec = 16000Hz), via a real encode/decode round trip --
                                    # not just read off hyperparams.yaml.


class DiscreteSpeechTokenizer(nn.Module):
    """Frozen wrapper around speechbrain's DiscreteSSL: waveform -> unit
    IDs (encode) and unit IDs -> waveform (decode), via a pretrained
    WavLM-large + k-means(1000) + UnitHiFiGAN pipeline. Nothing in this
    class is trained; it exists purely to give the rest of this codebase
    a clean, verified interface instead of depending on speechbrain's API
    directly in multiple places."""

    def __init__(
        self,
        ssl_model: str = "microsoft/wavlm-large",
        layer_num: int = 7,
        num_clusters: int = 1000,
        kmeans_dataset: str = "LibriSpeech",
        vocoder_repo_id: str = "speechbrain/hifigan-wavlm-k1000-LibriTTS",
        save_path: str = "pretrained_models/discrete_ssl",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        from speechbrain.integrations.audio_tokenizers.discrete_ssl import DiscreteSSL
        from speechbrain.integrations.huggingface.wavlm import WavLM

        self.device = device or torch.device("cpu")
        self.layer_num = layer_num
        self.vocab_size = num_clusters

        ssl_encoder = WavLM(
            source=ssl_model,
            save_path=save_path,
            output_all_hiddens=True,
            freeze=True,  # frozen per this module's whole design -- WavLM defaults to freeze=False otherwise
        )
        ssl_encoder.to(self.device)
        ssl_encoder.eval()

        self.model = DiscreteSSL(
            save_path=save_path,
            ssl_model=ssl_encoder,
            kmeans_dataset=kmeans_dataset,
            vocoder_repo_id=vocoder_repo_id,
            num_clusters=num_clusters,
            layers_num=[layer_num],
            device=str(self.device),
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
        org_tokens, _org_embedding, _processed_tokens = self.model.encode(waveform, SSL_layers=[self.layer_num])
        return org_tokens.squeeze(0).squeeze(-1).long()  # (1, T, 1) -> (T,)

    @torch.no_grad()
    def decode(self, unit_ids: torch.Tensor) -> torch.Tensor:
        """unit_ids: (T,) long tensor of unit IDs (as returned by encode --
        raw, per-layer-local cluster indices; decode applies its own
        layer-offset internally).
        Returns: (n_samples,) waveform at VOCODER_OUTPUT_SAMPLE_RATE (16kHz)."""
        tokens = unit_ids.view(1, -1, 1).to(self.device)  # (B=1, T, num_layers=1)
        waveform = self.model.decode(tokens, SSL_layers=[self.layer_num])  # (1, 1, n_samples)
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
