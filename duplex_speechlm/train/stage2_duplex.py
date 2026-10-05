"""Stage 2: duplex fine-tune on real otoSpeech two-channel conversation audio.

Per spec section 7, Stage 2 is what actually teaches backchannel/
interruption/overlap behavior -- it trains on real duplex audio, not
synthetic single-turn TTS pairs. This script skips Stage 1 (content
bootstrap) by direct instruction: we warm-start everything from frozen
WavLM + pretrained Qwen2.5-1.5B-Instruct + frozen Vocos, and train only
the new/adapted pieces (projector, duplex fusion, LoRA on Qwen, acoustic
head) directly on real duplex data.

RATE MISMATCH (resolved, not just flagged): WavLM runs at its native ~50Hz
and Vocos's real mel config (vocoder/mel_utils.py, verified against the
installed package) runs at ~93.75Hz (24kHz / hop_length 256) -- there is no
integer ratio between them. Since the acoustic head is a plain nn.Linear
with no time-mixing, resampling its output along the time axis to the
target mel length is equivalent (up to interpolation choice) to resampling
the hidden states first; this script does it on the acoustic head's output,
per-item, against each item's OWN true mel length (not a batch-wide
average), using the exact valid-frame counts from the masks below -- not an
approximation over padded regions. The model cannot genuinely produce more
time-resolution than WavLM's 50Hz gives it; this interpolation is a
resampling bridge between two fixed, externally-defined rates, not a
quality shortcut.

VARIABLE-LENGTH CHUNKS (resolved, not dropped): sessions are chopped into
fixed-length, non-overlapping chunks, but each session's trailing
remainder is KEPT (not discarded) as long as it's >= audio.min_chunk_s.
WavLMEncoder.forward returns a real frame-level attention mask (computed
by HF's own conv-stride-aware helper, not a guessed ratio), which is
threaded through the speech core's attention and through the mel loss so
padded frames never contribute to gradients or attention.

CHANNEL-ROLE ASSUMPTION (now a hard gate, not a silent guess): this script
refuses to start unless audio.channel_roles_verified: true is set in the
config, which should only happen after running
dataprep/print_channel_roles.py and confirming audio.user_channel /
audio.agent_channel against the real otoSpeech speaker-role metadata.

DUPLEX FUSION NO LONGER TAKES AGENT AUDIO AS INPUT (corrected design, not
just a refactor -- see speechcore/duplex_fusion.py's docstring for the
full reasoning). The previous version fed the real ground-truth agent
audio's embedding (shifted by one 20ms frame) into the model's own input
at every frame. Adjacent real-speech frames are highly autocorrelated, so
this gave the model an easy shortcut -- echo the ground truth forward --
instead of learning genuine content generation from the user audio and
its own understanding. It also meant training and real inference used
different mechanisms (real inference has no ground-truth agent audio to
lean on), which is exactly why generation-time self-feedback kept
producing noise no amount of inference-side signal processing fixed: the
model never actually learned to generate independently. The speech core's
own causal self-attention over its own past hidden states is what now
carries "memory of what I've been saying" -- the standard way any
autoregressive transformer conditions on its own history -- so only the
user stream is ever fed in as external input. This is a breaking change:
old checkpoints trained against the previous DuplexFusion signature are
not compatible and must be retrained from scratch.

SPECTRAL FLUX LOSS (addresses a SEPARATE, well-documented limitation of
plain L1 mel regression, not the leakage bug above): per MELLE
(arXiv:2407.08551) and Ren et al. ("Revisiting Over-Smoothness in Text to
Speech"), deterministic L1/L2 regression against mel targets tends to
produce over-smoothed, blurry predictions -- averaging over plausible
spectral detail minimizes L1 loss even though the average doesn't match
any real frame. A spectral-flux term (L1 loss on the frame-to-frame
first-order difference of predicted vs. target mel, weighted by
train.flux_loss_weight) directly penalizes under-predicting the target's
real temporal variation. This does not fully solve over-smoothing (MELLE
also uses a variational/latent-sampling module instead of a deterministic
readout, which this codebase does not implement) -- it is a cheap,
directionally-correct mitigation, not a complete fix.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python train/stage2_duplex.py`

from acoustichead.mel_head import MelHead
from common import ensure_dir, load_config, pick_device
from encoder.wavlm_encoder import WAVLM_SAMPLE_RATE, WavLMEncoder
from projector.projector import Projector
from speechcore.duplex_fusion import DuplexFusion
from speechcore.qwen_speech_core import QwenSpeechCore
from vocoder.mel_utils import VOCOS_SAMPLE_RATE
from vocoder.vocos_wrapper import VocosVocoder


class SessionChunkDataset(Dataset):
    """Indexes every chunk across every session in the manifest up front.
    Each session's audio is cut into non-overlapping chunk_duration_s
    chunks; the final, shorter remainder of each session is kept (not
    dropped) as its own, shorter chunk, as long as it's >= min_chunk_s.
    Chunks therefore have variable length -- see collate_variable_length
    and the training loop's masking for how that's handled correctly."""

    def __init__(
        self, manifest_path: str, chunk_duration_s: float, min_chunk_s: float, user_channel: int, agent_channel: int
    ):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if not manifest:
            raise RuntimeError(f"Manifest at {manifest_path} is empty -- run dataprep/preprocess_otospeech.py first.")

        self.index: List[Tuple[str, str, int, int]] = []  # (user_wav, agent_wav, start_sample, length_samples)
        native_sr = None
        for sess in manifest:
            sr = sess["sample_rate"]
            if native_sr is None:
                native_sr = sr
            elif sr != native_sr:
                raise RuntimeError(
                    f"Session {sess['session_id']} has sample_rate {sr}, expected {native_sr} -- "
                    f"this script assumes one consistent native sample rate across the whole manifest."
                )

            channel_wavs = {0: sess["speaker_0_wav"], 1: sess["speaker_1_wav"]}
            user_wav = str(Path(sess["session_dir"]) / channel_wavs[user_channel])
            agent_wav = str(Path(sess["session_dir"]) / channel_wavs[agent_channel])

            chunk_len = int(chunk_duration_s * sr)
            min_len = int(min_chunk_s * sr)
            n_samples = int(sess["duration_seconds"] * sr)
            n_full_chunks = n_samples // chunk_len
            for c in range(n_full_chunks):
                self.index.append((user_wav, agent_wav, c * chunk_len, chunk_len))

            remainder = n_samples - n_full_chunks * chunk_len
            if remainder >= min_len:
                self.index.append((user_wav, agent_wav, n_full_chunks * chunk_len, remainder))

        self.native_sr = native_sr
        if not self.index:
            raise RuntimeError(
                f"No chunks could be extracted from any session in the manifest -- sessions may all be "
                f"shorter than min_chunk_s, or the manifest is malformed."
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        user_wav, agent_wav, start, length = self.index[idx]
        user, sr_u = torchaudio.load(user_wav, frame_offset=start, num_frames=length)
        agent, sr_a = torchaudio.load(agent_wav, frame_offset=start, num_frames=length)
        assert sr_u == self.native_sr and sr_a == self.native_sr, "sample rate drifted from manifest-declared rate"
        return user.mean(dim=0), agent.mean(dim=0)  # mono, variable-length (length,) each


def collate_variable_length(batch):
    """Keeps each item as its own variable-length 1-D tensor instead of
    stacking -- stacking would require equal lengths, which chunks
    deliberately don't have (see the remainder-chunk handling above)."""
    users, agents = zip(*batch)
    return list(users), list(agents)


def build_models(cfg: dict, device: torch.device, dtype: torch.dtype):
    wavlm = WavLMEncoder(cfg["models"]["wavlm_name"], device=device)
    vocos = VocosVocoder(cfg["models"]["vocos_repo"], device=device)
    qwen_core = QwenSpeechCore(
        cfg["models"]["qwen_name"],
        device=device,
        use_lora=True,
        lora_r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        dtype=dtype,
    )
    hidden_size = qwen_core.hidden_size
    projector = Projector(d_in=768, d_out=hidden_size).to(device=device, dtype=dtype)
    fusion = DuplexFusion(d_in=hidden_size, d_hidden=hidden_size).to(device=device, dtype=dtype)
    acoustic_head = MelHead(d_in=hidden_size, n_mels=cfg["audio"]["n_mels"]).to(device=device, dtype=dtype)
    return wavlm, vocos, qwen_core, projector, fusion, acoustic_head


def trainable_param_groups(qwen_core, projector, fusion, acoustic_head):
    return (
        list(qwen_core.trainable_parameters())
        + list(projector.parameters())
        + list(fusion.parameters())
        + list(acoustic_head.parameters())
    )


def save_checkpoint(ckpt_dir: Path, step: int, qwen_core, projector, fusion, acoustic_head, optimizer):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    qwen_core.model.save_pretrained(str(ckpt_dir / "qwen_lora"))
    torch.save(
        {
            "step": step,
            "projector": projector.state_dict(),
            "fusion": fusion.state_dict(),
            "acoustic_head": acoustic_head.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        ckpt_dir / "train_state.pt",
    )
    print(f"[checkpoint] saved at step {step} -> {ckpt_dir}")


def try_resume(ckpt_dir: Path, qwen_core, projector, fusion, acoustic_head, optimizer, device) -> int:
    state_path = ckpt_dir / "train_state.pt"
    if not state_path.exists():
        return 0
    print(f"[resume] found checkpoint at {ckpt_dir}, loading...")
    state = torch.load(state_path, map_location=device)
    projector.load_state_dict(state["projector"])
    fusion.load_state_dict(state["fusion"])
    acoustic_head.load_state_dict(state["acoustic_head"])
    optimizer.load_state_dict(state["optimizer"])
    from peft import PeftModel

    qwen_core.model = PeftModel.from_pretrained(qwen_core.model.get_base_model(), str(ckpt_dir / "qwen_lora"))
    qwen_core.model.to(device)
    print(f"[resume] resumed at step {state['step']}")
    return state["step"]


def _require_channel_roles_verified(cfg: dict) -> None:
    if cfg["audio"].get("channel_roles_verified", False):
        return
    raise RuntimeError(
        "Refusing to train: audio.channel_roles_verified is false (or missing) in this config.\n"
        "audio.user_channel/agent_channel have not been confirmed against this dataset's real "
        "speaker-role metadata -- training on a swapped assumption would silently corrupt every result.\n"
        "Fix: run `python dataprep/print_channel_roles.py --manifest <manifest_path>`, read the printed "
        "speaker roles yourself, set user_channel/agent_channel correctly in this config, then set "
        "audio.channel_roles_verified: true."
    )


def train(cfg_path: str, resume: bool = True) -> None:
    cfg = load_config(cfg_path)
    _require_channel_roles_verified(cfg)

    device = pick_device(cfg["runtime"]["device"])
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg["runtime"]["dtype"]]
    ckpt_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])

    print(f"Device: {device}, dtype: {dtype}")
    wavlm, vocos, qwen_core, projector, fusion, acoustic_head = build_models(cfg, device, dtype)
    print("=== Qwen speech core ===")
    for k, v in qwen_core.verify_loaded().items():
        print(f"  {k}: {v}")

    dataset = SessionChunkDataset(
        manifest_path=cfg["paths"]["manifest_path"],
        chunk_duration_s=cfg["audio"]["chunk_duration_s"],
        min_chunk_s=cfg["audio"]["min_chunk_s"],
        user_channel=cfg["audio"]["user_channel"],
        agent_channel=cfg["audio"]["agent_channel"],
    )
    print(f"Dataset: {len(dataset)} chunks (target length {cfg['audio']['chunk_duration_s']}s, "
          f"trailing remainders >= {cfg['audio']['min_chunk_s']}s kept) at native sample rate {dataset.native_sr}Hz.")

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 2),
        drop_last=True,
        collate_fn=collate_variable_length,
    )

    optimizer = torch.optim.AdamW(
        trainable_param_groups(qwen_core, projector, fusion, acoustic_head),
        lr=cfg["train"]["lr"],
    )

    start_step = try_resume(ckpt_dir, qwen_core, projector, fusion, acoustic_head, optimizer, device) if resume else 0

    log_path = ckpt_dir / "train_log.csv"
    log_is_new = not log_path.exists()
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if log_is_new:
        log_writer.writerow(["step", "loss", "recon_loss", "flux_loss", "lr", "epoch", "wall_time_s"])

    max_steps = cfg["train"]["max_steps"]
    save_every = cfg["train"]["save_every"]
    log_every = cfg["train"].get("log_every", 10)
    t0 = time.time()
    step = start_step
    epoch = 0

    print(f"Starting training from step {step} toward max_steps {max_steps}...")
    while step < max_steps:
        epoch += 1
        for user_list, agent_list in loader:
            if step >= max_steps:
                break

            native_sr = dataset.native_sr
            # Deliberately kept on CPU here: WavLMEncoder/VocosVocoder both move
            # their inputs to the model's own device internally, and
            # WavLMEncoder's feature extractor needs CPU tensors (it calls
            # .numpy() on each row) -- moving to `device` first would break that.
            user_16k = [torchaudio.functional.resample(w, native_sr, WAVLM_SAMPLE_RATE) for w in user_list]
            agent_24k = [torchaudio.functional.resample(w, native_sr, VOCOS_SAMPLE_RATE) for w in agent_list]
            batch_size = len(user_16k)

            with torch.no_grad():
                # Only the user stream is ever encoded as model input now --
                # agent audio is used solely to build the loss target below,
                # never fed into the model (see the duplex-fusion note above).
                user_feat, frame_mask = wavlm(user_16k, WAVLM_SAMPLE_RATE)

                # Exact per-item mel targets (no batch-wide length assumption),
                # then zero-padded to the batch's max mel length with an explicit mask.
                mel_items = [vocos.mel_from_waveform(a.unsqueeze(0)).squeeze(0) for a in agent_24k]
                n_mels = mel_items[0].shape[0]
                t_mel_max = max(m.shape[-1] for m in mel_items)
                target_mel = torch.zeros(batch_size, n_mels, t_mel_max, device=device)
                mel_mask = torch.zeros(batch_size, t_mel_max, dtype=torch.bool, device=device)
                for i, m in enumerate(mel_items):
                    t_i = m.shape[-1]
                    target_mel[i, :, :t_i] = m
                    mel_mask[i, :t_i] = True

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                user_emb = projector(user_feat.to(dtype))
                fused = fusion(user_emb)

                hidden = qwen_core(fused, attention_mask=frame_mask)  # (B, T_wavlm, hidden)
                pred_mel_full = acoustic_head(hidden).transpose(1, 2).float()  # (B, n_mels, T_wavlm)

                # Resample each item's prediction from WavLM's rate to ITS OWN true
                # mel length, using only its valid (unpadded) region -- not a
                # batch-wide average, and never touching padded frames.
                pred_mel_aligned = torch.zeros_like(target_mel)
                for i in range(batch_size):
                    t_wavlm_i = int(frame_mask[i].sum().item())
                    t_mel_i = int(mel_mask[i].sum().item())
                    valid_pred = pred_mel_full[i : i + 1, :, :t_wavlm_i]
                    aligned = F.interpolate(valid_pred, size=t_mel_i, mode="linear", align_corners=False)
                    pred_mel_aligned[i, :, :t_mel_i] = aligned.squeeze(0)

                mask_f = mel_mask.unsqueeze(1).to(target_mel.dtype)  # (B, 1, T_mel_max)
                abs_diff = (pred_mel_aligned - target_mel).abs() * mask_f
                recon_loss = abs_diff.sum() / (mask_f.sum() * n_mels).clamp_min(1.0)

                # Spectral flux loss (per MELLE, arXiv:2407.08551, and Ren et al.
                # "Revisiting Over-Smoothness in TTS"): plain L1/L2 mel regression
                # is documented to cause over-smoothed/blurry output, because
                # averaging over plausible details minimizes L1 loss even though
                # it doesn't match any real frame. Penalizing the mismatch in
                # frame-to-frame variation directly discourages the model from
                # under-predicting the target's actual temporal variation.
                pred_flux = pred_mel_aligned[:, :, 1:] - pred_mel_aligned[:, :, :-1]
                target_flux = target_mel[:, :, 1:] - target_mel[:, :, :-1]
                flux_mask = mask_f[:, :, 1:]  # a flux frame is valid only if both frames behind it are
                flux_diff = (pred_flux - target_flux).abs() * flux_mask
                flux_loss = flux_diff.sum() / (flux_mask.sum() * n_mels).clamp_min(1.0)

                flux_weight = cfg["train"].get("flux_loss_weight", 0.5)
                loss = recon_loss + flux_weight * flux_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_param_groups(qwen_core, projector, fusion, acoustic_head), 1.0)
            optimizer.step()
            step += 1

            if step % log_every == 0:
                elapsed = time.time() - t0
                lr = optimizer.param_groups[0]["lr"]
                print(f"step {step}/{max_steps} | epoch {epoch} | loss {loss.item():.4f} "
                      f"(recon {recon_loss.item():.4f} + {flux_weight}*flux {flux_loss.item():.4f}) | "
                      f"elapsed {elapsed:.0f}s")
                log_writer.writerow([step, loss.item(), recon_loss.item(), flux_loss.item(), lr, epoch, elapsed])
                log_file.flush()

            if step % save_every == 0:
                save_checkpoint(ckpt_dir, step, qwen_core, projector, fusion, acoustic_head, optimizer)

        if step >= max_steps:
            break

    save_checkpoint(ckpt_dir, step, qwen_core, projector, fusion, acoustic_head, optimizer)
    log_file.close()
    print(f"Training finished at step {step}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2.yaml")
    parser.add_argument("--no_resume", action="store_true", help="Ignore any existing checkpoint and start fresh.")
    args = parser.parse_args()
    train(args.config, resume=not args.no_resume)
