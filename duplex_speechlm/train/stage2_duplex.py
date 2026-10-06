"""Stage 2: duplex fine-tune on real otoSpeech two-channel conversation audio.

Per spec section 7, Stage 2 is what actually teaches backchannel/
interruption/overlap behavior -- it trains on real duplex audio, not
synthetic single-turn TTS pairs. This script skips Stage 1 (content
bootstrap) by direct instruction.

ARCHITECTURE: DISCRETE UNITS, NOT CONTINUOUS MEL REGRESSION (a deliberate,
researched pivot -- see speechcore/discrete_tokenizer.py's docstring for
the full reasoning and citations). Both the user and agent streams are
encoded into discrete unit IDs by a frozen, pretrained WavLM-large +
k-means(1000) pipeline (speechbrain's DiscreteSSL). The speech core
(Qwen2.5-1.5B-Instruct, LoRA-adapted) is trained as an ordinary
autoregressive classifier: embed the user units, predict the agent's
units at each frame via cross-entropy. This matches how dGSLM and Moshi
-- the validated full-duplex speech-LM systems in the literature -- are
actually built, and avoids two separate real problems the previous
continuous-mel-regression version had:
  1. Over-smoothed/blurry output (a documented property of plain L1/L2
     mel regression -- MELLE, arXiv:2407.08551) is structurally impossible
     here: classification doesn't average over plausible targets the way
     deterministic regression does.
  2. The WavLM-50Hz-vs-Vocos-93.75Hz rate mismatch, and the per-item
     interpolation code it required, no longer exists: both the user and
     agent streams are unit sequences from the SAME WavLM+k-means
     pipeline, so they share an identical frame count by construction
     (asserted below, not just assumed).

ONLY THE USER STREAM IS EVER FED IN AS MODEL INPUT (carried over from the
previous fix, and still correct under this new architecture): the speech
core's own causal self-attention over its own past hidden states is what
carries "memory of what it's been saying" -- the standard way any
autoregressive transformer conditions on its own history. Training and
eval/generate_stage2.py's generation now use the exact same mechanism,
with no self-feedback audio loop anywhere.

BUDGET-DRIVEN CHOICE: the k-means quantizer and the unit-to-waveform
vocoder are pretrained, frozen, and never trained in this codebase --
training a neural audio codec or GAN vocoder from scratch would not fit
inside a single Colab A100 session's budget. Only the embedding, LoRA
adapters, and classification head are trained here.

CHANNEL-ROLE ASSUMPTION (still a hard gate, not a silent guess): this
script refuses to start unless audio.channel_roles_verified: true is set
in the config, which should only happen after running
dataprep/print_channel_roles.py and confirming audio.user_channel /
audio.agent_channel against the real otoSpeech speaker-role metadata.

BREAKING CHANGE: checkpoints from any earlier version of this script
(continuous-mel or the pre-fix duplex-fusion design) are not compatible
and must be retrained from scratch.
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

from acoustichead.unit_head import UnitHead
from common import ensure_dir, load_config, pick_device
from projector.unit_embedding import UnitEmbedding
from speechcore.discrete_tokenizer import DiscreteSpeechTokenizer
from speechcore.qwen_speech_core import QwenSpeechCore


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
    tokenizer = DiscreteSpeechTokenizer(
        ssl_model=cfg["discrete"]["ssl_model"],
        layer_num=cfg["discrete"]["layer_num"],
        num_clusters=cfg["discrete"]["num_clusters"],
        kmeans_dataset=cfg["discrete"]["kmeans_dataset"],
        vocoder_repo_id=cfg["discrete"]["vocoder_repo_id"],
        device=device,
    )
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
    vocab_size = cfg["discrete"]["num_clusters"]
    embedding = UnitEmbedding(vocab_size=vocab_size, d_out=hidden_size).to(device=device, dtype=dtype)
    unit_head = UnitHead(d_in=hidden_size, vocab_size=vocab_size).to(device=device, dtype=dtype)
    return tokenizer, qwen_core, embedding, unit_head


def trainable_param_groups(qwen_core, embedding, unit_head):
    return (
        list(qwen_core.trainable_parameters())
        + list(embedding.parameters())
        + list(unit_head.parameters())
    )


def save_checkpoint(ckpt_dir: Path, step: int, qwen_core, embedding, unit_head, optimizer):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    qwen_core.model.save_pretrained(str(ckpt_dir / "qwen_lora"))
    torch.save(
        {
            "step": step,
            "embedding": embedding.state_dict(),
            "unit_head": unit_head.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        ckpt_dir / "train_state.pt",
    )
    print(f"[checkpoint] saved at step {step} -> {ckpt_dir}")


def try_resume(ckpt_dir: Path, qwen_core, embedding, unit_head, optimizer, device) -> int:
    state_path = ckpt_dir / "train_state.pt"
    if not state_path.exists():
        return 0
    print(f"[resume] found checkpoint at {ckpt_dir}, loading...")
    state = torch.load(state_path, map_location=device)
    embedding.load_state_dict(state["embedding"])
    unit_head.load_state_dict(state["unit_head"])
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
    tokenizer, qwen_core, embedding, unit_head = build_models(cfg, device, dtype)
    print("=== Qwen speech core ===")
    for k, v in qwen_core.verify_loaded().items():
        print(f"  {k}: {v}")
    print(f"Discrete unit vocab size: {tokenizer.vocab_size}")

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
        trainable_param_groups(qwen_core, embedding, unit_head),
        lr=cfg["train"]["lr"],
    )

    start_step = try_resume(ckpt_dir, qwen_core, embedding, unit_head, optimizer, device) if resume else 0

    log_path = ckpt_dir / "train_log.csv"
    log_is_new = not log_path.exists()
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if log_is_new:
        log_writer.writerow(["step", "loss", "accuracy", "lr", "epoch", "wall_time_s"])

    max_steps = cfg["train"]["max_steps"]
    save_every = cfg["train"]["save_every"]
    log_every = cfg["train"].get("log_every", 10)
    native_sr = dataset.native_sr
    t0 = time.time()
    step = start_step
    epoch = 0

    print(f"Starting training from step {step} toward max_steps {max_steps}...")
    while step < max_steps:
        epoch += 1
        for user_list, agent_list in loader:
            if step >= max_steps:
                break

            # Encode both streams to discrete units. Per-item (not batched)
            # because DiscreteSSL's batching semantics for variable-length,
            # unpadded audio aren't something to assume without verifying --
            # correctness first, batch-encode as a speed optimization later
            # if throughput on the actual A100 run needs it.
            with torch.no_grad():
                user_units_list = [tokenizer.encode(u, native_sr) for u in user_list]
                agent_units_list = [tokenizer.encode(a, native_sr) for a in agent_list]

            # Both streams come from the same chunk duration through the same
            # WavLM+k-means pipeline, so their frame counts must match exactly
            # -- asserted, not assumed, so a real mismatch fails loudly instead
            # of silently misaligning user input against agent target.
            for i, (u, a) in enumerate(zip(user_units_list, agent_units_list)):
                assert u.shape[0] == a.shape[0], (
                    f"batch item {i}: user/agent unit counts differ ({u.shape[0]} vs {a.shape[0]}) -- "
                    f"this should be impossible for equal-length input audio, investigate the tokenizer."
                )

            batch_size = len(user_units_list)
            t_max = max(u.shape[0] for u in user_units_list)
            user_units = torch.zeros(batch_size, t_max, dtype=torch.long, device=device)
            target_units = torch.zeros(batch_size, t_max, dtype=torch.long, device=device)
            frame_mask = torch.zeros(batch_size, t_max, dtype=torch.long, device=device)
            for i, (u, a) in enumerate(zip(user_units_list, agent_units_list)):
                t_i = u.shape[0]
                user_units[i, :t_i] = u
                target_units[i, :t_i] = a
                frame_mask[i, :t_i] = 1

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                user_emb = embedding(user_units)  # (B, T, hidden)
                hidden = qwen_core(user_emb, attention_mask=frame_mask)  # (B, T, hidden)
                logits = unit_head(hidden).float()  # (B, T, vocab)

                flat_logits = logits.reshape(-1, logits.shape[-1])
                flat_targets = target_units.reshape(-1)
                flat_mask = frame_mask.reshape(-1).float()

                per_frame_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
                loss = (per_frame_loss * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)

                with torch.no_grad():
                    preds = flat_logits.argmax(dim=-1)
                    accuracy = ((preds == flat_targets).float() * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_param_groups(qwen_core, embedding, unit_head), 1.0)
            optimizer.step()
            step += 1

            if step % log_every == 0:
                elapsed = time.time() - t0
                lr = optimizer.param_groups[0]["lr"]
                print(f"step {step}/{max_steps} | epoch {epoch} | loss {loss.item():.4f} | "
                      f"unit-accuracy {accuracy.item():.3f} | elapsed {elapsed:.0f}s")
                log_writer.writerow([step, loss.item(), accuracy.item(), lr, epoch, elapsed])
                log_file.flush()

            if step % save_every == 0:
                save_checkpoint(ckpt_dir, step, qwen_core, embedding, unit_head, optimizer)

        if step >= max_steps:
            break

    save_checkpoint(ckpt_dir, step, qwen_core, embedding, unit_head, optimizer)
    log_file.close()
    print(f"Training finished at step {step}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2.yaml")
    parser.add_argument("--no_resume", action="store_true", help="Ignore any existing checkpoint and start fresh.")
    args = parser.parse_args()
    train(args.config, resume=not args.no_resume)
