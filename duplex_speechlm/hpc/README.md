# Running Stage 2 on the college H100 PBS cluster

This folder is prepared now so it's ready to go once you have cluster access;
you said that part comes later. Everything here assumes: SSH access, `scp`
for file transfer, and a PBS (`qsub`/`qstat`) batch scheduler -- adjust if
that turns out to be wrong.

## What's unconfirmed and must be edited before this will run

Every file below has an `EDIT ME` comment at the exact line that's a
placeholder, not a verified value:

- `setup_env.sh` -- module names (`module load python/3.11`, `module load cuda/12.1`) are guesses; run `module avail` on the cluster and fix them.
- `run_stage2.pbs` / `run_preprocess.pbs` -- the `#PBS -l select=...` GPU resource string and `#PBS -q` queue name are placeholders. PBS clusters vary a lot here (`ngpus=1` vs `select=1:ngpus=1:gpu_type=h100` vs a separate `-l gres=gpu:h100:1`, etc.) -- check `qsub --help`, cluster docs, or ask the admins, then edit both files.
- `configs/stage2_hpc.yaml` -- the two `/scratch/CHANGE_ME/...` paths need your actual scratch/home quota path.

Nothing here will silently run with wrong values -- wrong module names fail
loudly at `setup_env.sh` time, and a wrong PBS resource string either gets
rejected at `qsub` time or queues forever, not silently run on a CPU node.

## One-time setup, once you have access

```bash
# from your local machine, with the repo committed/pushed or just zipped up
scp -r duplex_speechlm/ you@cluster-login:~/s2s-engine/

ssh you@cluster-login
cd ~/s2s-engine
bash hpc/setup_env.sh          # builds .venv, installs requirements.txt
source .venv/bin/activate
huggingface-cli login          # needed once, for gated otoSpeech access
```

Edit `hpc/configs/stage2_hpc.yaml`'s two paths and `hpc/run_preprocess.pbs`'s
`PROCESSED_DIR` to match, and fix the PBS resource/queue lines noted above.

## Each run

```bash
# 1. Preprocess the dataset (CPU-only job, no GPU request needed/wanted)
qsub hpc/run_preprocess.pbs
qstat -u $USER                 # watch until it finishes

# 2. GATE -- confirm which channel is the user vs. agent side. train/stage2_duplex.py
# hard-refuses to start otherwise. Read the output yourself, then edit
# hpc/configs/stage2_hpc.yaml: set audio.user_channel/agent_channel to match what you
# saw, and only then set audio.channel_roles_verified: true.
python dataprep/print_channel_roles.py --manifest "$PROCESSED_DIR/manifest.json" --n_sessions 5

# 3. Train (GPU job)
qsub hpc/run_stage2.pbs
qstat -u $USER
```

Since you'll check results the next day: `run_stage2.pbs` logs to
`stage2_job.log` in the submit directory, and the training script itself
writes `checkpoint_dir/train_log.csv` (step, loss, **accuracy**, lr, epoch,
wall time -- accuracy is the discrete-unit prediction accuracy, a much
more directly readable signal than the old mel-regression loss was) plus
a checkpoint every `save_every` steps. If the job dies from hitting
`walltime` before `max_steps`, just `qsub hpc/run_stage2.pbs` again --
`train/stage2_duplex.py`'s `try_resume` picks up from the last checkpoint
automatically rather than restarting at step 0. Bump `#PBS -l walltime=` in
`run_stage2.pbs` if your queue allows longer single jobs and you'd rather
not chain resumes.

## Listening to what it actually generated

The cluster has no speakers, so this just produces WAV files to pull back
and listen to locally. `eval/generate_stage2.py` encodes the user audio
into discrete units and runs windowed, KV-cached generation -- the same
mechanism as training, no self-feedback loop -- on a real held-out
user-channel clip:

```bash
python eval/generate_stage2.py \
    --config hpc/configs/stage2_hpc.yaml \
    --user_wav /path/to/some/processed/session_X/speaker_0.wav \
    --window_s 0.4 --max_duration_s 30 \
    --out_dir outputs/generation_check
```

Then from your local machine:

```bash
scp you@cluster-login:~/s2s-engine/outputs/generation_check/*.wav .
```

A falling loss / rising accuracy in `train_log.csv` alone doesn't tell you
the output is intelligible or well-timed -- actually listening to
`generated_agent.wav` is the real check.

## Syncing checkpoints back

```bash
# from your local machine
scp -r you@cluster-login:~/s2s-engine/<checkpoint_dir>/stage2_discrete ./checkpoints_from_hpc
```

Or push just `train_log.csv` for a quick look without pulling full model
checkpoints:

```bash
scp you@cluster-login:~/s2s-engine/<checkpoint_dir>/stage2_discrete/train_log.csv .
```
