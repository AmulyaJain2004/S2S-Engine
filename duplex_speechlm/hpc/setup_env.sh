#!/usr/bin/env bash
# One-time environment setup on the H100 cluster login node.
# Run this ONCE interactively after scp-ing the repo over (see hpc/README.md).
# Do NOT run this inside the PBS job itself -- building a venv on a compute
# node wastes allocated GPU time for no reason.
set -euo pipefail

# EDIT ME: module names are cluster-specific. Check `module avail` yourself
# -- these are placeholders, not confirmed against your college's modules.
module load python/3.11 2>/dev/null || echo "[warn] module load python/3.11 failed -- check 'module avail' and edit this script"
module load cuda/12.1 2>/dev/null || echo "[warn] module load cuda/12.1 failed -- check 'module avail' and edit this script"

cd "$(dirname "$0")/.."   # repo root (duplex_speechlm/)

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Environment ready at $(pwd)/.venv"
echo "Next: huggingface-cli login (needed once, for gated otoSpeech access),"
echo "then edit hpc/configs/stage2_hpc.yaml paths, then qsub hpc/run_stage2.pbs"
