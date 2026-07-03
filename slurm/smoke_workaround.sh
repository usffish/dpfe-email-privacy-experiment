#!/bin/bash
# End-to-end smoke test of the /home-full workaround (GPT-2 Base, SMOKE mode).
# Validates: clone code into /tmp, stage enron_data from /home, download model
# to /tmp, train + attack at sigma 0 and 0.1, and PUSH results to GitHub.
# The token is read from the environment (never hardcoded); inject it at run:
#   ssh circe "srun -p muma_2021 --qos=muma21 --gres=gpu:1 --cpus-per-task=2 \
#     --mem=16G --time=02:00:00 --export=ALL,GITHUB_TOKEN=<tok> bash -s" \
#     < slurm/smoke_workaround.sh
set -e
set -o pipefail
REAL_HOME=/home/i/ismailj
JOBTMP=/tmp/dpfe_smoke_$$
mkdir -p "$JOBTMP"; cd "$JOBTMP"
export GITHUB_TOKEN="${GITHUB_TOKEN:-}"
export GITHUB_REPO="${GITHUB_REPO:-usffish/dpfe-email-privacy-experiment}"
export GITHUB_BRANCH="${GITHUB_BRANCH:-attack}"
echo "[smoke] node=$(hostname) cloning ${GITHUB_REPO}@${GITHUB_BRANCH}..."
git clone --depth 1 -b "$GITHUB_BRANCH" \
    "https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git" repo
cd repo
echo "[smoke] staging enron_data from /home..."
cp -r "${REAL_HOME}/dpfe-email-privacy-experiment/enron_data" enron_data
export CHECKPOINT_DIR="$JOBTMP/ckpt"
export HF_HOME="$JOBTMP/hf"
mkdir -p "$CHECKPOINT_DIR" "$HF_HOME"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 FRESH=0 SMOKE=1
export MODEL_NAME=gpt2
export OUTPUT_DIR=results/smoke-workaround
export MAX_NEW_TOKENS=40
export DP_NOISE_LEVELS=0,0.1
export DP_ATTACK_TYPE=zs_d_greedy
source "${REAL_HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate my_environment
echo "[smoke] running main.py (SMOKE=1)..."
python -u main.py
echo "[smoke] DONE — check results/smoke-workaround/smoke/table_11_results.json on the attack branch"
