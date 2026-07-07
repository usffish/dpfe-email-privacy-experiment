#!/bin/bash
# End-to-end smoke test — fast sanity check after any code change.
# Runs 500 emails, 50 pairs, 1 epoch, 64-token sequences, 2 attack types (~15 min on GPU).
# Submit from: /work_bgfs/i/ismailj/dpfe-email-privacy-experiment
#   sbatch --partition=muma_2021 --qos=muma21 --gres=gpu:1 --cpus-per-task=2 \
#     --mem=16G --time=01:00:00 --job-name=dpfe-smoke \
#     --output=/work_bgfs/i/ismailj/logs/%j.out \
#     --error=/work_bgfs/i/ismailj/logs/%j.err \
#     slurm/smoke_workaround.sh

REAL_HOME=/home/i/ismailj
WORKDIR=/work_bgfs/i/ismailj/dpfe-email-privacy-experiment

export HF_HOME=/work_bgfs/i/ismailj/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export FRESH=0
export SMOKE=1
export MODEL_NAME=EleutherAI/gpt-neo-125M
export OUTPUT_DIR=results/smoke

source ${REAL_HOME}/miniconda3/etc/profile.d/conda.sh
conda activate my_environment

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

cd "${WORKDIR}"
python -u main.py

echo "Done: $(date)"
