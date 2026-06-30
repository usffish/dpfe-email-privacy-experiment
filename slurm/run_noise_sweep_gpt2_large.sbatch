#!/bin/bash
#SBATCH --job-name=dpfe-sweep-gpt2l
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ismailj@usf.edu

# DP-SGD noise sweep for GPT-2 Large (774M), single best attack (zs_d_greedy).
# Trains one full fine-tune per sigma level then runs zs_d_greedy.
# Uses gpt2-large-hpo-v2 best config (Trial #32, val_loss=2.1984).
# Checkpoint/resume: results fetched from GitHub at startup; resubmit with FRESH=0.
# Estimated runtime: ~6h per sigma level x 10 levels = ~60h (1 submission).
# NOTE: /home is full — checkpoints go to /tmp, results pushed to GitHub after each sigma.

REAL_HOME=/home/i/ismailj
WORKDIR=${REAL_HOME}/dpfe-email-privacy-experiment

# All writes go to compute node /tmp — never touches the full /home filesystem.
export CHECKPOINT_DIR=/tmp/dpfe_${SLURM_JOB_ID}
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Results persist to GitHub (bypasses /home).
export GITHUB_TOKEN=$(cat /home/i/ismailj/.github_token 2>/dev/null || echo "")
export GITHUB_REPO=usffish/attack

export FRESH=0
export SMOKE=0

# gpt2-large-hpo-v2 best config (Trial #32, val_loss=2.1984)
export LEARNING_RATE=9.024263343579973e-06
export LR_SCHEDULE=cosine
export WEIGHT_DECAY=0.057443059201871274
export WARMUP_FRACTION=0.05337337560305046
export MAX_GRAD_NORM=1.4497073847836068
export GRAD_ACCUM_STEPS=2

export USE_LORA=0
export MODEL_NAME=gpt2-large
export OUTPUT_DIR=results/gpt2-large-noise-sweep
export MAX_EMAILS=50000
export EPOCHS=3
export SUBSET_PAIRS=3238
export MAX_NEW_TOKENS=100
export BATCH_SIZE=4
export MAX_LENGTH=256
export ATTACK_BATCH_SIZE=16

export DP_NOISE_LEVELS=0,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0,50.0
export DP_ATTACK_TYPE=zs_d_greedy

mkdir -p "${CHECKPOINT_DIR}" "${HF_HOME}"

source ${REAL_HOME}/miniconda3/etc/profile.d/conda.sh
conda activate my_environment

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"

cd "${WORKDIR}"
python -u main.py

echo "End: $(date)"
