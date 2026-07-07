#!/bin/bash
#SBATCH --job-name=dpfe-attacks-gpt2l
#SBATCH --output=/work_bgfs/i/ismailj/logs/%j.out
#SBATCH --error=/work_bgfs/i/ismailj/logs/%j.err
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ismailj@usf.edu

# Multi-attack-type experiment for GPT-2 Large (774M) — no DP noise.
# Trains once using gpt2-large-hpo-v2 best config (Trial #32, val_loss=2.1984),
# then runs all 15 attack strategies to compare effectiveness.
# Note: max_length=256 (OOM at 512 with batch_size>=8 for 774M model).
# Submit from: /work_bgfs/i/ismailj/dpfe-email-privacy-experiment
#   sbatch slurm/run_attacks_gpt2_large.sbatch

REAL_HOME=/home/i/ismailj
WORKDIR=/work_bgfs/i/ismailj/dpfe-email-privacy-experiment

export HF_HOME=/work_bgfs/i/ismailj/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

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
export OUTPUT_DIR=results/gpt2-large-attacks
export MAX_EMAILS=50000
export EPOCHS=3
export SUBSET_PAIRS=3238
export MAX_NEW_TOKENS=100
export BATCH_SIZE=4
export MAX_LENGTH=256
export ATTACK_BATCH_SIZE=16
export ATTACK_TYPES=all

source ${REAL_HOME}/miniconda3/etc/profile.d/conda.sh
conda activate my_environment

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"

cd "${WORKDIR}"
python -u main.py

echo "End: $(date)"
