#!/bin/bash
#SBATCH --job-name=dpfe-attacks
#SBATCH --output=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.out
#SBATCH --error=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.err
#SBATCH --partition=muma_2021
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ismailj@usf.edu

# Multi-attack-type experiment — no DP noise.
# Trains GPT-2 once, then runs all attack strategies to compare effectiveness.
# Swap MODEL_NAME / OUTPUT_DIR for gpt2-large run.

REAL_HOME=/home/i/ismailj
CONDA_ENV=${REAL_HOME}/miniconda3/envs/my_environment
WORKDIR=${REAL_HOME}/dpfe-email-privacy-experiment

export HF_HOME=${REAL_HOME}/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export FRESH=0          # set to 1 to wipe OUTPUT_DIR and start clean
export SMOKE=0          # set to 1 for a fast ~15 min smoke test
export GRAD_ACCUM_STEPS=8
export LEARNING_RATE=9.82e-05
export ATTACK_BATCH_SIZE=32
export LORA_R=64
export LORA_ALPHA=128
export USE_LORA=0       # full fine-tuning on RTX 6000 (24 GB VRAM)
export MODEL_NAME=gpt2
export OUTPUT_DIR=results/gpt2-base-attacks
export MAX_EMAILS=50000
export EPOCHS=3
export SUBSET_PAIRS=3238
export MAX_NEW_TOKENS=100
export BATCH_SIZE=16
export MAX_LENGTH=512
export MAX_GRAD_NORM=4.63
# ATTACK_TYPES: "all" runs all 15 default types (excludes context_* by default).
# Set to a comma-separated list to run a subset, e.g. "zs_d_greedy,fs_5_greedy".
# Include "context_50,context_100,context_200" explicitly to run context attacks.
export ATTACK_TYPES=all

source ${REAL_HOME}/miniconda3/etc/profile.d/conda.sh
conda activate my_environment

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "HOME:   ${HOME}  REAL_HOME: ${REAL_HOME}"
echo "CONDA_PREFIX: ${CONDA_PREFIX}"
echo "Start:  $(date)"

mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"
python -u main.py

echo "End: $(date)"
