#!/bin/bash
#SBATCH --job-name=dpfe-hpo-neo
#SBATCH --output=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.out
#SBATCH --error=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.err
#SBATCH --partition=muma_2021
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ismailj@usf.edu

# One HPO trial per job. Submit N in parallel with: bash submit_hpo.sh N run_hpo_gptneo.sbatch
# All jobs share the same study via JournalFileBackend on the shared filesystem.
# HyperBand prunes bad configs after epoch 1 — most jobs finish in ~30 min.

REAL_HOME=/home/i/ismailj
WORKDIR=${REAL_HOME}/dpfe-email-privacy-experiment

export HF_HOME=${REAL_HOME}/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ── Model and data (fixed across all trials) ──────────────────────────────────
export MODEL_NAME=EleutherAI/gpt-neo-125M
export DATA_DIR=enron_data
export OUTPUT_DIR=results/gpt-neo-125m-attacks
export MAX_LENGTH=128
export BATCH_SIZE=2
export GRAD_ACCUM_STEPS=8
export MAX_GRAD_NORM=1.0
export SEED=42
export USE_LORA=0

# ── HPO settings ──────────────────────────────────────────────────────────────
export HPO_STUDY_NAME=gpt-neo-hpo-v1
export HPO_STORAGE=/home/i/ismailj/dpfe-email-privacy-experiment/hpo_study.jsonl
export HPO_N_TRIALS=1
export HPO_EMAILS=10000
export HPO_PAIRS=3238
export HPO_ATTACK=zs_d_greedy
export ATTACK_BATCH_SIZE=32
export HPO_MAX_EPOCHS=5

source ${REAL_HOME}/miniconda3/etc/profile.d/conda.sh
conda activate my_environment

echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURMD_NODENAME}"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Study:   ${HPO_STUDY_NAME}"
echo "Storage: ${HPO_STORAGE}"
echo "Start:   $(date)"

mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"
python -u hpo_trial.py

echo "End: $(date)"
