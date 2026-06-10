#!/bin/bash
#SBATCH --job-name=dpfe-large
#SBATCH --output=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.out
#SBATCH --error=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.err
#SBATCH --partition=muma_2021
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ismailj@usf.edu

# Hardcoded real home — immune to SLURM inheriting HOME from submission env
REAL_HOME=/home/i/ismailj
CONDA_ENV=${REAL_HOME}/miniconda3/envs/my_environment
WORKDIR=${REAL_HOME}/dpfe-email-privacy-experiment

export HF_HOME=${REAL_HOME}/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# GPT-2 Large (774M) — float32 — LoRA fine-tuning with Opacus DP-SGD.
# Full run: 50k emails, 3 epochs, all 3238 attack pairs, 5 noise levels.
# Estimated ~100h total across all sigmas; checkpoint/resume handles restarts.
# Resubmit this script if the job hits the time limit — it will skip completed sigmas.
export FRESH=0          # set to 1 to wipe OUTPUT_DIR and start a clean run
export SMOKE=0          # set to 1 for a fast end-to-end check (~15 min, separate output dir)
export GRAD_ACCUM_STEPS=8
export LEARNING_RATE=2e-4
export ATTACK_BATCH_SIZE=32
export MODEL_NAME=gpt2-large
export OUTPUT_DIR=results/gpt2-large
export MAX_EMAILS=50000
export EPOCHS=3
export SUBSET_PAIRS=3238
export MAX_NEW_TOKENS=100
export BATCH_SIZE=2
export MAX_LENGTH=128

source ${REAL_HOME}/miniconda3/etc/profile.d/conda.sh
conda activate my_environment

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "HOME:   ${HOME}  REAL_HOME: ${REAL_HOME}"
echo "CONDA_PREFIX: ${CONDA_PREFIX}"
echo "LD_LIB: ${LD_LIBRARY_PATH}"
echo "Start:  $(date)"

mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"
python -u main.py

echo "End: $(date)"
