#!/bin/bash
#SBATCH --job-name=dpfe-base
#SBATCH --output=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.out
#SBATCH --error=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.err
#SBATCH --partition=snsm_itn19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ismailjhaveri@gmail.com

# Hardcoded real home — immune to SLURM inheriting HOME from submission env
REAL_HOME=/home/i/ismailj
CONDA_ENV=${REAL_HOME}/miniconda3/envs/my_environment
WORKDIR=${REAL_HOME}/dpfe-email-privacy-experiment

export HF_HOME=${REAL_HOME}/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# GPT-2 base (117M) — float32 — LoRA fine-tuning with Opacus DP-SGD.
# Full run: 50k emails, 3 epochs, all 3238 attack pairs, 5 noise levels.
# Estimated ~15h total; 24h limit should be sufficient in one submission.
# Same hyperparameters as run.sbatch (gpt2-large) for a fair comparison.
export FRESH=0          # set to 1 to wipe OUTPUT_DIR and start a clean run
export MODEL_NAME=gpt2
export OUTPUT_DIR=results/gpt2-base
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
