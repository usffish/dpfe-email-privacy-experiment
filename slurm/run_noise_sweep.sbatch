#!/bin/bash
#SBATCH --job-name=dpfe-noise-sweep
#SBATCH --output=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.out
#SBATCH --error=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.err
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ismailj@usf.edu

# DP-SGD noise sweep — Table 11 replication, `circe`-branch equivalent but
# full fine-tuning (no LoRA) + GPT-Neo-125M instead of QLoRA + GPT-Neo-1.3B.
# Trains one full fine-tune per noise level sigma (DP-SGD: clip + Gaussian
# noise on ALL trainable gradients), then runs DP_ATTACK_TYPE against each.
# Checkpoint/resume: completed sigma levels in table_11_results.json are
# skipped, and per-sigma model checkpoints are reused if present.

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
# gpt-neo-hpo-v2 best config (trial #20, val_loss=2.2413) — see README
export LEARNING_RATE=1.95e-05
export LR_SCHEDULE=cosine
export WEIGHT_DECAY=0.0799
export WARMUP_FRACTION=0.0780
export ATTACK_BATCH_SIZE=32
export USE_LORA=0       # full fine-tuning on RTX A6000 (48 GB VRAM)
export MODEL_NAME=EleutherAI/gpt-neo-125M
export OUTPUT_DIR=results/gpt-neo-125m-noise-sweep
export MAX_EMAILS=50000
export EPOCHS=3
export SUBSET_PAIRS=3238
export MAX_NEW_TOKENS=100
export BATCH_SIZE=32
export MAX_LENGTH=512
export MAX_GRAD_NORM=0.49

# DP-SGD noise sweep — switches main.py into Table 11 mode.
export DP_NOISE_LEVELS=0,0.0001,0.0005,0.002,0.005
export DP_ATTACK_TYPE=zs_d_greedy

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
