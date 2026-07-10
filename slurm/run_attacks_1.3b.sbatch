#!/bin/bash
#SBATCH --job-name=dpfe-attacks-1.3b
#SBATCH --output=/home/i/ismailj/logs/%j.out
#SBATCH --error=/home/i/ismailj/logs/%j.err
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ismailj@usf.edu

# Multi-attack-type experiment for GPT-Neo-1.3B — no DP noise.
# Trains once using gpt-neo-1.3b-hpo-512 best config (Trial #36, val_loss=1.6868),
# then runs all 15 attack strategies to compare effectiveness.
# Submit from: /work_bgfs/i/ismailj/dpfe-email-privacy-experiment
#   sbatch slurm/run_attacks_1.3b.sbatch

REAL_HOME=/home/i/ismailj
STORE=/work_bgfs/i/ismailj
TMPDIR=/tmp/dpfe-${SLURM_JOB_ID}
WORKDIR=${TMPDIR}/dpfe-email-privacy-experiment

# Copy code and model cache to local /tmp (work_bgfs not accessible on compute nodes)
mkdir -p "${TMPDIR}"
cp -r ${STORE}/dpfe-email-privacy-experiment "${TMPDIR}/"
export HF_HOME=${TMPDIR}/hf_cache
cp -r ${STORE}/hf_cache "${HF_HOME}"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export FRESH=0
export SMOKE=0

# gpt-neo-1.3b-hpo-512 best config (Trial #36, val_loss=1.6868)
export LEARNING_RATE=6.25e-06
export LR_SCHEDULE=linear
export WEIGHT_DECAY=0.04825697729590399
export WARMUP_FRACTION=0.04149201661525929
export MAX_GRAD_NORM=1.0816652190277265
export GRAD_ACCUM_STEPS=8

export USE_LORA=0
export MODEL_NAME=EleutherAI/gpt-neo-1.3B
export OUTPUT_DIR=results/gpt-neo-1.3b-attacks
export MAX_EMAILS=50000
export EPOCHS=3
export SUBSET_PAIRS=3238
export MAX_NEW_TOKENS=100
export BATCH_SIZE=4
export MAX_LENGTH=512
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

# Copy results back to persistent storage
RESULT_NAME=$(basename "${OUTPUT_DIR}")
mkdir -p "${STORE}/dpfe-email-privacy-experiment/results"
cp -r "${WORKDIR}/${OUTPUT_DIR}" "${STORE}/dpfe-email-privacy-experiment/results/${RESULT_NAME}"

echo "Results copied to ${STORE}/dpfe-email-privacy-experiment/results/${RESULT_NAME}"
echo "End: $(date)"
