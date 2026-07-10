#!/bin/bash
#SBATCH --job-name=dpfe-noise-sweep
#SBATCH --output=/home/i/ismailj/logs/%j.out
#SBATCH --error=/home/i/ismailj/logs/%j.err
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=60:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ismailj@usf.edu

# DP-SGD noise sweep — Table 11 replication.
# Full fine-tuning (no LoRA) + GPT-Neo-125M. Trains one model per sigma level
# (σ=0–50), then runs DP_ATTACK_TYPE against each. Checkpoint/resume: resubmit
# with FRESH=0 if job times out — completed sigma levels are skipped automatically.
# Submit from: /work_bgfs/i/ismailj/dpfe-email-privacy-experiment
#   sbatch slurm/run_noise_sweep.sbatch

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

# gpt-neo-hpo-v3 best config (trial #0, val_loss=2.2322) — see README
export LEARNING_RATE=1.32e-05
export LR_SCHEDULE=linear
export WEIGHT_DECAY=0.0729
export WARMUP_FRACTION=0.0166
export GRAD_ACCUM_STEPS=2
export MAX_GRAD_NORM=2.78
export ATTACK_BATCH_SIZE=32
export USE_LORA=0
export MODEL_NAME=EleutherAI/gpt-neo-125M
export OUTPUT_DIR=results/gpt-neo-125m-v3-noise-sweep
export MAX_EMAILS=50000
export EPOCHS=3
export SUBSET_PAIRS=3238
export MAX_NEW_TOKENS=100
export BATCH_SIZE=32
export MAX_LENGTH=512

export DP_NOISE_LEVELS=0,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0,50.0
export DP_ATTACK_TYPE=bracket_greedy

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
