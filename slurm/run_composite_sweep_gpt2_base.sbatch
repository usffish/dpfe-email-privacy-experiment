#!/bin/bash
#SBATCH --job-name=dpfe-comp-gpt2b
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ismailj@usf.edu

# Composite DP-SGD noise sweep for GPT-2 Base (117M).
# At each sigma level, trains one full fine-tune then unions hits across the 2
# attacks that collectively cover all 10 uniquely recoverable addresses:
#   bracket_greedy (8 unique), json_greedy (2 unique)
# Uses gpt2-base-hpo-v1 best config (Trial #36, val_loss=2.3565).
# Checkpoint/resume: resubmit with FRESH=0 if job times out — completed
# sigma levels in table_11_results.json are skipped automatically.
# Estimated runtime: ~2.5h per sigma level × 10 levels = ~25h total.

REAL_HOME=/home/i/ismailj
WORKDIR=${REAL_HOME}/dpfe-email-privacy-experiment

export CHECKPOINT_DIR=/tmp/dpfe_${SLURM_JOB_ID}
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export GITHUB_TOKEN=$(cat /home/i/ismailj/.github_token 2>/dev/null || echo "")
export GITHUB_REPO=usffish/attack

export FRESH=0
export SMOKE=0

# gpt2-base-hpo-v1 best config (Trial #36, val_loss=2.3565)
export LEARNING_RATE=0.0002304276211390074
export LR_SCHEDULE=cosine
export WEIGHT_DECAY=0.031225656699132093
export WARMUP_FRACTION=0.01092414683449043
export MAX_GRAD_NORM=0.4230167343685974
export GRAD_ACCUM_STEPS=4

export USE_LORA=0
export MODEL_NAME=gpt2
export OUTPUT_DIR=results/gpt2-base-composite-sweep
export MAX_EMAILS=50000
export EPOCHS=3
export SUBSET_PAIRS=3238
export MAX_NEW_TOKENS=100
export BATCH_SIZE=32
export MAX_LENGTH=512
export ATTACK_BATCH_SIZE=32

export GRADIENT_CHECKPOINTING=1
export DP_NOISE_LEVELS=0,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0,50.0
export DP_ATTACK_TYPE=composite
export COMPOSITE_ATTACK_TYPES=bracket_greedy,json_greedy

source ${REAL_HOME}/miniconda3/etc/profile.d/conda.sh
conda activate my_environment

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"

mkdir -p "${CHECKPOINT_DIR}" "${HF_HOME}"

cd "${WORKDIR}"
python -u main.py

echo "End: $(date)"
