#!/bin/bash
#SBATCH --job-name=dpfe-composite-sweep
#SBATCH --output=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.out
#SBATCH --error=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.err
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ismailj@usf.edu

# Composite-attack DP-SGD noise sweep.
# At each sigma level, trains one full fine-tune (v3 config) then runs the 6
# attack types that collectively cover all 15 uniquely recoverable addresses
# (bracket_greedy, zs_b_greedy, zs_d_greedy, zs_d_topk, json_greedy, zs_a_greedy)
# and unions the hits. This gives the highest possible attack success rate at
# each noise level — a stronger adversary than any single attack type.
# Time budget doubled vs single-attack sweep (6x more attack passes per sigma).

REAL_HOME=/home/i/ismailj
WORKDIR=${REAL_HOME}/dpfe-email-privacy-experiment

export HF_HOME=${REAL_HOME}/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export FRESH=0
export SMOKE=0
export GRAD_ACCUM_STEPS=2
# gpt-neo-hpo-v3 best config
export LEARNING_RATE=1.32e-05
export LR_SCHEDULE=linear
export WEIGHT_DECAY=0.0729
export WARMUP_FRACTION=0.0166
export ATTACK_BATCH_SIZE=32
export USE_LORA=0
export MODEL_NAME=EleutherAI/gpt-neo-125M
export OUTPUT_DIR=results/gpt-neo-125m-v3-composite-sweep
export MAX_EMAILS=50000
export EPOCHS=3
export SUBSET_PAIRS=3238
export MAX_NEW_TOKENS=100
export BATCH_SIZE=32
export MAX_LENGTH=512
export MAX_GRAD_NORM=2.78

export DP_NOISE_LEVELS=0,0.0001,0.0005,0.002,0.005,0.01,0.05,0.1
export DP_ATTACK_TYPE=composite   # runs all 6 coverage-optimal attack types, unions hits

source ${REAL_HOME}/miniconda3/etc/profile.d/conda.sh
conda activate my_environment

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"

mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"
python -u main.py

echo "End: $(date)"
