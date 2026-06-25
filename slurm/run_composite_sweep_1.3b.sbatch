#!/bin/bash
#SBATCH --job-name=dpfe-comp-1.3b
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

# Composite DP-SGD noise sweep for GPT-Neo-1.3B.
# At each sigma level, trains one full fine-tune then unions hits across the 7
# attacks that collectively cover all 57 uniquely recoverable addresses (1.3B run):
#   zs_d_beam5, bracket_greedy, zs_d_greedy, zs_b_greedy,
#   zs_c_greedy, zs_d_topk, json_greedy
# Uses gpt-neo-1.3b-hpo-512 best config (Trial #36, val_loss=1.6868).
# Checkpoint/resume: resubmit with FRESH=0 if job times out — completed
# sigma levels in table_11_results.json are skipped automatically.
# Estimated runtime: ~13.5h per sigma level × 10 levels = ~135h (2 submissions).

REAL_HOME=/home/i/ismailj
WORKDIR=${REAL_HOME}/dpfe-email-privacy-experiment

export HF_HOME=${REAL_HOME}/hf_cache
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
export OUTPUT_DIR=results/gpt-neo-1.3b-composite-sweep
export MAX_EMAILS=50000
export EPOCHS=3
export SUBSET_PAIRS=3238
export MAX_NEW_TOKENS=100
export BATCH_SIZE=4
export MAX_LENGTH=512
export ATTACK_BATCH_SIZE=16

export DP_NOISE_LEVELS=0,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0,50.0
export DP_ATTACK_TYPE=composite
export COMPOSITE_ATTACK_TYPES=zs_d_beam5,bracket_greedy,zs_d_greedy,zs_b_greedy,zs_c_greedy,zs_d_topk,json_greedy

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
