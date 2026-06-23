#!/bin/bash
#SBATCH --job-name=dpfe-hpo-neo-1.3b
#SBATCH --output=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.out
#SBATCH --error=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.err
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ismailj@usf.edu

# One HPO trial per job for GPT-Neo-1.3B.
# Submit N in parallel with: bash submit_hpo.sh N gpt-neo-1.3b-hpo-v1 slurm/run_hpo_gptneo_1.3b.sbatch
# Longer time limit than 125M trials due to larger model.

REAL_HOME=/home/i/ismailj
WORKDIR=${REAL_HOME}/dpfe-email-privacy-experiment

export HF_HOME=${REAL_HOME}/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export MODEL_NAME=EleutherAI/gpt-neo-1.3B
export DATA_DIR=enron_data
export OUTPUT_DIR=results/gpt-neo-1.3b-attacks
export MAX_LENGTH=512
export BATCH_SIZE=4
export GRAD_ACCUM_STEPS=8
export MAX_GRAD_NORM=1.0
export SEED=42
export USE_LORA=0

# v2: restricted search space after OOM testing — batch=8/512 and batch=32/512 both OOM'd
# Safe: batch<=8 at max_length=128/256; batch=2 at max_length=512 (untested, likely OK)
export HPO_BATCH_SIZE_CHOICES=2,4,8
export HPO_MAX_LENGTH_CHOICES=128,256
export HPO_STUDY_NAME=${HPO_STUDY_NAME:-gpt-neo-1.3b-hpo-v2}
export HPO_STORAGE=${HPO_STORAGE:-/home/i/ismailj/dpfe-email-privacy-experiment/hpo_study.jsonl}
export HPO_N_TRIALS=${HPO_N_TRIALS:-1}
export HPO_EMAILS=${HPO_EMAILS:-10000}
export HPO_PAIRS=${HPO_PAIRS:-3238}
export HPO_ATTACK=${HPO_ATTACK:-zs_d_greedy}
export ATTACK_BATCH_SIZE=${ATTACK_BATCH_SIZE:-16}
export HPO_MAX_EPOCHS=${HPO_MAX_EPOCHS:-3}

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
