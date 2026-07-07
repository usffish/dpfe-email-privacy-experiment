#!/bin/bash
#SBATCH --job-name=dpfe-hpo-gpt2-lg2
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

# HPO v2 for GPT-2 Large — restricted search space to avoid OOM on 44.5GB RTX A6000.
# v1 saw OOM at batch=8,16,32 with max_length=512. This study locks max_length to
# {128,256} and batch to {2,4,8} — all combinations are safe on this GPU.
# Submit N in parallel: bash submit_hpo.sh N gpt2-large-hpo-v2 slurm/run_hpo_gpt2_large_v2.sbatch

REAL_HOME=/home/i/ismailj
WORKDIR=${REAL_HOME}/dpfe-email-privacy-experiment

export HF_HOME=${REAL_HOME}/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export MODEL_NAME=gpt2-large
export DATA_DIR=enron_data
export OUTPUT_DIR=results/gpt2-large-hpo-v2
export USE_LORA=0

export HPO_BATCH_SIZE_CHOICES=2,4,8
export HPO_MAX_LENGTH_CHOICES=128,256
export HPO_STUDY_NAME=${HPO_STUDY_NAME:-gpt2-large-hpo-v2}
export HPO_STORAGE=${HPO_STORAGE:-/home/i/ismailj/dpfe-email-privacy-experiment/hpo_study.jsonl}
export HPO_N_TRIALS=${HPO_N_TRIALS:-1}
export HPO_EMAILS=${HPO_EMAILS:-10000}
export HPO_PAIRS=${HPO_PAIRS:-3238}
export HPO_ATTACK=${HPO_ATTACK:-zs_d_greedy}
export ATTACK_BATCH_SIZE=${ATTACK_BATCH_SIZE:-32}
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
