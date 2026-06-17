#!/bin/bash
#SBATCH --job-name=dpfe-seed-var
#SBATCH --output=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.out
#SBATCH --error=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.err
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ismailj@usf.edu

# Seed-variance check — is the v2→v3 val_loss gap (0.41%) real signal or noise?
# Trains the fixed gpt-neo-hpo-v3 winner config under several seeds and reports
# the std/spread of best val_loss. If std >= the v2→v3 gap, HPO is chasing noise.

REAL_HOME=/home/i/ismailj
WORKDIR=${REAL_HOME}/dpfe-email-privacy-experiment

export HF_HOME=${REAL_HOME}/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export MODEL_NAME=EleutherAI/gpt-neo-125M
export USE_LORA=0

# Seed-variance knobs (defaults baked into hpo_seed_variance.py: v3 winner config).
export SV_SEEDS=42,43,44,45      # 4 seeds — raise for a tighter std estimate
export SV_EMAILS=10000           # match HPO corpus
export SV_EPOCHS=5               # match HPO max_resource

source ${REAL_HOME}/miniconda3/etc/profile.d/conda.sh
conda activate my_environment

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"

mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"
python -u hpo_seed_variance.py

echo "End: $(date)"
