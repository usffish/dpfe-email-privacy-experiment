#!/bin/bash
#SBATCH --job-name=dpfe-qlora
#SBATCH --output=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.out
#SBATCH --error=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.err
#SBATCH --partition=snsm_itn19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
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

# gpt2-large in float16 (~1.5 GB) fits comfortably on the GTX 1070 Ti (8 GB).
# Reduced email count keeps wall time under 6h for a smoke-test run.
# For a full-scale experiment matching the paper, set MAX_EMAILS=50000 and EPOCHS=3.
export MODEL_NAME=gpt2-large
export MAX_EMAILS=3000
export EPOCHS=1
export SUBSET_PAIRS=200
export MAX_NEW_TOKENS=100


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
