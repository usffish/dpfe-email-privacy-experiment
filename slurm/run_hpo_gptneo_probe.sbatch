#!/bin/bash
#SBATCH --job-name=dpfe-hpo-probe
#SBATCH --output=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.out
#SBATCH --error=/home/i/ismailj/dpfe-email-privacy-experiment/logs/%j.err
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ismailj@usf.edu

# Long-context probe for GPT-Neo-125M: does max_length > 512 help further?
# Motivation: under the corrected (padding-masked) objective, max_length=512
# was the best config in gpt-neo-hpo-v2, and ~22% of emails are still
# truncated at 512 tokens (~10% at 1024). This probe searches {768, 1024}
# with the low-lr regime that won at 512, as a SEPARATE study (optuna can't
# extend a categorical's choices mid-study).
#
# Launch (after gpt-neo-hpo-v2 has converged):
#   python hpo/enqueue_len_probe.py    # optional: seed 4 trials from the v2 winner
#   bash submit_hpo.sh 8 gpt-neo-len-probe slurm/run_hpo_gptneo_probe.sbatch

REAL_HOME=/home/i/ismailj
WORKDIR=${REAL_HOME}/dpfe-email-privacy-experiment

export HF_HOME=${REAL_HOME}/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ── Model and data (fixed across all trials) ──────────────────────────────────
export MODEL_NAME=EleutherAI/gpt-neo-125M
export DATA_DIR=enron_data
export OUTPUT_DIR=results/gpt-neo-125m-attacks
export MAX_LENGTH=128
export BATCH_SIZE=2
export GRAD_ACCUM_STEPS=8
export MAX_GRAD_NORM=1.0
export SEED=42
export USE_LORA=0

# ── HPO settings ──────────────────────────────────────────────────────────────
# Use ':=' so a HPO_STUDY_NAME (etc.) passed in via `sbatch --export=ALL,VAR=...`
# is preserved instead of being clobbered by this script's default.
export HPO_STUDY_NAME=${HPO_STUDY_NAME:-gpt-neo-len-probe}
export HPO_STORAGE=${HPO_STORAGE:-/home/i/ismailj/dpfe-email-privacy-experiment/hpo_study.jsonl}
export HPO_N_TRIALS=${HPO_N_TRIALS:-1}
export HPO_EMAILS=${HPO_EMAILS:-10000}
export HPO_PAIRS=${HPO_PAIRS:-3238}
export HPO_ATTACK=${HPO_ATTACK:-zs_d_greedy}
export ATTACK_BATCH_SIZE=${ATTACK_BATCH_SIZE:-32}
export HPO_MAX_EPOCHS=${HPO_MAX_EPOCHS:-5}

# ── Probe search-space overrides ──────────────────────────────────────────────
# Longer contexts only; lr capped at 1e-4 (everything above ~6e-5 was pruned
# in gpt-neo-hpo-v2 — no point re-exploring the divergent high-lr region).
# batch_size 2/4 dropped (uniformly weak in v2). wd/warmup ceilings raised:
# the v2 winner sat at 0.096/0.096, within 5% of the old 0.1 caps.
export HPO_MAX_LENGTH_CHOICES=${HPO_MAX_LENGTH_CHOICES:-768,1024}
export HPO_BATCH_SIZE_CHOICES=${HPO_BATCH_SIZE_CHOICES:-8,16,32}
export HPO_LR_MIN=${HPO_LR_MIN:-5e-6}
export HPO_LR_MAX=${HPO_LR_MAX:-1e-4}
export HPO_WD_MAX=${HPO_WD_MAX:-0.3}
export HPO_WARMUP_MAX=${HPO_WARMUP_MAX:-0.2}

source ${REAL_HOME}/miniconda3/etc/profile.d/conda.sh
conda activate my_environment

echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURMD_NODENAME}"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Study:   ${HPO_STUDY_NAME}"
echo "Storage: ${HPO_STORAGE}"
echo "Lengths: ${HPO_MAX_LENGTH_CHOICES}  LR range: ${HPO_LR_MIN}-${HPO_LR_MAX}"
echo "Batch sizes: ${HPO_BATCH_SIZE_CHOICES}  wd_max: ${HPO_WD_MAX}  warmup_max: ${HPO_WARMUP_MAX}"
echo "Start:   $(date)"

mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"
python -u hpo_trial.py

echo "End: $(date)"
