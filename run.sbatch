#!/bin/bash
#SBATCH --job-name=dpfe-qlora
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ismailjhaveri@gmail.com

# ── Environment ────────────────────────────────────────────────────────────────
module purge
module load python/3.11
module load cuda/12.1   # adjust to whichever version CIRCE provides

# Point HuggingFace at scratch so model weights don't eat your home quota.
# Run download_model.py once before submitting this script.
export HF_HOME=/scratch/${USER}/hf_cache
export TRANSFORMERS_OFFLINE=1       # prevents any network calls on compute node
export TOKENIZERS_PARALLELISM=false # suppresses HF tokenizer fork warning

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR=/scratch/${USER}/dpfe-email-privacy-experiment
mkdir -p "${WORKDIR}/logs"

# ── Install dependencies (first run only — remove after venv is cached) ────────
if [ ! -d "${WORKDIR}/.venv" ]; then
    python -m venv "${WORKDIR}/.venv"
    source "${WORKDIR}/.venv/bin/activate"
    pip install --quiet torch transformers peft bitsandbytes opacus tabulate numpy
else
    source "${WORKDIR}/.venv/bin/activate"
fi

# ── Run ────────────────────────────────────────────────────────────────────────
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"

cd "${WORKDIR}"
python main.py

echo "End: $(date)"
