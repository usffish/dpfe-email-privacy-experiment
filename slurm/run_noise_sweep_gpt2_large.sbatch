#!/bin/bash
#SBATCH --job-name=dpfe-sweep-gpt2l
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ismailj@usf.edu

# DP-SGD noise sweep for GPT-2 Large (774M), single best attack (zs_d_greedy).
# Uses gpt2-large-hpo-v2 best config (Trial #32, val_loss=2.1984).
#
# /home-full workaround: this job never writes to /home (which is at 100%).
#   - code is cloned fresh from GitHub into compute-node /tmp
#   - model cache + checkpoints + predictions all live in /tmp
#   - training data + conda env are READ from /home (reads still work)
#   - results JSON is pushed to GitHub after each sigma (resume source of truth)
#   - a per-job run log is pushed to results/logs/<jobid>.log on exit
# Submit with the token injected (kept out of git):
#   ssh circe "sbatch --export=ALL,GITHUB_TOKEN=<tok>" < slurm/run_noise_sweep_gpt2_large.sbatch

set -e
set -o pipefail

REAL_HOME=/home/i/ismailj
JOBTMP=/tmp/dpfe_${SLURM_JOB_ID}
mkdir -p "${JOBTMP}"
RUNLOG="${JOBTMP}/run.log"

export GITHUB_TOKEN="${GITHUB_TOKEN:-}"
export GITHUB_REPO="${GITHUB_REPO:-usffish/dpfe-email-privacy-experiment}"
export GITHUB_BRANCH="${GITHUB_BRANCH:-attack}"

# On exit (success or failure) push the tail of the run log to GitHub for
# post-mortem visibility, since /home can't hold SLURM logs.
finish() {
  code=$?
  if [ -n "${GITHUB_TOKEN}" ] && [ -f "${RUNLOG}" ]; then
    SLURM_JOB_ID="${SLURM_JOB_ID}" REPO="${GITHUB_REPO}" BR="${GITHUB_BRANCH}" \
    TOK="${GITHUB_TOKEN}" LOG="${RUNLOG}" python3 - <<'PY' || true
import base64, json, os, urllib.request
tok, repo, br = os.environ["TOK"], os.environ["REPO"], os.environ["BR"]
jid = os.environ["SLURM_JOB_ID"]
data = open(os.environ["LOG"], "rb").read()[-60000:]
p = f"results/logs/{jid}.log"
body = json.dumps({"message": f"[auto] job log {jid}",
                   "content": base64.b64encode(data).decode(),
                   "branch": br}).encode()
req = urllib.request.Request(f"https://api.github.com/repos/{repo}/contents/{p}",
                             data=body, method="PUT",
                             headers={"Authorization": f"token {tok}",
                                      "Accept": "application/vnd.github.v3+json"})
try:
    urllib.request.urlopen(req, timeout=30)
    print(f"[log] pushed → {p}")
except Exception as e:
    print(f"[log] push failed: {e}")
PY
  fi
}
trap finish EXIT

# --- Clone up-to-date code into /tmp (the /home repo is stale + unwritable) ---
cd "${JOBTMP}"
git clone --depth 1 -b "${GITHUB_BRANCH}" \
    "https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git" repo
cd repo

# --- Training data: read from /home, stage into /tmp ---
cp -r "${REAL_HOME}/dpfe-email-privacy-experiment/enron_data" enron_data

# --- All heavy writes go to /tmp ---
export CHECKPOINT_DIR="${JOBTMP}/ckpt"
export HF_HOME="${JOBTMP}/hf_cache"
mkdir -p "${CHECKPOINT_DIR}" "${HF_HOME}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export FRESH=0
export SMOKE=0

export OUTPUT_DIR=results/gpt2-large-noise-sweep
# Seed resume state from any prior /home results (readable even when full).
mkdir -p "${OUTPUT_DIR}"
cp "${REAL_HOME}/dpfe-email-privacy-experiment/${OUTPUT_DIR}/table_11_results.json" \
   "${OUTPUT_DIR}/" 2>/dev/null || true

# gpt2-large-hpo-v2 best config (Trial #32, val_loss=2.1984)
export LEARNING_RATE=9.024263343579973e-06
export LR_SCHEDULE=cosine
export WEIGHT_DECAY=0.057443059201871274
export WARMUP_FRACTION=0.05337337560305046
export MAX_GRAD_NORM=1.4497073847836068
export GRAD_ACCUM_STEPS=2

export USE_LORA=0
export MODEL_NAME=gpt2-large
export MAX_EMAILS=50000
export EPOCHS=3
export SUBSET_PAIRS=3238
export MAX_NEW_TOKENS=100
export BATCH_SIZE=4
export MAX_LENGTH=256
export ATTACK_BATCH_SIZE=16

export DP_NOISE_LEVELS=0,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0,50.0
export DP_ATTACK_TYPE=zs_d_greedy

source "${REAL_HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate my_environment

echo "Job ID: ${SLURM_JOB_ID}"        | tee -a "${RUNLOG}"
echo "Node:   $(hostname)"            | tee -a "${RUNLOG}"
echo "Code:   ${JOBTMP}/repo"         | tee -a "${RUNLOG}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a "${RUNLOG}"
echo "Start:  $(date)"                | tee -a "${RUNLOG}"

python -u main.py 2>&1 | tee -a "${RUNLOG}"

echo "End: $(date)" | tee -a "${RUNLOG}"
