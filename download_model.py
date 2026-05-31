"""
Run this script on a CIRCE login node (or in an internet-connected job)
BEFORE submitting run.sbatch. It pre-downloads the model weights and
tokenizer into $HF_HOME so compute nodes can load them offline.

Usage:
    module load python/3.11 cuda/12.1
    pip install transformers peft bitsandbytes --quiet
    python download_model.py
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "EleutherAI/gpt-neo-1.3B"

# HF_HOME defaults to ~/.cache/huggingface — override via env var if you
# want weights stored on scratch instead of your home directory quota.
# export HF_HOME=/scratch/<netid>/hf_cache
cache_dir = os.environ.get("HF_HOME", None)

print(f"Downloading {MODEL} ...")
print(f"Cache dir: {cache_dir or '~/.cache/huggingface (default)'}")

AutoTokenizer.from_pretrained(MODEL, cache_dir=cache_dir)
AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=cache_dir)

print("Download complete. You can now submit run.sbatch.")
