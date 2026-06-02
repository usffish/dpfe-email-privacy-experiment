"""
Run this script on a CIRCE login node BEFORE submitting run.sbatch.
It pre-downloads the model weights and tokenizer into $HF_HOME so
compute nodes can load them offline (TRANSFORMERS_OFFLINE=1).

Usage:
    conda activate my_environment
    export HF_HOME=~/hf_cache
    python download_model.py
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "gpt2-large"

hf_home = os.environ.get("HF_HOME")
print(f"Downloading {MODEL} (~3 GB) ...")
print(f"HF_HOME: {hf_home or '~/.cache/huggingface (default)'}")

# Do NOT pass cache_dir — let HF_HOME env var control the path.
# Passing cache_dir directly places files one level too shallow, missing
# the hub/ subdirectory that HF_HOME expects.
AutoTokenizer.from_pretrained(MODEL)
AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)

print("Download complete. You can now submit run.sbatch.")
