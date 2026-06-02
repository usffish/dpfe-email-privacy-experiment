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

cache_dir = os.environ.get("HF_HOME", None)
print(f"Downloading {MODEL} (~3 GB) ...")
print(f"Cache dir: {cache_dir or '~/.cache/huggingface (default)'}")

AutoTokenizer.from_pretrained(MODEL, cache_dir=cache_dir)
AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=cache_dir, torch_dtype=torch.float16)

print("Download complete. You can now submit run.sbatch.")
