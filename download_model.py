"""
Run this script on a CIRCE login node BEFORE submitting run.sbatch or run_gpt2.sbatch.
It pre-downloads both model variants into $HF_HOME so compute nodes can load them
offline (TRANSFORMERS_OFFLINE=1).

Usage:
    conda activate my_environment
    export HF_HOME=~/hf_cache
    python download_model.py
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODELS = [
    ("gpt2", "~0.5 GB"),
    ("gpt2-large", "~3 GB"),
]

hf_home = os.environ.get("HF_HOME")
print(f"HF_HOME: {hf_home or '~/.cache/huggingface (default)'}\n")

for model_id, size in MODELS:
    print(f"Downloading {model_id} ({size}) ...")
    # Do NOT pass cache_dir — let HF_HOME env var control the path.
    # Passing cache_dir directly places files one level too shallow, missing
    # the hub/ subdirectory that HF_HOME expects.
    AutoTokenizer.from_pretrained(model_id)
    AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    print(f"  {model_id} done.\n")

print("Both models downloaded. You can now submit run.sbatch and run_gpt2.sbatch.")
