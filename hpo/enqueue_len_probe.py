"""
Seed the gpt-neo-len-probe study with trials based on the gpt-neo-hpo-v2
winner (trial #6: lr=1.1e-5, batch_size=16, max_length=512, linear,
wd=0.096, warmup=0.096, max_grad_norm=0.34), with max_length swapped to
768/1024 and a second, slightly higher lr per length.

Creates the study if it doesn't exist, so this is safe to run before the
first probe job. batch_size=16 is clamped to 8 at max_length=1024 by
hpo_trial.py's max_safe table.

Usage (on circe, from repo root):
    python hpo/enqueue_len_probe.py
    bash submit_hpo.sh 8 gpt-neo-len-probe slurm/run_hpo_gptneo_probe.sbatch
"""

import optuna

STUDY_NAME = "gpt-neo-len-probe"
STORAGE_PATH = "/home/i/ismailj/dpfe-email-privacy-experiment/hpo_study.jsonl"

# Non-length params from the gpt-neo-hpo-v2 best trial (#6).
BASE = {
    "batch_size": 16,
    "lr_schedule": "linear",
    "weight_decay": 0.09591109830041548,
    "warmup_fraction": 0.09561031797464463,
    "max_grad_norm": 0.3435116812896845,
}

PROBE_TRIALS = [
    {**BASE, "max_length": length, "learning_rate": lr}
    for length in (768, 1024)
    for lr in (1.1e-5, 3e-5)
]

if __name__ == "__main__":
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(STORAGE_PATH)
    )
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )

    for params in PROBE_TRIALS:
        study.enqueue_trial(params, skip_if_exists=False)
        print(f"Enqueued: max_length={params['max_length']} lr={params['learning_rate']:.1e}")

    print(f"Enqueued {len(PROBE_TRIALS)} trials into '{STUDY_NAME}'.")
