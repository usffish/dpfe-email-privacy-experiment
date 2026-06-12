"""
Seed attack-hpo-v5 (GPT-2, corrected objective) with the attack-hpo-v4
best config (trial #13, val_loss=1.1324 under the OLD buggy objective).

Running this config as trial #0 of v5 is the "quick check": it directly
measures the real (padding-masked, token-weighted) val loss of the v4
winner, telling us whether GPT-2's apparent advantage over GPT-Neo
(1.1324 vs 2.2413) survives the objective fix.

Usage (on circe):
    python enqueue_gpt2_v5_seed.py
    bash submit_hpo.sh 1 attack-hpo-v5 run_hpo.sbatch   # quick check
    bash submit_hpo.sh 8 attack-hpo-v5 run_hpo.sbatch   # then the sweep
"""

import optuna

STUDY_NAME = "attack-hpo-v5"
STORAGE_PATH = "/home/i/ismailj/dpfe-email-privacy-experiment/hpo_study.jsonl"

# attack-hpo-v4 best trial (#13), verbatim from the README table.
SEED_TRIALS = [
    {
        "learning_rate": 9.82e-05,
        "batch_size": 16,
        "max_length": 512,
        "lr_schedule": "linear",
        "weight_decay": 0.0637,
        "warmup_fraction": 0.0970,
        "max_grad_norm": 4.63,
    },
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

    for params in SEED_TRIALS:
        study.enqueue_trial(params, skip_if_exists=False)
        print(f"Enqueued: {params}")

    print(f"Enqueued {len(SEED_TRIALS)} trial(s) into '{STUDY_NAME}'.")
