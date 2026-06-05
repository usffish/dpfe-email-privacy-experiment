"""
BOHB HPO Trial — one optuna trial per SLURM job.
=================================================
Multiple SLURM jobs share the same study via a SQLite file on CIRCE's
shared filesystem. Each job samples one hyperparameter configuration,
trains the model, runs a fast attack evaluation, and reports back.

HyperBand pruning kills bad configurations early (after epoch 1) so
compute is focused on promising regions.

Usage:
    bash submit_hpo.sh 8        # submit 8 parallel jobs
    python view_hpo.py          # inspect results from any node
"""

import gc
import os
import sys

import optuna
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

# Import shared utilities from main.py without re-running run_experiment()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    CONFIG,
    EmailDataset,
    EnronDataProcessor,
    PrivacyAttack,
    make_nondomain_pool,
    set_seed,
    ts,
)

# ── HPO-specific config (all overridable via env vars) ────────────────────────

HPO = {
    "study_name": os.getenv("HPO_STUDY_NAME", "attack-training-hpo"),
    "storage":    os.getenv(
        "HPO_STORAGE",
        "/home/i/ismailj/dpfe-email-privacy-experiment/hpo_study.jsonl",
    ),
    "n_trials":   int(os.getenv("HPO_N_TRIALS", "1")),   # trials per job (keep at 1)
    "emails":     int(os.getenv("HPO_EMAILS", "10000")),  # reduced corpus for speed
    "pairs":      int(os.getenv("HPO_PAIRS", "200")),     # attack pairs for evaluation
    "attack":     os.getenv("HPO_ATTACK", "zs_d_greedy"), # attack to maximise
    "max_epochs": int(os.getenv("HPO_MAX_EPOCHS", "5")),  # HyperBand max_resource
}


# ── Training loop with per-epoch optuna reporting ────────────────────────────

def train_one_trial(trial, train_texts, tokenizer, device):
    """
    Sample hyperparameters, train, report per-epoch loss to HyperBand.
    Raises optuna.TrialPruned if HyperBand decides to kill this config early.
    Returns (model, epoch_losses).
    """
    # ── Sample hyperparameters ────────────────────────────────────────────────
    lr            = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32])
    schedule      = trial.suggest_categorical("lr_schedule", ["linear", "cosine"])
    weight_decay  = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_frac   = trial.suggest_float("warmup_fraction", 0.0, 0.1)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 5.0, log=True)
    max_length    = trial.suggest_categorical("max_length", [128, 256, 512])
    # epochs is NOT sampled — HyperBand controls budget via pruning after each epoch.
    # Full fine-tuning only: GPT-2 base (117M, ~2.1 GB base) leaves ~6 GB headroom
    # on the 1070 Ti so all batch sizes up to 32 fit safely.
    # max_length up to 512 adds ~3.2 GB activations — still within 8 GB budget.
    epochs = HPO["max_epochs"]
    accum_steps = CONFIG["grad_accum_steps"]

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"  lr={lr:.2e}  batch_size={batch_size}  max_length={max_length}  max_epochs={epochs}")
    print(f"  schedule={schedule}  weight_decay={weight_decay:.4f}  warmup={warmup_frac:.2f}")
    print(f"  max_grad_norm={max_grad_norm:.2f}")
    print(f"{'='*60}")

    # ── Build model (full fine-tuning) ────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"], torch_dtype=torch.float32
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Full fine-tuning: {n_params:,} parameters")

    model.train()
    dataset     = EmailDataset(train_texts, tokenizer, max_length)
    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    n_batches    = len(loader)
    steps_per_ep = max(1, n_batches // accum_steps)
    total_steps  = steps_per_ep * epochs
    warmup_steps = int(total_steps * warmup_frac)

    if schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        n_seen = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            (outputs.loss / accum_steps).backward()

            is_update = (
                (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == n_batches
            )
            if is_update:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += outputs.loss.item()
            n_seen += 1

        avg_loss = total_loss / max(n_seen, 1)
        epoch_losses.append(avg_loss)
        print(f"  {ts()}  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

        # Report to HyperBand — negative loss so higher = better
        trial.report(-avg_loss, step=epoch)
        if trial.should_prune():
            print(f"  Pruned at epoch {epoch+1} (loss {avg_loss:.4f})")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

    model.eval()
    return model, epoch_losses


# ── Objective function ────────────────────────────────────────────────────────

def objective(trial):
    # Different seed per trial so data shuffling varies
    set_seed(CONFIG["seed"] + trial.number)
    device = CONFIG["device"]

    # ── Load data ─────────────────────────────────────────────────────────────
    processor = EnronDataProcessor(CONFIG["data_dir"])
    processor.load_or_create_synthetic_data()

    train_texts  = processor.email_bodies[:HPO["emails"]]
    attack_pairs = processor.name_email_pairs[:HPO["pairs"]]
    print(f"  Data: {len(train_texts)} train emails, {len(attack_pairs)} attack pairs")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    # ── Train ─────────────────────────────────────────────────────────────────
    model, epoch_losses = train_one_trial(trial, train_texts, tokenizer, device)

    # ── Evaluate: run attack on held-out pairs ─────────────────────────────────
    nondomain_pool = make_nondomain_pool(attack_pairs)
    attacker = PrivacyAttack(tokenizer, device)

    attack_rate, correctness, num_hits = attacker.run_attack(
        model, attack_pairs, HPO["attack"],
        predictions_path=None,
        few_shot_pool=attack_pairs,
        nondomain_pool=nondomain_pool,
        context_dict=None,
        email_freq=None,
    )

    print(f"\n  Trial {trial.number} complete:")
    print(f"    Attack success rate : {attack_rate:.2f}% ({num_hits}/{len(attack_pairs)})")
    print(f"    Correctness         : {correctness:.1f}%")
    print(f"    Epoch losses        : {[f'{l:.4f}' for l in epoch_losses]}")

    # Attach extra info for view_hpo.py
    trial.set_user_attr("num_hits",     num_hits)
    trial.set_user_attr("correctness",  correctness)
    trial.set_user_attr("final_loss",   epoch_losses[-1])
    trial.set_user_attr("epoch_losses", epoch_losses)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return attack_rate


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Study : {HPO['study_name']}")
    print(f"DB    : {HPO['storage']}")
    print(f"Data  : {HPO['emails']} emails, {HPO['pairs']} pairs")
    print(f"Attack: {HPO['attack']}")
    print(f"Model : {CONFIG['model_name']}")

    # JournalFileStorage: append-only writes are NFS-safe (SQLite fails on NFS).
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(HPO["storage"])
    )
    print(f"Storage: {HPO['storage']}")

    study = optuna.create_study(
        study_name=HPO["study_name"],
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=HPO["max_epochs"],
            reduction_factor=3,
        ),
        load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Trials completed so far: {completed}")

    study.optimize(objective, n_trials=HPO["n_trials"])

    # Print best result found across all jobs
    try:
        best = study.best_trial
        print(f"\n{'='*60}")
        print(f"Best trial so far: #{best.number}")
        print(f"  Attack success rate : {best.value:.2f}%")
        print(f"  Params:")
        for k, v in best.params.items():
            print(f"    {k:<20} {v}")
    except ValueError:
        print("No completed trials yet.")


if __name__ == "__main__":
    main()
