"""
BOHB HPO Trial — one optuna trial per SLURM job.
=================================================
Objective: minimize validation loss on a 10% held-out split of the
training corpus. This is attack-type-agnostic — it measures how well
the model memorizes email text in general without biasing toward any
particular extraction prompt.

Attack success rate is still recorded as a user_attr for analysis but
is NOT the optimization target.

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
    "study_name": os.getenv("HPO_STUDY_NAME", "attack-hpo-v3"),
    "storage":    os.getenv(
        "HPO_STORAGE",
        "/home/i/ismailj/dpfe-email-privacy-experiment/hpo_study.jsonl",
    ),
    "n_trials":   int(os.getenv("HPO_N_TRIALS", "1")),   # trials per job (keep at 1)
    "emails":     int(os.getenv("HPO_EMAILS", "10000")),  # reduced corpus for speed
    "val_frac":   float(os.getenv("HPO_VAL_FRAC", "0.1")), # fraction held out for val loss
    "pairs":      int(os.getenv("HPO_PAIRS", "3238")),    # attack pairs (informational only)
    "attack":     os.getenv("HPO_ATTACK", "zs_d_greedy"), # attack stored as user_attr
    "max_epochs": int(os.getenv("HPO_MAX_EPOCHS", "3")),  # HyperBand max_resource
}


# ── Validation loss ───────────────────────────────────────────────────────────

def compute_val_loss(model, val_texts, tokenizer, max_length, batch_size, device):
    val_dataset = EmailDataset(val_texts, tokenizer, max_length)
    val_loader  = DataLoader(val_dataset, batch_size=max(1, batch_size), shuffle=False)
    model.eval()
    total_loss = 0.0
    n_seen = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            total_loss += outputs.loss.item()
            n_seen += 1
    model.train()
    return total_loss / max(n_seen, 1)


# ── Training loop with per-epoch optuna reporting ────────────────────────────

def train_one_trial(trial, train_texts, val_texts, tokenizer, device):
    """
    Sample hyperparameters, train, report per-epoch val loss to HyperBand.
    Raises optuna.TrialPruned if HyperBand decides to kill this config early.
    Returns (model, train_losses, val_losses).
    """
    # ── Sample hyperparameters ────────────────────────────────────────────────
    lr            = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32])
    max_length    = trial.suggest_categorical("max_length", [128, 256, 512])
    schedule      = trial.suggest_categorical("lr_schedule", ["linear", "cosine"])
    weight_decay  = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_frac   = trial.suggest_float("warmup_fraction", 0.0, 0.1)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 5.0, log=True)
    # epochs is NOT sampled — HyperBand controls budget via pruning after each epoch.
    epochs = HPO["max_epochs"]
    accum_steps = CONFIG["grad_accum_steps"]
    # Clamp batch_size to stay within 8 GB VRAM — activations scale as batch × seq_len².
    # Empirically verified safe limits on GTX 1070 Ti (8 GB), GPT-2 base full fine-tune:
    #   128 tokens × 32 batch — safe (attention cost 128²=small)
    #   256 tokens × 16 batch — OOM;  256 × 8 — safe
    #   512 tokens × 8  batch — OOM;  512 × 4 — safe
    max_safe = {128: 32, 256: 8, 512: 4}
    batch_size = min(batch_size, max_safe[max_length])
    trial.set_user_attr("effective_batch_size", batch_size)

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

    train_losses = []
    val_losses   = []

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

        avg_train_loss = total_loss / max(n_seen, 1)
        avg_val_loss   = compute_val_loss(model, val_texts, tokenizer, max_length, batch_size, device)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f"  {ts()}  Epoch {epoch+1}/{epochs} — train_loss: {avg_train_loss:.4f}  val_loss: {avg_val_loss:.4f}")

        # Report val_loss to HyperBand — direction=minimize so lower = better
        trial.report(avg_val_loss, step=epoch)
        if trial.should_prune():
            print(f"  Pruned at epoch {epoch+1} (val_loss {avg_val_loss:.4f})")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

    model.eval()
    return model, train_losses, val_losses


# ── Objective function ────────────────────────────────────────────────────────

def objective(trial):
    # Different seed per trial so data shuffling varies
    set_seed(CONFIG["seed"] + trial.number)
    device = CONFIG["device"]

    # ── Load data ─────────────────────────────────────────────────────────────
    processor = EnronDataProcessor(CONFIG["data_dir"])
    processor.load_or_create_synthetic_data()

    all_texts    = processor.email_bodies[:HPO["emails"]]
    split        = int(len(all_texts) * (1.0 - HPO["val_frac"]))
    train_texts  = all_texts[:split]
    val_texts    = all_texts[split:]
    attack_pairs = processor.name_email_pairs[:HPO["pairs"]]
    print(f"  Data: {len(train_texts)} train / {len(val_texts)} val emails, {len(attack_pairs)} attack pairs")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    # ── Train ─────────────────────────────────────────────────────────────────
    model, train_losses, val_losses = train_one_trial(trial, train_texts, val_texts, tokenizer, device)

    final_val_loss = val_losses[-1]

    # ── Attack eval (informational — not the optimization target) ──────────────
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
    print(f"    Val loss            : {final_val_loss:.4f}  (objective)")
    print(f"    Attack success rate : {attack_rate:.2f}% ({num_hits}/{len(attack_pairs)})  (informational)")
    print(f"    Correctness         : {correctness:.1f}%")
    print(f"    Train losses        : {[f'{l:.4f}' for l in train_losses]}")
    print(f"    Val losses          : {[f'{l:.4f}' for l in val_losses]}")

    # Attach extra info for view_hpo.py
    trial.set_user_attr("num_hits",      num_hits)
    trial.set_user_attr("correctness",   correctness)
    trial.set_user_attr("attack_rate",   attack_rate)
    trial.set_user_attr("final_val_loss", final_val_loss)
    trial.set_user_attr("train_losses",  train_losses)
    trial.set_user_attr("val_losses",    val_losses)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return final_val_loss


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Study   : {HPO['study_name']}")
    print(f"Storage : {HPO['storage']}")
    print(f"Data    : {HPO['emails']} emails ({int(HPO['val_frac']*100)}% held out for val loss)")
    print(f"Attack  : {HPO['attack']} (informational only)")
    print(f"Model   : {CONFIG['model_name']}")
    print(f"Objective: minimize validation loss")

    # JournalFileStorage: append-only writes are NFS-safe (SQLite fails on NFS).
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(HPO["storage"])
    )

    study = optuna.create_study(
        study_name=HPO["study_name"],
        storage=storage,
        direction="minimize",
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
        print(f"  Val loss  : {best.value:.4f}")
        print(f"  Params:")
        for k, v in best.params.items():
            print(f"    {k:<20} {v}")
    except ValueError:
        print("No completed trials yet.")


if __name__ == "__main__":
    main()
