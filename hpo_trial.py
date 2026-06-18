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
import random
import sys
import time

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
    "study_name": os.getenv("HPO_STUDY_NAME", "attack-hpo-v4"),
    "storage":    os.getenv(
        "HPO_STORAGE",
        "/home/i/ismailj/dpfe-email-privacy-experiment/hpo_study.jsonl",
    ),
    "n_trials":   int(os.getenv("HPO_N_TRIALS", "1")),   # trials per job (keep at 1)
    "emails":     int(os.getenv("HPO_EMAILS", "10000")),  # reduced corpus for speed
    "val_frac":   float(os.getenv("HPO_VAL_FRAC", "0.1")), # fraction held out for val loss
    "pairs":      int(os.getenv("HPO_PAIRS", "3238")),    # attack pairs (informational only)
    "attack":     os.getenv("HPO_ATTACK", "zs_d_greedy"), # attack stored as user_attr
    "max_epochs": int(os.getenv("HPO_MAX_EPOCHS", "5")),  # HyperBand max_resource
    # Search-space overrides for separate probe studies (e.g. longer contexts).
    # Optuna forbids changing a categorical's choices within an existing study,
    # so only set these for NEW studies — never for one that already has trials.
    "max_length_choices": [
        int(x) for x in os.getenv("HPO_MAX_LENGTH_CHOICES", "128,256,512").split(",")
    ],
    "batch_size_choices": [
        int(x) for x in os.getenv("HPO_BATCH_SIZE_CHOICES", "2,4,8,16,32").split(",")
    ],
    # grad_accum_steps is NOT a search variable in gpt-neo-hpo-v2; adding it
    # here requires a new study name (Optuna forbids extending a study's param
    # space after trials exist). Default covers no-accum → production value.
    "accum_steps_choices": [
        int(x) for x in os.getenv("HPO_ACCUM_STEPS_CHOICES", "1,2,4,8").split(",")
    ],
    "lr_min":     float(os.getenv("HPO_LR_MIN", "5e-6")),
    "lr_max":     float(os.getenv("HPO_LR_MAX", "5e-4")),
    "wd_max":     float(os.getenv("HPO_WD_MAX", "0.1")),
    "warmup_max": float(os.getenv("HPO_WARMUP_MAX", "0.1")),
}


# ── Validation loss ───────────────────────────────────────────────────────────

def compute_val_loss(model, val_texts, tokenizer, max_length, batch_size, device):
    # Token-weighted mean: HF's causal-LM loss is the mean over each batch's
    # non-ignored (shifted) targets, so averaging batch means would weight
    # tokens in sparsely-filled batches more — and the bias would vary with
    # the trial's batch_size. Recover per-batch sums and divide by the total
    # target count instead.
    val_dataset = EmailDataset(val_texts, tokenizer, max_length)
    val_loader  = DataLoader(val_dataset, batch_size=max(1, batch_size), shuffle=False)
    model.eval()
    total_loss   = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=labels,
            )
            n_targets = (labels[..., 1:] != -100).sum().item()
            total_loss   += outputs.loss.item() * n_targets
            total_tokens += n_targets
    model.train()
    return total_loss / max(total_tokens, 1)


# ── Training loop with per-epoch optuna reporting ────────────────────────────

def train_one_trial(trial, train_texts, val_texts, tokenizer, device):
    """
    Sample hyperparameters, train, report per-epoch val loss to HyperBand.
    Raises optuna.TrialPruned if HyperBand decides to kill this config early.
    Returns (model, train_losses, val_losses).
    """
    # ── Sample hyperparameters ────────────────────────────────────────────────
    lr            = trial.suggest_float("learning_rate", HPO["lr_min"], HPO["lr_max"], log=True)
    batch_size    = trial.suggest_categorical("batch_size", HPO["batch_size_choices"])
    max_length    = trial.suggest_categorical("max_length", HPO["max_length_choices"])
    schedule      = trial.suggest_categorical("lr_schedule", ["linear", "cosine"])
    weight_decay  = trial.suggest_float("weight_decay", 0.0, HPO["wd_max"])
    warmup_frac   = trial.suggest_float("warmup_fraction", 0.0, HPO["warmup_max"])
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 5.0, log=True)
    accum_steps   = trial.suggest_categorical("grad_accum_steps", HPO["accum_steps_choices"])
    # epochs is NOT sampled — HyperBand controls budget via pruning after each epoch.
    epochs = HPO["max_epochs"]
    # Clamp batch_size to stay within VRAM — activations scale as batch × seq_len².
    # Empirically verified on RTX A6000 (48 GB), GPT-Neo-125M full fine-tune,
    # one forward+backward pass:
    #   128 tokens × 32 batch —  8.7 GB
    #   256 tokens × 32 batch — 17.5 GB
    #   512 tokens × 32 batch — 36.8 GB;  512 × 64 — OOM
    # 32 is the largest value in the batch_size search space and fits at all
    # three lengths, so no clamping is needed on this GPU.
    # 768/1024 caps are conservative extrapolations from the 512×32 measurement
    # (per-sample activation cost grows superlinearly with seq_len) — not yet
    # empirically verified.
    max_safe = {128: 32, 256: 32, 512: 32, 768: 16, 1024: 8}
    batch_size = min(batch_size, max_safe[max_length])
    trial.set_user_attr("effective_batch_size", batch_size * accum_steps)

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"  lr={lr:.2e}  batch={batch_size}×{accum_steps}accum={batch_size*accum_steps}eff  max_length={max_length}  max_epochs={epochs}")
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

    best_val_loss = min(val_losses)
    best_epoch    = val_losses.index(best_val_loss) + 1  # 1-indexed

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
    print(f"    Best val loss       : {best_val_loss:.4f}  (epoch {best_epoch})  (objective)")
    print(f"    Attack success rate : {attack_rate:.2f}% ({num_hits}/{len(attack_pairs)})  (informational)")
    print(f"    Correctness         : {correctness:.1f}%")
    print(f"    Train losses        : {[f'{l:.4f}' for l in train_losses]}")
    print(f"    Val losses          : {[f'{l:.4f}' for l in val_losses]}")

    # Attach extra info for view_hpo.py
    trial.set_user_attr("num_hits",      num_hits)
    trial.set_user_attr("correctness",   correctness)
    trial.set_user_attr("attack_rate",   attack_rate)
    trial.set_user_attr("best_val_loss", best_val_loss)
    trial.set_user_attr("best_epoch",    best_epoch)
    trial.set_user_attr("train_losses",  train_losses)
    trial.set_user_attr("val_losses",    val_losses)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return best_val_loss


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Study   : {HPO['study_name']}")
    print(f"Storage : {HPO['storage']}")
    print(f"Data    : {HPO['emails']} emails ({int(HPO['val_frac']*100)}% held out for val loss)")
    print(f"Attack  : {HPO['attack']} (informational only)")
    print(f"Model   : {CONFIG['model_name']}")
    print(f"Objective: minimize validation loss")

    # Stagger create_study() across parallel jobs — when many jobs hit the
    # journal file's _sync_with_backend() at the same instant, concurrent
    # NFS appends can interleave and corrupt a log line (seen as a
    # UnicodeDecodeError on a NUL-padded line on the next read).
    time.sleep(random.uniform(0, 20))

    # JournalFileStorage: append-only writes are NFS-safe (SQLite fails on NFS).
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(HPO["storage"])
    )

    # multivariate=True + group=True models hyperparameter *interactions*
    # (e.g. effective LR depends jointly on learning_rate × batch_size ×
    # grad_accum_steps) instead of sampling each param independently — the
    # default univariate TPE misses exactly the couplings v3 introduced.
    # min_resource=2 softens HyperBand: a trial must survive 2 epochs before
    # it can be pruned, so configs are judged on a real val-loss trajectory
    # rather than a noisy epoch-1 reading. This feeds TPE more *completed*
    # trials, which it needs to build a useful surrogate. Sampler/pruner are
    # not part of the frozen search space, so changing them on the existing
    # gpt-neo-hpo-v3 study is allowed.
    study = optuna.create_study(
        study_name=HPO["study_name"],
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=2,
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
        print(f"  Best val loss : {best.value:.4f}")
        print(f"  Params:")
        for k, v in best.params.items():
            print(f"    {k:<20} {v}")
    except ValueError:
        print("No completed trials yet.")


if __name__ == "__main__":
    main()
