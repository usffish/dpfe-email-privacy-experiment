"""
Seed-variance check for HPO.
============================
Trains the SAME fixed hyperparameter config (default: gpt-neo-hpo-v3 winner,
trial #0) several times with different random seeds, and reports the spread of
best val_loss across seeds.

Purpose: decide whether the val_loss differences HPO is optimizing are real
signal or just seed-to-seed noise. The whole v2→v3 sweep moved best val_loss
from 2.2413 to 2.2322 — a 0.41% gap. If the std (or spread) of val_loss across
seeds at a FIXED config is comparable to that gap, then HPO is chasing noise:
any reasonable config is equivalent and we should stop optimizing val_loss.

Everything mirrors hpo_trial.py's training loop exactly (same optimizer,
scheduler, gradient accumulation, token-weighted val loss) so the numbers are
directly comparable to the study's reported best_val_loss values.

Usage (SLURM): sbatch slurm/run_seed_variance.sbatch
Override seeds/config via SV_* env vars (see below).
"""

import gc
import os
import statistics
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import CONFIG, EmailDataset, EnronDataProcessor, set_seed, ts
from hpo_trial import compute_val_loss  # token-weighted, identical to HPO

# ── Fixed config — gpt-neo-hpo-v3 winner (trial #0). Overridable via env. ──────
FIXED = {
    "learning_rate":    float(os.getenv("SV_LR", "1.3164591902732453e-05")),
    "batch_size":       int(os.getenv("SV_BATCH_SIZE", "32")),
    "max_length":       int(os.getenv("SV_MAX_LENGTH", "512")),
    "lr_schedule":      os.getenv("SV_LR_SCHEDULE", "linear"),
    "weight_decay":     float(os.getenv("SV_WEIGHT_DECAY", "0.07287548218471876")),
    "warmup_fraction":  float(os.getenv("SV_WARMUP_FRACTION", "0.016563744877478272")),
    "max_grad_norm":    float(os.getenv("SV_MAX_GRAD_NORM", "2.7811500810608725")),
    "grad_accum_steps": int(os.getenv("SV_GRAD_ACCUM_STEPS", "2")),
    "epochs":           int(os.getenv("SV_EPOCHS", "5")),  # match HPO max_resource
}
SEEDS    = [int(x) for x in os.getenv("SV_SEEDS", "42,43,44,45").split(",")]
EMAILS   = int(os.getenv("SV_EMAILS", "10000"))     # match HPO corpus
VAL_FRAC = float(os.getenv("SV_VAL_FRAC", "0.1"))    # match HPO val split

# Reference gap to compare against: |v3 - v2| / v2 = |2.2322 - 2.2413| / 2.2413.
REF_GAP_PCT = float(os.getenv("SV_REF_GAP_PCT", "0.41"))


def train_once(seed, train_texts, val_texts, tokenizer, device):
    """Train the fixed config under one seed; return best (min) val_loss."""
    set_seed(seed)
    cfg = FIXED

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"], torch_dtype=torch.float32
    ).to(device)
    model.train()

    dataset = EmailDataset(train_texts, tokenizer, cfg["max_length"])
    loader  = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
    )

    accum        = cfg["grad_accum_steps"]
    n_batches    = len(loader)
    steps_per_ep = max(1, n_batches // accum)
    total_steps  = steps_per_ep * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_fraction"])

    if cfg["lr_schedule"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    val_losses = []
    for epoch in range(cfg["epochs"]):
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(loader):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            (outputs.loss / accum).backward()
            is_update = ((batch_idx + 1) % accum == 0 or (batch_idx + 1) == n_batches)
            if is_update:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    cfg["max_grad_norm"],
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        vl = compute_val_loss(
            model, val_texts, tokenizer, cfg["max_length"], cfg["batch_size"], device
        )
        val_losses.append(vl)
        print(f"  {ts()}  seed {seed}  epoch {epoch+1}/{cfg['epochs']}  val_loss {vl:.4f}")

    best = min(val_losses)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best


def main():
    device = CONFIG["device"]

    processor = EnronDataProcessor(CONFIG["data_dir"])
    processor.load_or_create_synthetic_data()
    all_texts   = processor.email_bodies[:EMAILS]
    split       = int(len(all_texts) * (1.0 - VAL_FRAC))
    train_texts = all_texts[:split]
    val_texts   = all_texts[split:]

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    print("=" * 60)
    print("Seed-variance check — gpt-neo-hpo-v3 config (fixed)")
    print("=" * 60)
    print(f"  {len(train_texts)} train / {len(val_texts)} val emails, "
          f"{FIXED['epochs']} epochs each")
    print(f"  config: lr={FIXED['learning_rate']:.2e}  "
          f"batch={FIXED['batch_size']}×{FIXED['grad_accum_steps']}accum  "
          f"max_length={FIXED['max_length']}  schedule={FIXED['lr_schedule']}")
    print(f"  seeds: {SEEDS}")
    print(f"  reference gap (v2→v3): {REF_GAP_PCT:.2f}%")

    results = {}
    for seed in SEEDS:
        print(f"\n{'-'*60}\nSeed {seed}\n{'-'*60}")
        results[seed] = train_once(seed, train_texts, val_texts, tokenizer, device)
        print(f"  → seed {seed} best val_loss: {results[seed]:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    vals       = list(results.values())
    mean       = statistics.mean(vals)
    std        = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    spread     = max(vals) - min(vals)
    std_pct    = 100 * std / mean
    spread_pct = 100 * spread / mean

    print(f"\n{'='*60}")
    print("Seed-variance summary")
    print(f"{'='*60}")
    for seed, v in results.items():
        print(f"  seed {seed}: {v:.4f}")
    print(f"  mean   : {mean:.4f}")
    print(f"  std    : {std:.4f}  ({std_pct:.2f}%)")
    print(f"  spread : {spread:.4f}  ({spread_pct:.2f}%)  [max - min]")
    print(f"  ref gap: {REF_GAP_PCT:.2f}%  (v2→v3 best val_loss improvement)")

    print(f"\n{'='*60}")
    if std_pct >= REF_GAP_PCT:
        print("VERDICT: NOISE DOMINATES.")
        print(f"  Seed std ({std_pct:.2f}%) >= v2→v3 gap ({REF_GAP_PCT:.2f}%).")
        print("  The HPO 'improvement' is within seed noise — any reasonable")
        print("  config is equivalent. Stop optimizing val_loss; reuse v2.")
    else:
        print("VERDICT: SIGNAL EXCEEDS NOISE.")
        print(f"  Seed std ({std_pct:.2f}%) < v2→v3 gap ({REF_GAP_PCT:.2f}%).")
        print("  Config differences are larger than seed noise — continued HPO")
        print("  (softer pruning + multivariate TPE) is worthwhile.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
