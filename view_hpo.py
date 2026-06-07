"""
View HPO results from any node.

Usage:
    python view_hpo.py                        # default study
    python view_hpo.py --study my-study       # named study
    python view_hpo.py --top 5                # show top N trials
    python view_hpo.py --best                 # print best params only (for scripting)
"""

import argparse
import os

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_STORAGE = "/home/i/ismailj/dpfe-email-privacy-experiment/hpo_study.jsonl"
DEFAULT_STUDY = "attack-hpo-v4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study",   default=DEFAULT_STUDY)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--top",     type=int, default=10)
    parser.add_argument("--best",    action="store_true",
                        help="Print best params only (machine-readable)")
    args = parser.parse_args()

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(args.storage)
    )

    try:
        study = optuna.load_study(study_name=args.study, storage=storage)
    except Exception as e:
        print(f"Could not load study '{args.study}': {e}")
        return

    trials = study.trials
    complete  = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    running   = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
    failed    = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]

    if args.best:
        try:
            best = study.best_trial
            print(f"val_loss={best.value:.4f}")
            for k, v in best.params.items():
                print(f"{k}={v}")
        except ValueError:
            print("no_completed_trials")
        return

    print(f"\nStudy: {args.study}  (objective: min val_loss across epochs)")
    print(f"  Complete: {len(complete)}  Pruned: {len(pruned)}  "
          f"Running: {len(running)}  Failed: {len(failed)}")

    if not complete:
        print("\nNo completed trials yet.")
        if running:
            print(f"{len(running)} trial(s) currently running.")
        return

    # Sort by best val_loss ascending (lower = better)
    ranked = sorted(complete, key=lambda t: t.value)

    print(f"\n{'─'*95}")
    print(f"{'Rank':<5}{'Trial':<7}{'BestVL':>8}{'Ep':>4}{'Attack%':>9}{'Hits':>6}{'Correct%':>10}  Params")
    print(f"{'─'*95}")

    for rank, t in enumerate(ranked[:args.top], 1):
        hits       = t.user_attrs.get("num_hits", "?")
        correct    = t.user_attrs.get("correctness", None)
        attack_pct = t.user_attrs.get("attack_rate", None)
        best_epoch = t.user_attrs.get("best_epoch", "?")
        correct_str    = f"{correct:.1f}%"    if correct    is not None else "?"
        attack_pct_str = f"{attack_pct:.2f}%" if attack_pct is not None else "?"
        params = "  ".join(f"{k}={v}" for k, v in t.params.items())
        print(f"{rank:<5}{t.number:<7}{t.value:>8.4f}{best_epoch:>4}{attack_pct_str:>9}{hits:>6}{correct_str:>10}  {params}")

    print(f"{'─'*90}")

    # Best trial detail
    best = study.best_trial
    best_epoch = best.user_attrs.get("best_epoch", "?")
    print(f"\nBest: Trial #{best.number}  val_loss={best.value:.4f}  (epoch {best_epoch})")
    print("  Hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:<22} {v}")

    val_losses   = best.user_attrs.get("val_losses")
    train_losses = best.user_attrs.get("train_losses")
    if val_losses:
        print(f"  Val losses  : {[f'{l:.4f}' for l in val_losses]}")
    if train_losses:
        print(f"  Train losses: {[f'{l:.4f}' for l in train_losses]}")

    print(f"\nTo use best params in run_attacks.sbatch:  (val_loss={best.value:.4f})")
    lr            = best.params.get("learning_rate")
    batch_size    = best.params.get("batch_size")
    max_grad_norm = best.params.get("max_grad_norm")
    max_length    = best.params.get("max_length")
    if lr:
        print(f"  export LEARNING_RATE={lr:.2e}")
    if batch_size:
        print(f"  export BATCH_SIZE={batch_size}")
    if max_grad_norm:
        print(f"  export MAX_GRAD_NORM={max_grad_norm:.2f}")
    if max_length:
        print(f"  export MAX_LENGTH={max_length}")
    print(f"  export USE_LORA=0")

    # Importance analysis if enough trials
    if len(complete) >= 5:
        print(f"\nHyperparameter importance ({len(complete)} trials):")
        try:
            importance = optuna.importance.get_param_importances(study)
            for param, imp in importance.items():
                bar = "█" * int(imp * 30)
                print(f"  {param:<22} {imp:.3f}  {bar}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
