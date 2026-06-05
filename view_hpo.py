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

DEFAULT_STORAGE = (
    "sqlite:////home/i/ismailj/dpfe-email-privacy-experiment/hpo_study.db"
)
DEFAULT_STUDY = "attack-training-hpo"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study",   default=DEFAULT_STUDY)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--top",     type=int, default=10)
    parser.add_argument("--best",    action="store_true",
                        help="Print best params only (machine-readable)")
    args = parser.parse_args()

    try:
        study = optuna.load_study(study_name=args.study, storage=args.storage)
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
            print(f"attack_success_rate={best.value:.4f}")
            for k, v in best.params.items():
                print(f"{k}={v}")
        except ValueError:
            print("no_completed_trials")
        return

    print(f"\nStudy: {args.study}")
    print(f"  Complete: {len(complete)}  Pruned: {len(pruned)}  "
          f"Running: {len(running)}  Failed: {len(failed)}")

    if not complete:
        print("\nNo completed trials yet.")
        if running:
            print(f"{len(running)} trial(s) currently running.")
        return

    # Sort by attack success rate descending
    ranked = sorted(complete, key=lambda t: t.value, reverse=True)

    print(f"\n{'─'*80}")
    print(f"{'Rank':<5}{'Trial':<7}{'Attack%':>8}{'Hits':>6}{'Correct%':>10}"
          f"{'Loss':>8}  Params")
    print(f"{'─'*80}")

    for rank, t in enumerate(ranked[:args.top], 1):
        hits    = t.user_attrs.get("num_hits", "?")
        correct = t.user_attrs.get("correctness", None)
        loss    = t.user_attrs.get("final_loss", None)
        correct_str = f"{correct:.1f}%" if correct is not None else "?"
        loss_str    = f"{loss:.4f}"     if loss    is not None else "?"
        params = "  ".join(f"{k}={v}" for k, v in t.params.items())
        print(f"{rank:<5}{t.number:<7}{t.value:>7.2f}%{hits:>6}{correct_str:>10}"
              f"{loss_str:>8}  {params}")

    print(f"{'─'*80}")

    # Best trial detail
    best = study.best_trial
    print(f"\nBest: Trial #{best.number}  Attack {best.value:.2f}%")
    print("  Hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:<22} {v}")

    epoch_losses = best.user_attrs.get("epoch_losses")
    if epoch_losses:
        print(f"  Epoch losses: {[f'{l:.4f}' for l in epoch_losses]}")

    print(f"\nTo use best params in run_attacks.sbatch:")
    lr         = best.params.get("learning_rate")
    batch_size = best.params.get("batch_size")
    if lr:
        print(f"  export LEARNING_RATE={lr:.2e}")
    if batch_size:
        print(f"  export BATCH_SIZE={batch_size}")
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
