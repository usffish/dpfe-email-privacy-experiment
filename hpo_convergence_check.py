"""
HPO convergence checker — called by the monitoring loop each wakeup.

Outputs one of:
  STATUS running <n_running> <n_complete> <n_pruned> <n_failed>
  STATUS batch_done <n_complete> <best_val_loss> <improvement_pct> <no_improve_batches>
  STATUS converged <n_complete> <best_val_loss>
  STATUS error <message>

Convergence: best val_loss improved < 1% for 2 consecutive batches of 8 trials.

State is persisted in hpo_v3_state.json in the working directory.
"""

import json
import os
import sys
import subprocess

STUDY   = os.getenv("HPO_STUDY_NAME", "gpt-neo-hpo-v3")
STORAGE = os.getenv(
    "HPO_STORAGE",
    "/home/i/ismailj/dpfe-email-privacy-experiment/hpo_study.jsonl",
)
STATE_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hpo_v3_state.json")
BATCH_SIZE  = 8
IMPROVE_THRESHOLD = 0.01   # 1% improvement threshold
CONVERGE_BATCHES  = 2      # consecutive non-improving batches to declare convergence
SBATCH_FILE = "slurm/run_hpo_gptneo.sbatch"

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"prev_batch_best": None, "no_improve_batches": 0, "last_completed": 0}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_study_stats():
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(STORAGE)
    )
    s = optuna.load_study(study_name=STUDY, storage=storage)
    trials   = s.trials
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned   = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    running  = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
    failed   = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
    best_val = min((t.value for t in complete), default=None)
    best_trial = min(complete, key=lambda t: t.value) if complete else None
    return {
        "n_complete": len(complete),
        "n_pruned":   len(pruned),
        "n_running":  len(running),
        "n_failed":   len(failed),
        "best_val":   best_val,
        "best_trial": best_trial,
        "complete":   complete,
    }


def check_running_errors():
    """Return list of (jobid, err_snippet) for any HPO jobs with non-empty .err files."""
    result = subprocess.run(
        ["squeue", "-u", "ismailj", "-h", "-o", "%i %j"],
        capture_output=True, text=True
    )
    errors = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2 and "hpo" in parts[1].lower():
            jobid = parts[0]
            err_path = f"/home/i/ismailj/dpfe-email-privacy-experiment/logs/{jobid}.err"
            if os.path.exists(err_path) and os.path.getsize(err_path) > 0:
                with open(err_path) as f:
                    snippet = f.read(500).strip()
                errors.append((jobid, snippet))
    return errors


def hpo_jobs_running():
    result = subprocess.run(
        ["squeue", "-u", "ismailj", "-h", "-o", "%j"],
        capture_output=True, text=True
    )
    return any("hpo" in line.lower() for line in result.stdout.splitlines())


def submit_batch():
    for _ in range(BATCH_SIZE):
        subprocess.run(
            ["sbatch", f"--export=ALL,HPO_STUDY_NAME={STUDY}", SBATCH_FILE],
            capture_output=True
        )


def main():
    try:
        # ── Check for errors in currently running jobs ────────────────────────
        errs = check_running_errors()
        if errs:
            for jobid, snippet in errs:
                print(f"ERROR job {jobid}: {snippet[:200]}")

        # ── If HPO jobs still running, just report progress ───────────────────
        if hpo_jobs_running():
            stats = get_study_stats()
            print(f"STATUS running {stats['n_running']} {stats['n_complete']} "
                  f"{stats['n_pruned']} {stats['n_failed']}")
            if errs:
                sys.exit(1)
            return

        # ── Batch complete — check convergence ────────────────────────────────
        stats = get_study_stats()

        if stats["n_failed"] > 0:
            print(f"STATUS error {stats['n_failed']} failed trials in study")
            sys.exit(1)

        if stats["n_complete"] == 0:
            print("STATUS error no completed trials found")
            sys.exit(1)

        state = load_state()
        current_best = stats["best_val"]
        prev_best    = state["prev_batch_best"]

        if prev_best is None:
            improvement = 1.0  # first batch — always continue
        else:
            improvement = (prev_best - current_best) / prev_best

        no_improve = state["no_improve_batches"]
        if improvement < IMPROVE_THRESHOLD:
            no_improve += 1
        else:
            no_improve = 0

        print(f"STATUS batch_done {stats['n_complete']} {current_best:.4f} "
              f"{improvement*100:.2f} {no_improve}")

        if stats["best_trial"]:
            p = stats["best_trial"].params
            print(f"BEST trial={stats['best_trial'].number} val_loss={current_best:.4f}")
            for k, v in p.items():
                print(f"  {k}={v}")

        # ── Converged? ────────────────────────────────────────────────────────
        if no_improve >= CONVERGE_BATCHES:
            print(f"STATUS converged {stats['n_complete']} {current_best:.4f}")
            return

        # ── Not converged — submit next batch ─────────────────────────────────
        save_state({
            "prev_batch_best":   current_best,
            "no_improve_batches": no_improve,
            "last_completed":    stats["n_complete"],
        })
        submit_batch()
        print(f"SUBMITTED {BATCH_SIZE} more trials")

    except Exception as e:
        print(f"STATUS error {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
