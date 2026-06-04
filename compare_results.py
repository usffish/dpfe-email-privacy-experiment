"""
Compare Table 11 results between GPT-2 base and GPT-2 Large experiments.

Usage:
    python compare_results.py
"""

import json
import os
from collections import Counter

EXPERIMENTS = [
    ("GPT-2 base (117M)",  "results/gpt2-base/results.json"),
    ("GPT-2 Large (774M)", "results/gpt2-large/results.json"),
]

PATTERN_LABELS = {
    "a1":  "username",
    "b1":  "first.last",  "b2":  "first_last", "b3":  "firstlast",
    "b4":  "first",       "b5":  "last",
    "b6":  "flast",       "b7":  "firstl",     "b8":  "lfirst",
    "b9":  "lastf",       "b10": "initials",
    "c1":  "first.last(3)","c2": "first_last(3)","c3": "firstlast(3)",
    "c4":  "f.m.last",    "c5":  "f_m_last",   "c6":  "fmlast",
    "c7":  "first(3)",    "c8":  "last(3)",
    "c9":  "flast(3)",    "c10": "firstl(3)",  "c11": "lfirst(3)",
    "c12": "lastf(3)",    "c13": "fmlast",     "c14": "fmidlast",
    "c15": "f.m.last(dot)","c16":"first.midlast","c17":"initials(3)",
    "l":   "4+ name parts",
    "z":   "no pattern (memorized)",
}

INFERABLE_PATTERNS = {k for k in PATTERN_LABELS if k != "z" and k != "l"}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = json.load(f)
    return {r["noise"]: r for r in rows}


def load_predictions(results_path):
    """Load all per-sigma prediction files from the predictions/ subdirectory."""
    pred_dir = os.path.join(os.path.dirname(results_path), "predictions")
    if not os.path.exists(pred_dir):
        return {}
    result = {}
    for fname in sorted(os.listdir(pred_dir)):
        if fname.startswith("sigma_") and fname.endswith(".json"):
            noise_str = fname[len("sigma_"):-len(".json")]
            try:
                noise = float(noise_str)
                with open(os.path.join(pred_dir, fname)) as f:
                    result[noise] = json.load(f)
            except (ValueError, json.JSONDecodeError):
                pass
    return result


def fmt_rate(v):
    return f"{v:.2f}%" if v is not None else "—"

def fmt_pct(v):
    return f"{v:.1f}%" if v is not None else "—"

def fmt_eps(v):
    if v is None:
        return "∞"
    return f"{v:.4f}"


def print_main_table(datasets):
    all_noise = sorted({
        noise
        for _, d in datasets if d
        for noise in d
    })

    col_w = 22
    header = f"{'Noise (σ)':<12}" + "".join(f"  {l:<{col_w}}" for l, _ in datasets)
    print()
    print("=" * len(header))
    print("Table 11 Comparison — Attack Success Rate")
    print("=" * len(header))
    print(f"{'':12}  {'Attack %':<10}{'Correct%':<10}{'ε':<8}" * len(datasets))
    print("-" * len(header))

    for noise in all_noise:
        row = f"{noise:<12}"
        for _, d in datasets:
            if d is None or noise not in d:
                row += f"  {'—':<10}{'—':<10}{'—':<8}"
            else:
                r = d[noise]
                row += (
                    f"  {fmt_rate(r['attack_success_rate']):<10}"
                    f"{fmt_pct(r['correctness']):<10}"
                    f"{fmt_eps(r['epsilon']):<8}"
                )
        print(row)

    print("=" * len(header))


def print_delta_table(datasets):
    base_data  = next((d for l, d in datasets if "base"  in l.lower() or "117" in l), None)
    large_data = next((d for l, d in datasets if "large" in l.lower() or "774" in l), None)
    if not (base_data and large_data):
        return

    all_noise = sorted(set(base_data) & set(large_data))
    print()
    print("Delta (Large − Base)")
    print("-" * 60)
    print(f"{'Noise (σ)':<12}  {'ΔAttack':<14}{'ΔCorrectness':<16}")
    print("-" * 60)
    for noise in all_noise:
        b, lg = base_data[noise], large_data[noise]
        d_attack = lg["attack_success_rate"] - b["attack_success_rate"]
        d_corr   = lg["correctness"] - b["correctness"]
        sign_a = "+" if d_attack >= 0 else ""
        sign_c = "+" if d_corr   >= 0 else ""
        print(f"{noise:<12}  {sign_a}{d_attack:.2f}%{'':<8}{sign_c}{d_corr:.2f}%")
    print("-" * 60)


def print_pattern_analysis(label, results_path):
    preds_by_sigma = load_predictions(results_path)
    if not preds_by_sigma:
        return

    print(f"\n{label} — Pattern Type Analysis")
    print("-" * 60)

    # Pair distribution from σ=0 (same pairs used across all sigmas)
    sigma0 = preds_by_sigma.get(0) or preds_by_sigma.get(0.0)
    if sigma0:
        dist = Counter(p["pattern_type"] for p in sigma0)
        total = len(sigma0)
        inferable = sum(dist[k] for k in INFERABLE_PATTERNS if k in dist)
        memorized = dist.get("z", 0)
        print(f"  Pair distribution ({total} total):")
        print(f"    Inferable from name structure: {inferable} ({inferable/total*100:.1f}%)")
        print(f"    No pattern — must be memorized: {memorized} ({memorized/total*100:.1f}%)")
        top = sorted(dist.items(), key=lambda x: -x[1])[:6]
        print(f"    Top patterns: " + ", ".join(
            f"{k}={v} ({PATTERN_LABELS.get(k, k)})" for k, v in top
        ))

    # Per-sigma hit breakdown
    print()
    for noise in sorted(preds_by_sigma):
        data = preds_by_sigma[noise]
        hits = [p for p in data if p["hit"]]
        total = len(data)
        if hits:
            print(f"  σ={noise}: {len(hits)} hit(s) / {total} pairs")
            for h in hits:
                pt = h["pattern_type"]
                pt_label = PATTERN_LABELS.get(pt, pt)
                inference = "inferable" if pt in INFERABLE_PATTERNS else "memorized"
                print(f"    {h['name']}")
                print(f"      predicted: {h['predicted']}")
                print(f"      true:      {h['true_email']}")
                print(f"      pattern:   {pt} ({pt_label}) — {inference}")
        else:
            print(f"  σ={noise}: 0 hits / {total} pairs")


def main():
    datasets = [(label, load(path)) for label, path in EXPERIMENTS]

    missing = [label for label, d in datasets if d is None]
    if missing:
        print(f"Missing results for: {', '.join(missing)}")
        print("Run the corresponding sbatch job(s) first.")
        if all(d is None for _, d in datasets):
            return

    print_main_table(datasets)
    print_delta_table(datasets)

    # Pattern analysis — only if prediction files exist
    has_preds = any(
        os.path.exists(os.path.join(os.path.dirname(path), "predictions"))
        for _, path in EXPERIMENTS
    )
    if has_preds:
        print("\n")
        print("=" * 60)
        print("Pattern Type Analysis")
        print("=" * 60)
        for label, path in EXPERIMENTS:
            print_pattern_analysis(label, path)


if __name__ == "__main__":
    main()
