"""
Compare Table 11 results between GPT-2 base and GPT-2 Large experiments.

Usage:
    python compare_results.py
"""

import json
import os

EXPERIMENTS = [
    ("GPT-2 base (117M)",  "results/gpt2-base/results.json"),
    ("GPT-2 Large (774M)", "results/gpt2-large/results.json"),
]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = json.load(f)
    return {r["noise"]: r for r in rows}


def fmt_rate(v):
    return f"{v:.2f}%" if v is not None else "—"

def fmt_pct(v):
    return f"{v:.1f}%" if v is not None else "—"

def fmt_eps(v):
    if v is None:
        return "∞"
    return f"{v:.4f}"


def main():
    datasets = [(label, load(path)) for label, path in EXPERIMENTS]

    missing = [label for label, d in datasets if d is None]
    if missing:
        print(f"Missing results for: {', '.join(missing)}")
        print("Run the corresponding sbatch job(s) first.")
        if all(d is None for _, d in datasets):
            return

    all_noise = sorted({
        noise
        for _, d in datasets if d
        for noise in d
    })

    col_w = 22
    model_cols = [label for label, _ in datasets]

    header = f"{'Noise (σ)':<12}" + "".join(f"  {m:<{col_w}}" for m in model_cols)
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

    # Delta table — only if both are present
    base_data  = next((d for l, d in datasets if "base"  in l.lower() or "117" in l), None)
    large_data = next((d for l, d in datasets if "large" in l.lower() or "774" in l), None)
    if base_data and large_data:
        print()
        print("Delta (Large − Base)")
        print("-" * 60)
        print(f"{'Noise (σ)':<12}  {'ΔAttack':<14}{'ΔCorrectness':<16}")
        print("-" * 60)
        for noise in all_noise:
            if noise not in base_data or noise not in large_data:
                continue
            b, lg = base_data[noise], large_data[noise]
            d_attack = lg["attack_success_rate"] - b["attack_success_rate"]
            d_corr   = lg["correctness"] - b["correctness"]
            sign_a = "+" if d_attack >= 0 else ""
            sign_c = "+" if d_corr   >= 0 else ""
            print(f"{noise:<12}  {sign_a}{d_attack:.2f}%{'':<8}{sign_c}{d_corr:.2f}%")
        print("-" * 60)


if __name__ == "__main__":
    main()
