"""
Compare Table 11 results across model runs.

Usage:
    python compare_results.py                         # compares gpt2 vs gpt2-large
    python compare_results.py results/gpt2/table_11_results.json results/gpt2-large/table_11_results.json
"""

import sys
import json
import os
from tabulate import tabulate


DEFAULT_PATHS = [
    "results/gpt2/table_11_results.json",
    "results/gpt2-large/table_11_results.json",
]


def load_results(path):
    with open(path) as f:
        return json.load(f)


def model_label(path):
    # results/gpt2-large/table_11_results.json → gpt2-large
    parts = path.split(os.sep)
    if len(parts) >= 2:
        return parts[-2]
    return path


def compare(paths):
    datasets = []
    for p in paths:
        if not os.path.exists(p):
            print(f"Missing: {p}")
            continue
        data = {r["noise"]: r for r in load_results(p)}
        datasets.append((model_label(p), data))

    if not datasets:
        print("No result files found.")
        return

    all_noise = sorted({n for _, d in datasets for n in d})

    # ── Attack Success Rate ──────────────────────────────────────────────────
    print("\nAttack Success Rate (%)")
    headers = ["σ"] + [label for label, _ in datasets]
    rows = []
    for noise in all_noise:
        row = [str(noise)]
        for _, data in datasets:
            r = data.get(noise)
            row.append(f"{r['attack_success_rate']:.2f}" if r else "—")
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="center"))

    # ── Privacy Enhancement ──────────────────────────────────────────────────
    print("\nPrivacy Enhancement (%)")
    headers = ["σ"] + [label for label, _ in datasets]
    rows = []
    for noise in all_noise:
        row = [str(noise)]
        for _, data in datasets:
            r = data.get(noise)
            row.append(f"{r['privacy_enhancement']:.0f}" if r else "—")
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="center"))

    # ── Correctness ──────────────────────────────────────────────────────────
    print("\nCorrectness (%)")
    headers = ["σ"] + [label for label, _ in datasets]
    rows = []
    for noise in all_noise:
        row = [str(noise)]
        for _, data in datasets:
            r = data.get(noise)
            row.append(f"{r['correctness']:.2f}" if r else "—")
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="center"))

    # ── Privacy Budget ───────────────────────────────────────────────────────
    print("\nPrivacy Budget ε (δ = 1e-5)")
    headers = ["σ"] + [label for label, _ in datasets]
    rows = []
    for noise in all_noise:
        row = [str(noise)]
        for _, data in datasets:
            r = data.get(noise)
            if r is None:
                row.append("—")
            elif r.get("epsilon") is None:
                row.append("∞")
            else:
                row.append(f"{r['epsilon']:.4f}")
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="center"))

    # ── p(G) adversary success probability ───────────────────────────────────
    # Equation from DPFE paper Table 13: p(G) = e^ε / (1 + e^ε)
    any_epsilon = any(
        r.get("epsilon") is not None
        for _, data in datasets
        for r in data.values()
    )
    if any_epsilon:
        import math
        print("\nAdversary Success Probability p(G) = e^ε / (1 + e^ε)  [DPFE Table 13]")
        headers = ["σ"] + [label for label, _ in datasets]
        rows = []
        for noise in all_noise:
            row = [str(noise)]
            for _, data in datasets:
                r = data.get(noise)
                if r is None or r.get("epsilon") is None:
                    row.append("—")
                else:
                    eps = r["epsilon"]
                    p_g = math.exp(eps) / (1 + math.exp(eps)) if eps < 700 else 1.0
                    row.append(f"{p_g:.4f}")
            rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt="grid", stralign="center"))


if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PATHS
    compare(paths)
