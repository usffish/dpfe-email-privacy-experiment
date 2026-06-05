"""
Compare multi-attack-type results for email extraction experiments.

Usage:
    python compare_results.py                          # ranked table
    python compare_results.py --csv                    # also export CSV
    python compare_results.py results/gpt2-base-attacks results/gpt2-large-attacks
"""

import json
import os
import sys
import argparse
import csv
from collections import Counter

GROUPS = {
    "Zero-shot templates": ["zs_a_greedy", "zs_b_greedy", "zs_c_greedy", "zs_d_greedy"],
    "Few-shot (Enron domain)": ["fs_1_greedy", "fs_2_greedy", "fs_5_greedy"],
    "Few-shot (non-domain)":   ["fs_1_nondomain_greedy", "fs_2_nondomain_greedy", "fs_5_nondomain_greedy"],
    "Decoding variants":       ["zs_d_beam5", "zs_d_topk"],
    "Novel methods":           ["bracket_greedy", "json_greedy", "domain_hint_greedy"],
    "Context injection":       ["context_50", "context_100", "context_200"],
}

PATTERN_LABELS = {
    "a1":  "username",
    "b1":  "first.last",    "b2":  "first_last",  "b3":  "firstlast",
    "b4":  "first",         "b5":  "last",
    "b6":  "flast",         "b7":  "firstl",      "b8":  "lfirst",
    "b9":  "lastf",         "b10": "initials",
    "c1":  "first.last(3)", "c2":  "first_last(3)","c3": "firstlast(3)",
    "c4":  "f.m.last",      "c5":  "f_m_last",    "c6":  "fmlast",
    "c7":  "first(3)",      "c8":  "last(3)",
    "c9":  "flast(3)",      "c10": "firstl(3)",   "c11": "lfirst(3)",
    "c12": "lastf(3)",      "c13": "fmlast",      "c14": "fmidlast",
    "c15": "f.m.last(dot)", "c16": "first.midlast","c17":"initials(3)",
    "l":   "4+ name parts",
    "z":   "no pattern (memorized)",
}

INFERABLE_PATTERNS = {k for k in PATTERN_LABELS if k not in ("z", "l")}

DEFAULT_DIRS = [
    "results/gpt2-base-attacks",
    "results/gpt2-large-attacks",
]


def load_results(output_dir):
    path = os.path.join(output_dir, "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = json.load(f)
    return {r["attack_type"]: r for r in rows}


def load_predictions(output_dir, attack_type):
    path = os.path.join(output_dir, "predictions", f"{attack_type}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt_rate(v):
    return f"{v:.2f}%" if v is not None else "—"

def fmt_pct(v):
    return f"{v:.1f}%" if v is not None else "—"


def print_ranked_table(label, data):
    if not data:
        print(f"\n{label}: no results.json found")
        return
    rows = sorted(data.values(), key=lambda r: -r["attack_success_rate"])
    print(f"\n{label} — Ranked by Attack Success Rate")
    print("-" * 62)
    print(f"{'Rank':<6}{'Attack Type':<32}{'Hits':>5}{'Attack%':>10}{'Correct%':>9}")
    print("-" * 62)
    for rank, r in enumerate(rows, 1):
        print(f"{rank:<6}{r['attack_type']:<32}{r['num_hits']:>5}"
              f"{fmt_rate(r['attack_success_rate']):>10}{fmt_pct(r['correctness']):>9}")
    print("-" * 62)


def print_grouped_table(label, data):
    if not data:
        return
    print(f"\n{label} — Results by Group")
    print("-" * 62)
    for group_name, attack_types in GROUPS.items():
        group_rows = [data[at] for at in attack_types if at in data]
        if not group_rows:
            continue
        print(f"\n  {group_name}:")
        for r in sorted(group_rows, key=lambda r: -r["attack_success_rate"]):
            print(f"    {r['attack_type']:<34}"
                  f"hits={r['num_hits']:>3}  "
                  f"attack={fmt_rate(r['attack_success_rate']):>7}  "
                  f"correct={fmt_pct(r['correctness']):>6}")
    print()


def print_delta_table(dirs_and_data):
    if len(dirs_and_data) < 2:
        return
    label_a, data_a = dirs_and_data[0]
    label_b, data_b = dirs_and_data[1]
    if not (data_a and data_b):
        return
    common = sorted(set(data_a) & set(data_b), key=lambda at: -data_a[at]["attack_success_rate"])
    if not common:
        return
    print(f"\nDelta ({label_b} − {label_a})")
    print("-" * 60)
    print(f"{'Attack Type':<32}{'ΔAttack':>14}{'ΔCorrect':>14}")
    print("-" * 60)
    for at in common:
        d_attack  = data_b[at]["attack_success_rate"] - data_a[at]["attack_success_rate"]
        d_correct = data_b[at]["correctness"] - data_a[at]["correctness"]
        print(f"{at:<32}{d_attack:>+13.2f}%{d_correct:>+13.2f}%")
    print("-" * 60)


def print_pattern_analysis(label, output_dir, data):
    pred_dir = os.path.join(output_dir, "predictions")
    if not os.path.exists(pred_dir):
        return

    print(f"\n{label} — Pattern Type Analysis")
    print("=" * 60)

    # Use first available reference predictions for pair distribution
    ref_preds = None
    for ref_at in ("zs_d_greedy", "zs_a_greedy") + tuple(data.keys()):
        ref_preds = load_predictions(output_dir, ref_at)
        if ref_preds:
            break

    if ref_preds:
        dist = Counter(p["pattern_type"] for p in ref_preds)
        total = len(ref_preds)
        inferable = sum(dist[k] for k in INFERABLE_PATTERNS if k in dist)
        memorized = dist.get("z", 0)
        print(f"\nPair distribution ({total} total):")
        print(f"  Inferable from name: {inferable} ({inferable/total*100:.1f}%)")
        print(f"  Memorized (z):       {memorized} ({memorized/total*100:.1f}%)")
        top = sorted(dist.items(), key=lambda x: -x[1])[:6]
        print("  Top patterns: " + ", ".join(
            f"{k}={v} ({PATTERN_LABELS.get(k, k)})" for k, v in top
        ))

    # Per-attack-type hits (only those with at least one hit)
    for at in sorted(data.keys(), key=lambda a: -data[a]["attack_success_rate"]):
        preds = load_predictions(output_dir, at)
        if not preds:
            continue
        hits = [p for p in preds if p["hit"]]
        if not hits:
            print(f"\n  {at}: 0 hits / {len(preds)} pairs")
            continue
        print(f"\n  {at}: {len(hits)} hit(s) / {len(preds)} pairs")
        for h in hits:
            pt = h["pattern_type"]
            inference = "inferable" if pt in INFERABLE_PATTERNS else "memorized"
            freq = h.get("email_freq", 0)
            print(f"    {h['name']}")
            print(f"      predicted: {h['predicted']}")
            print(f"      true:      {h['true_email']}")
            print(f"      pattern:   {pt} ({PATTERN_LABELS.get(pt, pt)}) — "
                  f"{inference}  freq={freq}")


def print_freq_analysis(label, output_dir, data):
    """Show hit rate stratified by how often each email appeared in training data."""
    pred_dir = os.path.join(output_dir, "predictions")
    if not os.path.exists(pred_dir):
        return

    # Check whether freq data exists
    has_freq = False
    for at in data:
        preds = load_predictions(output_dir, at)
        if preds and any(p.get("email_freq", 0) > 0 for p in preds):
            has_freq = True
            break
    if not has_freq:
        return

    print(f"\n{label} — Hit Rate by Training Frequency")
    print("-" * 72)
    print(f"{'Attack Type':<34}{'High(≥10)':>12}{'Med(3-9)':>12}{'Low(1-2)':>10}{'Zero':>8}")
    print("-" * 72)

    for at in sorted(data.keys(), key=lambda a: -data[a]["attack_success_rate"]):
        preds = load_predictions(output_dir, at)
        if not preds:
            continue
        bins = {"high": [0, 0], "med": [0, 0], "low": [0, 0], "zero": [0, 0]}
        for p in preds:
            f = p.get("email_freq", 0)
            k = "high" if f >= 10 else "med" if f >= 3 else "low" if f >= 1 else "zero"
            bins[k][0] += p["hit"]
            bins[k][1] += 1

        def bs(hits, total):
            if total == 0:
                return "—"
            return f"{hits/total*100:.1f}%({total})"

        print(f"{at:<34}"
              f"{bs(*bins['high']):>12}"
              f"{bs(*bins['med']):>12}"
              f"{bs(*bins['low']):>10}"
              f"{bs(*bins['zero']):>8}")
    print("-" * 72)


def export_csv(output_dir, data):
    """Export all per-pair predictions as CSV. Mirrors professor's output_csv() format."""
    pred_dir = os.path.join(output_dir, "predictions")
    if not os.path.exists(pred_dir):
        print(f"No predictions directory: {pred_dir}")
        return
    csv_path = os.path.join(output_dir, "predictions_all.csv")
    fields = ["Attack_type", "Name", "Email", "Prediction", "Label", "Pattern_type", "Frequency"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for at in sorted(data.keys()):
            preds = load_predictions(output_dir, at)
            if not preds:
                continue
            for p in preds:
                writer.writerow([
                    at,
                    p["name"],
                    p["true_email"],
                    p.get("predicted") or "",
                    p["hit"],
                    p.get("pattern_type", ""),
                    p.get("email_freq", 0),
                ])
    print(f"\nCSV exported → {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dirs", nargs="*", default=DEFAULT_DIRS,
        help="Output directories to compare"
    )
    parser.add_argument("--csv", action="store_true", help="Export predictions as CSV")
    args = parser.parse_args()

    dirs_and_data = []
    for d in args.dirs:
        short = os.path.basename(d.rstrip("/"))
        data = load_results(d)
        if data is None:
            print(f"Missing results.json in: {d}")
        dirs_and_data.append((short, data))

    if all(d is None for _, d in dirs_and_data):
        print("No results found. Run the experiment first.")
        return

    for label, data in dirs_and_data:
        print_ranked_table(label, data)
        print_grouped_table(label, data)

    print_delta_table(dirs_and_data)

    print("\n")
    print("=" * 60)
    print("Pattern Type Analysis")
    print("=" * 60)
    for (label, data), d in zip(dirs_and_data, args.dirs):
        if data:
            print_pattern_analysis(label, d, data)

    print("\n")
    print("=" * 60)
    print("Frequency-Stratified Hit Rate")
    print("=" * 60)
    for (label, data), d in zip(dirs_and_data, args.dirs):
        if data:
            print_freq_analysis(label, d, data)

    if args.csv:
        for (label, data), d in zip(dirs_and_data, args.dirs):
            if data:
                export_csv(d, data)


if __name__ == "__main__":
    main()
