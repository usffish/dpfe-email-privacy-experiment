"""
Plot DP-SGD noise sweep results: hits and val loss vs sigma.
Usage: python plot_noise_sweep.py <results_json> [output_png]
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_noise_sweep.py <results_json> [output_png]")
        sys.exit(1)

    results_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else results_path.replace(".json", "_plot.png")

    with open(results_path) as f:
        data = json.load(f)

    data = sorted(data, key=lambda r: r["noise"])
    sigmas = [r["noise"] for r in data]
    hits = [r["num_hits"] for r in data]
    val_losses = [r.get("val_loss") for r in data]
    perplexities = [r.get("perplexity") for r in data]

    has_val_loss = any(v is not None for v in val_losses)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_hits = "#e74c3c"
    color_loss = "#2980b9"

    x = np.arange(len(sigmas))
    sigma_labels = [str(s) for s in sigmas]

    ax1.bar(x, hits, color=color_hits, alpha=0.7, label="Hits (composite attack)")
    ax1.set_xlabel("DP-SGD Noise Level (σ)", fontsize=12)
    ax1.set_ylabel("Number of Hits", color=color_hits, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_hits)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sigma_labels, rotation=45, ha="right")
    ax1.set_ylim(bottom=0)

    for i, h in enumerate(hits):
        ax1.text(i, h + 0.2, str(h), ha="center", va="bottom", fontsize=9, color=color_hits)

    if has_val_loss:
        ax2 = ax1.twinx()
        valid = [(i, v) for i, v in enumerate(val_losses) if v is not None]
        xi, yi = zip(*valid)
        ax2.plot(xi, yi, color=color_loss, marker="o", linewidth=2, markersize=6, label="Val Loss")
        ax2.set_ylabel("Validation Loss", color=color_loss, fontsize=12)
        ax2.tick_params(axis="y", labelcolor=color_loss)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    plt.title("Composite Attack Hits & Val Loss vs. DP-SGD Noise (σ)\nGPT-Neo-125M, 6-attack union, full fine-tune", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
