#!/usr/bin/env python3
"""Render comparison figures from comparison_*_by_snr.csv.

F1  accuracy vs SNR (M1/M3/M4 + M5 oracle)
F3  accuracy vs mean payload bytes (goal-oriented efficiency)
Saves both PNG (preview) and PDF (paper) into the figures dir.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "M1_image":    ("#ffb454", "o", "M1 traditional (full image, JPEG+LDPC)"),
    "M3_token":    ("#9aa7b4", "s", "M3 GO-SG digital token"),
    "M4_adaptive": ("#5ad19a", "D", "M4 ours (adaptive selection)"),
    "M5_oracle":   ("#4ea1ff", "^", "M5 oracle (upper bound)"),
}
ORDER = ["M5_oracle", "M4_adaptive", "M3_token", "M1_image"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="outputs/figures/comparison")
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    data = defaultdict(list)  # method -> list[(snr, acc, bytes)]
    for r in csv.DictReader(open(args.csv)):
        data[r["method"]].append((float(r["snr_db"]), float(r["accuracy"]), float(r["mean_payload_bytes"])))
    for m in data:
        data[m].sort()

    # ---- F1 accuracy vs SNR ----
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in ORDER:
        if m not in data:
            continue
        col, mk, lab = STYLE[m]
        xs = [d[0] for d in data[m]]; ys = [d[1] for d in data[m]]
        ls = "--" if m == "M5_oracle" else "-"
        ax.plot(xs, ys, ls, color=col, marker=mk, label=lab, linewidth=2, markersize=6)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("VQA answer accuracy")
    ax.set_title("Accuracy vs SNR (Rician K=6 dB)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_dir}/F1_acc_snr_{args.tag}.{ext}", dpi=140)
    plt.close(fig)

    # ---- F3 accuracy vs payload bytes (log-x) ----
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in ORDER:
        if m not in data:
            continue
        col, mk, lab = STYLE[m]
        xs = [max(d[2], 1) for d in data[m]]; ys = [d[1] for d in data[m]]
        ax.scatter(xs, ys, color=col, marker=mk, s=55, label=lab, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("mean payload (bytes / query)"); ax.set_ylabel("VQA answer accuracy")
    ax.set_title("Goal-oriented efficiency: accuracy vs transmission cost")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_dir}/F3_acc_bytes_{args.tag}.{ext}", dpi=140)
    plt.close(fig)

    print(f"wrote F1/F3 ({args.tag}) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
