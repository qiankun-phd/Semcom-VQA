#!/usr/bin/env python3
"""Final comparison figures from comparison_all.csv (tidy long format).

F1  3-panel accuracy vs SNR (one panel per channel), all methods
F2  cliff effect (one channel): naive digital cliffs vs graceful adaptive
F4  per-question-type grouped bars at a fixed SNR (one channel)
F5  Pareto: accuracy vs channel uses per query (bandwidth-fair across analog/digital)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "M0_errorfree": ("#444444", "*", "M0 error-free image (ideal channel)"),
    "M0_naive":    ("#ff6b6b", "x", "M0 naive digital (fixed-rate LDPC)"),
    "M1_image":    ("#ffb454", "o", "M1 traditional (rate-adaptive image)"),
    "M2_analog":   ("#c678dd", "v", "M2 uncoded analog (JSCC-lite)"),
    "M3_token":    ("#9aa7b4", "s", "M3 GO-SG token"),
    "M4_adaptive": ("#5ad19a", "D", "M4 ours (adaptive)"),
    "M5_oracle":   ("#4ea1ff", "^", "M5 oracle (service upper bound)"),
}
ORDER = ["M0_errorfree", "M5_oracle", "M4_adaptive", "M2_analog", "M3_token", "M1_image", "M0_naive"]
DASHED = {"M5_oracle": "--", "M0_errorfree": ":"}
CH_TITLE = {"awgn": "AWGN", "rayleigh": "Rayleigh", "rician": "Rician K=6 dB"}


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["snr_db"] = float(r["snr_db"]); r["accuracy"] = float(r["accuracy"])
        r["mean_payload_bytes"] = float(r["mean_payload_bytes"]); r["lcb"] = float(r["lcb"]); r["ucb"] = float(r["ucb"])
        r["mean_channel_uses"] = float(r.get("mean_channel_uses") or 0.0)
        r["cbr"] = float(r.get("cbr") or 0.0)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="outputs/figures/comparison")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--bar-snr", type=float, default=5.0)
    ap.add_argument("--pareto-channel", default="rician")
    ap.add_argument("--cliff-channel", default="rayleigh")
    args = ap.parse_args()
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load(args.csv)
    channels = [c for c in ["awgn", "rayleigh", "rician"] if any(r["channel"] == c for r in rows)]

    # ---- F1: 3-panel acc vs SNR (qtype=all, split=test) ----
    fig, axes = plt.subplots(1, len(channels), figsize=(5.2 * len(channels), 4.3), sharey=True)
    if len(channels) == 1:
        axes = [axes]
    for ax, ch in zip(axes, channels):
        for m in ORDER:
            pts = sorted([(r["snr_db"], r["accuracy"]) for r in rows
                          if r["channel"] == ch and r["method"] == m and r["qtype"] == "all"])
            if not pts:
                continue
            col, mk, lab = STYLE[m]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], DASHED.get(m, "-"), color=col, marker=mk,
                    label=lab, linewidth=2, markersize=5)
        ax.set_title(CH_TITLE.get(ch, ch)); ax.set_xlabel("SNR (dB)"); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("VQA answer accuracy")
    axes[-1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Accuracy vs SNR across channels (held-out test set)", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_dir}/F1_acc_snr_3panel_{args.tag}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- F2: cliff effect (one channel): naive digital cliffs vs graceful adaptive ----
    cliff_ch = args.cliff_channel
    if any(r["channel"] == cliff_ch for r in rows):
        fig, ax = plt.subplots(figsize=(6.6, 4.3))
        for m in ["M0_errorfree", "M5_oracle", "M4_adaptive", "M1_image", "M2_analog", "M0_naive"]:
            pts = sorted([(r["snr_db"], r["accuracy"]) for r in rows
                          if r["channel"] == cliff_ch and r["method"] == m and r["qtype"] == "all"])
            if not pts:
                continue
            col, mk, lab = STYLE[m]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], DASHED.get(m, "-"), color=col, marker=mk,
                    label=lab, linewidth=2, markersize=5)
        ax.axhline(0.5, ls=":", color="#888", lw=1)
        ax.text(0.99, 0.5, " chance≈0.5", transform=ax.get_yaxis_transform(),
                va="bottom", ha="right", color="#888", fontsize=8)
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("VQA answer accuracy")
        ax.set_title(f"Cliff effect: naive digital vs graceful adaptive ({CH_TITLE.get(cliff_ch, cliff_ch)})")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7.5, loc="lower right")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"{args.out_dir}/F2_cliff_{args.tag}.{ext}", dpi=140)
        plt.close(fig)

    # ---- F4: per-question-type grouped bars at fixed SNR (rician) ----
    ch = args.pareto_channel
    qtypes = sorted({r["qtype"] for r in rows if r["qtype"] not in ("all",)})
    methods = [m for m in ORDER if any(r["method"] == m for r in rows)]
    import numpy as np
    fig, ax = plt.subplots(figsize=(7, 4.3))
    x = np.arange(len(qtypes)); w = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        vals = []
        for qt in qtypes:
            v = [r["accuracy"] for r in rows if r["channel"] == ch and r["method"] == m
                 and r["qtype"] == qt and abs(r["snr_db"] - args.bar_snr) < 1e-6]
            vals.append(v[0] if v else 0)
        col, mk, lab = STYLE[m]
        ax.bar(x + i * w, vals, w, color=col, label=lab)
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(qtypes)
    ax.set_ylabel("accuracy"); ax.set_title(f"Per-question-type accuracy ({CH_TITLE.get(ch, ch)}, SNR={args.bar_snr:g} dB)")
    ax.legend(fontsize=7.5, ncol=2); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_dir}/F4_qtype_{args.tag}.{ext}", dpi=140)
    plt.close(fig)

    # ---- F5: Pareto acc vs channel uses per query (unified analog/digital axis) ----
    use_uses = any(r["mean_channel_uses"] > 0 for r in rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    pareto_methods = (["M0_naive", "M1_image", "M2_analog", "M3_token", "M4_adaptive"]
                      if use_uses else ["M1_image", "M3_token", "M4_adaptive"])
    xkey = "mean_channel_uses" if use_uses else "mean_payload_bytes"
    for m in pareto_methods:
        pts = [(r[xkey], r["accuracy"]) for r in rows
               if r["channel"] == ch and r["method"] == m and r["qtype"] == "all" and r[xkey] > 0]
        if not pts:
            continue
        col, mk, lab = STYLE[m]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, s=60, label=lab, zorder=3)
    ax.set_xscale("log")
    if use_uses:
        ax.set_xlabel("mean complex channel uses per query (CBR = uses / 2.88M source symbols)")
    else:
        ax.set_xlabel("mean payload (bytes / query)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Goal-oriented efficiency ({CH_TITLE.get(ch, ch)})")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_dir}/F5_pareto_{args.tag}.{ext}", dpi=140)
    plt.close(fig)

    print(f"wrote F1(3-panel)/F2/F4/F5 [{args.tag}] -> {args.out_dir}; channels={channels}")


if __name__ == "__main__":
    raise SystemExit(main())
