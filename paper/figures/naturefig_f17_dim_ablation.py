#!/usr/bin/env python3
"""naturefig F17: routing-dimension ablation, dot plot (TGCN).

Table-to-figure conversion of the manuscript's Table `tab:dimablation`:
pooled test accuracy of the LCB selector under six conditioning-feature
sets becomes a horizontal dot plot with a reference line at the
question-type-only selector -- question type carries the decision, every
further feature moves pooled accuracy by at most +-0.003.

Data of record only: outputs/reports/ablation_mechanism_v3.csv
(ablation=3a_featureset, channel=pooled rows; n = 16848).

Style: nature-figure conventions -- Times-family serif (Nimbus Roman on
the render host), dots not truncated bars (the axis does not start at 0),
Type-42 fonts. Outputs PDF + SVG + PNG (300 dpi).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/figures/naturefig"

ROWS = [  # csv variant key -> display label (top-to-bottom, table order)
    ("none", "none (global best level)"),
    ("qt", "question type only"),
    ("qt+snr  (current M4)", "qt + SNR (as deployed)"),
    ("qt+snr+view", "qt + SNR + viewpoint"),
    ("qt+snr+view+risk", "qt + SNR + viewpoint + risk"),
    ("qt+snr+freshness", "qt + SNR + freshness"),
]
C_QT, C_REST = "#5ad19a", "#5a6b7c"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    accs = {}
    for r in csv.DictReader(open(
            REPO / "outputs/reports/ablation_mechanism_v3.csv")):
        if r["ablation"] == "3a_featureset" and r["channel"] == "pooled":
            accs[r["variant"]] = float(r["test_accuracy"])
            assert int(r["n"]) == 16848
    ref = accs["qt"]

    fig, ax = plt.subplots(figsize=(3.35, 1.9))
    ys = range(len(ROWS), 0, -1)
    ax.axvline(ref, color=C_QT, lw=0.9, ls="--", zorder=2)
    for y, (key, lab) in zip(ys, ROWS):
        a = accs[key]
        hero = key == "qt"
        ax.plot(a, y, "D" if hero else "o", ms=5 if hero else 4,
                color=C_QT if hero else C_REST, zorder=4)
        d = a - ref
        ax.annotate("ref." if hero else f"{d:+.3f}", (a, y),
                    textcoords="offset points",
                    xytext=(6, -2.6), fontsize=6.2,
                    color="#1f7a54" if hero else "0.35")
        print(f"{lab:32s} {a:.4f}  ({a - ref:+.4f})")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([lab for _, lab in ROWS], fontsize=7)
    ax.set_xlabel("pooled test accuracy (3 channels, $n{=}16848$)")
    ax.set_xlim(0.628, 0.6775)
    ax.set_ylim(0.4, len(ROWS) + 0.6)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.annotate("further features: $\\leq\\pm0.003$",
                xy=(0.6595, 4.4), fontsize=6.2, color="0.35", ha="right")
    fig.tight_layout()
    for ext, dpi in (("pdf", None), ("svg", None), ("png", 300)):
        fig.savefig(OUT / f"F17_dim_ablation.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"wrote {OUT}/F17_dim_ablation.[pdf|svg|png]")


if __name__ == "__main__":
    main()
