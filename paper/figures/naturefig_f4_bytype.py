#!/usr/bin/env python3
"""F4: per-question-type *grouped bars* at fixed SNR (original style).

Faithful recreation of the F4 block in make_comparison_figures_v2.py:
grouped bars + Wilson yerr, same colors/labels/layout — only export is
sharper (PDF + 400 dpi PNG for slides). No line/marker restyle.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    # Times New Roman (Times-family fallbacks for Linux render hosts)
    "font.family": "serif",
    "font.serif": [
        "Times New Roman", "Nimbus Roman", "Liberation Serif",
        "STIXGeneral", "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    "figure.dpi": 150,
    "savefig.dpi": 300,
})
import matplotlib.pyplot as plt
import numpy as np

STYLE = {
    "M0_errorfree": ("#444444", "*", "Error-free image (ideal)"),
    "M0_naive": ("#ff6b6b", "x", "Fixed-rate image"),
    "M1_image": ("#ffb454", "o", "Rate-adaptive image"),
    "M2_analog": ("#c678dd", "v", "Uncoded analog"),
    "M3_token": ("#9aa7b4", "s", "Fixed token"),
    "M4_adaptive": ("#5ad19a", "D", "Evidence routing (proposed)"),
    "M5_oracle": ("#4ea1ff", "^", "Oracle (upper bound)"),
}
ORDER = [
    "M0_errorfree", "M5_oracle", "M4_adaptive", "M2_analog",
    "M3_token", "M1_image", "M0_naive",
]
CH_TITLE = {
    "awgn": "AWGN",
    "rayleigh": "Rayleigh",
    "rician": "Rician K=6 dB",
}


def find_csv() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        Path.home() / "phd_research/vqa_semcom/outputs/reports/comparison_v3_5qt.csv",
        here.parents[1] / "outputs/reports/comparison_v3_5qt.csv",
        here.parent / "comparison_v3_5qt.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("comparison_v3_5qt.csv not found")


def load(path: Path):
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        r["snr_db"] = float(r["snr_db"])
        r["accuracy"] = float(r["accuracy"])
        r["lcb"] = float(r["lcb"])
        r["ucb"] = float(r["ucb"])
    return rows


def main() -> None:
    bar_snr = 5.0
    channel = "rician"
    rows = load(find_csv())

    qtypes = sorted({r["qtype"] for r in rows if r["qtype"] not in ("all",)})
    methods = [m for m in ORDER if any(r["method"] == m for r in rows)]

    # Original single-column size from make_comparison_figures_v2.py
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(qtypes))
    w = 0.8 / max(len(methods), 1)

    for i, m in enumerate(methods):
        vals, lo, hi = [], [], []
        for qt in qtypes:
            rr = [
                r for r in rows
                if r["channel"] == channel
                and r["method"] == m
                and r["qtype"] == qt
                and abs(r["snr_db"] - bar_snr) < 1e-6
            ]
            if rr:
                vals.append(rr[0]["accuracy"])
                lo.append(rr[0]["accuracy"] - rr[0]["lcb"])
                hi.append(rr[0]["ucb"] - rr[0]["accuracy"])
            else:
                vals.append(0.0)
                lo.append(0.0)
                hi.append(0.0)
        col, _mk, lab = STYLE[m]
        ax.bar(
            x + i * w, vals, w, color=col, label=lab,
            yerr=[lo, hi],
            error_kw=dict(ecolor="0.3", lw=0.6, capsize=1.4),
        )
        print(m, list(zip(qtypes, [f"{v:.3f}" for v in vals])))

    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(qtypes, fontsize=6.8)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    ax.text(
        0.99, 0.98,
        f"{CH_TITLE.get(channel, channel)}, SNR = {bar_snr:g} dB",
        transform=ax.transAxes, va="top", ha="right", fontsize=6.5,
        fontweight="bold",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, fontsize=6,
        frameon=False, handlelength=1.0, columnspacing=0.8,
        labelspacing=0.3, bbox_to_anchor=(0.5, 1.02),
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(rect=(0, 0, 1, 0.82), pad=0.4)

    out_dirs = [
        Path.home() / "phd_research/vqa_semcom/outputs/figures/naturefig",
        Path(__file__).resolve().parent,
    ]
    for out in out_dirs:
        out.mkdir(parents=True, exist_ok=True)
        # vector + sharp raster
        fig.savefig(out / "F4_bytype.pdf")
        fig.savefig(out / "F4_bytype.png", dpi=300)
        fig.savefig(out / "F4_bytype.svg")
        print(f"wrote {out}/F4_bytype.[pdf|png|svg]")
    plt.close(fig)


if __name__ == "__main__":
    main()
