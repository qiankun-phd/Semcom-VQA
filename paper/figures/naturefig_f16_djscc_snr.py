#!/usr/bin/env python3
"""naturefig F16: learned DJSCC baseline in the Rician grid (TGCN).

Table-to-figure conversion of the manuscript's Table `tab:m6`: the five
rows become five accuracy-vs-SNR lines on the Rician test grid.  Rows
marked with a double dagger in the table (presence+counting subset, the
analog baseline's coverage) are drawn with open markers and broken lines.

Data of record only -- no recomputation:
  outputs/reports/comparison_v3_5qt.csv  (M1_image, M2_analog, M4_adaptive;
      rician / qtype=all / split=test rows -- same source as the table)
  outputs/reports/p1_m6_results.json     (DJSCC per_snr, per_snr_2type)

Style: nature-figure conventions -- Times-family serif (Nimbus Roman on
the render host), grayscale-safe linestyles (same per-method styles as
every quantitative figure), Type-42 fonts. Outputs PDF + SVG + PNG (300 dpi).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 6.5,
    "lines.linewidth": 1.0, "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/figures/naturefig"
DDAG = "‡"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acc = {}
    for r in csv.DictReader(open(REPO / "outputs/reports/comparison_v3_5qt.csv")):
        if (r["channel"] == "rician" and r["qtype"] == "all"
                and r["split"] == "test"):
            acc.setdefault(r["method"], {})[float(r["snr_db"])] = \
                float(r["accuracy"])
    m6 = json.loads((REPO / "outputs/reports/p1_m6_results.json").read_text())
    acc["M6_djscc"] = {float(s): v["acc"] for s, v in m6["per_snr"].items()}
    acc["M6_djscc_2t"] = {float(s): v["acc"]
                          for s, v in m6["per_snr_2type"].items()}

    series = [  # key, color, marker, ls, open?, label
        ("M4_adaptive", "#5ad19a", "D", "-", False, "Type routing (proposed)"),
        ("M1_image", "#e8962f", "o", "-", False, "Rate-adaptive image"),
        ("M6_djscc", "#8b5e3c", "P", "-.", False, "DJSCC (learned)"),
        ("M6_djscc_2t", "#8b5e3c", "P", ":", True,
         f"DJSCC{DDAG} (presence+counting)"),
        ("M2_analog", "#8e6bb5", "v", "--", True,
         f"Uncoded analog{DDAG}"),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    for key, c, mk, ls, opn, lb in series:
        xs = sorted(acc[key])
        ys = [acc[key][s] for s in xs]
        hero = key == "M4_adaptive"
        ax.plot(xs, ys, marker=mk, color=c, ls=ls, label=lb,
                mfc="white" if opn else c,
                lw=1.4 if hero else 0.9, ms=3.8 if hero else 3.2,
                zorder=6 if hero else 3)
        print(key, [f"{y:.4f}" for y in ys])
    ax.annotate("routing stays 7.6–9.3 pts ahead", xy=(2.5, 0.695),
                fontsize=6.2, color="#1f7a54", ha="center")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("accuracy (Rician, test)")
    ax.set_xticks(sorted(acc["M4_adaptive"]))
    ax.set_ylim(0.44, 0.72)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.text(0.985, 0.03,
            f"{DDAG}: presence+counting subset\n(analog coverage)",
            transform=ax.transAxes, fontsize=6.0, color="0.35",
            ha="right", va="bottom")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.13), frameon=False,
              handlelength=1.7, labelspacing=0.25)
    fig.tight_layout()
    for ext, dpi in (("pdf", None), ("svg", None), ("png", 300)):
        fig.savefig(OUT / f"F16_djscc_snr.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"wrote {OUT}/F16_djscc_snr.[pdf|svg|png]")


if __name__ == "__main__":
    main()
