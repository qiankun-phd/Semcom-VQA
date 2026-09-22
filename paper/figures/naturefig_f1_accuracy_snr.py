#!/usr/bin/env python3
"""naturefig F1: 3-panel accuracy vs SNR (AWGN / Rayleigh / Rician).

Publication redraw of make_comparison_figures_v2.py F1 block.
Plot-only from comparison_v3_5qt.csv (qtype=all); data unchanged.

Style aligned with F4/F8/F10 restyle (2026-07-23):
  Times New Roman, Type-42, clean spines, panel tags outside axes,
  high-res PNG@400 dpi (+ optional pdftoppm@400).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.8,
    "lines.linewidth": 1.15,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": [
        "Times New Roman", "Nimbus Roman", "Liberation Serif",
        "STIXGeneral", "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})
import matplotlib.pyplot as plt

STYLE = {
    "M0_errorfree": ("#444444", "*", "Error-free image"),
    "M0_naive": ("#d1495b", "x", "Fixed-rate image"),
    "M1_image": ("#e8962f", "o", "Rate-adaptive image"),
    "M2_analog": ("#8e6bb5", "v", "Uncoded analog"),
    "M3_token": ("#5a6b7c", "s", "Fixed token"),
    "M4_adaptive": ("#5ad19a", "D", "Type routing (proposed)"),
    "M5_oracle": ("#4ea1ff", "^", "Oracle (upper bound)"),
}
ORDER = [
    "M0_errorfree", "M5_oracle", "M4_adaptive", "M2_analog",
    "M3_token", "M1_image", "M0_naive",
]
DASHED = {"M5_oracle": "--", "M0_errorfree": ":"}
CH_ORDER = ("awgn", "rayleigh", "rician")
CH_TITLE = {"awgn": "AWGN", "rayleigh": "Rayleigh", "rician": "Rician K=6 dB"}
BAND_METHODS = {"M4_adaptive", "M1_image", "M3_token"}


def find_csv(cli: str | None) -> Path:
    if cli:
        p = Path(cli)
        if p.is_file():
            return p
        raise FileNotFoundError(cli)
    here = Path(__file__).resolve().parent
    for p in (
        here / "comparison_v3_5qt.csv",
        Path.home() / "phd_research/vqa_semcom/outputs/reports/comparison_v3_5qt.csv",
        here.parents[1] / "outputs/reports/comparison_v3_5qt.csv",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError("comparison_v3_5qt.csv not found")


def load_rows(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "channel": r["channel"],
                "method": r["method"],
                "qtype": r["qtype"],
                "snr_db": float(r["snr_db"]),
                "accuracy": float(r["accuracy"]),
                "lcb": float(r["lcb"]),
                "ucb": float(r["ucb"]),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    csv_path = find_csv(args.csv)
    rows = load_rows(csv_path)
    channels = [c for c in CH_ORDER if any(r["channel"] == c for r in rows)]
    if not channels:
        raise RuntimeError(f"no channels in {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # double-column width; slightly taller for legend + panel tags
    fig, axes = plt.subplots(
        1, len(channels), figsize=(7.16, 2.15), sharey=True,
    )
    if len(channels) == 1:
        axes = [axes]

    for k, (ax, ch) in enumerate(zip(axes, channels)):
        for m in ORDER:
            pts = sorted(
                [
                    (r["snr_db"], r["accuracy"], r["lcb"], r["ucb"])
                    for r in rows
                    if r["channel"] == ch and r["method"] == m and r["qtype"] == "all"
                ],
                key=lambda p: p[0],
            )
            if not pts:
                continue
            col, mk, lab = STYLE[m]
            hero = m == "M4_adaptive"
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(
                xs, ys, DASHED.get(m, "-"), color=col, marker=mk, label=lab,
                markersize=4.0 if hero else 3.2,
                markeredgewidth=0.45 if hero else 0.35,
                markeredgecolor="white" if hero else col,
                lw=1.65 if hero else 1.05,
                zorder=6 if hero else 3,
            )
            if m in BAND_METHODS:
                ax.fill_between(
                    xs, [p[2] for p in pts], [p[3] for p in pts],
                    color=col, alpha=0.18 if hero else 0.11,
                    linewidth=0, zorder=2,
                )

        # annotations
        if k == len(channels) - 1:
            ef = [
                r for r in rows
                if r["channel"] == ch and r["method"] == "M0_errorfree"
                and r["qtype"] == "all"
            ]
            if ef:
                y0 = ef[0]["accuracy"]
                ax.annotate(
                    f"error-free ref. {y0:.3f}", xy=(18.5, y0),
                    xytext=(19.6, 0.472), ha="right", fontsize=6.0,
                    color="#444444",
                    arrowprops=dict(
                        arrowstyle="-", lw=0.55, color="#444444",
                        alpha=0.65, shrinkA=0.5, shrinkB=1.0,
                    ),
                )
        if k == 0:
            nv = sorted(
                [
                    (r["snr_db"], r["accuracy"])
                    for r in rows
                    if r["channel"] == ch and r["method"] == "M0_naive"
                    and r["qtype"] == "all"
                ]
            )
            if len(nv) >= 2:
                ax.annotate(
                    "digital cliff", xy=nv[1],
                    xytext=(nv[1][0] + 2.2, nv[1][1] - 0.055),
                    fontsize=6.0, color="#d1495b",
                    arrowprops=dict(
                        arrowstyle="-", lw=0.55, color="#d1495b",
                        alpha=0.85, shrinkA=0.5, shrinkB=1.5,
                    ),
                )

        ax.text(
            0.04, 0.06, CH_TITLE.get(ch, ch), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=7.5, fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25", fc="white", ec="0.75",
                alpha=0.92, lw=0.5,
            ),
        )
        # panel tag outside axes — avoid crop (same fix as F6)
        ax.text(
            -0.02, 1.08, f"({chr(97 + k)})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="right",
            clip_on=False,
        )
        ax.set_xlabel("SNR (dB)")
        ax.set_xlim(-6.5, 21.5)
        ax.set_ylim(0.36, 0.80)
        ax.set_xticks([-5, 0, 5, 10, 15, 20])
        ax.set_yticks([0.4, 0.5, 0.6, 0.7])
        # keep full box (top/right spines) — original F1 / paper style
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.6)
        ax.grid(True, alpha=0.28)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=0.55)

    axes[0].set_ylabel("VQA answer accuracy")

    handles, labels = axes[-1].get_legend_handles_labels()
    # stable legend order = ORDER
    by_lab = dict(zip(labels, handles))
    ordered_labs = [STYLE[m][2] for m in ORDER if STYLE[m][2] in by_lab]
    ordered_h = [by_lab[l] for l in ordered_labs]
    fig.legend(
        ordered_h, ordered_labs, loc="upper center", ncol=4, fontsize=6.8,
        frameon=False, borderaxespad=0.15, handlelength=1.85,
        columnspacing=0.95, handletextpad=0.4,
        bbox_to_anchor=(0.5, 1.06),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86), pad=0.35, w_pad=0.55)

    pdf_path = out_dir / "F1_accuracy_snr.pdf"
    fig.savefig(pdf_path)
    fig.savefig(out_dir / "F1_accuracy_snr.svg")
    fig.savefig(out_dir / "F1_accuracy_snr.png", dpi=400)
    plt.close(fig)

    try:
        subprocess.run(
            [
                "pdftoppm", "-png", "-r", "400", "-singlefile",
                str(pdf_path), str(out_dir / "F1_accuracy_snr"),
            ],
            check=False, capture_output=True,
        )
    except FileNotFoundError:
        pass

    # stage to paper_semcom/figures if running from scripts/
    paper_fig = Path(__file__).resolve().parent
    if paper_fig.name == "figures":
        pass  # already writing here when --out-dir is figures
    print(f"data: {csv_path}")
    print(f"wrote F1_accuracy_snr.[pdf,svg,png] -> {out_dir}")


if __name__ == "__main__":
    main()
