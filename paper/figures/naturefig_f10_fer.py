#!/usr/bin/env python3
"""naturefig F10: residual frame-loss probability vs SNR.

Token information-outage (measured mean s1 payload) vs measured fixed-rate
LDPC FER, three channels. Data: p1_fer_payload.json.

Style aligned with F4/F8/F14 naturefig restyle (2026-07-23):
  Times New Roman, Type-42, clean spines, high-res PNG@400 dpi.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.5,
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
    "savefig.pad_inches": 0.02,
})
import matplotlib.pyplot as plt
import numpy as np

# Channel palette consistent with F1-family cool/warm split
COLORS = {
    "awgn": "#4ea1ff",
    "rayleigh": "#d1495b",
    "rician": "#5ad19a",
}
CH_ORDER = ["awgn", "rayleigh", "rician"]
FLOOR = 2e-5


def find_json(cli: str | None) -> Path:
    if cli:
        p = Path(cli)
        if p.is_file():
            return p
        raise FileNotFoundError(cli)
    here = Path(__file__).resolve().parent
    for p in (
        here / "p1_fer_payload.json",
        Path.home() / "phd_research/vqa_semcom/outputs/reports/p1_fer_payload.json",
        here.parents[1] / "outputs/reports/p1_fer_payload.json",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError("p1_fer_payload.json not found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-json", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    path = find_json(args.from_json)
    data = json.loads(path.read_text())
    curves = data["curves"]
    ldpc = data["report"]["ldpc_fixed_rate_fer_measured"]
    grid = np.asarray(curves["snr_grid"], dtype=float)

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.45, 2.40))

    # Token outage (solid)
    for name in CH_ORDER:
        key = f"tokenMeas_{name}"
        if key not in curves:
            continue
        y = np.asarray(curves[key], dtype=float)
        c = COLORS[name]
        if np.all(y <= FLOOR):
            ax.semilogy(
                grid, np.full_like(grid, FLOOR), color=c, ls="-", lw=1.35,
                label=f"Fixed token outage ({name.upper() if name=='awgn' else name.title()}) $=0$", zorder=3,
            )
            continue
        ax.semilogy(
            grid, np.maximum(y, FLOOR), color=c, ls="-", lw=1.25,
            label=f"Fixed token outage ({name.upper() if name=='awgn' else name.title()})", zorder=3,
        )

    # Fixed-rate LDPC FER (dashed + markers)
    for name in CH_ORDER:
        if name not in ldpc:
            continue
        c = COLORS[name]
        # keys like "−5dB" / "-5dB" / "5dB"
        items = []
        for k, v in ldpc[name].items():
            s = k.replace("dB", "").replace("−", "-").strip()
            items.append((float(s), float(v)))
        items.sort()
        xs = [a for a, _ in items]
        ys = np.maximum([b for _, b in items], FLOOR)
        ax.semilogy(
            xs, ys, color=c, ls="--", marker="o", ms=3.2,
            markeredgewidth=0.4, markeredgecolor="white",
            lw=1.15, label=f"Fixed-rate image FER ({name.upper() if name=='awgn' else name.title()})", zorder=4,
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("residual frame-loss probability")
    ax.set_ylim(FLOOR, 2.0)
    ax.set_xlim(-6, 21)
    ax.set_xticks([-5, 0, 5, 10, 15, 20])
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(True, which="major", alpha=0.28)
    ax.grid(True, which="minor", alpha=0.12)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.55)
    ax.legend(
        frameon=False, ncol=1, loc="lower left",
        handlelength=1.9, borderaxespad=0.4, labelspacing=0.25,
    )

    fig.tight_layout(pad=0.25)

    pdf_path = out_dir / "F10_fer.pdf"
    fig.savefig(pdf_path)
    fig.savefig(out_dir / "F10_fer.svg")
    fig.savefig(out_dir / "F10_fer.png", dpi=400)
    plt.close(fig)

    try:
        subprocess.run(
            [
                "pdftoppm", "-png", "-r", "400", "-singlefile",
                str(pdf_path), str(out_dir / "F10_fer"),
            ],
            check=False, capture_output=True,
        )
    except FileNotFoundError:
        pass

    # print key bins for caption QA
    bins = [-5, 0, 5, 10, 15, 20]
    print(f"data: {path}")
    for name in CH_ORDER:
        key = f"tokenMeas_{name}"
        if key not in curves:
            continue
        y = np.asarray(curves[key], dtype=float)
        g = list(grid)
        parts = []
        for s in bins:
            if s in g:
                parts.append(f"{s}dB={y[g.index(s)]:.2e}")
        print(f"  tokenMeas {name}: " + ", ".join(parts))
        if name in ldpc:
            print(f"  LDPC      {name}: " + ", ".join(
                f"{k}={v:.3f}" for k, v in sorted(
                    ldpc[name].items(),
                    key=lambda kv: float(kv[0].replace("dB", "").replace("−", "-")),
                )
            ))
    print(f"wrote F10_fer.[pdf,svg,png] -> {out_dir}")


if __name__ == "__main__":
    main()
