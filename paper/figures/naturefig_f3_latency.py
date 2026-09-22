#!/usr/bin/env python3
"""naturefig F3: end-to-end latency breakdown (Rayleigh).

Stacked bars per SNR: upload (solid) + tx-side detector (//) + inference (..).
Data: latency_breakdown.csv (from build_latency_breakdown.py).

Style aligned with F4/F8/F10 naturefig restyle (2026-07-23):
  Times New Roman, Type-42, clean spines, high-res PNG@400 dpi.
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
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.2,
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

ORDER = ["M0_naive", "M1_image", "M2_analog", "M3_token", "M4_adaptive"]
STYLE = {
    "M0_naive": "#d1495b",
    "M1_image": "#e8962f",
    "M2_analog": "#8e6bb5",
    "M3_token": "#5a6b7c",
    "M4_adaptive": "#5ad19a",
}
LABEL = {
    "M0_naive": "Fixed-rate image",
    "M1_image": "Rate-adaptive image",
    "M2_analog": "Uncoded analog",
    "M3_token": "Fixed token",
    "M4_adaptive": "Type routing (proposed)",
}
CH_TAG = {"awgn": "AWGN", "rayleigh": "Rayleigh", "rician": "Rician K=6 dB"}


def find_csv(cli: str | None) -> Path:
    if cli:
        p = Path(cli)
        if p.is_file():
            return p
        raise FileNotFoundError(cli)
    here = Path(__file__).resolve().parent
    for p in (
        here / "latency_breakdown.csv",
        Path.home() / "phd_research/vqa_semcom/outputs/reports/latency_breakdown.csv",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError("latency_breakdown.csv not found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-csv", default=None)
    ap.add_argument("--channel", default="rayleigh")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    path = find_csv(args.from_csv)
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["channel"] != args.channel:
                continue
            rows.append({
                "method": r["method"],
                "snr_db": float(r["snr_db"]),
                "upload_s": float(r["upload_s"]),
                "txside_s": float(r["txside_s"]),
                "inference_s": float(r["inference_s"]),
                "total_s": float(r["total_s"]),
            })
    if not rows:
        raise RuntimeError(f"no rows for channel={args.channel} in {path}")

    methods = [m for m in ORDER if any(r["method"] == m for r in rows)]
    snrs = sorted({r["snr_db"] for r in rows})
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.50, 2.45))
    x = np.arange(len(snrs))
    w = 0.82 / max(len(methods), 1)

    for i, m in enumerate(methods):
        ups, txs, infs = [], [], []
        for s in snrs:
            rr = next((r for r in rows if r["method"] == m and r["snr_db"] == s), None)
            ups.append(rr["upload_s"] if rr else 0.0)
            txs.append(rr["txside_s"] if rr else 0.0)
            infs.append(rr["inference_s"] if rr else 0.0)
        base = np.asarray(ups)
        mid = np.asarray(txs)
        xpos = x + i * w
        ax.bar(xpos, ups, w, color=STYLE[m], label=LABEL[m],
               edgecolor="white", linewidth=0.25, zorder=3)
        ax.bar(xpos, txs, w, bottom=base, color=STYLE[m], alpha=0.58,
               hatch="//", edgecolor="white", linewidth=0.25, zorder=3)
        ax.bar(xpos, infs, w, bottom=base + mid, color=STYLE[m], alpha=0.32,
               hatch="..", edgecolor="white", linewidth=0.25, zorder=3)

    ax.set_yscale("log")
    ymax = max(r["total_s"] for r in rows)
    ax.set_ylim(bottom=0.02, top=ymax * 14)
    ax.set_xticks(x + 0.5 * w * (len(methods) - 1))
    ax.set_xticklabels([f"{s:g}" for s in snrs])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("latency per query (s)")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(True, axis="y", which="both", alpha=0.28, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.55)

    ch_name = CH_TAG.get(args.channel, args.channel)
    ax.text(
        0.0, 1.03,
        f"{ch_name}  |  solid = upload   // = tx-side detector   .. = inference",
        transform=ax.transAxes, va="bottom", ha="left",
        fontsize=5.8, color="0.35",
    )
    ax.legend(
        fontsize=5.9, ncol=2, loc="upper right",
        handlelength=1.15, labelspacing=0.22, columnspacing=0.65,
        borderpad=0.28, frameon=True, framealpha=0.92,
        edgecolor="0.85", fancybox=False,
    )

    fig.tight_layout(pad=0.3)

    pdf_path = out_dir / "F3_latency.pdf"
    fig.savefig(pdf_path)
    fig.savefig(out_dir / "F3_latency.svg")
    fig.savefig(out_dir / "F3_latency.png", dpi=400)
    plt.close(fig)

    try:
        subprocess.run(
            [
                "pdftoppm", "-png", "-r", "400", "-singlefile",
                str(pdf_path), str(out_dir / "F3_latency"),
            ],
            check=False, capture_output=True,
        )
    except FileNotFoundError:
        pass

    # QA: paper numbers at -5 dB
    print(f"data: {path}")
    for m in methods:
        rr = next((r for r in rows if r["method"] == m and abs(r["snr_db"] + 5) < 1e-9), None)
        if rr:
            print(
                f"  {m:12s} @-5dB  upload={rr['upload_s']:.3f}  "
                f"tx={rr['txside_s']:.3f}  inf={rr['inference_s']:.3f}  "
                f"tot={rr['total_s']:.3f}"
            )
    print(f"wrote F3_latency.[pdf,svg,png] -> {out_dir}")


if __name__ == "__main__":
    main()
