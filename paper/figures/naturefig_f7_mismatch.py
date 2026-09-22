#!/usr/bin/env python3
"""naturefig F7: CSI-mismatch heatmaps (assumed SNR × true SNR).

Policy picks service from (qtype, SNR_assumed); outcome is read at
SNR_true. Physical link still adapts to true SNR — isolates scheduler
CSI error. Data: mismatch_matrix.csv (qtype=all cells).

Style aligned with F4/F8/F14 naturefig restyle (2026-07-23):
  Times New Roman, Type-42, shared color scale, high-res PNG@400 dpi.
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
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.55,
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

CH_ORDER = ["awgn", "rayleigh", "rician"]
CH_TAG = {"awgn": "AWGN", "rayleigh": "Rayleigh", "rician": "Rician K=6 dB"}


def find_csv(cli: str | None) -> Path:
    if cli:
        p = Path(cli)
        if p.is_file():
            return p
        raise FileNotFoundError(cli)
    here = Path(__file__).resolve().parent
    for p in (
        here / "mismatch_matrix.csv",
        Path.home() / "phd_research/vqa_semcom/outputs/reports/mismatch_matrix.csv",
        here.parents[1] / "outputs/reports/mismatch_matrix.csv",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError("mismatch_matrix.csv not found")


def load_mats(path: Path):
    """Return {channel: (snrs_sorted, matrix[ia, it])} for qtype=all."""
    cells: dict[tuple, float] = {}
    snr_set: set[float] = set()
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["qtype"] != "all":
                continue
            ch = r["channel"]
            sa = float(r["snr_assumed_db"])
            st = float(r["snr_true_db"])
            snr_set.add(sa)
            snr_set.add(st)
            cells[(ch, sa, st)] = float(r["accuracy"])
    snrs = sorted(snr_set)
    mats = {}
    for ch in CH_ORDER:
        mat = np.full((len(snrs), len(snrs)), np.nan)
        for i, sa in enumerate(snrs):
            for j, st in enumerate(snrs):
                v = cells.get((ch, sa, st))
                if v is not None:
                    mat[i, j] = v
        if np.isfinite(mat).any():
            mats[ch] = (snrs, mat)
    return mats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-csv", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    csv_path = find_csv(args.from_csv)
    mats = load_mats(csv_path)
    chs = [c for c in CH_ORDER if c in mats]
    if not chs:
        raise RuntimeError(f"no channels in {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    allv = np.concatenate([mats[c][1].ravel() for c in chs])
    allv = allv[np.isfinite(allv)]
    vmin, vmax = float(allv.min()), float(allv.max())
    # slight pad so colorbar isn't edge-clipped
    pad = 0.002
    vmin_p, vmax_p = vmin - pad, vmax + pad
    thr = vmin + 0.55 * (vmax - vmin)

    # IEEE single-column width family (~3.5 in) → slightly wider for 3 panels
    fig, axes = plt.subplots(
        1, len(chs), figsize=(3.60, 1.58), squeeze=False,
        constrained_layout=True,
    )
    im = None
    for k, (ax, ch) in enumerate(zip(axes[0], chs)):
        snrs, mat = mats[ch]
        im = ax.imshow(
            mat, origin="lower", cmap="viridis", aspect="auto",
            vmin=vmin_p, vmax=vmax_p, interpolation="nearest",
        )
        ax.set_xticks(range(len(snrs)))
        ax.set_xticklabels([f"{s:g}" for s in snrs], fontsize=5.5)
        ax.set_yticks(range(len(snrs)))
        if k == 0:
            ax.set_yticklabels([f"{s:g}" for s in snrs], fontsize=5.5)
            ax.set_ylabel("assumed SNR (dB)", fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("true SNR (dB)", fontsize=7)
        ax.tick_params(length=2.0, width=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
        ax.text(
            0.02, 1.05, CH_TAG.get(ch, ch),
            transform=ax.transAxes, va="bottom", ha="left",
            fontsize=6.5, fontweight="bold",
        )
        for i in range(len(snrs)):
            for j in range(len(snrs)):
                v = mat[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=4.4,
                    color="white" if v < thr else "0.05",
                    fontweight="medium",
                )

    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=5.5, length=2.0, width=0.5)
    cbar.outline.set_linewidth(0.55)

    pdf_path = out_dir / "F7_mismatch.pdf"
    fig.savefig(pdf_path)
    fig.savefig(out_dir / "F7_mismatch.svg")
    fig.savefig(out_dir / "F7_mismatch.png", dpi=400)
    plt.close(fig)

    try:
        subprocess.run(
            [
                "pdftoppm", "-png", "-r", "400", "-singlefile",
                str(pdf_path), str(out_dir / "F7_mismatch"),
            ],
            check=False, capture_output=True,
        )
    except FileNotFoundError:
        pass

    # quick sanity print
    for ch in chs:
        snrs, mat = mats[ch]
        diag = [mat[i, i] for i in range(len(snrs))]
        off = [mat[i, j] for i in range(len(snrs)) for j in range(len(snrs)) if i != j]
        print(
            f"[{ch}] diag={min(diag):.3f}–{max(diag):.3f}  "
            f"worst off={min(off):.3f}  mean off={sum(off)/len(off):.3f}"
        )
    print(f"data: {csv_path}")
    print(f"wrote F7_mismatch.[pdf,svg,png] -> {out_dir}")


if __name__ == "__main__":
    main()
