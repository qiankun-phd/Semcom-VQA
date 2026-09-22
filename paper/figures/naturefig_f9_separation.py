#!/usr/bin/env python3
"""naturefig F9: U-space safety bridge — separation minimum & capacity loss.

Two-panel figure (Rayleigh, peak load, shared C2, 2-sigma):
  (top)    d_TC separation minimum vs SNR
  (bottom) airspace-capacity loss (%) vs SNR

Data: separation_v2/sepcap_peak_shared.csv (M/G/1 companion analysis).

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
    "legend.fontsize": 6.3,
    "lines.linewidth": 1.2,
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

# Match make_p1_figures.py / paper-facing order
ORDER = ["M0_naive", "M1_image", "M4_adaptive", "M2_analog", "M3_token"]
STYLE = {
    "M0_naive": dict(color="#7f7f7f", ls=":", marker="v", label="Fixed-rate image"),
    "M1_image": dict(color="#d62728", ls="-", marker="o", label="Rate-adaptive image"),
    "M4_adaptive": dict(color="#1f77b4", ls="-", marker="s", label="Evidence routing (proposed)"),
    "M2_analog": dict(color="#ff7f0e", ls="--", marker="^", label="Uncoded analog"),
    "M3_token": dict(color="#2ca02c", ls="-", marker="D", label="Fixed token"),
}


def find_csv(cli: str | None) -> Path:
    if cli:
        p = Path(cli)
        if p.is_file():
            return p
        raise FileNotFoundError(cli)
    here = Path(__file__).resolve().parent
    for p in (
        here / "sepcap_peak_shared.csv",
        Path.home() / "phd_research/vqa_semcom/outputs/reports/separation_v2/sepcap_peak_shared.csv",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError("sepcap_peak_shared.csv not found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-csv", default=None)
    ap.add_argument("--channel", default="rayleigh")
    ap.add_argument("--sigma-band", type=float, default=2.0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    path = find_csv(args.from_csv)
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["channel"] != args.channel:
                continue
            if abs(float(r["sigma_band"]) - args.sigma_band) > 1e-9:
                continue
            if r.get("load") not in (None, "", "peak") and r.get("load") != "peak":
                # file is already peak_shared; still filter if column present
                if r.get("load") != "peak":
                    continue
            rows.append({
                "service": r["service"],
                "snr_db": float(r["snr_db"]),
                "d_TC_m": float(r["d_TC_m"]),
                "d_TC_default_m": float(r["d_TC_default_m"]),
                "rel_capacity": float(r["rel_capacity"]),
            })
    if not rows:
        raise RuntimeError(f"no rows for {args.channel} sigma={args.sigma_band}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    default = float(rows[0]["d_TC_default_m"])
    services = [s for s in ORDER if any(r["service"] == s for r in rows)]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.50, 3.40), sharex=True,
        gridspec_kw={"hspace": 0.12},
        constrained_layout=True,
    )

    for svc in services:
        g = sorted(
            [r for r in rows if r["service"] == svc],
            key=lambda r: r["snr_db"],
        )
        st = STYLE[svc]
        xs = [r["snr_db"] for r in g]
        dtc = [r["d_TC_m"] for r in g]
        loss = [100.0 * (1.0 - r["rel_capacity"]) for r in g]
        ax1.plot(
            xs, dtc, color=st["color"], ls=st["ls"], marker=st["marker"],
            ms=3.4, markeredgewidth=0.35, markeredgecolor="white",
            lw=1.25, label=st["label"], zorder=3,
        )
        ax2.plot(
            xs, loss, color=st["color"], ls=st["ls"], marker=st["marker"],
            ms=3.4, markeredgewidth=0.35, markeredgecolor="white",
            lw=1.25, label=st["label"], zorder=3,
        )

    # BUBBLES baseline
    ax1.axhline(default, color="0.15", ls="--", lw=0.85, zorder=2)
    ax1.annotate(
        f"BUBBLES baseline ({default:.1f} m)",
        xy=(7.2, default),
        xytext=(6.0, default + 8.5),
        fontsize=6.2, color="0.2",
        arrowprops=dict(arrowstyle="-", lw=0.45, color="0.4", shrinkA=0, shrinkB=1),
    )
    # headroom for annotations / curves
    all_d = [r["d_TC_m"] for r in rows]
    ax1.set_ylim(min(all_d) - 4, max(all_d) + 12)
    ax1.set_ylabel(r"separation minimum $d_{\mathrm{TC}}$ (m)")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, alpha=0.28, zorder=0)
    ax1.set_axisbelow(True)
    ax1.tick_params(length=2.5, width=0.55)
    ax1.text(
        0.02, 0.96,
        r"Rayleigh, peak load, shared C2, $2\sigma$",
        transform=ax1.transAxes, va="top", ha="left",
        fontsize=6.3, color="0.3",
    )

    ax2.axhline(0.0, color="0.15", ls="--", lw=0.85, zorder=2)
    ax2.set_ylabel("airspace-capacity loss (%)")
    ax2.set_xlabel("SNR (dB)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, alpha=0.28, zorder=0)
    ax2.set_axisbelow(True)
    ax2.tick_params(length=2.5, width=0.55)
    ax2.set_xticks([-5, 0, 5, 10, 15, 20])
    ax2.legend(
        frameon=False, ncol=1, loc="upper right",
        handlelength=1.9, labelspacing=0.22, borderaxespad=0.35,
    )

    pdf_path = out_dir / "F9_separation.pdf"
    fig.savefig(pdf_path)
    fig.savefig(out_dir / "F9_separation.svg")
    fig.savefig(out_dir / "F9_separation.png", dpi=400)
    plt.close(fig)

    try:
        subprocess.run(
            [
                "pdftoppm", "-png", "-r", "400", "-singlefile",
                str(pdf_path), str(out_dir / "F9_separation"),
            ],
            check=False, capture_output=True,
        )
    except FileNotFoundError:
        pass

    print(f"data: {path}")
    print(f"baseline d_TC_default = {default:.2f} m")
    for svc in services:
        rr = next(
            (r for r in rows if r["service"] == svc and abs(r["snr_db"] + 5) < 1e-9),
            None,
        )
        if rr:
            loss = 100.0 * (1.0 - rr["rel_capacity"])
            print(
                f"  {svc:12s} @-5dB  d_TC={rr['d_TC_m']:.2f} m  "
                f"Δ={rr['d_TC_m'] - default:+.2f} m  loss={loss:.2f}%"
            )
    print(f"wrote F9_separation.[pdf,svg,png] -> {out_dir}")


if __name__ == "__main__":
    main()
