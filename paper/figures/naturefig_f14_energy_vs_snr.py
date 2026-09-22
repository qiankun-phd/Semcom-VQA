#!/usr/bin/env python3
"""naturefig F14: joint energy per answered question vs SNR (TGCN).

Restyled to match F1/F5/F8/F17 naturefig conventions:
  Times-family serif, Type-42 fonts, figsize 3.35x2.65, lw/ms as F1,
  clean annotations from the original asset.

P_tx 0.1--1 W sensitivity (tab:jpa) is shown as *point-wise* error bars
on fixed token / evidence routing / rate-adaptive image only (true
physical range via E = uses/B * P_tx + e_cmp; not a continuous ribbon).

Data of record: outputs/energy/energy_summary.json
Outputs: PDF + SVG + PNG (300 dpi) under outputs/figures/naturefig/
and paper_semcom/figures/ when present.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "lines.linewidth": 1.1,
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

STYLE = {
    "M0_naive": ("#ff6b6b", "x", "Fixed-rate image", ":"),
    "M1_image": ("#ffb454", "o", "Rate-adaptive image", "-"),
    "M2_analog": ("#c678dd", "v", "Uncoded analog", "--"),
    "M6_djscc": ("#8b5e3c", "P", "DJSCC (learned)", "-."),
    "M3_token": ("#9aa7b4", "s", "Fixed token", "-"),
    "M4_adaptive": ("#5ad19a", "D", "Evidence routing (proposed)", "-"),
}
ORDER = [
    "M0_naive", "M1_image", "M2_analog", "M6_djscc",
    "M4_adaptive", "M3_token",
]
BAND = ("M3_token", "M4_adaptive", "M1_image")
P_LO, P_HI = 0.1, 1.0


def find_repo() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1],  # paper_semcom/
        Path.home() / "phd_research/vqa_semcom",
        here.parents[2] / "vqa_semcom" if len(here.parents) > 2 else here,
    ]
    for repo in candidates:
        if (repo / "outputs/energy/energy_summary.json").is_file():
            return repo
    return Path.home() / "phd_research/vqa_semcom"


def cell(pm_m: dict, s: float) -> dict:
    for k in (str(s), str(float(s)), str(int(s)) if float(s).is_integer() else None):
        if k is not None and k in pm_m:
            return pm_m[k]
    raise KeyError(s)


def main() -> None:
    repo = find_repo()
    es = json.loads((repo / "outputs/energy/energy_summary.json").read_text())
    pm, prm = es["per_method"], es["params"]
    bw = float(prm["bandwidth_hz"])
    assert [P_LO, P_HI] == [prm["p_tx_grid_w"][0], prm["p_tx_grid_w"][-1]]

    snrs = sorted(float(s) for s in pm["M4_adaptive"])

    # Match F1/F17 single-column naturefig size
    fig, ax = plt.subplots(figsize=(3.35, 2.65))

    for m in ORDER:
        c, mk, lb, ls = STYLE[m]
        hero = m == "M4_adaptive"
        xs = [s for s in snrs if str(s) in pm[m] or str(float(s)) in pm[m]]
        ys = [cell(pm[m], s)["j_per_answer"] for s in xs]

        if m in BAND:
            lo, hi = [], []
            for s in xs:
                v = cell(pm[m], s)
                lo.append(v["uses"] / bw * P_LO + v["e_cmp_j"])
                hi.append(v["uses"] / bw * P_HI + v["e_cmp_j"])
            yerr = [
                [max(y - a, 1e-12) for y, a in zip(ys, lo)],
                [max(b - y, 1e-12) for y, b in zip(ys, hi)],
            ]
            # Point-wise only; moderate weight so bars read without muddying.
            ax.errorbar(
                xs, ys, yerr=yerr, fmt="none",
                ecolor=c, elinewidth=1.15, capsize=2.6, capthick=1.0,
                alpha=0.85, zorder=4,
            )
            print(m, f"@{xs[0]:g}dB {lo[0]:.3f}-{hi[0]:.3f} J")

        ax.plot(
            xs, ys,
            marker=mk, color=c, label=lb, ls=ls,
            lw=1.4 if hero else 0.9,
            ms=3.6 if hero else 3.0,
            markeredgewidth=0.5,
            zorder=6 if hero else 3,
        )
        print(m, [f"{y:.3f}" for y in ys])

    def j_at(m: str, s: float = 10.0) -> float:
        return cell(pm[m], s)["j_per_answer"]

    # Original manuscript annotations (wording unchanged)
    ax.annotate(
        "",
        xy=(10, j_at("M4_adaptive")),
        xytext=(10, j_at("M1_image")),
        arrowprops=dict(
            arrowstyle="<->", lw=0.7, color="0.3", shrinkA=1, shrinkB=1,
        ),
    )
    ax.annotate(
        "$2.2\\times$ at\nevery SNR",
        xy=(9.4, 20.0), fontsize=6.2, color="0.25", ha="right",
    )
    ax.annotate(
        "shaded: $P_{\\mathrm{tx}}$ swept 0.1–1 W",
        xy=(0.03, 0.585), xycoords="axes fraction",
        fontsize=6.0, color="0.35",
    )

    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("energy per answered question (J)")
    ax.set_xticks(snrs)
    ax.set_ylim(0.3, 60)
    ax.grid(True, which="both", alpha=0.25)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.text(
        0.03, 0.97, "Rician K=6 dB, incremental compute",
        transform=ax.transAxes, fontsize=6.5, va="top",
    )
    ax.legend(
        loc="center right", bbox_to_anchor=(1.0, 0.40), frameon=False,
        handlelength=1.6, labelspacing=0.25, borderpad=0.2,
    )
    fig.tight_layout()

    out_dirs = [repo_out for repo_out in (
        repo / "outputs/figures/naturefig",
        Path(__file__).resolve().parent,
    )]
    for out in out_dirs:
        out.mkdir(parents=True, exist_ok=True)
        for ext, dpi in (("pdf", None), ("svg", None), ("png", 300)):
            fig.savefig(out / f"F14_energy_vs_snr.{ext}", dpi=dpi)
        # Extra sharp PNG for slides
        fig.savefig(out / "F14_energy_vs_snr_300.png", dpi=300)
        print(f"wrote {out}/F14_energy_vs_snr.[pdf|svg|png]")

    plt.close(fig)


if __name__ == "__main__":
    main()
