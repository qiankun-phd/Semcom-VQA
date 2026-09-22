#!/usr/bin/env python3
"""naturefig F14 with the per-sample MLP router series (2026-08-05).

Derived from naturefig_f14_energy_vs_snr.py; adds "Per-sample router (proposed)"
from outputs/energy/w14_mlp_10seed_series.csv (Rician; three-seed mean, error bars =
per-seed min--max energy, unlike the P_tx-sweep bars of the fixed methods).
Ratio annotation updated to quote both rungs (3.1x router / 2.2x type).
"""
from __future__ import annotations

import csv
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
    "M0_naive": ("#d1495b", "x", "Fixed-rate image", ":"),
    "M1_image": ("#e8962f", "o", "Rate-adaptive image", "-"),
    "M2_analog": ("#8e6bb5", "v", "Uncoded analog", "--"),
    "M6_djscc": ("#8b5e3c", "P", "DJSCC (learned)", "-."),
    "M3_token": ("#5a6b7c", "s", "Fixed token", "-"),
    "M4_adaptive": ("#5ad19a", "D", "Type routing (proposed)", "-"),
    "M8_mlp": ("#1f7a53", "^", "Per-sample router (proposed)", "-"),
}
ORDER = [
    "M0_naive", "M1_image", "M2_analog", "M6_djscc",
    "M4_adaptive", "M8_mlp", "M3_token",
]
BAND = ("M3_token", "M4_adaptive", "M1_image")
P_LO, P_HI = 0.1, 1.0


def find_repo() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1],
        Path.home() / "phd_research/vqa_semcom",
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

    mlp = {}
    for r in csv.DictReader(open(repo / "outputs/energy/w14_mlp_10seed_series.csv")):
        if r["channel"] != "rician":
            continue
        mlp[float(r["snr_db"])] = (float(r["j_mean"]), float(r["j_min"]),
                                   float(r["j_max"]))

    snrs = sorted(float(s) for s in pm["M4_adaptive"])
    fig, ax = plt.subplots(figsize=(3.35, 2.65))

    for m in ORDER:
        c, mk, lb, ls = STYLE[m]
        if m == "M8_mlp":
            ys = [mlp[s][0] for s in snrs]
            yerr = [[y - mlp[s][1] for y, s in zip(ys, snrs)],
                    [mlp[s][2] - y for y, s in zip(ys, snrs)]]
            ax.errorbar(snrs, ys, yerr=yerr, fmt="none", ecolor=c,
                        elinewidth=1.0, capsize=2.2, capthick=0.9,
                        alpha=0.75, zorder=5)
            ax.plot(snrs, ys, marker=mk, color=c, label=lb, ls=ls,
                    lw=1.4, ms=3.8, markeredgewidth=0.5, zorder=7)
            continue
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
            ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor=c,
                        elinewidth=1.15, capsize=2.6, capthick=1.0,
                        alpha=0.85, zorder=4)
        ax.plot(xs, ys, marker=mk, color=c, label=lb, ls=ls,
                lw=1.1 if hero else 0.9, ms=3.2 if hero else 3.0,
                markeredgewidth=0.5, zorder=6 if hero else 3)

    def j_at(m: str, s: float = 10.0) -> float:
        return cell(pm[m], s)["j_per_answer"]

    ax.annotate(
        "",
        xy=(10, mlp[10.0][0]),
        xytext=(10, j_at("M1_image")),
        arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.3",
                        shrinkA=1, shrinkB=1),
    )
    ax.annotate(
        "$3.5\\times$ (router), $2.2\\times$ (type) at every SNR",
        xy=(1.8, 21.0), fontsize=6.0, color="0.25", ha="center",
        va="center",
    )
    ax.annotate(
        "shaded: $P_{\\mathrm{tx}}$ swept 0.1–1 W;\nrouter bars: seed min–max",
        xy=(0.03, 0.555), xycoords="axes fraction",
        fontsize=5.6, color="0.35",
    )

    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("energy per answered question (J)")
    ax.set_xticks(snrs)
    ax.set_ylim(0.3, 60)
    ax.grid(True, which="both", alpha=0.25)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.text(0.03, 0.97, "Rician K=6 dB, incremental compute",
            transform=ax.transAxes, fontsize=6.5, va="top")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), frameon=False,
              handlelength=1.6, labelspacing=0.25, borderpad=0.2,
              fontsize=6.0)
    fig.tight_layout()

    out = repo / "outputs/figures/naturefig"
    out.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("pdf", None), ("svg", None), ("png", 300)):
        fig.savefig(out / f"F14_energy_vs_snr.{ext}", dpi=dpi)
    print(f"wrote {out}/F14_energy_vs_snr.[pdf|svg|png]")
    plt.close(fig)


if __name__ == "__main__":
    main()
