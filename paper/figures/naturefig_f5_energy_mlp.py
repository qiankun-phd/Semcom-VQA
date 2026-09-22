#!/usr/bin/env python3
"""naturefig F5+F11 with the per-sample MLP router series (2026-08-05).

Derived from naturefig_f5_energy.py (content baseline unchanged); adds the
method-of-record series "Per-sample routing (proposed)" from
outputs/energy/w8_mlp_series.csv (three-seed mean; horizontal bars in (a) =
per-seed min--max energy). Everything else identical to the parent script.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]

BANDWIDTH_HZ = 1.0e6
P_TX_HEAD = 0.5
E_DET_LO_J = 7.0 * 0.015
E_DET_HI_J = 15.0 * 0.050
E_DET_MID_J = 0.5 * (E_DET_LO_J + E_DET_HI_J)
F_IMG_M4 = 410.0 / 936.0
M6_USES = 9.2215e4
M6_ACC = {-5.0: 0.564, 0.0: 0.578, 5.0: 0.592, 10.0: 0.599, 15.0: 0.600,
          20.0: 0.596}

STYLE = {
    "M0_naive":    ("#d1495b", "x", "Fixed-rate image", ":"),
    "M1_image":    ("#e8962f", "o", "Rate-adaptive image", "-"),
    "M2_analog":   ("#8e6bb5", "v", "Uncoded analog", "--"),
    "M6_djscc":    ("#8b5e3c", "P", "DJSCC (learned)", "-."),
    "M3_token":    ("#5a6b7c", "s", "Fixed token", "-"),
    "M4_adaptive": ("#5ad19a", "D", "Type routing (proposed)", "-"),
    "M8_mlp":      ("#1f7a53", "^", "Per-sample routing (proposed)", "-"),
}
ORDER = ["M8_mlp", "M4_adaptive", "M3_token", "M6_djscc", "M2_analog",
         "M1_image", "M0_naive"]
F_IMG = {"M0_naive": 1.0, "M1_image": 1.0, "M2_analog": 1.0, "M6_djscc": 1.0,
         "M3_token": 0.0, "M4_adaptive": F_IMG_M4}


def load_rows(csv_path: Path, channel: str):
    out: dict[str, dict[float, dict]] = {}
    for r in csv.DictReader(open(csv_path)):
        if r["channel"] != channel or r["qtype"] != "all" or r["split"] != "test":
            continue
        out.setdefault(r["method"], {})[float(r["snr_db"])] = {
            "acc": float(r["accuracy"]),
            "uses": float(r["mean_channel_uses"] or 0.0),
            "lcb": float(r["lcb"]), "ucb": float(r["ucb"]),
        }
    out["M6_djscc"] = {s: {"acc": a, "uses": M6_USES, "lcb": None, "ucb": None}
                       for s, a in M6_ACC.items()}
    return out


def load_mlp(csv_path: Path, channel: str):
    out: dict[float, dict] = {}
    for r in csv.DictReader(open(csv_path)):
        if r["channel"] != channel:
            continue
        out[float(r["snr_db"])] = {
            "acc": float(r["acc_mean"]),
            "j": float(r["j_mean"]),
            "j_min": float(r["j_min"]), "j_max": float(r["j_max"]),
            "apj": float(r["apj_mean"]),
        }
    return out


def energy_j(method, uses, p_tx, e_vlm, e_det=E_DET_MID_J):
    e_tx = uses / BANDWIDTH_HZ * p_tx
    f = F_IMG[method]
    return e_tx + f * e_vlm + (1.0 - f) * e_det


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/reports/comparison_v3_5qt.csv")
    ap.add_argument("--power-json", default="outputs/energy/gpu_power_phases.json")
    ap.add_argument("--mlp-csv", default="outputs/energy/w8_mlp_series.csv")
    ap.add_argument("--out-dir", default="outputs/figures/naturefig")
    ap.add_argument("--channel", default="rician")
    args = ap.parse_args()
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(REPO / args.csv, args.channel)
    mlp = load_mlp(REPO / args.mlp_csv, args.channel)
    power = json.loads((REPO / args.power_json).read_text())
    e_vlm_inc = power["phases"]["vlm"]["joule_per_item_incremental"]
    snrs = sorted(rows["M4_adaptive"].keys())

    E = {m: {s: energy_j(m, rows[m][s]["uses"], P_TX_HEAD, e_vlm_inc)
             for s in snrs if s in rows[m]} for m in ORDER if m != "M8_mlp"}

    # ---- (a) F5: accuracy vs J/answer, log x -------------------------------
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.axvline(e_vlm_inc, color="0.6", lw=0.6, ls=":", zorder=0)
    ax.text(e_vlm_inc * 0.88, 0.435, "VLM compute lower bound",
            rotation=90, fontsize=5.5, color="0.4", ha="right", va="bottom")
    for m in ORDER:
        c, mk, lb, ls = STYLE[m]
        if m == "M8_mlp":
            xs = [mlp[s]["j"] for s in snrs]
            ys = [mlp[s]["acc"] for s in snrs]
            ax.plot(xs, ys, marker=mk, color=c, label=lb, ls=ls,
                    lw=1.4, ms=3.8, zorder=8)
            for s in snrs:  # per-seed energy range, horizontal
                ax.hlines(mlp[s]["acc"], mlp[s]["j_min"], mlp[s]["j_max"],
                          color=c, alpha=0.45, lw=1.6, zorder=5)
            continue
        xs = [E[m][s] for s in snrs if s in E[m]]
        ys = [rows[m][s]["acc"] for s in snrs if s in E[m]]
        hero = m == "M4_adaptive"
        ax.plot(xs, ys, marker=mk, color=c, label=lb, ls=ls,
                lw=1.1 if hero else 0.9,
                ms=3.2 if hero else 3.0, zorder=6 if hero else 3)
        if hero:
            for s in snrs:
                ax.vlines(E[m][s], rows[m][s]["lcb"], rows[m][s]["ucb"],
                          color=c, alpha=0.55, lw=0.9, zorder=4)
        if m == "M3_token":
            for s in snrs:
                lo = energy_j(m, rows[m][s]["uses"], P_TX_HEAD, e_vlm_inc,
                              E_DET_LO_J)
                hi = energy_j(m, rows[m][s]["uses"], P_TX_HEAD, e_vlm_inc,
                              E_DET_HI_J)
                ax.hlines(rows[m][s]["acc"], lo, hi, color=c, alpha=0.4,
                          lw=2.2, zorder=2)
    frontier_csv = REPO / "outputs/energy" / f"c1_frontier_{args.channel}_mlp.csv"
    if not frontier_csv.exists():
        frontier_csv = REPO / "outputs/energy" / f"c1_frontier_{args.channel}.csv"
    if frontier_csv.exists():
        fr = sorted((float(r["E_j"]), float(r["acc"]))
                    for r in csv.DictReader(open(frontier_csv)))
        ax.plot([p[0] for p in fr], [p[1] for p in fr], color="#2f2f2f",
                ls="--", lw=0.9, marker=".", ms=2.2, zorder=7,
                label="Budget-tunable routing ($\\lambda$ sweep)")
        ax.annotate("energy price $\\lambda$ sweep", (0.55, 0.702),
                    fontsize=5.5, color="#2f2f2f", ha="left")
    ax.annotate("fixed-rate @$-5$ dB:\ndigital cliff (FER${\\approx}$1)",
                (E["M0_naive"][snrs[0]], rows["M0_naive"][snrs[0]]["acc"]),
                textcoords="offset points", xytext=(-8, 2), fontsize=5.0,
                color=STYLE["M0_naive"][0], ha="right")
    s_lo, s_hi = snrs[0], snrs[-1]
    for m, s, fx, dy, ha, va in (
            ("M1_image",    s_hi, 0.88, +0.030, "right", "bottom"),
            ("M1_image",    s_lo, 0.68, -0.016, "right", "center")):
        c = STYLE[m][0]
        x, y = E[m][s], rows[m][s]["acc"]
        ax.annotate(f"{s:+.0f} dB", xy=(x, y), xytext=(x * fx, y + dy),
                    fontsize=5.5, color=c, ha=ha, va=va,
                    arrowprops=dict(arrowstyle="-", lw=0.45, color=c,
                                    alpha=0.65, shrinkA=0.4, shrinkB=1.6))
    for s, fx, dy, ha, va in ((s_hi, 0.74, +0.020, "right", "bottom"),
                              (s_lo, 0.72, -0.024, "right", "top")):
        c = STYLE["M8_mlp"][0]
        x, y = mlp[s]["j"], mlp[s]["acc"]
        ax.annotate(f"{s:+.0f} dB", xy=(x, y), xytext=(x * fx, y + dy),
                    fontsize=5.5, color=c, ha=ha, va=va,
                    arrowprops=dict(arrowstyle="-", lw=0.45, color=c,
                                    alpha=0.65, shrinkA=0.4, shrinkB=1.6))
    ax.set_xscale("log")
    ax.set_xlabel("joint energy per answer (J)", fontsize=8)
    ax.set_ylabel("VQA accuracy (test)", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    ax.text(0.03, 0.97, "Rician K=6 dB", transform=ax.transAxes, fontsize=6.5,
            va="top")
    ax.legend(fontsize=5.9, loc="lower left", handlelength=1.4,
              labelspacing=0.22, borderpad=0.3)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(out_dir / f"F5_pareto_energy.{ext}")
    fig.savefig(out_dir / "F5_pareto_energy.png", dpi=600)
    plt.close(fig)

    # ---- (b) F11: answers per joule vs SNR ---------------------------------
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for m in ORDER:
        c, mk, lb, ls = STYLE[m]
        if m == "M8_mlp":
            ys = [mlp[s]["apj"] for s in snrs]
            ax.plot(snrs, ys, marker=mk, color=c, label=lb, ls=ls,
                    lw=1.4, ms=3.6, zorder=6)
            continue
        xs = [s for s in snrs if s in E[m]]
        ys = [rows[m][s]["acc"] / E[m][s] for s in xs]
        ax.plot(xs, ys, marker=mk, color=c, label=lb, ls=ls,
                lw=1.1 if m == "M4_adaptive" else 0.9, ms=3.2,
                zorder=5 if m == "M4_adaptive" else 3)
    apj = lambda m, s: rows[m][s]["acc"] / E[m][s]
    ax.annotate(f"{apj('M3_token', 20.0):.2f}", (-4.7, apj("M3_token", -5.0)),
                textcoords="offset points", xytext=(0, -9), fontsize=5.5,
                color=STYLE["M3_token"][0])
    ax.annotate("0.061--0.075".replace("--", "–"), (-4.7, mlp[-5.0]["apj"]),
                textcoords="offset points", xytext=(0, 5), fontsize=5.5,
                color=STYLE["M8_mlp"][0])
    ax.annotate("0.043--0.047".replace("--", "–"),
                (-4.7, apj("M4_adaptive", -5.0)),
                textcoords="offset points", xytext=(0, -10), fontsize=5.5,
                color="#2e9e6e")
    ax.annotate("$\\approx$0.018 (all full-image pipelines)",
                (2.0, apj("M1_image", 10.0)),
                textcoords="offset points", xytext=(0, -11), fontsize=5.5,
                color=STYLE["M1_image"][0])
    ax.set_yscale("log")
    ax.set_ylim(top=4.0)
    ax.set_xlabel("SNR (dB)", fontsize=8)
    ax.set_ylabel("correct answers per joule", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    ax.text(0.02, 0.97, f"Rician K=6 dB, $P_{{\\mathrm{{tx}}}}$={P_TX_HEAD} W",
            transform=ax.transAxes, fontsize=6.5, va="top")
    ax.legend(fontsize=5.9, loc="center right", bbox_to_anchor=(1.0, 0.64),
              handlelength=1.4, labelspacing=0.22, borderpad=0.3)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(out_dir / f"F11_answers_per_joule.{ext}")
    fig.savefig(out_dir / "F11_answers_per_joule.png", dpi=600)
    plt.close(fig)
    print(f"wrote F5_pareto_energy + F11_answers_per_joule -> {out_dir}")


if __name__ == "__main__":
    main()
