#!/usr/bin/env python3
"""naturefig F13: per-answer energy decomposition, stacked bars (TGCN).

Table-to-figure conversion batch: NEW figure (no table predecessor). Each
evidence path of the manuscript's energy frontier becomes one stacked bar,
split into its three measured/parameterized components:

    radio (transmit)  |  detector compute  |  VLM compute

Bars (worst link, Rician K=6 dB, -5 dB, P_tx = 0.5 W, incremental billing):
  fixed token, budget-tunable routing operating point (MLP lambda sweep,
  the saturation point about 0.69 at 3.2 J), evidence routing (type rule),
  DJSCC (learned), rate-adaptive image.

Data of record only -- no recomputation:
  outputs/energy/energy_summary.json   (per-method e_tx / e_cmp / totals,
                                        params: E_VLM = 32.310 J,
                                        E_det mid-band = 0.4275 J, f_img)
  outputs/energy/gpu_power_phases.json (measured phase powers, cross-check)
  outputs/energy/c1_frontier_rician_mlp.csv (MLP lambda-sweep operating point; the split
      uses frac_image with the same E_VLM / E_det constants).

Style: nature-figure conventions -- Times-family serif (Nimbus Roman on the
render host), grayscale-safe hatches, direct value labels, Type-42 fonts.
Outputs PDF + SVG + PNG (300 dpi).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "hatch.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/figures/naturefig"

# component palette: same energy-split colors as the Fig. 1 schematic
C_VLM = "#4a5763"   # inference (VLM) compute -- dark
C_DET = "#7f9ab8"   # detector compute -- mid blue-grey
C_TX = "#e8c268"    # radio -- light, hatched (grayscale-safe)
SNR_KEY = "-5.0"    # headline worst link (matches the text's 15.3 / 34.4 J)
LAMBDA_E_ANCHOR = 3.2  # MLP lambda-sweep saturation point of the text ("about 0.69 at a third of the router's energy"); same sweep as the dashed curve of Fig. 3


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    es = json.loads((REPO / "outputs/energy/energy_summary.json").read_text())
    pm, prm = es["per_method"], es["params"]
    e_vlm, e_det = prm["e_vlm_incremental_j"], prm["e_det_mid_j"]

    # cross-check the constants against the measured power file of record
    power = json.loads((REPO / "outputs/energy/gpu_power_phases.json")
                       .read_text())
    assert abs(power["phases"]["vlm"]["joule_per_item_incremental"]
               - e_vlm) < 1e-9

    def split(m):
        """(radio, det, vlm) J/answer for a fixed method at the headline SNR."""
        v = pm[m][SNR_KEY]
        f = {"M3_token": 0.0, "M4_adaptive": prm["f_img_m4"],
             "M1_image": 1.0, "M6_djscc": 1.0}[m]
        det, vlm = (1.0 - f) * e_det, f * e_vlm
        assert abs(v["e_cmp_j"] - det - vlm) < 1e-6, m
        return v["e_tx_j"], det, vlm, v["j_per_answer"], v["acc"]

    # lambda operating point: row of the frontier CSV of record closest to
    # the MLP-sweep saturation anchor (about 0.69 at 3.2 J)
    fr = list(csv.DictReader(open(REPO / "outputs/energy/c1_frontier_rician_mlp.csv")))
    row = min(fr, key=lambda r: abs(float(r["E_j"]) - LAMBDA_E_ANCHOR))
    fL, eL, aL = (float(row["frac_image"]), float(row["E_j"]),
                  float(row["acc"]))
    vlmL, detL = fL * e_vlm, (1.0 - fL) * e_det
    txL = eL - vlmL - detL

    bars = [  # (display label, radio, det, vlm, total, acc)
        ("Fixed\ntoken", *split("M3_token")),
        ("Budget-tunable\nrouting ($\\lambda$ point)", txL, detL, vlmL, eL, aL),
        ("Type\nrouting", *split("M4_adaptive")),
        ("DJSCC\n(learned)", *split("M6_djscc")),
        ("Rate-adaptive\nimage", *split("M1_image")),
    ]
    for b in bars:
        print(f"{b[0].replace(chr(10), ' '):24s} radio {b[1]:7.3f}  "
              f"det {b[2]:6.3f}  vlm {b[3]:7.3f}  total {b[4]:7.3f}  "
              f"acc {b[5]:.3f}")

    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    xs = range(len(bars))
    for i, (_, tx, det, vlm, tot, acc) in enumerate(bars):
        ax.bar(i, vlm, 0.62, color=C_VLM, zorder=3)
        ax.bar(i, det, 0.62, bottom=vlm, color=C_DET, zorder=3)
        ax.bar(i, tx, 0.62, bottom=vlm + det, color=C_TX, hatch="///",
               edgecolor="#8a6d2f", lw=0.4, zorder=3)
        ax.annotate(f"{tot:.3g} J", (i, tot), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7, fontweight="bold")
        ax.annotate(f"acc {acc:.3f}", (i, tot), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=6.2, color="0.35")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars], fontsize=6.8)
    ax.set_ylabel("energy per answered question (J)")
    ax.set_ylim(0, 41.5)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.text(0.02, 0.97, "Rician K=6 dB, $-5$ dB, "
            "$P_{\\mathrm{tx}}{=}0.5$ W", transform=ax.transAxes,
            fontsize=6.5, va="top")
    ax.legend(handles=[
        Patch(fc=C_VLM, label="VLM compute (image branch)"),
        Patch(fc=C_DET, label="detector compute"),
        Patch(fc=C_TX, hatch="///", ec="#8a6d2f", label="radio (transmit)")],
        loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False,
        handlelength=1.2, labelspacing=0.3)

    # inset: the token bar split is invisible at frontier scale -- zoom it
    axi = ax.inset_axes([0.28, 0.21, 0.155, 0.36])
    tx, det, vlm = bars[0][1], bars[0][2], bars[0][3]
    axi.bar(0, det, 0.5, color=C_DET, zorder=3)
    axi.bar(0, tx, 0.5, bottom=det, color=C_TX, hatch="///",
            edgecolor="#8a6d2f", lw=0.4, zorder=3)
    axi.set_ylim(0, 0.48)
    axi.set_xlim(-0.42, 0.95)
    axi.set_xticks([])
    axi.set_yticks([0, 0.2, 0.4])
    axi.tick_params(labelsize=5.5, length=1.5, pad=1)
    axi.set_title("token bar,\nzoomed (J)", fontsize=5.5, pad=2)
    axi.annotate(f"radio\n{tx:.4f}", (0.30, det + tx - 0.005),
                 fontsize=5.2, ha="left", va="top")
    axi.annotate(f"detector\n{det:.3f}", (0.30, det * 0.5),
                 fontsize=5.2, ha="left", va="center")
    fig.tight_layout()
    for ext, dpi in (("pdf", None), ("svg", None), ("png", 300)):
        fig.savefig(OUT / f"F13_energy_stack.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"wrote {OUT}/F13_energy_stack.[pdf|svg|png]")


if __name__ == "__main__":
    main()
