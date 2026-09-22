#!/usr/bin/env python3
"""naturefig F14 (paper Fig. 2), redesign 2026-08-24: make SNR do work.

(a) Radio (transmit) energy per answered question vs SNR, log axis, with
    the two measured compute levels as horizontal references (VLM forward
    pass 32.3 J incremental; detector 0.43 J Jetson mid-band). Shows the
    ~12x SNR swing of the rate-adaptive terms AND that they sit an order of
    magnitude below the compute line. Token line carries its declared
    airtime range (compact frame .. full slot) as a band at low SNR.
(b) Regime map: radio share of the per-answer energy budget for the
    rate-adaptive image path vs SNR under (B, eta) in
    {1 MHz, 100 kHz} x {1, 0.25}. Pure accounting from the same data:
    E_tx scales with 1/B and 1/eta (Eq. energy of the paper).
    The 50% line marks radio/compute parity.

Data of record only: energy_summary.json (per-method e_tx / e_cmp per SNR,
Rician K=6 dB, P_tx = 0.5 W, B = 1 MHz). No new measurements.
Output: F14_energy_vs_snr.[pdf|svg|png] (same filename as before so the
\\evfig reference in 06_experiments.tex is unchanged).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.3,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.25,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "energy_summary.json"

# unified palette (see MANIFEST 2026-08-23)
SERIES = {
    "M0_naive":    ("#d1495b", "x", "Fixed-rate image", ":"),
    "M1_image":    ("#e8962f", "o", "Rate-adaptive image", "-"),
    "M2_analog":   ("#8e6bb5", "v", "Uncoded analog", "--"),
    "M6_djscc":    ("#8b5e3c", "P", "DJSCC (learned)", "-."),
    "M4_adaptive": ("#5ad19a", "D", "Type routing (proposed)", "-"),
    "M3_token":    ("#5a6b7c", "s", "Fixed token", "-"),
}
C_INK = "#2a3340"


def main() -> None:
    d = json.loads(DATA.read_text())
    p = d["params"]
    B_hz = p["bandwidth_hz"]            # 1e6
    ptx = p["p_tx_headline_w"]          # 0.5
    e_vlm = p["e_vlm_incremental_j"]    # 32.31
    e_det = p["e_det_mid_j"]            # 0.4275
    per = d["per_method"]
    snrs = sorted(float(s) for s in per["M1_image"].keys())

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.1, 2.15))

    # ---------------- (a) radio term vs SNR ----------------
    for key, (c, mk, lab, ls) in SERIES.items():
        ys = [per[key][f"{s:.1f}"]["e_tx_j"] for s in snrs]
        ax_a.plot(snrs, ys, color=c, marker=mk, ls=ls, ms=3.6, label=lab, zorder=3)
    # token slot-spread range band (compact frame .. full slot), declared in §III
    slot_j = 0.3 * ptx                       # tau * P_tx = 0.15 J at B tau symbols
    tok = [per["M3_token"][f"{s:.1f}"]["e_tx_j"] for s in snrs]
    ax_a.fill_between(snrs, tok, [slot_j] * len(snrs), color="#5a6b7c", alpha=0.10,
                      lw=0, zorder=1)
    ax_a.text(snrs[0] + 0.3, 0.02, "token airtime range (compact frame → full slot)",
              fontsize=5.6, color="#5a6b7c", va="center")
    # measured compute references
    ax_a.axhline(e_vlm, color=C_INK, ls="--", lw=0.8, zorder=2)
    ax_a.text(snrs[-1], e_vlm * 1.18, f"VLM inference, measured ({e_vlm:.1f} J)",
              fontsize=6, color=C_INK, ha="right", va="bottom")
    ax_a.axhline(e_det, color=C_INK, ls=":", lw=0.8, zorder=2)
    ax_a.text(6.0, e_det * 1.18, f"on-board detector ({e_det:.2f} J)",
              fontsize=6, color=C_INK, ha="left", va="bottom")
    # swing annotation for the rate-adaptive image
    e0 = per["M1_image"][f"{snrs[0]:.1f}"]["e_tx_j"]
    e1 = per["M1_image"][f"{snrs[-1]:.1f}"]["e_tx_j"]
    ax_a.annotate("", xy=(snrs[-1], e1), xytext=(snrs[-1], e0),
                  arrowprops=dict(arrowstyle="<->", lw=0.7, color="#e8962f"))
    ax_a.text(snrs[-1] - 0.7, 1.55, f"{e0 / e1:.1f}× over the grid",
              fontsize=6, color="#e8962f", ha="right", va="center")
    ax_a.set_yscale("log")
    ax_a.set_ylim(3e-3, 90)
    ax_a.set_xlabel("SNR (dB)")
    ax_a.set_ylabel("radio energy per answered question (J)")
    ax_a.set_xticks(snrs)
    ax_a.grid(True, which="major", lw=0.4, alpha=0.35)
    ax_a.text(0.02, 0.975, f"Rician K=6 dB, $P_\\mathrm{{tx}}$={ptx:g} W, $B$=1 MHz",
              transform=ax_a.transAxes, fontsize=6.3, va="top")
    ax_a.text(0.0, 1.03, "(a)", transform=ax_a.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")

    # ---------------- (b) regime map: radio share of the image path ----------------
    regimes = [
        (1e6, 1.00, "#e8962f", "-",  "B = 1 MHz, η = 1 (headline)"),
        (1e6, 0.25, "#e8962f", "--", "B = 1 MHz, η = 0.25"),
        (1e5, 1.00, "#d1495b", "-",  "B = 100 kHz, η = 1"),
        (1e5, 0.25, "#d1495b", "--", "B = 100 kHz, η = 0.25"),
    ]
    for B, eta, c, ls, lab in regimes:
        k = (B_hz / B) / eta
        share = []
        for s in snrs:
            etx = per["M1_image"][f"{s:.1f}"]["e_tx_j"] * k
            share.append(100.0 * etx / (etx + e_vlm))
        ax_b.plot(snrs, share, color=c, ls=ls, marker="o", ms=3.2, label=lab, zorder=3)
    ax_b.axhline(50, color=C_INK, ls=":", lw=0.8)
    ax_b.text(snrs[0], 52, "radio = compute (parity)", fontsize=6, color=C_INK,
              ha="left", va="bottom")
    ax_b.set_ylim(0, 100)
    ax_b.set_xlabel("SNR (dB)")
    ax_b.set_ylabel("radio share of per-answer energy (%)")
    ax_b.set_xticks(snrs)
    ax_b.grid(True, lw=0.4, alpha=0.35)
    ax_b.legend(loc="upper right", frameon=False, handlelength=2.2)
    ax_b.text(0.02, 0.975, "rate-adaptive image path;\naccounting extrapolation in $B$ and η",
              transform=ax_b.transAxes, fontsize=6.3, va="top")
    ax_b.text(0.0, 1.03, "(b)", transform=ax_b.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")

    for ax in (ax_a, ax_b):
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.6)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               handlelength=2.0, columnspacing=1.2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.90))
    for ext in ("pdf", "svg"):
        fig.savefig(HERE / f"F14_energy_vs_snr.{ext}")
    fig.savefig(HERE / "F14_energy_vs_snr.png", dpi=400)
    # print the numbers the caption/text quote
    print(f"image radio term: {e0:.2f} J @ {snrs[0]:g} dB -> {e1:.3f} J @ {snrs[-1]:g} dB ({e0/e1:.1f}x)")
    for B, eta, *_ in regimes:
        k = (B_hz / B) / eta
        etx = per["M1_image"][f"{snrs[0]:.1f}"]["e_tx_j"] * k
        print(f"  B={B/1e3:g} kHz eta={eta}: worst-link radio {etx:.1f} J, share {100*etx/(etx+e_vlm):.1f}%")
    print("wrote F14_energy_vs_snr.[pdf|svg|png] ->", HERE)


if __name__ == "__main__":
    main()
