#!/usr/bin/env python3
"""Fig. 8 (paper 2): corrected M/G/1 safety bridge from separation_v2 data.

Input : figures/data/sepcap_{peak,nominal}_shared.csv  (E7v2, server 160)
Output: figures/fig8_safetybridge.pdf

Rayleigh channel, SAIL I-II, 2-sigma band (WP5 convention).  Two panels:
(a) tactical-contact separation d_TC vs SNR, (b) relative airspace capacity
d_TC_default/d_TC vs SNR, for s2 image and s1 token under peak and nominal
shared-C2 load, with the BUBBLES default-comm baseline marked.  The corrected
coupling adds M/G/1 queue wait W on top of the official 1.8 s mean, so no
curve can beat the baseline -- the honest, asymmetric story.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "pdf.fonttype": 42,
})


def load(load_name: str):
    rows = list(csv.DictReader((DATA / f"sepcap_{load_name}_shared.csv").open()))
    out = {}
    for r in rows:
        if r["channel"] != "rayleigh" or int(r["sigma_band"]) != 2:
            continue
        svc = r["service"]
        if svc not in ("M1_image", "M3_token"):
            continue
        out.setdefault(svc, []).append((float(r["snr_db"]), float(r["d_TC_m"]),
                                        float(r["rel_capacity"]), float(r["d_TC_default_m"])))
    for svc in out:
        out[svc].sort()
    return out


peak = load("peak")
nominal = load("nominal")
default_dtc = peak["M1_image"][0][3]  # 370.49 (2 sigma)

STYLES = {
    ("M1_image", "peak"): dict(color="#c0392b", ls="-", marker="o", ms=2.8,
                               label="$s_2$ image, peak load"),
    ("M1_image", "nominal"): dict(color="#c0392b", ls="--", marker="o", ms=2.8,
                                  mfc="white", label="$s_2$ image, nominal load"),
    ("M3_token", "peak"): dict(color="#1e8449", ls="-", marker="s", ms=2.8,
                               label="$s_1$ token, peak load"),
    ("M3_token", "nominal"): dict(color="#1e8449", ls="--", marker="s", ms=2.8,
                                  mfc="white", label="$s_1$ token, nominal load"),
}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))

ax = axes[0]
for (svc, ld), st in STYLES.items():
    data = (peak if ld == "peak" else nominal)[svc]
    ax.plot([d[0] for d in data], [d[1] for d in data], lw=1.1, **st)
ax.axhline(default_dtc, color="k", ls=":", lw=0.9)
ax.annotate("BUBBLES default comm ($1.8$\\,s): $370.5$\\,m",
            xy=(0.03, default_dtc), xycoords=("axes fraction", "data"),
            xytext=(0, 4), textcoords="offset points", ha="left",
            fontsize=6.5)
ax.set_ylim(365, 400)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("contact distance $d_{TC}$ (m)")
ax.set_title("(a) separation minimum ($2\\sigma$)", loc="left", fontsize=8)
ax.legend(frameon=False, loc="upper right", ncol=1)

ax = axes[1]
for (svc, ld), st in STYLES.items():
    data = (peak if ld == "peak" else nominal)[svc]
    ax.plot([d[0] for d in data], [d[2] for d in data], lw=1.1, **st)
ax.axhline(1.0, color="k", ls=":", lw=0.9)
ax.annotate("baseline capacity $=1.0$ (not exceedable)",
            xy=(0.98, 1.0), xycoords=("axes fraction", "data"),
            xytext=(0, -7.5), textcoords="offset points", ha="right", fontsize=6.5)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("relative capacity $d_{TC}^{\\mathrm{def}}/d_{TC}$")
ax.set_title("(b) airspace capacity", loc="left", fontsize=8)

fig.tight_layout(pad=0.4)
fig.savefig(HERE / "fig8_safetybridge.pdf", bbox_inches="tight")
print("wrote", HERE / "fig8_safetybridge.pdf")
