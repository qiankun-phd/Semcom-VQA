#!/usr/bin/env python3
"""M4 figures for paper 2 from paper2_numbers_v8.json.

fig3_m4_axes.pdf : 2x2 panels (UAV count / arrival load / link attenuation /
                   interference floor), admitted mission success vs axis,
                   peak-trained arms evaluated zero-shot (no retraining).
fig4_zeroshot.pdf: three-point zero-shot bars (unseen soft-conflict profile,
                   x1.5 load, unseen low-SNR blockage profile).

Usage: python3 _make_m4_figures.py path/to/paper2_numbers_v8.json
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
JSON = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data" / "paper2_numbers_v8.json"
d = json.load(open(JSON))
M4 = d["m4_generalization"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "legend.fontsize": 6.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "pdf.fonttype": 42,
})

METRIC = "admitted_mission_success_rate"
ARMS = [
    ("proposed", "Proposed", "#c0392b", "o", "-"),
    ("fixed_penalty", "Fixed-penalty", "#7d3c98", "^", "-"),
    ("no_lagrangian", "No-Lagrangian", "#2471a3", "v", "-"),
    ("bl_semantic_greedy", "Semantic greedy", "#616a6b", "s", "--"),
    ("bl_oracle_escalation_aware", "Esc-aware oracle", "#1e8449", "d", ":"),
]

AXES = [
    ("uav", "UAV count", lambda t: int(t[1:]), "n"),
    ("arrival", "queries per episode", lambda t: int(t[1:]), "t"),
    ("snr", "excess link loss (dB)", lambda t: float(t[1:].replace("p", ".")), "x"),
    ("i0", "interference floor (dBm)", lambda t: -float(t[1:]), "f"),
]


def series(axis, arm):
    pts = M4.get(axis, {})
    xs, ys, es = [], [], []
    _, _, decode, prefix = next(a for a in AXES if a[0] == axis)
    for tag in sorted(pts, key=decode):
        blk = pts[tag].get(arm)
        if not blk:
            continue
        c = blk["metrics"].get(METRIC)
        if not c:
            continue
        xs.append(decode(tag))
        ys.append(c["mean"])
        es.append(c["std"])
    return np.array(xs), np.array(ys), np.array(es)


fig, axes2 = plt.subplots(2, 2, figsize=(7.1, 4.0))
for panel, (axis, xlabel, decode, prefix) in zip(axes2.flat, AXES):
    for arm, label, color, marker, ls in ARMS:
        xs, ys, es = series(axis, arm)
        if len(xs) == 0:
            continue
        panel.plot(xs, ys, color=color, marker=marker, ms=3, lw=1.1, ls=ls, label=label)
        if es.any():
            panel.fill_between(xs, ys - es, ys + es, color=color, alpha=0.15, lw=0)
    panel.set_xlabel(xlabel)
    panel.set_ylabel("mission success (admitted)")
    panel.set_ylim(-0.03, 1.0)
    if axis in ("uav", "arrival"):
        from matplotlib.ticker import MaxNLocator
        panel.xaxis.set_major_locator(MaxNLocator(integer=True))
letters = ["(a) fleet size", "(b) arrival load", "(c) link attenuation", "(d) interference floor"]
for panel, t in zip(axes2.flat, letters):
    panel.set_title(t, loc="left", fontsize=8)
axes2.flat[0].legend(frameon=False, loc="upper right", ncol=1, fontsize=6.2)
fig.tight_layout(pad=0.5)
fig.savefig(HERE / "fig3_m4_axes.pdf", bbox_inches="tight")
print("wrote fig3_m4_axes.pdf")

# ------------------------------------------------------------- zero-shot bars
ZS = [("soft", "unseen soft-conflict\nprofile"), ("load15", "$1.5\\times$ arrival\nload"),
      ("lowsnr", "unseen low-SNR\nblockage profile")]
fig, ax = plt.subplots(figsize=(3.5, 2.2))
width = 0.15
xbase = np.arange(len(ZS))
for i, (arm, label, color, marker, ls) in enumerate(ARMS):
    ys, es = [], []
    for tag, _ in ZS:
        blk = M4.get("zs", {}).get(tag, {}).get(arm)
        c = blk["metrics"].get(METRIC) if blk else None
        ys.append(c["mean"] if c else np.nan)
        es.append(c["std"] if c else 0.0)
    ax.bar(xbase + (i - 2) * width, ys, width, yerr=es, color=color, label=label,
           error_kw=dict(lw=0.7, capsize=1.5))
ax.set_xticks(xbase)
ax.set_xticklabels([z[1] for z in ZS], fontsize=6.5)
ax.set_ylabel("mission success (admitted)")
ax.legend(frameon=False, fontsize=6.0, ncol=2, loc="upper left")
ax.set_ylim(0, 1.05)
fig.tight_layout(pad=0.4)
fig.savefig(HERE / "fig4_zeroshot.pdf", bbox_inches="tight")
print("wrote fig4_zeroshot.pdf")
