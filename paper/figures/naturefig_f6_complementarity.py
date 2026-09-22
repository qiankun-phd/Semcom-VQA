#!/usr/bin/env python3
"""naturefig F6: evidence-question complementarity, TGCN two-panel restore.

(a) Per-type token vs image accuracy pairs (VisDrone, three channels pooled,
    test split) with Wilson 95% intervals per arm -- the raw complementarity
    the router exploits; threshold carries its "boundary type" tag.
(b) The token-image gap Delta per type on BOTH datasets (VisDrone vs
    DroneVehicle) with image-clustered bootstrap 95% CIs, read from
    outputs/reports/paper1_stats.json (the numbers of record behind the
    manuscript's Table `tab:comp`; 5000 resamples, image-level clusters).
    Sign consistency across datasets is the message: 4/5 types agree,
    threshold is the boundary type whose sign flips.

Panel (a) accuracies are recomputed from the canonical merged logs
outputs/vlm/v3_0_{awgn,rayleigh,rician}_predictions.csv exactly as
scripts/make_complementarity_fig.py does (string image_id, crc32 test
split); they are cross-checked at runtime against the JSON point values so
figure and table can never drift apart.

Style: nature-figure conventions -- 7.16 in final width (one-column TGCN
layout), grayscale-safe (filled vs open markers, direct labels), no
in-figure titles, Type-42 fonts.  Outputs SVG + PDF + PNG (300 dpi).
"""
from __future__ import annotations

import csv
import json
import math
import zlib
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
CH = ["awgn", "rayleigh", "rician"]
ORDER = ["counting", "comparison", "co_presence", "threshold", "presence"]
LABELS = ["counting", "comparison", "co-\npresence", "threshold\n(boundary)",
          "presence"]

C_TOKEN = "#5a6b7c"    # token: dark slate (filled square)
C_IMAGE = "#e8962f"    # image: orange (open circle reads in grayscale)
C_VD = "#2a3340"       # VisDrone (filled)
C_DV = "#7f9ab8"       # DroneVehicle (open)
C_SYM = "#5ad19a"
C_INK = "#2a3340"


def ok(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def is_test(image_id, frac: float = 0.2) -> bool:
    return (zlib.crc32(str(image_id).encode()) % 100) < int(frac * 100)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(mid - half, 0.0), min(mid + half, 1.0)


def per_arm_counts():
    """(qtype, service) -> [k, n] over pooled 3-channel TEST decisions,
    exactly mirroring scripts/make_complementarity_fig.py."""
    dec, qt = defaultdict(dict), {}
    for c in CH:
        p = REPO / f"outputs/vlm/v3_0_{c}_predictions.csv"
        for r in csv.DictReader(open(p)):
            if r["service_level"] not in ("1", "2"):
                continue
            if not is_test(r["image_id"]):
                continue
            # channel in the key: one decision per (task, channel), so the pool
            # is a true 3-channel pool (not first-channel-wins / AWGN-only)
            key = (c, r["image_id"], r["question"], r["snr_bin"])
            dec[key].setdefault(r["service_level"], ok(r["correct"]))
            qt[key] = r["question_type"]
    agg = defaultdict(lambda: {"1": [0, 0], "2": [0, 0]})
    for key, sv in dec.items():
        for s in ("1", "2"):
            if s in sv:
                agg[qt[key]][s][0] += int(sv[s])
                agg[qt[key]][s][1] += 1
    return agg


def main() -> None:
    out_dir = REPO / "outputs/figures/naturefig"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads((REPO / "outputs/reports/paper1_stats.json")
                       .read_text())
    vd = stats["table3_delta_visdrone"]
    dv = stats["table3_delta_dronevehicle"]

    agg = per_arm_counts()
    # runtime cross-check: recomputed accuracy == numbers of record
    for q in ORDER:
        (kt, nt), (ki, ni) = agg[q]["1"], agg[q]["2"]
        assert abs(kt / nt - vd[q]["token_acc"]) < 5e-4, (q, "token")
        assert abs(ki / ni - vd[q]["image_acc"]) < 5e-4, (q, "image")

    # top margin leaves room for (a)/(b) panel tags (was 0.97 → labels clipped)
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.16, 2.90),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.28,
                     "left": 0.07, "right": 0.99, "top": 0.88,
                     "bottom": 0.18})
    xs = range(len(ORDER))

    # ---- (a) token vs image accuracy pairs, Wilson 95% ---------------------
    for i, q in enumerate(ORDER):
        (kt, nt), (ki, ni) = agg[q]["1"], agg[q]["2"]
        at, ai = kt / nt, ki / ni
        lo_t, hi_t = wilson(kt, nt)
        lo_i, hi_i = wilson(ki, ni)
        ax_a.plot([i - 0.13, i + 0.13], [at, ai], color="0.55", lw=0.9,
                  zorder=2)
        ax_a.errorbar(i - 0.13, at, yerr=[[at - lo_t], [hi_t - at]],
                      fmt="s", ms=4.5, color=C_TOKEN, ecolor=C_TOKEN,
                      elinewidth=0.9, capsize=2, zorder=3)
        ax_a.errorbar(i + 0.13, ai, yerr=[[ai - lo_i], [hi_i - ai]],
                      fmt="o", ms=4.5, mfc="white", color=C_IMAGE,
                      ecolor=C_IMAGE, elinewidth=0.9, capsize=2, zorder=3)
    ax_a.set_xticks(list(xs))
    ax_a.set_xticklabels(LABELS)
    ax_a.set_ylabel("accuracy (3 channels pooled, test)")
    ax_a.set_ylim(0.18, 0.95)
    ax_a.grid(axis="y", alpha=0.3)
    ax_a.axvspan(-0.5, 3.5, color=C_SYM, alpha=0.10, zorder=0)
    ax_a.text(1.5, 0.915, "symbolic reasoning", ha="center", fontsize=7.5,
              color="#1f7a54")
    # keep "perception" inside axes (ha=right) so right edge is not clipped
    ax_a.text(4.45, 0.915, "perception", ha="right", fontsize=7.5,
              color="#8a6d3b")
    ax_a.legend(handles=[
        Line2D([], [], marker="s", ls="", ms=4.5, color=C_TOKEN,
               label="Fixed token (symbolic decoder)"),
        Line2D([], [], marker="o", ls="", ms=4.5, mfc="white",
               color=C_IMAGE, label="Rate-adaptive image (frozen VLM)")],
        loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=7,
        handletextpad=0.3)
    # panel tag outside axes box; clip_on=False prevents crop by axes patch
    ax_a.text(
        -0.02, 1.06, "(a)", transform=ax_a.transAxes,
        fontsize=9, fontweight="bold", va="bottom", ha="right", clip_on=False,
    )

    # ---- (b) Delta with clustered-bootstrap CI, both datasets --------------
    # Enhanced (table-to-figure batch): this panel now IS the manuscript's
    # former Table `tab:comp` -- each point carries its numeric Delta of
    # record, with a dagger when the clustered-bootstrap CI crosses zero.
    ax_b.axhline(0, color=C_INK, lw=0.8)
    for i, q in enumerate(ORDER):
        for dx, src, color, filled in ((-0.13, vd, C_VD, True),
                                       (+0.13, dv, C_DV, False)):
            d, (lo, hi) = src[q]["delta"], src[q]["ci95"]
            ax_b.errorbar(i + dx, d, yerr=[[d - lo], [hi - d]], fmt="D",
                          ms=4, color=color, mfc=color if filled else "white",
                          ecolor=color, elinewidth=1.0, capsize=2, zorder=3)
            dag = "$^\\dagger$" if src[q]["ci_crosses_zero"] else ""
            above = filled  # VisDrone label above its CI, DroneVehicle below
            ax_b.annotate(f"{d:+.3f}{dag}",
                          xy=(i + dx, hi if above else lo),
                          textcoords="offset points",
                          xytext=(0, 2.5 if above else -3.0),
                          ha="center", va="bottom" if above else "top",
                          fontsize=5.8, color=color, zorder=4)
    ax_b.set_xticks(list(xs))
    ax_b.set_xticklabels(LABELS)
    ax_b.set_ylabel("$\\Delta$ = acc(token) $-$ acc(image)")
    ax_b.set_ylim(-0.17, 0.31)
    ax_b.grid(axis="y", alpha=0.3)
    ax_b.annotate("sign flips across datasets\n(the one boundary type)",
                  xy=(3.13, dv["threshold"]["delta"] + 0.09),
                  xytext=(4.35, 0.255), fontsize=7, color=C_INK, ha="right",
                  arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4"))
    ax_b.text(0.98, 0.03, "$\\Delta>0$: token wins", transform=ax_b.transAxes,
              ha="right", va="bottom", fontsize=7, color="0.35")
    ax_b.legend(handles=[
        Line2D([], [], marker="D", ls="", ms=4, color=C_VD,
               label="VisDrone (primary)"),
        Line2D([], [], marker="D", ls="", ms=4, color=C_DV, mfc="white",
               label="DroneVehicle")],
        loc="lower left", bbox_to_anchor=(0.0, 0.02), fontsize=7,
        handletextpad=0.3)
    ax_b.text(
        -0.02, 1.06, "(b)", transform=ax_b.transAxes,
        fontsize=9, fontweight="bold", va="bottom", ha="right", clip_on=False,
    )

    for ax in (ax_a, ax_b):
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    print("=== panel (a) per-arm (pooled 3ch test) ===")
    for q in ORDER:
        (kt, nt), (ki, ni) = agg[q]["1"], agg[q]["2"]
        print(f"  {q:12s} token {kt}/{nt}={kt/nt:.4f}  "
              f"image {ki}/{ni}={ki/ni:.4f}  Delta={kt/nt-ki/ni:+.4f}")
    print("=== panel (b) from paper1_stats.json (Table tab:comp) ===")
    for q in ORDER:
        print(f"  {q:12s} VD {vd[q]['delta']:+.4f} {vd[q]['ci95']}   "
              f"DV {dv[q]['delta']:+.4f} {dv[q]['ci95']}")

    # pad_inches keeps (a)/(b) tags outside the axes from being cropped
    for ext, dpi in (("svg", None), ("pdf", None), ("png", 400)):
        fig.savefig(
            out_dir / f"F6_complementarity.{ext}",
            dpi=dpi, bbox_inches="tight", pad_inches=0.04,
        )
    # also stage into paper_semcom/figures when present
    paper_fig = Path(__file__).resolve().parent
    if paper_fig.name == "figures" and (paper_fig / "MANIFEST.md").is_file():
        for ext in ("svg", "pdf", "png"):
            src = out_dir / f"F6_complementarity.{ext}"
            if src.is_file():
                (paper_fig / f"F6_complementarity.{ext}").write_bytes(src.read_bytes())
        print(f"also staged -> {paper_fig}/F6_complementarity.[svg|pdf|png]")
    print(f"wrote {out_dir}/F6_complementarity.[svg|pdf|png]")


if __name__ == "__main__":
    main()
