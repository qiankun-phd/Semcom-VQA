#!/usr/bin/env python3
"""naturefig F15: cross-VLM per-type accuracy, grouped bars (TGCN).

Table-to-figure conversion of the manuscript's Table `tab:crossvlm`
(Rician): per question type, one VLM-agnostic token bar (the symbolic
decoder, identical by construction for every receiver) next to the image
accuracy of the three receiver VLMs (Qwen2-VL-2B, Qwen2.5-VL-3B,
SmolVLM-Instruct). Each group carries the table's routing verdict
(-> token / -> image / boundary).

Data of record only: outputs/reports/paper1_stats.json, key
`table5_crossvlm` -- the numbers behind the manuscript table (token column
= the 2B row's full-n token accuracy, exactly as printed).

Style: nature-figure conventions -- Times-family serif (Nimbus Roman on
the render host), grayscale-safe (token bar hatched, model bars in
lightness-ordered fills), Type-42 fonts. Outputs PDF + SVG + PNG (300 dpi).
"""
from __future__ import annotations

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

ORDER = ["presence", "counting", "comparison", "co_presence", "threshold"]
LABELS = ["presence", "counting", "comparison", "co-presence",
          "threshold\n(boundary)"]
VERDICT = {"presence": "$\\to$ image", "counting": "$\\to$ token",
           "comparison": "$\\to$ token", "co_presence": "$\\to$ token",
           "threshold": "boundary"}
C_TOKEN = "#5a6b7c"
C_2B, C_3B, C_SMOL = "#f4b860", "#e8962f", "#9c5f28"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t5 = json.loads((REPO / "outputs/reports/paper1_stats.json")
                    .read_text())["table5_crossvlm"]
    token = {q: t5["qwen2vl_2b"][q]["token"] for q in ORDER}
    img = {mdl: {q: t5[mdl][q]["image"] for q in ORDER}
           for mdl in ("qwen2vl_2b", "qwen25vl_3b", "smolvlm")}

    fig, ax = plt.subplots(figsize=(4.9, 2.75))
    w = 0.19
    for i, q in enumerate(ORDER):
        ax.bar(i - 1.5 * w, token[q], w, color=C_TOKEN, hatch="///",
               edgecolor="white", lw=0.3, zorder=3)
        ax.bar(i - 0.5 * w, img["qwen2vl_2b"][q], w, color=C_2B, zorder=3)
        ax.bar(i + 0.5 * w, img["qwen25vl_3b"][q], w, color=C_3B, zorder=3)
        ax.bar(i + 1.5 * w, img["smolvlm"][q], w, color=C_SMOL, zorder=3)
        top = max(token[q], *(img[m][q] for m in img))
        ax.annotate(VERDICT[q], (i, top), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=6.5,
                    color="#1f7a54" if "token" in VERDICT[q] else
                    ("#8a6d3b" if "image" in VERDICT[q] else "0.35"))
        print(f"{q:12s} token {token[q]:.3f}  " +
              "  ".join(f"{m} {img[m][q]:.3f}" for m in img))
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("accuracy (Rician, test)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.legend(handles=[
        Patch(fc=C_TOKEN, hatch="///", ec="white",
              label="Fixed token (symbolic decoder)"),
        Patch(fc=C_2B, label="Rate-adaptive image, Qwen2-VL-2B"),
        Patch(fc=C_3B, label="Rate-adaptive image, Qwen2.5-VL-3B"),
        Patch(fc=C_SMOL, label="Rate-adaptive image, SmolVLM-Instruct")],
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
        handlelength=1.2, labelspacing=0.3, columnspacing=0.9)
    fig.tight_layout()
    for ext, dpi in (("pdf", None), ("svg", None), ("png", 300)):
        fig.savefig(OUT / f"F15_crossvlm.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"wrote {OUT}/F15_crossvlm.[pdf|svg|png]")


if __name__ == "__main__":
    main()
