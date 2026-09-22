"""Shared styles for the 2026-09 manuscript figure revision.

Method colors are independent of component colors. Never relabel LUT data
as the training-free rule. Use markers/line styles as redundant encodings.
"""
from __future__ import annotations

import matplotlib as mpl

METHODS = {
    "mlp": ("#0072B2", "o", "-", "Per-sample routing (MLP)"),
    "rule": ("#009E73", "^", "-", "Type rule"),
    "lut": ("#56B4E9", "D", "--", "Type routing (LUT)"),
    "image": ("#E69F00", "s", "-", "Rate-adaptive image"),
    "token": ("#626970", "v", "-.", "Fixed detection evidence"),
    "linear": ("#CC79A7", "p", "--", "Linear routing"),
    "djscc": ("#D55E00", "P", "-.", "DJSCC"),
    "analog": ("#8172B3", "x", ":", "Uncoded analog"),
    "fixed_image": ("#8C8C8C", "+", ":", "Fixed-rate image"),
    "oracle": ("#202020", "*", "--", "Oracle"),
    "clean_image": ("#202020", None, ":", "Error-free image"),
}
COMPONENTS = {
    "radio": ("#E3BB66", "///"),
    "detector": ("#A8BDC9", "xx"),
    "vlm": ("#505A66", ""),
}


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.65,
        "lines.linewidth": 1.35,
        "lines.markersize": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })


def style_axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E2E5E8", linewidth=0.55)
    ax.tick_params(direction="out", length=3, width=0.6)
