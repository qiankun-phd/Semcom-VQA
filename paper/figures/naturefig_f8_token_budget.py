#!/usr/bin/env python3
"""naturefig F8: top-t token-budget sweep (TGCN / group-meeting 11.3).

Publication-figure redraw of draw_f8() in build_token_budget_sweep.py.
Fast path only: reads token_budget_full.csv (no recompute).

Style aligned with F4 / F14 naturefig restyle (2026-07-23):
  Times New Roman (serif fallbacks), Type-42 PDF fonts, clean spines,
  panel letters (a)(b)(c), claim annotations, high-res PNG via PDF@400 dpi.

Three panels (Rician K=6 dB):
  (a) accuracy vs mean tokens sent at SNR = -5 dB
  (b) same at SNR = 20 dB
  (c) accuracy vs airtime per query (ms, B = 1 MHz) at SNR = 5 dB
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
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
from matplotlib.ticker import FixedLocator, NullFormatter

QT_SYMBOLIC = ("counting", "comparison", "co_presence", "threshold")
COLORS = {
    "counting": "#d1495b",
    "comparison": "#4ea1ff",
    "co_presence": "#5ad19a",
    "threshold": "#8e6bb5",
    "all": "#444444",
}
DISPLAY = {"co_presence": "co-presence"}
CH_TAG = {"awgn": "AWGN", "rayleigh": "Rayleigh", "rician": "Rician K=6 dB"}


def find_csv(cli: str | None) -> Path:
    if cli:
        p = Path(cli)
        if p.is_file():
            return p
        raise FileNotFoundError(cli)
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "token_budget_full.csv",
        Path.home() / "phd_research/vqa_semcom/outputs/reports/token_budget_full.csv",
        here.parents[1] / "outputs/reports/token_budget_full.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("token_budget_full.csv not found")


def default_out_dirs() -> list[Path]:
    here = Path(__file__).resolve().parent
    outs = [here]
    server = Path.home() / "phd_research/vqa_semcom/outputs/figures/naturefig"
    if server.parent.is_dir() or str(Path.home()).startswith("/home"):
        outs.append(server)
    return outs


def load_rows(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "channel": r["channel"],
                "t_budget": str(r["t_budget"]),
                "snr_db": float(r["snr_db"]),
                "qtype": r["qtype"],
                "accuracy": float(r["accuracy"]),
                "n": int(r["n"]),
                "mean_tokens_sent": float(r["mean_tokens_sent"]),
                "mean_payload_bytes": float(r["mean_payload_bytes"]),
                "mean_channel_uses": float(r["mean_channel_uses"]),
            })
    return rows


def style_ax(ax) -> None:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(True, alpha=0.28, linestyle="-", which="major")
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-csv", default=None)
    ap.add_argument("--fig-channel", default="rician")
    ap.add_argument(
        "--out-dir", default=None,
        help="Primary output dir (also always writes beside this script).",
    )
    args = ap.parse_args()

    csv_path = find_csv(args.from_csv)
    rows = load_rows(csv_path)
    ch = args.fig_channel
    sub = [r for r in rows if r["channel"] == ch]
    if not sub:
        raise RuntimeError(f"no rows for channel={ch} in {csv_path}")

    out_dirs = default_out_dirs()
    if args.out_dir:
        out_dirs.insert(0, Path(args.out_dir))
    # unique preserve order
    seen: set[str] = set()
    out_dirs = [d for d in out_dirs if not (str(d) in seen or seen.add(str(d)))]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    f_lab, f_tick, f_leg, f_tag = 8, 7.5, 7, 7
    # Double-column width (~7.16 in) x short height — same family as original F8
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), sharey=True)

    series: dict[tuple[int, str], list[tuple[float, float, str]]] = {}
    for pi, (ax, snr) in enumerate(zip(axes[:2], (-5.0, 20.0))):
        for qt in list(QT_SYMBOLIC) + ["all"]:
            pts = sorted(
                [
                    (r["mean_tokens_sent"], r["accuracy"], r["t_budget"])
                    for r in sub
                    if abs(r["snr_db"] - snr) < 1e-9 and r["qtype"] == qt
                ],
                key=lambda p: p[0],
            )
            if not pts:
                continue
            series[(pi, qt)] = pts
            hero = qt == "all"
            ax.plot(
                [p[0] for p in pts], [p[1] for p in pts], "o-",
                color=COLORS[qt], label=DISPLAY.get(qt, qt),
                linewidth=1.4 if hero else 1.05,
                markersize=3.0 if hero else 2.5,
                markeredgewidth=0.4,
                markeredgecolor="white" if hero else COLORS[qt],
                zorder=5 if hero else 3,
            )
        ax.set_xscale("log")
        ax.set_xlabel("mean tokens sent (top-$t$)", fontsize=f_lab)
        style_ax(ax)
        ax.text(
            0.04, 0.06, f"SNR = {snr:g} dB", transform=ax.transAxes,
            va="bottom", ha="left", fontsize=f_tag, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75",
                      alpha=0.92, lw=0.5),
        )

    def budget_point(pi: int, qt: str, budget: str):
        for x, y, tb in series.get((pi, qt), []):
            if tb == budget:
                return x, y
        return None

    # Claim annotations (values from CSV; not hand-tuned numbers)
    for pi in (0, 1):
        pk = budget_point(pi, "threshold", "32")
        if pk:
            axes[pi].scatter(
                [pk[0]], [pk[1]], s=42, facecolors="none",
                edgecolors=COLORS["threshold"], linewidths=1.0, zorder=6,
            )
            if pi == 0:
                axes[pi].annotate(
                    "peak $t{=}32$", xy=pk,
                    xytext=(pk[0] * 1.28, pk[1] + 0.082),
                    fontsize=6.0, color=COLORS["threshold"], ha="left",
                    arrowprops=dict(
                        arrowstyle="-", lw=0.55, color=COLORS["threshold"],
                        alpha=0.75, shrinkA=0.5, shrinkB=2.0,
                    ),
                )
    sat = budget_point(0, "comparison", "3")
    if sat:
        axes[0].annotate(
            "saturates $t{=}3$", xy=sat,
            xytext=(sat[0] * 0.36, sat[1] - 0.093),
            fontsize=6.0, color=COLORS["comparison"], ha="left", va="top",
            arrowprops=dict(
                arrowstyle="-", lw=0.55, color=COLORS["comparison"],
                alpha=0.75, shrinkA=0.5, shrinkB=1.5,
            ),
        )
    cnt = series.get((0, "counting"))
    if cnt:
        axes[0].text(
            cnt[-1][0] * 1.12, cnt[-1][1] - 0.15,
            "still climbing at $t{=}48$", fontsize=6.0,
            color=COLORS["counting"], ha="right", va="top",
        )

    # (c) cost view at mid-SNR
    ax = axes[2]
    for qt in list(QT_SYMBOLIC) + ["all"]:
        pts = sorted(
            [
                (r["mean_channel_uses"], r["accuracy"])
                for r in sub
                if abs(r["snr_db"] - 5.0) < 1e-9 and r["qtype"] == qt
            ],
            key=lambda p: p[0],
        )
        if not pts:
            continue
        hero = qt == "all"
        ax.plot(
            [p[0] for p in pts], [p[1] for p in pts], "o-",
            color=COLORS[qt], label=DISPLAY.get(qt, qt),
            linewidth=1.4 if hero else 1.05,
            markersize=3.0 if hero else 2.5,
            markeredgewidth=0.4,
            markeredgecolor="white" if hero else COLORS[qt],
            zorder=5 if hero else 3,
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1.2e4, 1.4e4, 1.6e4, 1.8e4]))
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _p: f"{v / 1e3:g}")
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("airtime per query (ms)", fontsize=f_lab)
    style_ax(ax)
    ax.text(
        0.04, 0.06, "SNR = 5 dB", transform=ax.transAxes,
        va="bottom", ha="left", fontsize=f_tag, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75",
                  alpha=0.92, lw=0.5),
    )

    for k, a in enumerate(axes):
        a.tick_params(labelsize=f_tick, length=2.5, width=0.55)
        a.set_ylim(0.18, 0.92)
        a.text(
            -0.02, 1.04, f"({chr(97 + k)})", transform=a.transAxes,
            fontsize=8.5, fontweight="bold", va="bottom", ha="right",
        )
    axes[0].set_ylabel("accuracy", fontsize=f_lab)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=5, fontsize=f_leg,
        frameon=False, handlelength=1.6, columnspacing=1.1,
        handletextpad=0.4, bbox_to_anchor=(0.48, 1.03),
    )
    fig.text(
        0.995, 0.98, CH_TAG.get(ch, ch), ha="right", va="top",
        fontsize=f_tag, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=0.35, w_pad=0.55)

    primary = out_dirs[0]
    pdf_path = primary / "F8_token_budget.pdf"
    fig.savefig(pdf_path)
    fig.savefig(primary / "F8_token_budget.svg")
    # Direct high-res PNG
    fig.savefig(primary / "F8_token_budget.png", dpi=400)
    plt.close(fig)

    # Sharper slide PNG via PDF raster (same path as F4 restyle)
    try:
        subprocess.run(
            [
                "pdftoppm", "-png", "-r", "400", "-singlefile",
                str(pdf_path), str(primary / "F8_token_budget"),
            ],
            check=False, capture_output=True,
        )
    except FileNotFoundError:
        pass

    # Mirror to other out dirs
    for d in out_dirs[1:]:
        for ext in ("pdf", "svg", "png"):
            src = primary / f"F8_token_budget.{ext}"
            if src.is_file():
                (d / f"F8_token_budget.{ext}").write_bytes(src.read_bytes())

    print(f"data: {csv_path}")
    print(f"wrote F8_token_budget.[pdf,svg,png] -> {', '.join(str(d) for d in out_dirs)}")


if __name__ == "__main__":
    main()
