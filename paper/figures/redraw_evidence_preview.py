#!/usr/bin/env python3
"""Plot existing summaries only. No training, inference, or recalibration.

Produces isolated revision previews; does not replace manuscript figures.
Run from any directory: python3 paper/figures/redraw_evidence_preview.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pubfig as pf
from pubfig.specs import FigureSpec

from publication_style import METHODS, apply_style, style_axis

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "outputs/figure_revision_20260907"
FIGURES = OUT / "figures"
SOURCES = [PAPER / "data/w14_mlp_10seed_series.csv",
           PAPER / "data/w14_mlp_10seed.json",
           PAPER / "data/w11_mlp_deepening.json",
           PAPER / "figures/comparison_v3_5qt.csv"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def export(fig, name: str, height_mm: float) -> None:
    spec = FigureSpec(name="communications", font_family="Times New Roman")
    pf.batch_export(fig, FIGURES / name, spec=spec, formats=("pdf", "svg", "png"),
                    width=182, height_mm=height_mm, dpi=400, trim=False)
    plt.close(fig)


def accuracy_preview() -> None:
    mlp = read_csv(SOURCES[0])
    base = read_csv(SOURCES[3])
    seeds = json.loads(SOURCES[1].read_text())["params"]["seeds"]
    assert len(seeds) == 10
    # Explicit axes preserve the reserved legend/footer space during export.
    fig = plt.figure(figsize=(182 / 25.4, 73 / 25.4))
    axes = [fig.add_axes([0.075 + i * 0.308, 0.22, 0.282, 0.49]) for i in range(3)]
    summary = []
    for i, (ax, channel) in enumerate(zip(axes, ["awgn", "rayleigh", "rician"])):
        rows = sorted((r for r in mlp if r["channel"] == channel),
                      key=lambda r: float(r["snr_db"]))
        assert len(rows) == 6
        x = np.array([float(r["snr_db"]) for r in rows])
        for method, source_id in [("mlp", None), ("lut", "M4_adaptive"),
                                  ("image", "M1_image"), ("token", "M3_token")]:
            color, marker, linestyle, label = METHODS[method]
            if source_id is None:
                y = np.array([float(r["acc_mean"]) for r in rows])
                lo = [float(r["acc_min"]) for r in rows]
                hi = [float(r["acc_max"]) for r in rows]
                assert np.all(np.array(lo) <= y) and np.all(y <= np.array(hi))
                ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
            else:
                records = sorted((r for r in base if r["channel"] == channel
                                  and r["method"] == source_id
                                  and r["qtype"] == "all" and r["split"] == "test"),
                                 key=lambda r: float(r["snr_db"]))
                assert len(records) == 6 and {int(r["n"]) for r in records} == {936}
                assert [float(r["snr_db"]) for r in records] == x.tolist()
                y = np.array([float(r["accuracy"]) for r in records])
                if method == "lut":
                    assert np.allclose(y, [float(r["acc_m4"]) for r in rows], atol=0.00015)
            ax.plot(x, y, color=color, marker=marker, linestyle=linestyle,
                    label=label, linewidth=1.65 if method == "mlp" else 1.25,
                    markerfacecolor="white", markeredgewidth=1,
                    zorder=5 if method == "mlp" else 3)
            summary.extend((channel, snr, method, acc) for snr, acc in zip(x, y))
        style_axis(ax)
        ax.set_xlim(-6, 21)
        ax.set_xticks(x)
        ax.set_ylim(0.55, 0.74)
        ax.set_yticks([0.55, 0.60, 0.65, 0.70])
        if i:
            ax.tick_params(labelleft=False)
        ax.set_xlabel("SNR (dB)")
        title = {"awgn": "AWGN", "rayleigh": "Rayleigh", "rician": "Rician (K = 6 dB)"}[channel]
        ax.set_title(f"({chr(97+i)}) {title}", pad=9)
    axes[0].set_ylabel("Answer accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 1),
               ncol=2, frameon=False, columnspacing=2.4, handlelength=2.4)
    fig.text(0.53, 0.015, "MLP: mean and min-max across 10 seeds; shading is not a confidence interval.",
             ha="center", fontsize=7.5, color="#41464B")
    export(fig, "accuracy_snr_preview", 73)
    with (OUT / "accuracy_values.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "snr_db", "method", "accuracy"])
        writer.writerows(summary)


def ablation_preview() -> None:
    source = json.loads(SOURCES[2].read_text())
    assert source["seeds"] == [0, 1, 2]
    data = source["ablation_rician_mlp"]
    labels = [("full", "Full MLP"), ("drop_qtype", "Without question type"),
              ("drop_class", "Without object class"),
              ("drop_view_risk", "Without viewpoint / risk"),
              ("drop_snr", "Without SNR"),
              ("drop_detector", "Without detector features"),
              ("drop_polarity", "Without polarity features")]
    fig = plt.figure(figsize=(182 / 25.4, 86 / 25.4))
    ax = fig.add_axes([0.29, 0.19, 0.69, 0.63])
    for i, (key, label) in enumerate(labels):
        color = METHODS["mlp"][0] if key == "full" else "#626970"
        ax.errorbar(data[key]["acc_mean"] * 100, i,
                    xerr=data[key]["acc_std"] * 100, fmt="o", color=color,
                    capsize=3, elinewidth=1.1, markersize=5,
                    markerfacecolor=color if key == "full" else "white")
        ax.text(71.03, i, f'{100 * data[key]["acc_mean"]:.2f} ± {100 * data[key]["acc_std"]:.2f}',
                va="center", fontsize=8, color=color)
    ax.axvline(data["full"]["acc_mean"] * 100, color=METHODS["mlp"][0],
               linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_yticks(range(len(labels)), [label for _, label in labels])
    ax.invert_yaxis()
    ax.set_xlim(67.2, 72.25)
    ax.set_xticks([67.5, 68, 68.5, 69, 69.5, 70, 70.5])
    ax.set_xlabel("Answer accuracy (%)")
    ax.set_title("Rician channel: MLP feature ablation", loc="left", pad=16)
    ax.grid(axis="x", color="#E2E5E8", linewidth=0.55)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.text(71.03, -0.7, "Mean ± SD", fontsize=8, color="#41464B")
    fig.text(0.57, 0.02, "Error bars: standard deviation across 3 seeds; not confidence intervals.",
             ha="center", fontsize=7.5, color="#41464B")
    export(fig, "mlp_ablation_preview", 86)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    apply_style()
    accuracy_preview()
    ablation_preview()
    manifest = {str(p.relative_to(PAPER)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in SOURCES}
    (OUT / "source_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved two preview figures (PDF/SVG/PNG) under {FIGURES}")


if __name__ == "__main__":
    main()
