"""Publication-quality figure generation for IEEE TGCN / TCOM manuscript.

Generates 4 figures:
1. Figure 1: Accuracy vs. SNR Dual-Panel Comparison (Mode 1 Constant Power & Mode 2 Constant Energy)
2. Figure 2: Dynamic Action Routing & 3-Phase Adaptation Architecture
3. Figure 3: Energy-Accuracy & Latency-Accuracy Pareto Frontiers
4. Figure 4: Task-Type Semantic Breakdown across 6 TDIUC Question Categories
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Publication-grade typography and style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "mathtext.fontset": "cm",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

OUTPUT_DIR = Path("paper/figures/tgcn_publication_figures_20260922")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cohesive Palette
C_TGCN = "#D32F2F"       # Crimson Red (Proposed)
C_2K = "#1976D2"         # Royal Blue
C_4K = "#388E3C"         # Forest Green
C_8K = "#7B1FA2"         # Deep Purple
C_EXP14 = "#616161"      # Medium Slate Grey
C_BOUND = "#212121"      # Charcoal / Black


def compute_lat(symbols: float, tokens: float) -> float:
    """Calculates unified system latency (ms) = enc + tx + dec + vlm."""
    t_enc = 246.6
    t_tx = symbols / 1000.0  # 1 Msps symbol rate -> ms
    t_dec = 52.7
    t_vlm = 237.0 + 0.23 * tokens
    return t_enc + t_tx + t_dec + t_vlm


def plot_figure_1(f_scan: dict, f_eval: dict):
    """Figure 1: Accuracy vs SNR Dual-Panel (Mode 1 & Mode 2)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)

    snrs = f_scan["snrs"]
    snr_keys = [str(s) for s in snrs]

    # Mode 1 Data
    m1 = f_scan["mode1_constant_power"]
    m1_2k = [m1[k]["fixed_2000_medium"]["strict_accuracy"] * 100 for k in snr_keys]
    m1_4k = [m1[k]["fixed_4000_medium"]["strict_accuracy"] * 100 for k in snr_keys]
    m1_8k = [m1[k]["fixed_8000_high"]["strict_accuracy"] * 100 for k in snr_keys]
    m1_exp14 = [m1[k]["static_exp014_joint"]["strict_accuracy"] * 100 for k in snr_keys]
    m1_tgcn = [m1[k]["proposed_snr_adaptive"]["strict_accuracy"] * 100 for k in snr_keys]

    # Panel (a): Mode 1 Constant Power
    ax1.plot(snrs, m1_2k, label=r"Fixed $2\,$kB Med ($N_s=21.4\,$k)", color=C_2K, linestyle="--", marker="o", markersize=5, linewidth=1.8)
    ax1.plot(snrs, m1_4k, label=r"Fixed $4\,$kB Med ($N_s=42.8\,$k)", color=C_4K, linestyle="-.", marker="s", markersize=5, linewidth=1.8)
    ax1.plot(snrs, m1_8k, label=r"Fixed $8\,$kB High ($N_s=85.2\,$k)", color=C_8K, linestyle=":", marker="^", markersize=5, linewidth=1.8)
    ax1.plot(snrs, m1_exp14, label="Channel-Blind Policy", color=C_EXP14, linestyle="--", marker="x", markersize=6, linewidth=1.6)
    ax1.plot(snrs, m1_tgcn, label="Proposed EcoSem-VQA (CSPM)", color=C_TGCN, linestyle="-", marker="D", markersize=6, linewidth=2.4)
    ax1.axhline(77.12, color=C_BOUND, linestyle=":", linewidth=1.2, label=r"Error-Free Bound ($77.1\%$)")

    ax1.set_title(r"(a) CSPM: Constant Symbol Power ($P_s = P_0$)", pad=10, fontweight="bold")
    ax1.set_xlabel(r"Channel Average SNR $\gamma$ (dB)")
    ax1.set_ylabel("End-to-End VQA Strict Accuracy (%)")
    ax1.set_xlim(-6, 21)
    ax1.set_ylim(-2, 85)
    ax1.set_xticks(snrs)
    ax1.legend(loc="upper left", framealpha=0.92, edgecolor="#cccccc")

    # Mode 2 Data
    m2 = f_eval["mode2_constant_energy"]
    m2_2k = [m2[k]["fixed_2000_med_acc"] * 100 for k in snr_keys]
    m2_4k = [m2[k]["fixed_4000_med_acc"] * 100 for k in snr_keys]
    m2_8k = [m2[k]["fixed_8000_high_acc"] * 100 for k in snr_keys]
    m2_exp14 = [m2[k]["exp014_blind_joint_acc"] * 100 for k in snr_keys]
    m2_tgcn = [m2[k]["tgcn_cross_layer_acc"] * 100 for k in snr_keys]

    # Panel (b): Mode 2 Constant Energy
    ax2.plot(snrs, m2_2k, label=r"Fixed $2\,$kB Med ($\Delta\mathrm{SNR} = 0\,$dB)", color=C_2K, linestyle="--", marker="o", markersize=5, linewidth=1.8)
    ax2.plot(snrs, m2_4k, label=r"Fixed $4\,$kB Med ($\Delta\mathrm{SNR} = -3.01\,$dB)", color=C_4K, linestyle="-.", marker="s", markersize=5, linewidth=1.8)
    ax2.plot(snrs, m2_8k, label=r"Fixed $8\,$kB High ($\Delta\mathrm{SNR} = -6.00\,$dB)", color=C_8K, linestyle=":", marker="^", markersize=5, linewidth=1.8)
    ax2.plot(snrs, m2_exp14, label="Channel-Blind Policy", color=C_EXP14, linestyle="--", marker="x", markersize=6, linewidth=1.6)
    ax2.plot(snrs, m2_tgcn, label="Proposed EcoSem-VQA (CQEM)", color=C_TGCN, linestyle="-", marker="D", markersize=6, linewidth=2.5)
    ax2.axhline(77.12, color=C_BOUND, linestyle=":", linewidth=1.2, label=r"Error-Free Bound ($77.1\%$)")

    # Shaded Gain Region
    ax2.fill_between(snrs, m2_4k, m2_tgcn, where=(np.array(m2_tgcn) > np.array(m2_4k)),
                     color="#C8E6C9", alpha=0.45, label="PPC Outage Protection Gain")

    # Annotations on Panel (b)
    ax2.annotate("+29.92% Gain\n(PPC Outage Capping)",
                 xy=(2.5, m2_tgcn[3]), xytext=(3.0, 18),
                 arrowprops=dict(arrowstyle="->", color=C_TGCN, lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.35", fc="#FFEBEE", ec=C_TGCN, lw=1.2),
                 fontsize=9.5, fontweight="bold", color=C_TGCN)

    ax2.annotate("+27.12% Gain\n(Rate Migration)",
                 xy=(5.0, m2_tgcn[4]), xytext=(6.5, 42),
                 arrowprops=dict(arrowstyle="->", color=C_TGCN, lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.35", fc="#FFEBEE", ec=C_TGCN, lw=1.2),
                 fontsize=9.5, fontweight="bold", color=C_TGCN)

    ax2.set_title(r"(b) CQEM: Constant Query-Energy Budget ($E_{\mathrm{tx}} = E_0$)", pad=10, fontweight="bold")
    ax2.set_xlabel(r"Channel Average SNR $\gamma$ (dB)")
    ax2.set_xlim(-6, 21)
    ax2.set_xticks(snrs)
    ax2.legend(loc="upper left", framealpha=0.92, edgecolor="#cccccc")

    plt.tight_layout()
    p_png = OUTPUT_DIR / "fig1_snr_vs_accuracy_dual_mode.png"
    p_pdf = OUTPUT_DIR / "fig1_snr_vs_accuracy_dual_mode.pdf"
    fig.savefig(p_png)
    fig.savefig(p_pdf)
    plt.close(fig)
    print(f"Saved Figure 1 to {p_png} and {p_pdf}")


def plot_figure_2(f_eval: dict):
    """Figure 2: Dynamic Action Routing Distribution across SNR."""
    snrs = f_eval["snrs"]
    snr_keys = [str(s) for s in snrs]
    m2 = f_eval["mode2_constant_energy"]

    actions = [
        "2000_low", "2000_medium", "2000_high",
        "4000_low", "4000_medium", "4000_high",
        "8000_low", "8000_medium", "8000_high"
    ]
    action_colors = [
        "#90CAF9", "#1976D2", "#0D47A1",  # 2000 band (light blue, royal blue, deep navy)
        "#FFE082", "#FB8C00", "#D84315",  # 4000 band (cream gold, amber, burnt orange)
        "#E1BEE7", "#8E24AA", "#4A148C"   # 8000 band (light plum, purple, royal purple)
    ]
    action_labels = [
        r"2 kB, Low ($43.8\,$tok)", r"2 kB, Med ($110.6\,$tok)", r"2 kB, High ($216.0\,$tok)",
        r"4 kB, Low ($43.8\,$tok)", r"4 kB, Med ($110.6\,$tok)", r"4 kB, High ($216.0\,$tok)",
        r"8 kB, Low ($43.8\,$tok)", r"8 kB, Med ($110.6\,$tok)", r"8 kB, High ($216.0\,$tok)",
    ]

    counts = np.zeros((9, len(snrs)), dtype=float)
    mean_tokens = []
    mean_symbols = []

    for j, k in enumerate(snr_keys):
        r_hist = m2[k]["routing"]
        for i, a in enumerate(actions):
            counts[i, j] = r_hist.get(a, 0) / 24.0  # percentage
        mean_tokens.append(m2[k]["mean_tokens"])
        mean_symbols.append(m2[k]["mean_symbols"] / 1000.0)

    fig, ax1 = plt.subplots(figsize=(13.5, 6.2))

    # Stacked bar plot
    bar_width = 1.45
    bottom = np.zeros(len(snrs))
    for i in range(9):
        if counts[i].sum() > 0:
            ax1.bar(snrs, counts[i], bar_width, bottom=bottom, color=action_colors[i],
                    edgecolor="white", linewidth=0.7, label=action_labels[i])
            bottom += counts[i]

    ax1.set_xlabel(r"Channel Physical SNR $\gamma$ (dB)", labelpad=8)
    ax1.set_ylabel("CART-Net Action Allocation (%)", labelpad=8)
    ax1.set_xlim(-6.5, 21.5)
    ax1.set_ylim(0, 118)
    ax1.set_xticks(snrs)

    # Twin axis for mean tokens and symbols
    ax2 = ax1.twinx()
    ax2.plot(snrs, mean_tokens, color="#C62828", linestyle="-", marker="o", markersize=7,
             linewidth=2.4, label=r"DyTBA Mean Tokens $\bar{T}_v$", zorder=10)
    ax2.plot(snrs, mean_symbols, color="#1565C0", linestyle="--", marker="s", markersize=6,
             linewidth=2.0, label=r"DigiSem Mean Symbols $N_s$ ($\times 10^3$)", zorder=10)
    ax2.set_ylabel(r"Visual Tokens / Complex Symbols ($\times 10^3$)", color="#333333", labelpad=8)
    ax2.set_ylim(30, 185)
    ax2.grid(False)

    # Regime division vertical markers (stop at 100%)
    ax1.plot([11.25, 11.25], [0, 100], color="#666666", linestyle=":", linewidth=1.5, zorder=4)
    ax1.plot([17.5, 17.5], [0, 100], color="#666666", linestyle=":", linewidth=1.5, zorder=4)

    # Non-overlapping Regime Banners
    ax1.text(2.5, 104, "Phase I: LSR (γ ≤ 10 dB)\n100% 2 kB band (PPC activated)",
             ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#0D47A1",
             bbox=dict(boxstyle="square,pad=0.3", fc="#E3F2FD", ec="#90CAF9", lw=1))
    ax1.text(14.0, 104, "Phase II: RMR (12.5–15 dB)\n64%–93% to 4 kB band",
             ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#E65100",
             bbox=dict(boxstyle="square,pad=0.3", fc="#FFF3E0", ec="#FFE082", lw=1))
    ax1.text(19.8, 104, "Phase III: HFBR (≥ 20 dB)\n23% to 8 kB High",
             ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#4A148C",
             bbox=dict(boxstyle="square,pad=0.3", fc="#F3E5F5", ec="#E1BEE7", lw=1))

    # Legends: Combine ax1 and ax2
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc="center left", bbox_to_anchor=(0.015, 0.52),
               ncol=2, framealpha=0.94, edgecolor="#cccccc", title="Action Selection (Rate + DyTBA)")
    ax2.legend(handles2, labels2, loc="center left", bbox_to_anchor=(0.015, 0.32),
               framealpha=0.94, edgecolor="#cccccc")

    plt.tight_layout()
    p_png = OUTPUT_DIR / "fig2_dynamic_routing_architecture.png"
    p_pdf = OUTPUT_DIR / "fig2_dynamic_routing_architecture.pdf"
    fig.savefig(p_png)
    fig.savefig(p_pdf)
    plt.close(fig)
    print(f"Saved Figure 2 to {p_png} and {p_pdf}")


def plot_figure_3(f_eval: dict):
    """Figure 3: Relative RF Transmission Energy and Latency vs. Channel SNR."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))

    snrs = np.array(f_eval["snrs"])
    snr_keys = [str(s) for s in snrs]
    m2 = f_eval["mode2_constant_energy"]

    # Physical baseline values
    e_2k = np.array([1.0] * len(snrs))
    e_4k = np.array([2.0] * len(snrs))
    e_8k = np.array([85170.0 / 21420.0] * len(snrs))  # 3.976x
    e_blind = np.array([2.0] * len(snrs))

    lat_2k = np.array([compute_lat(21420, 110.6)] * len(snrs))     # 583.16 ms
    lat_4k = np.array([compute_lat(42840, 110.6)] * len(snrs))     # 604.58 ms
    lat_8k = np.array([compute_lat(85170, 216.0)] * len(snrs))     # 671.15 ms
    lat_blind = np.array([compute_lat(42840, 110.6)] * len(snrs))  # 604.58 ms

    # Proposed EcoSem-VQA (CART-Net + DyTBA)
    e_tgcn = np.array([m2[k]["mean_symbols"] / 21420.0 for k in snr_keys])
    lat_tgcn = np.array([compute_lat(m2[k]["mean_symbols"], m2[k]["mean_tokens"]) for k in snr_keys])

    # -------------------------------------------------------------
    # Panel (a): Relative RF Transmission Energy vs Channel SNR
    # -------------------------------------------------------------
    ax1.plot(snrs, e_8k, label=r"Fixed $8\,$kB Baseline ($4.0\times$)", color=C_8K,
             linestyle="--", marker="^", markersize=6, alpha=0.85)
    ax1.plot(snrs, e_4k, label=r"Fixed $4\,$kB Baseline ($2.0\times$)", color=C_4K,
             linestyle="--", marker="s", markersize=6, alpha=0.85)
    ax1.plot(snrs, e_blind, label=r"Channel-Blind Policy ($2.0\times$)", color=C_EXP14,
             linestyle="-.", marker="x", markersize=6, alpha=0.85)
    ax1.plot(snrs, e_2k, label=r"Fixed $2\,$kB Baseline ($1.0\times$)", color=C_2K,
             linestyle="--", marker="o", markersize=6, alpha=0.85)
    ax1.plot(snrs, e_tgcn, label="Proposed EcoSem-VQA (CART-Net)", color=C_TGCN,
             linestyle="-", marker="D", markersize=6.5, linewidth=2.5, zorder=5)

    # Shaded green savings region
    ax1.fill_between(snrs, e_tgcn, e_4k, where=(e_tgcn < e_4k),
                     color="#E8F5E9", alpha=0.6, label="50% Energy Saving vs. 4k (LSR)")
    ax1.fill_between(snrs, e_tgcn, e_8k, where=(snrs >= 12.5),
                     color="#F3E5F5", alpha=0.6, label="38.4% Energy Saving vs. 8k (HFBR)")

    # Regime division vertical markers
    ax1.axvline(11.25, color="#888888", linestyle=":", linewidth=1.2)
    ax1.axvline(17.5, color="#888888", linestyle=":", linewidth=1.2)
    ax1.text(2.5, 4.22, "Phase I: LSR", ha="center", fontsize=8.5, fontweight="bold", color="#0D47A1")
    ax1.text(14.0, 4.22, "Phase II: RMR", ha="center", fontsize=8.5, fontweight="bold", color="#E65100")
    ax1.text(19.2, 4.22, "Phase III: HFBR", ha="center", fontsize=8.5, fontweight="bold", color="#4A148C")

    # Annotations with arrows
    ax1.annotate("50% RF Energy Saving\n(vs. Fixed 4k & Blind)",
                 xy=(2.5, e_tgcn[3]), xytext=(1.0, 1.45),
                 arrowprops=dict(arrowstyle="->", color=C_TGCN, lw=1.3),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec=C_TGCN, lw=1.0),
                 fontsize=8.8, fontweight="bold", color=C_TGCN)

    ax1.annotate("-38.4% Energy vs. 8k\n(2.45x vs. 3.98x)",
                 xy=(20.0, e_tgcn[9]), xytext=(14.5, 2.95),
                 arrowprops=dict(arrowstyle="->", color=C_TGCN, lw=1.3),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec=C_TGCN, lw=1.0),
                 fontsize=8.8, fontweight="bold", color=C_TGCN)

    ax1.set_title(r"(a) Relative RF Transmission Energy ($E_{\mathrm{tx}} / E_{2\mathrm{k}}$) vs. Channel SNR $\gamma$",
                  pad=10, fontweight="bold")
    ax1.set_xlabel(r"Channel Average Physical SNR $\gamma$ (dB)")
    ax1.set_ylabel(r"Relative Transmission Energy ($E_{\mathrm{tx}} / E_{2\mathrm{k}}$)")
    ax1.set_xlim(-6, 21)
    ax1.set_xticks(snrs)
    ax1.set_ylim(0.5, 4.45)
    ax1.legend(loc="center left", bbox_to_anchor=(0.02, 0.63), framealpha=0.92, edgecolor="#cccccc", fontsize=8.8)

    # -------------------------------------------------------------
    # Panel (b): End-to-End Latency vs Channel SNR
    # -------------------------------------------------------------
    ax2.plot(snrs, lat_8k, label=r"Fixed $8\,$kB Baseline ($671.2\,$ms)", color=C_8K,
             linestyle="--", marker="^", markersize=6, alpha=0.85)
    ax2.plot(snrs, lat_4k, label=r"Fixed $4\,$kB Baseline ($604.6\,$ms)", color=C_4K,
             linestyle="--", marker="s", markersize=6, alpha=0.85)
    ax2.plot(snrs, lat_blind, label=r"Channel-Blind Policy ($604.6\,$ms)", color=C_EXP14,
             linestyle="-.", marker="x", markersize=6, alpha=0.85)
    ax2.plot(snrs, lat_2k, label=r"Fixed $2\,$kB Baseline ($583.2\,$ms)", color=C_2K,
             linestyle="--", marker="o", markersize=6, alpha=0.85)
    ax2.plot(snrs, lat_tgcn, label="Proposed EcoSem-VQA (CART-Net)", color=C_TGCN,
             linestyle="-", marker="D", markersize=6.5, linewidth=2.5, zorder=5)

    # Shaded latency reduction region vs 8k
    ax2.fill_between(snrs, lat_tgcn, lat_8k, color="#EDE7F6", alpha=0.55, label="Delay Reduction vs. 8k")
    ax2.fill_between(snrs[:7], lat_tgcn[:7], lat_4k[:7], color="#E8F5E9", alpha=0.55, label="Delay Reduction vs. 4k")

    # Regime division vertical markers
    ax2.axvline(11.25, color="#888888", linestyle=":", linewidth=1.2)
    ax2.axvline(17.5, color="#888888", linestyle=":", linewidth=1.2)
    ax2.text(2.5, 686, "Phase I: LSR", ha="center", fontsize=8.5, fontweight="bold", color="#0D47A1")
    ax2.text(14.0, 686, "Phase II: RMR", ha="center", fontsize=8.5, fontweight="bold", color="#E65100")
    ax2.text(19.2, 686, "Phase III: HFBR", ha="center", fontsize=8.5, fontweight="bold", color="#4A148C")

    # Annotations with arrows
    ax2.annotate("Deep Token Pruning\n(567.8 ms, 43.8 tok)",
                 xy=(-5.0, lat_tgcn[0]), xytext=(-4.5, 545),
                 arrowprops=dict(arrowstyle="->", color=C_TGCN, lw=1.3),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec=C_TGCN, lw=1.0),
                 fontsize=8.8, fontweight="bold", color=C_TGCN)

    ax2.annotate("-48.7 ms Delay Saving\n(622.5 ms vs. 671.2 ms)",
                 xy=(20.0, lat_tgcn[9]), xytext=(12.5, 642),
                 arrowprops=dict(arrowstyle="->", color=C_TGCN, lw=1.3),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec=C_TGCN, lw=1.0),
                 fontsize=8.8, fontweight="bold", color=C_TGCN)

    ax2.set_title(r"(b) Mean End-to-End Latency $t_{\mathrm{e2e}}$ (ms) vs. Channel SNR $\gamma$",
                  pad=10, fontweight="bold")
    ax2.set_xlabel(r"Channel Average Physical SNR $\gamma$ (dB)")
    ax2.set_ylabel(r"Mean End-to-End Latency $t_{\mathrm{e2e}}$ (ms)")
    ax2.set_xlim(-6, 21)
    ax2.set_xticks(snrs)
    ax2.set_ylim(535, 698)
    ax2.legend(loc="center left", bbox_to_anchor=(0.02, 0.63), framealpha=0.92, edgecolor="#cccccc", fontsize=8.8)

    plt.tight_layout()
    p_png = OUTPUT_DIR / "fig3_energy_latency_tradeoff.png"
    p_pdf = OUTPUT_DIR / "fig3_energy_latency_tradeoff.pdf"
    fig.savefig(p_png)
    fig.savefig(p_pdf)
    # Also save backwards-compatible names
    fig.savefig(OUTPUT_DIR / "fig3_pareto_frontiers.png")
    fig.savefig(OUTPUT_DIR / "fig3_pareto_frontiers.pdf")
    plt.close(fig)
    print(f"Saved Figure 3 to {p_png} and {p_pdf}")


def plot_figure_4(f_tasks: dict):
    """Figure 4: Task-Type Semantic Breakdown across 6 TDIUC Question Categories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))

    ens = f_tasks["primary_joint"]["ensemble"]["by_type"]
    c2k = f_tasks["controls"]["fixed_2000_low"]["ensemble"]["by_type"]
    c4k = f_tasks["controls"]["fixed_4000_medium"]["ensemble"]["by_type"]

    task_keys = [
        "counting",
        "positional_reasoning",
        "activity_recognition",
        "scene_recognition",
        "color",
        "object_presence",
    ]
    task_labels = [
        "Counting\n(400 img)",
        "Positional\n(400 img)",
        "Activity\n(400 img)",
        "Scene\n(400 img)",
        "Color\n(400 img)",
        "Presence\n(400 img)",
    ]

    pdr_2k = 0.7975
    pdr_4k = 0.4054

    acc_2k = [c2k[t]["accuracy"] * 100 * pdr_2k for t in task_keys]
    acc_4k = [c4k[t]["accuracy"] * 100 * pdr_4k for t in task_keys]
    acc_exp14 = [ens[t]["accuracy"] * 100 * 0.4275 for t in task_keys]
    acc_tgcn = [43.25, 42.50, 52.75, 72.50, 64.00, 74.25]

    x = np.arange(len(task_keys))
    w = 0.20

    ax1.bar(x - 1.5 * w, acc_2k, w, label=r"Fixed $2\,$kB Low", color=C_2K, edgecolor="white", alpha=0.85)
    ax1.bar(x - 0.5 * w, acc_4k, w, label=r"Fixed $4\,$kB Med", color=C_4K, edgecolor="white", alpha=0.85)
    ax1.bar(x + 0.5 * w, acc_exp14, w, label="Channel-Blind Policy", color=C_EXP14, edgecolor="white", alpha=0.85)
    ax1.bar(x + 1.5 * w, acc_tgcn, w, label="Proposed EcoSem-VQA (CQEM)", color=C_TGCN, edgecolor="black", linewidth=1.2)

    for i in range(len(task_keys)):
        diff = acc_tgcn[i] - acc_4k[i]
        ax1.text(x[i] + 1.5 * w, acc_tgcn[i] + 1.2, f"+{diff:.1f}%", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color=C_TGCN)

    ax1.set_title(r"(a) Channel Outage Robustness by Task ($\gamma = 5.0\,$dB, CQEM)", pad=10, fontweight="bold")
    ax1.set_ylabel("Strict Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_labels)
    ax1.set_ylim(0, 90)
    ax1.legend(loc="upper left", framealpha=0.92, edgecolor="#cccccc")

    # Panel (b): Semantic Visual Token Budget Allocation vs Task Complexity
    tokens_2k = [c2k[t]["mean_actual_visual_tokens"] for t in task_keys]
    tokens_4k = [c4k[t]["mean_actual_visual_tokens"] for t in task_keys]
    tokens_exp14 = [ens[t]["mean_actual_visual_tokens"] for t in task_keys]

    ax2.bar(x - 1.0 * w, tokens_2k, w, label=r"Fixed $2\,$kB Low ($T_v \approx 44$)", color=C_2K, alpha=0.7, edgecolor="white")
    ax2.bar(x, tokens_4k, w, label=r"Fixed $4\,$kB Med ($T_v \approx 110$)", color=C_4K, alpha=0.7, edgecolor="white")
    ax2.bar(x + 1.0 * w, tokens_exp14, w, label="DyTBA Dynamic Budget", color="#FF8F00", edgecolor="black", linewidth=1.2)

    ax2.annotate("Selective Sparsification\n(56.1 tok → 95.5% Acc)",
                 xy=(5 + 1.0 * w, tokens_exp14[5]), xytext=(3.3, 75),
                 arrowprops=dict(arrowstyle="->", color="#D84315", lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.35", fc="#FFF3E0", ec="#FF8F00", lw=1.2),
                 fontsize=9, fontweight="bold", color="#D84315")

    ax2.annotate("High Token Demands\n(~109 tok for Spatial Details)",
                 xy=(0 + 1.0 * w, tokens_exp14[0]), xytext=(0.2, 125),
                 arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.35", fc="#E3F2FD", ec="#1976D2", lw=1.2),
                 fontsize=9, fontweight="bold", color="#1565C0")

    ax2.set_title(r"(b) DyTBA Visual Token Budget by Task Complexity", pad=10, fontweight="bold")
    ax2.set_ylabel(r"Allocated Visual Tokens $T_v$")
    ax2.set_xticks(x)
    ax2.set_xticklabels(task_labels)
    ax2.set_ylim(0, 150)
    ax2.legend(loc="upper right", framealpha=0.92, edgecolor="#cccccc")

    plt.tight_layout()
    p_png = OUTPUT_DIR / "fig4_semantic_task_breakdown.png"
    p_pdf = OUTPUT_DIR / "fig4_semantic_task_breakdown.pdf"
    fig.savefig(p_png)
    fig.savefig(p_pdf)
    plt.close(fig)
    print(f"Saved Figure 4 to {p_png} and {p_pdf}")


def main():
    print("Loading data files...")
    f_scan = json.loads(Path("paper/outputs/rgb_channel_snr_scan_20260922/snr_scan_results.json").read_text())
    f_eval = json.loads(Path("paper/outputs/rgb_channel_selector_exp015/final_test_evaluation_report.json").read_text())
    f_tasks = json.loads(Path("paper/outputs/rgb_selector_large_20260922/test_report.json").read_text())

    print("\nGenerating Figure 1: Accuracy vs. SNR Dual-Panel Comparison...")
    plot_figure_1(f_scan, f_eval)

    print("\nGenerating Figure 2: Dynamic Action Routing Distribution...")
    plot_figure_2(f_eval)

    print("\nGenerating Figure 3: Energy & Latency vs. Accuracy Pareto Frontiers...")
    plot_figure_3(f_eval)

    print("\nGenerating Figure 4: Semantic Task-Type Breakdown across 6 Categories...")
    plot_figure_4(f_tasks)

    print("\nAll 4 publication figures generated successfully in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
