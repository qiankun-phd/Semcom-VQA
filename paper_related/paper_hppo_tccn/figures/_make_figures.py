"""Generate the §VI figure set for the TCCN extension.

⚠️ Preliminary mode: this script ships **synthetic** mock data shaped to
match what we expect MA-HPPO and the five baselines to produce. Real
training logs from D10--D11 will replace the synthetic arrays in-place
without changing the rendering code, so the visual / typography /
caption work done now is not wasted.

Each figure is written to ``figures/<name>.pdf`` (vector). Colors use
the Okabe-Ito colorblind-safe palette. Run from the paper directory:

    python figures/_make_figures.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# --------------------------------------------------------------------------
# Style / colors
# --------------------------------------------------------------------------
OKABE_ITO = {
    "ours":   "#0072B2",  # blue
    "mappo":  "#E69F00",  # orange
    "paddpg": "#009E73",  # bluish green
    "pdqn":   "#D55E00",  # vermillion
    "hppo":   "#CC79A7",  # reddish purple
    "greedy": "#56B4E9",  # sky blue
}
LABELS = {
    "ours":   "MA-HPPO (ours)",
    "mappo":  "MAPPO-hybrid",
    "paddpg": "PADDPG",
    "pdqn":   "PDQN",
    "hppo":   "HPPO ($M{=}1$)",
    "greedy": "Greedy",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

OUT = Path(__file__).parent
RNG = np.random.default_rng(20260508)


# --------------------------------------------------------------------------
# Mock data generators (replace with log readers in D10)
# --------------------------------------------------------------------------
def _conv_curve(steps, asymptote, learning_speed, std_scale, oscillation=0.0):
    """Smooth concave-up convergence + std band + optional oscillation."""
    t = steps / steps[-1]
    mu = asymptote * (1 - np.exp(-learning_speed * t))
    if oscillation > 0:
        mu = mu + oscillation * np.sin(8 * np.pi * t) * (1 - t) ** 1.5
    sd = std_scale * (0.6 + 0.4 * np.exp(-2 * t))
    noise = RNG.standard_normal(steps.shape) * sd * 0.3
    return mu + noise, sd


def fig3_convergence():
    fig, ax = plt.subplots(figsize=(5.0, 3.0), constrained_layout=True)
    steps = np.linspace(0, 150_000, 300)
    specs = {
        "ours":   dict(asymptote=180, learning_speed=4.0, std_scale=8,  oscillation=0,  ls="-"),
        "mappo":  dict(asymptote=140, learning_speed=3.5, std_scale=12, oscillation=0,  ls="--"),
        "hppo":   dict(asymptote=120, learning_speed=3.0, std_scale=10, oscillation=0,  ls="-."),
        "paddpg": dict(asymptote=70,  learning_speed=2.0, std_scale=18, oscillation=12, ls=(0, (3, 1, 1, 1))),
        "pdqn":   dict(asymptote=20,  learning_speed=1.0, std_scale=22, oscillation=18, ls=(0, (5, 2))),
    }
    for k, kw in specs.items():
        ls = kw.pop("ls")
        mu, sd = _conv_curve(steps, **kw)
        ax.plot(steps / 1000, mu, color=OKABE_ITO[k], label=LABELS[k], lw=1.2, ls=ls)
        ax.fill_between(steps / 1000, mu - sd, mu + sd, color=OKABE_ITO[k], alpha=0.15)
    # Greedy = flat reference line at lower asymptote
    ax.axhline(35, color=OKABE_ITO["greedy"], ls=":", lw=1.0, label=LABELS["greedy"])
    ax.set_xlabel("training iteration ($\\times 10^{3}$)")
    ax.set_ylabel("episodic reward")
    ax.set_xlim(0, 150)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.savefig(OUT / "fig3_convergence.pdf")
    plt.close(fig)


def fig4_scaling():
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), constrained_layout=True)
    M_vals = [1, 2, 4]
    N_vals = [3, 6, 9]
    # Joint cost matrix (lower better) for ours and mappo across (M,N).
    base = np.array([[0.42, 0.38, 0.36],
                     [0.34, 0.30, 0.28],
                     [0.30, 0.26, 0.23]])
    ours = base + RNG.standard_normal(base.shape) * 0.005
    mappo = base * 1.18 + RNG.standard_normal(base.shape) * 0.007
    for ax, mat, label in zip(axes, [ours, mappo], ["MA-HPPO (ours)", "MAPPO-hybrid"]):
        im = ax.imshow(mat, cmap="viridis_r", aspect="auto", vmin=0.20, vmax=0.55)
        ax.set_xticks(range(len(N_vals)), labels=[str(n) for n in N_vals])
        ax.set_yticks(range(len(M_vals)), labels=[str(m) for m in M_vals])
        ax.set_xlabel("$N$ (UEs)")
        ax.set_ylabel("$M$ (UAVs)")
        ax.set_title(label)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=8)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.03)
    cbar.set_label("joint cost $O(t)$ ($\\downarrow$)")
    fig.savefig(OUT / "fig4_scaling.pdf")
    plt.close(fig)


def fig5_trajectory():
    fig = plt.figure(figsize=(6.0, 2.8), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")

    # Two UAVs spiraling toward UE clusters.
    t = np.linspace(0, 1, 200)
    uav0_x = 200 * np.cos(2 * np.pi * t) * (1 - t * 0.7) + 100
    uav0_y = 200 * np.sin(2 * np.pi * t) * (1 - t * 0.7) + 50
    uav1_x = -180 * np.cos(2 * np.pi * t + 0.5) * (1 - t * 0.5) - 100
    uav1_y = -180 * np.sin(2 * np.pi * t + 0.5) * (1 - t * 0.5) + 80
    z = np.full_like(t, 50.0)

    # 4 UEs scattered.
    ue_x = np.array([220, 60, -150, -200])
    ue_y = np.array([100, -90, 100, -50])
    ue_z = np.zeros(4) + 1.5

    # 2-D
    ax1.plot(uav0_x, uav0_y, color=OKABE_ITO["ours"], lw=1.0, label="UAV 0")
    ax1.plot(uav1_x, uav1_y, color=OKABE_ITO["mappo"], lw=1.0, label="UAV 1")
    ax1.scatter(uav0_x[0], uav0_y[0], marker="o", color=OKABE_ITO["ours"], s=18)
    ax1.scatter(uav0_x[-1], uav0_y[-1], marker="*", color=OKABE_ITO["ours"], s=70)
    ax1.scatter(uav1_x[0], uav1_y[0], marker="o", color=OKABE_ITO["mappo"], s=18)
    ax1.scatter(uav1_x[-1], uav1_y[-1], marker="*", color=OKABE_ITO["mappo"], s=70)
    ax1.scatter(ue_x, ue_y, marker="^", color="black", s=24, label="UEs")
    ax1.scatter(0, 0, marker="s", color=OKABE_ITO["pdqn"], s=40, label="BS")
    ax1.set_xlabel("$x$ (m)")
    ax1.set_ylabel("$y$ (m)")
    ax1.set_aspect("equal")
    ax1.legend(loc="upper right", frameon=False, fontsize=7)
    ax1.set_title("(a) 2-D trajectory")

    # 3-D
    ax2.plot(uav0_x, uav0_y, z, color=OKABE_ITO["ours"], lw=1.0)
    ax2.plot(uav1_x, uav1_y, z, color=OKABE_ITO["mappo"], lw=1.0)
    ax2.scatter(ue_x, ue_y, ue_z, color="black", marker="^", s=18)
    ax2.scatter([0], [0], [15], color=OKABE_ITO["pdqn"], marker="s", s=30)
    ax2.set_xlabel("$x$ (m)")
    ax2.set_ylabel("$y$ (m)")
    ax2.set_zlabel("$z$ (m)")
    ax2.set_title("(b) 3-D trajectory")
    ax2.view_init(elev=20, azim=-45)

    fig.savefig(OUT / "fig5_trajectory.pdf")
    plt.close(fig)


def fig6_similarity_cdf():
    fig, ax = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)
    n = 2000
    x = np.linspace(0, 1, 256)
    samples = {
        "ours":   np.clip(RNG.beta(8, 2.5, n), 0, 1),
        "mappo":  np.clip(RNG.beta(5, 3, n), 0, 1),
        "hppo":   np.clip(RNG.beta(4, 3, n), 0, 1),
        "paddpg": np.clip(RNG.beta(3, 3.5, n), 0, 1),
        "pdqn":   np.clip(RNG.beta(2.0, 6, n), 0, 1),
    }
    for k, s in samples.items():
        s_sorted = np.sort(s)
        cdf = np.linspace(0, 1, len(s_sorted))
        ax.plot(s_sorted, cdf, color=OKABE_ITO[k], label=LABELS[k], lw=1.3)
    ax.axvline(0.5, color="grey", ls="--", lw=0.8, alpha=0.7)
    ax.text(0.515, 0.05, "$\\xi_{\\mathrm{th}}{=}0.5$", color="grey", fontsize=8)
    ax.set_xlabel("per-task semantic similarity $\\xi$")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(OUT / "fig6_similarity_cdf.pdf")
    plt.close(fig)


def fig7_sensitivity():
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), constrained_layout=True)
    # (a) alpha sweep
    alphas = np.linspace(0, 1, 11)
    cost = 0.30 + 0.05 * np.cos(np.pi * alphas) + RNG.standard_normal(11) * 0.005
    delay = 1 - alphas + RNG.standard_normal(11) * 0.02
    energy = alphas + RNG.standard_normal(11) * 0.02
    ax = axes[0]
    ax.plot(alphas, cost, "o-", color=OKABE_ITO["ours"], label="$O(t)$", lw=1.0)
    ax.plot(alphas, delay, "s--", color=OKABE_ITO["paddpg"], label="$\\widetilde{T}$", lw=1.0, alpha=0.85)
    ax.plot(alphas, energy, "^--", color=OKABE_ITO["pdqn"], label="$\\widetilde{E}$", lw=1.0, alpha=0.85)
    ax.set_xlabel("trade-off $\\alpha$")
    ax.set_ylabel("normalized metric")
    ax.set_title("(a) sensitivity to $\\alpha$")
    ax.legend(loc="center right", frameon=False)
    # (b) bandwidth sweep
    Bs = np.array([1, 2, 5, 10, 20]) * 1e6
    cost_b = 0.55 - 0.06 * np.log10(Bs / 1e6) + RNG.standard_normal(5) * 0.01
    sim_b = 0.5 + 0.08 * np.log10(Bs / 1e6) + RNG.standard_normal(5) * 0.005
    sr_b = np.clip(0.45 + 0.12 * np.log10(Bs / 1e6) + RNG.standard_normal(5) * 0.01, 0, 1)
    ax = axes[1]
    ax.plot(Bs / 1e6, cost_b, "o-", color=OKABE_ITO["ours"], label="$O(t)$", lw=1.0)
    ax.plot(Bs / 1e6, sim_b, "s--", color=OKABE_ITO["mappo"], label="$\\overline{\\xi}$", lw=1.0, alpha=0.85)
    ax.plot(Bs / 1e6, sr_b, "^--", color=OKABE_ITO["hppo"], label="$\\mathrm{SR}$", lw=1.0, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("$B_c$ per channel (MHz)")
    ax.set_ylabel("metric")
    ax.set_title("(b) sensitivity to $B_c$")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(OUT / "fig7_sensitivity.pdf")
    plt.close(fig)


def fig8_robustness():
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), constrained_layout=True)
    # (a) channel noise dB
    ax = axes[0]
    perturb = np.array([0, 2, 5, 8, 12])
    for k in ["ours", "mappo", "hppo", "paddpg"]:
        slope = {"ours": 0.06, "mappo": 0.085, "hppo": 0.10, "paddpg": 0.14}[k]
        intercept = {"ours": 0.30, "mappo": 0.36, "hppo": 0.40, "paddpg": 0.45}[k]
        y = intercept + slope * perturb / 12 + RNG.standard_normal(perturb.shape) * 0.005
        ax.plot(perturb, y, "o-", color=OKABE_ITO[k], label=LABELS[k], lw=1.0)
    ax.set_xlabel("channel noise perturbation $\\Delta$ (dB)")
    ax.set_ylabel("joint cost $O(t)$ ($\\downarrow$)")
    ax.set_title("(a) channel-noise robustness")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    # (b) UE-count generalization
    ax = axes[1]
    Ns = np.array([3, 4, 5, 6, 7, 8, 9])
    for k in ["ours", "mappo", "hppo", "paddpg"]:
        slope = {"ours": 0.012, "mappo": 0.022, "hppo": 0.030, "paddpg": 0.045}[k]
        base = {"ours": 0.30, "mappo": 0.34, "hppo": 0.38, "paddpg": 0.43}[k]
        y = base + slope * (Ns - 4) + RNG.standard_normal(Ns.shape) * 0.005
        ax.plot(Ns, y, "o-", color=OKABE_ITO[k], label=LABELS[k], lw=1.0)
    ax.axvline(4, color="grey", ls="--", lw=0.8, alpha=0.7)
    ax.text(4.1, 0.27, "trained at $N{=}4$", color="grey", fontsize=7)
    ax.set_xlabel("evaluation $N$ (UEs)")
    ax.set_ylabel("joint cost $O(t)$ ($\\downarrow$)")
    ax.set_title("(b) UE-count generalization")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    fig.savefig(OUT / "fig8_robustness.pdf")
    plt.close(fig)


def fig9_jamming():
    fig, ax = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)
    jam_db = np.array([0, 3, 6, 9, 12, 15])
    for k in ["ours", "mappo", "hppo", "paddpg", "pdqn"]:
        # success rate vs jamming
        sr_floor = {"ours": 0.86, "mappo": 0.74, "hppo": 0.64, "paddpg": 0.42, "pdqn": 0.18}[k]
        decay = {"ours": 0.025, "mappo": 0.040, "hppo": 0.050, "paddpg": 0.055, "pdqn": 0.012}[k]
        y = sr_floor - decay * jam_db + RNG.standard_normal(jam_db.shape) * 0.012
        y = np.clip(y, 0, 1)
        ax.plot(jam_db, y, "o-", color=OKABE_ITO[k], label=LABELS[k], lw=1.0)
    ax.set_xlabel("jamming strength (dBm)")
    ax.set_ylabel("success rate $\\mathrm{SR}$ ($\\uparrow$)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", frameon=False, fontsize=7, ncol=2)
    fig.savefig(OUT / "fig9_jamming.pdf")
    plt.close(fig)


def main():
    print("Rendering figures into", OUT)
    for fn in [fig3_convergence, fig4_scaling, fig5_trajectory,
               fig6_similarity_cdf, fig7_sensitivity, fig8_robustness,
               fig9_jamming]:
        print(f"  - {fn.__name__}")
        fn()
    print("Done.")


if __name__ == "__main__":
    main()
