"""Render the MA-HPPO architecture diagram as fig2_arch.pdf.

Block-diagram in matplotlib (vector PDF, IEEEtran-compatible).
Run: python3 _make_arch.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent
plt.rcParams.update({
    "font.family": "serif",
    "font.size":   8,
    "axes.linewidth": 0.6,
})

# Okabe-Ito accents kept consistent with fig3_convergence.
C_TRUNK   = "#0072B2"   # blue
C_DISC    = "#E69F00"   # orange
C_CONT    = "#009E73"   # bluish green
C_CRITIC  = "#CC79A7"   # reddish purple
C_ENV     = "#56B4E9"   # sky blue


def _box(ax, xy, wh, text, fc, ec="black"):
    x, y = xy
    w, h = wh
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=0.7, edgecolor=ec, facecolor=fc, alpha=0.85,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)


def _arrow(ax, xy0, xy1, ls="-"):
    a = FancyArrowPatch(
        xy0, xy1,
        arrowstyle="-|>", mutation_scale=8,
        linewidth=0.6, color="black", linestyle=ls, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(6.4, 2.8), constrained_layout=True)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    # Left: environment + state
    _box(ax, (0.1, 1.7), (1.4, 1.2),
         "Multi-UAV\nDec-POMDP\nenv (§III)", C_ENV)
    ax.text(0.8, 1.55, "$s_t$", ha="center", va="top", fontsize=8.5)

    # Trunk encoder
    _box(ax, (2.0, 1.7), (1.6, 1.2),
         "Shared trunk\n$\\phi_\\psi$\nMLP $L\\!\\times\\!H$", C_TRUNK)

    # Discrete head (top branch)
    _box(ax, (4.4, 3.3), (2.0, 1.0),
         "Discrete heads\n(Dueling, $|\\mathcal{A}^{d}|=2(M\\!+\\!N)$)", C_DISC)
    ax.text(5.4, 4.45, "role-aware (MA-HPPO) /", ha="center", fontsize=6.5)
    ax.text(5.4, 4.25, "shared+index $e_m$ (MAPPO-hybrid)",
            ha="center", fontsize=6.5)

    # Continuous head (middle branch)
    _box(ax, (4.4, 1.7), (2.0, 1.0),
         "Continuous head\nReparam.\n$\\mathcal{N}(\\mu,\\sigma^{2}I)$", C_CONT)

    # Critic (bottom branch)
    _box(ax, (4.4, 0.2), (2.0, 1.0),
         "Critic $V_\\phi$\n(centralized,\n global state)", C_CRITIC)

    # Action / environment closure
    _box(ax, (7.0, 1.7), (1.7, 1.2),
         "Joint action\n$a_t=(a^d_t,a^c_t)$", C_ENV)
    _box(ax, (8.95, 1.7), (1.0, 1.2), "reward\n$r_t$", C_ENV)

    # Arrows
    _arrow(ax, (1.5, 2.3), (2.0, 2.3))                 # env -> trunk
    _arrow(ax, (3.6, 2.6), (4.4, 3.7))                 # trunk -> discrete
    _arrow(ax, (3.6, 2.3), (4.4, 2.2))                 # trunk -> continuous
    _arrow(ax, (3.6, 2.0), (4.4, 0.8))                 # trunk -> critic
    _arrow(ax, (6.4, 3.7), (7.0, 2.5))                 # disc -> action
    _arrow(ax, (6.4, 2.2), (7.0, 2.2))                 # cont -> action
    _arrow(ax, (8.7, 2.3), (8.95, 2.3))                # action -> reward
    # feedback dashed
    _arrow(ax, (9.45, 1.7), (9.45, 0.5), ls="--")
    _arrow(ax, (9.45, 0.5), (0.8, 0.5), ls="--")
    _arrow(ax, (0.8, 0.5), (0.8, 1.7), ls="--")
    ax.text(5.0, 0.35, "feedback to next slot $t{+}1$", fontsize=6.5,
            ha="center", style="italic", color="#444")

    # CTDE annotation
    ax.annotate(
        "CTDE: trunk + critic centralized; heads decentralized at exec.",
        xy=(5.0, 4.85), ha="center", fontsize=7, color="#222",
    )

    out = OUT / "fig2_arch.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
