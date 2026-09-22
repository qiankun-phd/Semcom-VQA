#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# from build_evidence_complementarity output (pooled 3ch, test)
qt = ["counting", "co_presence", "comparison", "threshold", "presence"]
delta = [0.194, 0.122, 0.111, 0.021, -0.068]
reasoning = ["symbolic", "symbolic", "symbolic", "symbolic", "perceptual"]
colors = ["#5ad19a" if r == "symbolic" else "#ffb454" for r in reasoning]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
bars = ax.bar(qt, delta, color=colors, edgecolor="#2a3340")
ax.axhline(0, color="#2a3340", lw=1)
for b, d in zip(bars, delta):
    ax.text(b.get_x() + b.get_width()/2, d + (0.008 if d >= 0 else -0.018),
            f"{d:+.3f}", ha="center", fontsize=9)
ax.set_ylabel("token-gain  Δ = acc(token) − acc(image)")
ax.set_title("Evidence–question complementarity (pooled AWGN/Rayleigh/Rician, test)")
ax.text(0.02, 0.95, "Δ>0 → detector token wins\nΔ<0 → image wins",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="#f2f6ff", ec="#4ea1ff"))
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#5ad19a", label="symbolic (count-based)"),
                   Patch(color="#ffb454", label="perceptual (existence)")], loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"outputs/figures/comparison/F6_complementarity.{ext}", dpi=140)
print("wrote F6_complementarity")
