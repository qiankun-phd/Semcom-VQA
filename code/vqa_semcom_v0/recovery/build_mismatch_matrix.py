#!/usr/bin/env python3
"""P3: policy-level CSI-mismatch matrix (assumed SNR x true SNR).

The M4 selector picks a service from the policy cell (qtype, SNR_assumed)
while the outcome is read from the prediction logs at SNR_true.  The
physical link itself stays rate-adaptive at the true SNR, so the matrix
isolates the SCHEDULER's robustness to CSI error (stale/erroneous SNR at
selection time).  Fully offline: every (task, snr, service) outcome is
already logged.  Reference protocol: ADJSCC eval always feeds the true
test SNR to the conditioning input and never measures this axis.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import build_comparison_v2 as bc  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="outputs/vlm")
    ap.add_argument("--prefix", default="v3_0")
    ap.add_argument("--out", default="outputs/reports/mismatch_matrix.csv")
    ap.add_argument("--out-dir", default="outputs/figures/comparison")
    ap.add_argument("--tag", default="v3")
    args = ap.parse_args()

    all_rows = []
    mats = {}  # channel -> (snrs, matrix of acc for qtype=all)
    for ch in bc.CHANNELS:
        pred = f"{args.pred_dir}/{args.prefix}_{ch}_predictions.csv"
        if not os.path.exists(pred):
            print(f"skip {ch}: no {pred}")
            continue
        tasks, qtype = bc.load_channel(pred)
        pol = bc.learn_policy(tasks, qtype, split_test=False)
        snrs = sorted({k[2] for k in tasks}, key=bc.snr_val)

        # test tasks indexed by (image, question) -> {snr_bin: services}
        by_iq = defaultdict(dict)
        for key, svcs in tasks.items():
            if bc.is_test(key[0]):
                by_iq[(key[0], key[1])][key[2]] = svcs

        mat = [[None] * len(snrs) for _ in snrs]
        for ia, sa in enumerate(snrs):
            for it, st in enumerate(snrs):
                agg = defaultdict(lambda: [0, 0, 0, 0.0])  # qbucket -> k,n,bytes,uses
                for (img, q), snr_map in by_iq.items():
                    if st not in snr_map:
                        continue
                    svcs = snr_map[st]
                    qt = qtype[(img, q, st)]
                    sel = pol.get((qt, sa))
                    if sel is None or sel not in svcs:
                        continue
                    c, b = svcs[sel]
                    for qb in ("all", qt):
                        a = agg[qb]
                        a[0] += int(c); a[1] += 1; a[2] += b
                        a[3] += bc.uses_digital(sel, ch, bc.snr_val(st), b)
                for qb, (k, n, b, u) in sorted(agg.items()):
                    if not n:
                        continue
                    all_rows.append([ch, bc.snr_val(sa), bc.snr_val(st), qb,
                                     round(k / n, 4), n, round(b / n, 1), round(u / n, 1)])
                    if qb == "all":
                        mat[ia][it] = k / n
        mats[ch] = (snrs, mat)
        diag = [mat[i][i] for i in range(len(snrs))]
        off = [mat[i][j] for i in range(len(snrs)) for j in range(len(snrs))
               if i != j and mat[i][j] is not None]
        print(f"[{ch}] diagonal (matched CSI): "
              + " ".join(f"{v:.3f}" for v in diag)
              + f" | worst off-diagonal: {min(off):.3f} | mean off-diag: {sum(off)/len(off):.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "snr_assumed_db", "snr_true_db", "qtype",
                    "accuracy", "n", "mean_payload_bytes", "mean_channel_uses"])
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows -> {args.out}")

    # ---- F7: heatmaps (qtype=all), one panel per channel ----
    chs = [c for c in bc.CHANNELS if c in mats]
    fig, axes = plt.subplots(1, len(chs), figsize=(4.6 * len(chs), 4.2), squeeze=False)
    for ax, ch in zip(axes[0], chs):
        snrs, mat = mats[ch]
        vals = [[v if v is not None else float("nan") for v in row] for row in mat]
        im = ax.imshow(vals, origin="lower", cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(snrs))); ax.set_xticklabels([f"{bc.snr_val(s):g}" for s in snrs])
        ax.set_yticks(range(len(snrs))); ax.set_yticklabels([f"{bc.snr_val(s):g}" for s in snrs])
        ax.set_xlabel("true SNR (dB)"); ax.set_ylabel("assumed SNR at selection (dB)")
        ax.set_title({"awgn": "AWGN", "rayleigh": "Rayleigh", "rician": "Rician K=6 dB"}.get(ch, ch))
        for i in range(len(snrs)):
            for j in range(len(snrs)):
                if vals[i][j] == vals[i][j]:
                    ax.text(j, i, f"{vals[i][j]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if vals[i][j] < 0.64 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Policy robustness to CSI error: accuracy under assumed-vs-true SNR "
                 "(test set, link stays rate-adaptive at true SNR)", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_dir}/F7_mismatch_{args.tag}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote F7_mismatch_{args.tag} -> {args.out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
