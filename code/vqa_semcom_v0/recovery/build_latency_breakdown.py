#!/usr/bin/env python3
"""Latency breakdown per method x SNR (GO-SG style): upload + tx-side sensing + inference.

  * upload_s   : airtime = mean complex channel uses / bandwidth (from comparison csv,
                 which charges every method in the same channel-use currency);
  * txside_s   : transmitter-side semantic extraction (detector) latency, token path only;
  * inference_s: receiver-side VLM/LLM answer latency (measured in the eval runs).

Reads the tidy comparison csv (for channel uses) + the raw prediction csvs (for
measured latencies), reports the TEST split only, and draws F3 stacked bars.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_comparison_v2 as bc  # noqa: E402  (reuse split/policy/uses logic)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def mean_latencies_by_service(pred_path, test_frac):
    """(service, snr_bin) -> (mean latency_sec, mean detector_latency_sec)."""
    agg = defaultdict(lambda: [0.0, 0.0, 0])
    for r in csv.DictReader(open(pred_path)):
        if not bc.is_test(r["image_id"], test_frac):
            continue
        key = (r["service_level"], r["snr_bin"])
        a = agg[key]
        a[0] += float(r.get("latency_sec") or 0.0)
        a[1] += float(r.get("detector_latency_sec") or 0.0)
        a[2] += 1
    return {k: (v[0] / v[2], v[1] / v[2]) for k, v in agg.items() if v[2]}


def m4_service_shares(pred_path, test_frac):
    """snr_bin -> fraction of test tasks routed to s1 by the learned policy."""
    tasks, qtype = bc.load_channel(pred_path)
    pol = bc.learn_policy(tasks, qtype, split_test=False)
    cnt = defaultdict(lambda: [0, 0])
    for key, svcs in tasks.items():
        if not bc.is_test(key[0], test_frac):
            continue
        snr = key[2]
        sel = pol.get((qtype[key], snr))
        if sel in svcs:
            cnt[snr][1] += 1
            cnt[snr][0] += int(sel == "1")
    return {snr: (k / n if n else 0.0) for snr, (k, n) in cnt.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison-csv", default="outputs/reports/comparison_all.csv")
    ap.add_argument("--pred-dir", default="outputs/vlm")
    ap.add_argument("--prefix", default="v2_0")
    ap.add_argument("--naive-prefix", default="v2_0")
    ap.add_argument("--channel", default="rayleigh")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out", default="outputs/reports/latency_breakdown.csv")
    ap.add_argument("--out-dir", default="outputs/figures/comparison")
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()
    ch = args.channel

    # channel uses per query from the tidy csv (qtype=all)
    uses = {}
    for r in csv.DictReader(open(args.comparison_csv)):
        if r["channel"] == ch and r["qtype"] == "all":
            uses[(r["method"], float(r["snr_db"]))] = float(r.get("mean_channel_uses") or 0.0)

    main_pred = f"{args.pred_dir}/{args.prefix}_{ch}_predictions.csv"
    lat = mean_latencies_by_service(main_pred, args.test_frac)
    shares = m4_service_shares(main_pred, args.test_frac)

    rows = []  # channel, method, snr_db, upload_s, txside_s, inference_s, total_s

    def emit(method, snr_bin, upload, txside, inference):
        rows.append([ch, method, bc.snr_val(snr_bin), round(upload, 4), round(txside, 4),
                     round(inference, 4), round(upload + txside + inference, 4)])

    snr_bins = sorted({k[1] for k in lat}, key=bc.snr_val)
    for snr in snr_bins:
        s = bc.snr_val(snr)
        if ("2", snr) in lat:
            inf, _ = lat[("2", snr)]
            emit("M1_image", snr, uses.get(("M1_image", s), 0.0) / bc.BANDWIDTH_HZ, 0.0, inf)
        if ("1", snr) in lat:
            inf, det = lat[("1", snr)]
            emit("M3_token", snr, uses.get(("M3_token", s), 0.0) / bc.BANDWIDTH_HZ, det, inf)
        if ("1", snr) in lat and ("2", snr) in lat:
            w1 = shares.get(snr, 0.0)
            inf1, det1 = lat[("1", snr)]
            inf2, _ = lat[("2", snr)]
            emit("M4_adaptive", snr, uses.get(("M4_adaptive", s), 0.0) / bc.BANDWIDTH_HZ,
                 w1 * det1, w1 * inf1 + (1 - w1) * inf2)

    m2 = f"{args.pred_dir}/m2_analog_{ch}_predictions.csv"
    if os.path.exists(m2):
        agg = defaultdict(lambda: [0.0, 0])
        for r in csv.DictReader(open(m2)):
            if not bc.is_test(r["image_id"], args.test_frac):
                continue
            a = agg[r["snr_bin"]]; a[0] += float(r.get("latency_sec") or 0.0); a[1] += 1
        for snr, (tot, n) in sorted(agg.items(), key=lambda kv: bc.snr_val(kv[0])):
            emit("M2_analog", snr, uses.get(("M2_analog", bc.snr_val(snr)), 0.0) / bc.BANDWIDTH_HZ,
                 0.0, tot / n)

    nv = f"{args.pred_dir}/{args.naive_prefix}_{ch}_naive_predictions.csv"
    if os.path.exists(nv):
        agg = defaultdict(lambda: [0.0, 0])
        for r in csv.DictReader(open(nv)):
            if r.get("service_level") != "2" or not bc.is_test(r["image_id"], args.test_frac):
                continue
            a = agg[r["snr_bin"]]; a[0] += float(r.get("latency_sec") or 0.0); a[1] += 1
        for snr, (tot, n) in sorted(agg.items(), key=lambda kv: bc.snr_val(kv[0])):
            emit("M0_naive", snr, uses.get(("M0_naive", bc.snr_val(snr)), 0.0) / bc.BANDWIDTH_HZ,
                 0.0, tot / n)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "method", "snr_db", "upload_s", "txside_s", "inference_s", "total_s"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")

    # ---- F3: stacked latency bars per method across SNR ----
    import numpy as np
    methods = [m for m in ("M0_naive", "M1_image", "M2_analog", "M3_token", "M4_adaptive")
               if any(r[1] == m for r in rows)]
    snrs = sorted({r[2] for r in rows})
    style = {
        "M0_naive": "#ff6b6b", "M1_image": "#ffb454", "M2_analog": "#c678dd",
        "M3_token": "#9aa7b4", "M4_adaptive": "#5ad19a",
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(snrs)); w = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        ups, txs, infs = [], [], []
        for s in snrs:
            rr = [r for r in rows if r[1] == m and r[2] == s]
            ups.append(rr[0][3] if rr else 0); txs.append(rr[0][4] if rr else 0)
            infs.append(rr[0][5] if rr else 0)
        base = np.array(ups); mid = np.array(txs)
        ax.bar(x + i * w, ups, w, color=style[m], label=m.replace("_", " "))
        ax.bar(x + i * w, txs, w, bottom=base, color=style[m], alpha=0.55, hatch="//")
        ax.bar(x + i * w, infs, w, bottom=base + mid, color=style[m], alpha=0.30, hatch="..")
    ax.set_yscale("log")
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels([f"{s:g}" for s in snrs])
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("latency per query (s, log)")
    ax.set_title(f"End-to-end latency breakdown ({ch}); solid=upload, //=tx-side detector, ..=inference")
    ax.grid(True, alpha=0.3, axis="y", which="both"); ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_dir}/F3_latency_{args.tag}.{ext}", dpi=140)
    print(f"wrote F3_latency_{args.tag} -> {args.out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
