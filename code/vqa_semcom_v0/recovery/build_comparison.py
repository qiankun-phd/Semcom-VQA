#!/usr/bin/env python3
"""Build the head-to-head comparison (M1/M3/M4/M5) from a predictions CSV.

Methods (organised per the comparison plan):
  M1 traditional SSCC      = service s2 (rate-adaptive JPEG over the link)
  M3 GO-SG digital token   = service s1 (detector evidence token)
  M4 ours (adaptive)       = per-(question_type, snr) pick the service whose LUT
                             Wilson-LCB accuracy is highest, then score the ACTUAL
                             per-task correctness of the chosen service
  M5 oracle (upper bound)  = per-task best service (genie)
  M0 cache (if s0 present)  reported separately

Run on the pilot now to validate the logic; re-run on the full CSV later.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

SERVICE_TO_METHOD = {"1": "M3_token", "2": "M1_image", "0": "M0_cache", "3": "M3b_roi"}


def wilson_lcb(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def is_correct(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def snr_key(label: str) -> float:
    return float(str(label).replace("dB", "").replace("db", ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out-prefix", default="outputs/reports/comparison")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.predictions)))
    # group by task key (image_id, question, snr_bin) -> {service: (correct, payload)}
    tasks = defaultdict(dict)
    qtype_of = {}
    for r in rows:
        key = (r["image_id"], r["question"], r["snr_bin"])
        svc = r["service_level"]
        tasks[key][svc] = (is_correct(r["correct"]), int(r.get("payload_bytes") or 0))
        qtype_of[key] = r.get("question_type", "")

    # --- build LUT policy: per (question_type, snr_bin) pick best-LCB service ---
    cell = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # (qtype,snr)->svc->[k,n]
    for key, svcs in tasks.items():
        qt, snr = qtype_of[key], key[2]
        for svc, (corr, _) in svcs.items():
            if svc in ("1", "2"):  # policy chooses among token/image
                cell[(qt, snr)][svc][1] += 1
                cell[(qt, snr)][svc][0] += int(corr)
    policy = {}
    for cs, svcd in cell.items():
        best_svc, best_lcb = None, -1.0
        for svc, (k, n) in svcd.items():
            lcb = wilson_lcb(k, n)
            if lcb > best_lcb:
                best_lcb, best_svc = lcb, svc
        policy[cs] = best_svc

    # --- aggregate accuracy per snr per method ---
    snrs = sorted({k[2] for k in tasks}, key=snr_key)
    methods = ["M1_image", "M3_token", "M4_adaptive", "M5_oracle"]
    agg = {m: defaultdict(lambda: [0, 0, 0]) for m in methods}  # method->snr->[k,n,bytes]

    for key, svcs in tasks.items():
        qt, snr = qtype_of[key], key[2]
        # M1 image (s2), M3 token (s1)
        if "2" in svcs:
            c, b = svcs["2"]; agg["M1_image"][snr][0] += int(c); agg["M1_image"][snr][1] += 1; agg["M1_image"][snr][2] += b
        if "1" in svcs:
            c, b = svcs["1"]; agg["M3_token"][snr][0] += int(c); agg["M3_token"][snr][1] += 1; agg["M3_token"][snr][2] += b
        # M4 adaptive: chosen service by policy
        chosen = policy.get((qt, snr))
        if chosen in svcs:
            c, b = svcs[chosen]; agg["M4_adaptive"][snr][0] += int(c); agg["M4_adaptive"][snr][1] += 1; agg["M4_adaptive"][snr][2] += b
        # M5 oracle: best over {s1,s2}
        cand = [(svcs[s][0], svcs[s][1]) for s in ("1", "2") if s in svcs]
        if cand:
            best = max(cand, key=lambda t: int(t[0]))
            agg["M5_oracle"][snr][0] += int(best[0]); agg["M5_oracle"][snr][1] += 1
            # oracle "cost" = cost of whichever service it picked
            agg["M5_oracle"][snr][2] += best[1]

    # --- print table ---
    print("policy[(qtype,snr)] -> chosen service:")
    for cs in sorted(policy, key=lambda x: (x[0], snr_key(x[1]))):
        print("  %-10s %-6s -> s%s" % (cs[0], cs[1], policy[cs]))
    print("\naccuracy by method x SNR (n per cell, mean payload bytes):")
    hdr = "%-13s" % "SNR" + "".join("%14s" % s for s in snrs)
    print(hdr)
    for m in methods:
        line = "%-13s" % m
        for s in snrs:
            k, n, b = agg[m][s]
            line += "%14s" % (f"{k/n:.3f}" if n else "-")
        print(line)
    print("\nmean payload bytes by method x SNR:")
    print(hdr)
    for m in methods:
        line = "%-13s" % m
        for s in snrs:
            k, n, b = agg[m][s]
            line += "%14s" % (f"{b/n:.0f}" if n else "-")
        print(line)

    # write tidy CSV for figures
    out = args.out_prefix + "_by_snr.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "snr_db", "accuracy", "n", "mean_payload_bytes"])
        for m in methods:
            for s in snrs:
                k, n, b = agg[m][s]
                if n:
                    w.writerow([m, snr_key(s), round(k / n, 4), n, round(b / n, 1)])
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
