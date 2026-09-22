#!/usr/bin/env python3
"""Multi-channel comparison with train/test split + per-question-type breakdown.

For each channel:
  * group predictions by task (image_id, question, snr) -> {service: (correct, bytes)}
  * split tasks by image_id into train (policy learning) / test (reporting), 20% test
  * learn M4 policy on TRAIN: per (question_type, snr) the service with best Wilson-LCB
  * report on TEST: M1 (s2) / M3 (s1) / M4 (adaptive) / M5 (oracle) accuracy + payload
  * merge M2 analog (if its csv exists) -> accuracy per snr (+qtype)
  * merge M0_naive fixed-rate digital (cliff baseline) and M0_errorfree clean upper bound
Outputs one tidy long CSV consumed by the figure scripts.

Bandwidth accounting (unified across analog/digital so the Pareto axis is fair):
  * every method is charged in COMPLEX CHANNEL USES per query;
  * s2 rate-adaptive digital: uses = bits / ergodic_spectral_efficiency(snr, fading)
    (the link fits the JPEG to the achievable rate; below the JPEG floor the
    payload exceeds one slot and the extra airtime is charged honestly);
  * s1 token and M0_naive fixed-rate digital: LDPC r=1/2 x BPSK -> 0.5 bit/use;
    for M0_naive the transmitter always sends the FULL image (delivered bytes
    shrink at low SNR), so uses are charged on the per-image original size,
    approximated by the max delivered bytes across SNR bins;
  * M2 analog: native channel_uses column (one slot = B*tau = 300k uses);
  * M0_errorfree: ideal infinite-capacity channel -> uses reported as 0.
  CBR = mean complex channel uses / mean source dimension (3*H*W).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import zlib
from collections import defaultdict

CHANNELS = ["awgn", "rayleigh", "rician"]

SE_TOKEN = 0.5  # bits per complex use: LDPC rate 1/2 x BPSK (digital_link defaults)
MEAN_SOURCE_REALS = 2_882_890.0  # mean 3*H*W over the 548 VisDrone-DET val images
BANDWIDTH_HZ = 1.0e6  # LinkConfig.bandwidth_hz (one complex use = 1/B seconds)

_SE_CACHE: dict[tuple[str, float], float] = {}


def se_s2(ch: str, snr) -> float:
    """Ergodic spectral efficiency (bits/complex use) of the rate-adaptive s2 link."""
    key = (ch, float(snr))
    if key not in _SE_CACHE:
        sys.path.insert(0, "src")
        from vqa_semcom.degradation.digital_link import FadingConfig, ergodic_spectral_efficiency
        _SE_CACHE[key] = max(1e-6, ergodic_spectral_efficiency(float(snr), FadingConfig(kind=ch)))
    return _SE_CACHE[key]


def uses_digital(service: str, ch: str, snr, payload_bytes: float) -> float:
    if service == "2":
        return payload_bytes * 8.0 / se_s2(ch, snr)
    return payload_bytes * 8.0 / SE_TOKEN


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def correct(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def snr_val(s):
    return float(str(s).replace("dB", "").replace("db", ""))


def is_test(image_id, frac=0.2):
    return (zlib.crc32(str(image_id).encode()) % 100) < int(frac * 100)


def load_channel(path):
    """task key -> {service: (correct, bytes)}, plus qtype map."""
    tasks = defaultdict(dict)
    qtype = {}
    for r in csv.DictReader(open(path)):
        key = (r["image_id"], r["question"], r["snr_bin"])
        tasks[key][r["service_level"]] = (correct(r["correct"]), int(r.get("payload_bytes") or 0))
        qtype[key] = r.get("question_type", "")
    return tasks, qtype


def learn_policy(tasks, qtype, split_test):
    """per (qtype, snr) -> best-LCB service among {1,2}, learned on TRAIN tasks."""
    cell = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for key, svcs in tasks.items():
        if is_test(key[0]) == split_test:  # learn on train only -> split_test=False
            continue
        cs = (qtype[key], key[2])
        for s in ("1", "2"):
            if s in svcs:
                cell[cs][s][1] += 1
                cell[cs][s][0] += int(svcs[s][0])
    pol = {}
    for cs, sv in cell.items():
        best, blcb = None, -1
        for s, (k, n) in sv.items():
            _, lcb, _ = wilson(k, n)
            if lcb > blcb:
                blcb, best = lcb, s
        pol[cs] = best
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="outputs/vlm")
    ap.add_argument("--prefix", default="v2_0")
    ap.add_argument("--naive-prefix", default="v2_0",
                    help="prefix of the fixed-rate naive predictions (only run on the main task set)")
    ap.add_argument("--clean-file", default=None,
                    help="clean (error-free) predictions csv; default <pred-dir>/<prefix>_clean_predictions.csv")
    ap.add_argument("--out", default="outputs/reports/comparison_all.csv")
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    out_rows = []  # channel, method, snr_db, qtype, split, accuracy, n, lcb, ucb, bytes, uses, cbr

    def emit(ch, method, snr, qt, split, k, n, b, u):
        if n == 0:
            return
        p, lcb, ucb = wilson(k, n)
        mean_u = u / n
        out_rows.append([ch, method, snr_val(snr), qt, split, round(p, 4), n,
                         round(lcb, 4), round(ucb, 4), round(b / n, 1),
                         round(mean_u, 1), round(mean_u / MEAN_SOURCE_REALS, 6)])

    clean_default = f"{args.pred_dir}/{args.prefix}_clean_predictions.csv"
    clean_path = args.clean_file or clean_default

    for ch in CHANNELS:
        pred = f"{args.pred_dir}/{args.prefix}_{ch}_predictions.csv"
        if not os.path.exists(pred):
            print(f"skip {ch}: no {pred}")
            continue
        tasks, qtype = load_channel(pred)
        pol = learn_policy(tasks, qtype, split_test=False)
        snrs = sorted({k[2] for k in tasks}, key=snr_val)
        # accumulators: method -> snr -> qt -> [k, n, bytes, uses]
        acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0.0])))
        for key, svcs in tasks.items():
            if not is_test(key[0], args.test_frac):
                continue  # report on TEST only
            qt, snr = qtype[key], key[2]
            for qbucket in ("all", qt):
                if "2" in svcs:
                    c, b = svcs["2"]; a = acc["M1_image"][snr][qbucket]
                    a[0] += int(c); a[1] += 1; a[2] += b; a[3] += uses_digital("2", ch, snr_val(snr), b)
                if "1" in svcs:
                    c, b = svcs["1"]; a = acc["M3_token"][snr][qbucket]
                    a[0] += int(c); a[1] += 1; a[2] += b; a[3] += uses_digital("1", ch, snr_val(snr), b)
                ch_sel = pol.get((qt, snr))
                if ch_sel in svcs:
                    c, b = svcs[ch_sel]; a = acc["M4_adaptive"][snr][qbucket]
                    a[0] += int(c); a[1] += 1; a[2] += b; a[3] += uses_digital(ch_sel, ch, snr_val(snr), b)
                cand = [(s, svcs[s]) for s in ("1", "2") if s in svcs]
                if cand:
                    bs, best = max(cand, key=lambda t: int(t[1][0]))
                    a = acc["M5_oracle"][snr][qbucket]
                    a[0] += int(best[0]); a[1] += 1; a[2] += best[1]
                    a[3] += uses_digital(bs, ch, snr_val(snr), best[1])
        for method, sd in acc.items():
            for snr, qd in sd.items():
                for qt, (k, n, b, u) in qd.items():
                    emit(ch, method, snr, qt, "test", k, n, b, u)

        # ---- merge M2 analog if present (native channel_uses; bytes not meaningful) ----
        m2 = f"{args.pred_dir}/m2_analog_{ch}_predictions.csv"
        if os.path.exists(m2):
            macc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0.0]))
            for r in csv.DictReader(open(m2)):
                if not is_test(r["image_id"], args.test_frac):
                    continue
                snr, qt = r["snr_bin"], r.get("question_type", "")
                cu = float(r.get("channel_uses") or 0)
                for qb in ("all", qt):
                    a = macc[snr][qb]; a[0] += int(correct(r["correct"])); a[1] += 1; a[3] += cu
            for snr, qd in macc.items():
                for qt, (k, n, b, u) in qd.items():
                    emit(ch, "M2_analog", snr, qt, "test", k, n, b, u)
        else:
            print(f"note: M2 analog not yet available for {ch}")

        # ---- merge NAIVE fixed-rate LDPC digital (cliffs), service s2 only ----
        nv = f"{args.pred_dir}/{args.naive_prefix}_{ch}_naive_predictions.csv"
        if os.path.exists(nv):
            nrows = [r for r in csv.DictReader(open(nv)) if r.get("service_level") == "2"]
            # transmitter always sends the full image at fixed rate; approximate the
            # per-image original size by the max delivered bytes across SNR bins
            orig = defaultdict(int)
            for r in nrows:
                orig[r["image_id"]] = max(orig[r["image_id"]], int(r.get("payload_bytes") or 0))
            nacc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0.0]))
            for r in nrows:
                if not is_test(r["image_id"], args.test_frac):
                    continue
                snr, qt = r["snr_bin"], r.get("question_type", "")
                b = int(r.get("payload_bytes") or 0)
                u = orig[r["image_id"]] * 8.0 / SE_TOKEN
                for qb in ("all", qt):
                    a = nacc[snr][qb]; a[0] += int(correct(r["correct"])); a[1] += 1; a[2] += b; a[3] += u
            for snr, qd in nacc.items():
                for qt, (k, n, b, u) in qd.items():
                    emit(ch, "M0_naive", snr, qt, "test", k, n, b, u)
        else:
            print(f"note: naive digital not yet available for {ch}")

        # ---- merge CLEAN error-free upper bound (channel-independent horizontal line) ----
        if os.path.exists(clean_path):
            cacc = defaultdict(lambda: [0, 0, 0])
            for r in csv.DictReader(open(clean_path)):
                if r.get("service_level") != "2":
                    continue
                if not is_test(r["image_id"], args.test_frac):
                    continue
                qt = r.get("question_type", "")
                b = int(r.get("payload_bytes") or 0)
                for qb in ("all", qt):
                    a = cacc[qb]; a[0] += int(correct(r["correct"])); a[1] += 1; a[2] += b
            for snr in snrs:
                for qt, (k, n, b) in cacc.items():
                    emit(ch, "M0_errorfree", snr, qt, "test", k, n, b, 0.0)
        else:
            print(f"note: clean error-free predictions not yet available ({clean_path})")
        print(f"{ch}: policy={ {k: pol[k] for k in sorted(pol)} }")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "method", "snr_db", "qtype", "split", "accuracy", "n",
                    "lcb", "ucb", "mean_payload_bytes", "mean_channel_uses", "cbr"])
        w.writerows(out_rows)
    print(f"\nwrote {len(out_rows)} rows -> {args.out}")

    # quick console summary (qtype=all, test)
    for ch in CHANNELS:
        sub = [r for r in out_rows if r[0] == ch and r[3] == "all"]
        if not sub:
            continue
        snrs = sorted({r[2] for r in sub})
        methods = sorted({r[1] for r in sub})
        print(f"\n[{ch}] accuracy (test, all qtypes):")
        print("  %-14s" % "SNR" + "".join("%8.0f" % s for s in snrs))
        for m in methods:
            line = "  %-14s" % m
            for s in snrs:
                v = [r[5] for r in sub if r[1] == m and r[2] == s]
                line += "%8s" % (f"{v[0]:.3f}" if v else "-")
            print(line)


if __name__ == "__main__":
    raise SystemExit(main())
