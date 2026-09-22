#!/usr/bin/env python3
"""P2: per-sample learned service selector (OracleNet-style predictor vs LUT).

Two calibrated per-service correctness predictors P(correct | features, service)
are trained on the TRAIN split of the existing prediction logs (per-sample
online-rollout labels, PADC OracleNet recipe with MSE->BCE); the policy picks
argmax-probability per sample (cost-aware: prefers the cheap token service
within a margin).  Features are restricted to what the scheduler can see at
selection time: question type/class, transmitter-side raw detector count,
task metadata, SNR.  Evaluation is a pure offline re-selection over already
evaluated (task, snr, service) cells -- directly comparable to M3/M1/M4/M5.

The M4<->M5 oracle gap (~7.5 pts) is entirely per-sample: this measures how
much of it per-sample features can recover beyond question-type routing.
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

import numpy as np  # noqa: E402

QTYPES = ["presence", "counting", "comparison", "co_presence", "threshold"]
CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
           "awning-tricycle", "bus", "motor"]
VIEWS = ["poor", "fair", "good"]
RISKS = ["normal", "elevated", "critical"]


def featurize(r):
    f = []
    f += [1.0 if r["question_type"] == q else 0.0 for q in QTYPES]
    f += [1.0 if r.get("target_class") == c else 0.0 for c in CLASSES]
    f += [1.0 if r.get("view_quality_bin") == v else 0.0 for v in VIEWS]
    f += [1.0 if r.get("risk_level") == k else 0.0 for k in RISKS]
    snr = bc.snr_val(r["snr_bin"])
    f.append(snr / 20.0)
    raw = float(r.get("raw_detector_count") or 0)
    f.append(min(raw, 60.0) / 60.0)
    f.append(1.0 if raw > 0 else 0.0)
    f.append(float(r.get("density_score") or 0) / 200.0 if r.get("density_score") else 0.0)
    f.append(1.0 if r.get("presence_polarity") == "negative" else 0.0)
    return f


def load(path):
    """(img,question,snr) -> {'feat': x, 's1': bool, 's2': bool, 'qt': str, 'img': id, 'bytes': {svc: b}}"""
    groups = {}
    for r in csv.DictReader(open(path)):
        if r.get("service_level") not in ("1", "2"):
            continue
        key = (r["image_id"], r["question"], r["snr_bin"])
        g = groups.setdefault(key, {"qt": r["question_type"], "img": r["image_id"],
                                    "snr": r["snr_bin"], "bytes": {}, "feat": None})
        g[f's{r["service_level"]}'] = bc.correct(r["correct"])
        g["bytes"][r["service_level"]] = int(r.get("payload_bytes") or 0)
        if r["service_level"] == "1":  # s1 row carries transmitter-side detector features
            g["feat"] = featurize(r)
    return {k: g for k, g in groups.items()
            if g["feat"] is not None and "s1" in g and "s2" in g}


def fit_logistic(X, y, l2=1e-3, iters=400, lr=0.5):
    X = np.asarray(X); y = np.asarray(y, dtype=float)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
    return w


def predict(w, X):
    Xb = np.hstack([np.asarray(X), np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-Xb @ w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="outputs/vlm")
    ap.add_argument("--prefix", default="v3_0")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="prefer s1 (cheap) unless p2 - p1 > margin")
    ap.add_argument("--out", default="outputs/reports/persample_policy.csv")
    args = ap.parse_args()

    out_rows = []
    for ch in bc.CHANNELS:
        pred = f"{args.pred_dir}/{args.prefix}_{ch}_predictions.csv"
        if not os.path.exists(pred):
            print(f"skip {ch}")
            continue
        groups = load(pred)
        train = [g for k, g in groups.items() if not bc.is_test(g["img"])]
        test = [g for k, g in groups.items() if bc.is_test(g["img"])]

        Xtr = [g["feat"] for g in train]
        w1 = fit_logistic(Xtr, [g["s1"] for g in train])
        w2 = fit_logistic(Xtr, [g["s2"] for g in train])

        # LUT baseline policy (M4) for the same split, from bc
        tasks, qtype = bc.load_channel(pred)
        pol = bc.learn_policy(tasks, qtype, split_test=False)

        Xte = [g["feat"] for g in test]
        p1, p2 = predict(w1, Xte), predict(w2, Xte)
        agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # method -> qt -> [k,n,bytes]
        for g, q1, q2 in zip(test, p1, p2):
            sel_ml = "1" if (q2 - q1) <= args.margin else "2"
            sel_lut = pol.get((g["qt"], g["snr"]), "2")
            oracle = g["s1"] or g["s2"]
            for qb in ("all", g["qt"]):
                a = agg["ML_persample"][qb]
                a[0] += int(g[f"s{sel_ml}"]); a[1] += 1; a[2] += g["bytes"].get(sel_ml, 0)
                a = agg["M4_lut"][qb]
                a[0] += int(g[f"s{sel_lut}"]); a[1] += 1; a[2] += g["bytes"].get(sel_lut, 0)
                agg["M3_token"][qb][0] += int(g["s1"]); agg["M3_token"][qb][1] += 1
                agg["M1_image"][qb][0] += int(g["s2"]); agg["M1_image"][qb][1] += 1
                agg["M5_oracle"][qb][0] += int(oracle); agg["M5_oracle"][qb][1] += 1
        print(f"\n[{ch}] test accuracy (pooled over SNR):")
        for m in ("M3_token", "M1_image", "M4_lut", "ML_persample", "M5_oracle"):
            k, n, b = agg[m]["all"]
            print(f"  {m:13s} {k/n:.4f}  (n={n}, mean_bytes={b/max(n,1):.0f})")
            out_rows.append([ch, m, "all", round(k / n, 4), n, round(b / max(n, 1), 1)])
        for qt in QTYPES:
            k4, n4, _ = agg["M4_lut"][qt]
            km, nm, _ = agg["ML_persample"][qt]
            if n4:
                d = km / nm - k4 / n4
                print(f"    {qt:12s} lut={k4/n4:.4f} ml={km/nm:.4f} delta={d:+.4f}")
                out_rows.append([ch, "delta_ml_vs_lut", qt, round(d, 4), n4, 0])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "method", "qtype", "accuracy", "n", "mean_payload_bytes"])
        w.writerows(out_rows)
    print(f"\nwrote {len(out_rows)} rows -> {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
