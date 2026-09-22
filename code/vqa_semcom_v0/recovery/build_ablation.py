#!/usr/bin/env python3
"""Ablation of the M4 service-selection mechanism (reuses existing predictions).

3a  Which LUT/conditioning dimensions matter for token-vs-image selection?
    nested feature sets {} < {qt} < {qt,snr} < {qt,snr,view} < {qt,snr,view,risk}
    (freshness is reported separately: it has 0 effect on s1/s2 selection)
3b  Selection criterion: Wilson-LCB vs plain mean accuracy.

Same train/test split (crc32 image_id, 20% test) and same LCB selection as the
main comparison, so numbers are comparable to comparison_all.csv.
"""
from __future__ import annotations
import argparse, csv, math, os, zlib
from collections import defaultdict

CHANNELS = ["awgn", "rayleigh", "rician"]
ATTRS = {"qt": "question_type", "snr": "snr_bin", "view": "view_quality_bin", "risk": "risk_level"}
FEATURE_SETS = [
    ("none", []),
    ("qt", ["qt"]),
    ("qt+snr  (current M4)", ["qt", "snr"]),
    ("qt+snr+view", ["qt", "snr", "view"]),
    ("qt+snr+view+risk", ["qt", "snr", "view", "risk"]),
    ("qt+snr+freshness", ["qt", "snr", "fresh"]),  # show freshness adds nothing
]


def wilson_lcb(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    return (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)


def correct(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def is_test(image_id, frac=0.2):
    return (zlib.crc32(str(image_id).encode()) % 100) < int(frac * 100)


def load_instances(path):
    """instance (img,q,snr) -> {attrs..., 's': {service: correct}}; freshness kept per-service-set as note."""
    inst = {}
    for r in csv.DictReader(open(path)):
        if r["service_level"] not in ("1", "2"):
            continue
        key = (r["image_id"], r["question"], r["snr_bin"])
        d = inst.setdefault(key, {"img": r["image_id"], "qt": r["question_type"], "snr": r["snr_bin"],
                                  "view": r.get("view_quality_bin", ""), "risk": r.get("risk_level", ""),
                                  "fresh": r.get("freshness_bin", ""), "s": {}})
        # replicated across freshness; first occurrence wins (all identical for s1/s2)
        d["s"].setdefault(r["service_level"], correct(r["correct"]))
    return inst


def cell_key(d, feats):
    return tuple(d[f] for f in feats)


def learn_policy(inst, feats, criterion):
    cell = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for d in inst.values():
        if is_test(d["img"]):
            continue  # train only
        ck = cell_key(d, feats)
        for s, c in d["s"].items():
            cell[ck][s][1] += 1
            cell[ck][s][0] += int(c)
    # global fallback (best service overall on train)
    gl = defaultdict(lambda: [0, 0])
    for d in inst.values():
        if is_test(d["img"]):
            continue
        for s, c in d["s"].items():
            gl[s][1] += 1; gl[s][0] += int(c)
    def score(k, n):
        return wilson_lcb(k, n) if criterion == "lcb" else (k / n if n else -1)
    g_best = max(gl, key=lambda s: score(gl[s][0], gl[s][1])) if gl else "1"
    pol = {}
    for ck, sv in cell.items():
        pol[ck] = max(sv, key=lambda s: score(sv[s][0], sv[s][1]))
    return pol, g_best


def eval_policy(inst, feats, pol, g_best):
    k = n = 0
    for d in inst.values():
        if not is_test(d["img"]):
            continue
        if not ({"1", "2"} <= set(d["s"])):
            continue  # need both candidates for a fair selection
        chosen = pol.get(cell_key(d, feats), g_best)
        k += int(d["s"][chosen]); n += 1
    return k, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="outputs/vlm")
    ap.add_argument("--prefix", default="v2_0")
    ap.add_argument("--out", default="outputs/reports/ablation_mechanism.csv")
    args = ap.parse_args()

    inst_by_ch = {}
    for ch in CHANNELS:
        p = f"{args.pred_dir}/{args.prefix}_{ch}_predictions.csv"
        if os.path.exists(p):
            inst_by_ch[ch] = load_instances(p)

    rows = []
    # ---- 3a: feature-set ablation (LCB selection) ----
    print("=== 3a  M4 test accuracy by conditioning feature set (LCB selection) ===")
    print("%-22s %8s %8s %8s %8s" % ("feature set", *CHANNELS, "pooled"))
    for name, feats in FEATURE_SETS:
        line = "%-22s" % name
        pk = pn = 0
        for ch in CHANNELS:
            inst = inst_by_ch.get(ch)
            if not inst:
                line += "%8s" % "-"; continue
            pol, gb = learn_policy(inst, feats, "lcb")
            k, n = eval_policy(inst, feats, pol, gb)
            pk += k; pn += n
            line += "%8.3f" % (k / n if n else 0)
            rows.append(["3a_featureset", name, ch, "lcb", round(k/n, 4) if n else 0, n])
        line += "%8.3f" % (pk / pn if pn else 0)
        rows.append(["3a_featureset", name, "pooled", "lcb", round(pk/pn, 4) if pn else 0, pn])
        print(line)

    # ---- 3b: LCB vs mean for the current feature set (qt+snr) ----
    print("\n=== 3b  selection criterion: LCB vs mean (feature set = qt+snr) ===")
    print("%-10s %8s %8s %8s %8s" % ("criterion", *CHANNELS, "pooled"))
    for crit in ("lcb", "mean"):
        line = "%-10s" % crit
        pk = pn = 0
        for ch in CHANNELS:
            inst = inst_by_ch.get(ch)
            if not inst:
                line += "%8s" % "-"; continue
            pol, gb = learn_policy(inst, ["qt", "snr"], crit)
            k, n = eval_policy(inst, ["qt", "snr"], pol, gb)
            pk += k; pn += n
            line += "%8.3f" % (k / n if n else 0)
            rows.append(["3b_criterion", crit, ch, crit, round(k/n, 4) if n else 0, n])
        line += "%8.3f" % (pk / pn if pn else 0)
        rows.append(["3b_criterion", crit, "pooled", crit, round(pk/pn, 4) if pn else 0, pn])
        print(line)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ablation", "variant", "channel", "criterion", "test_accuracy", "n"])
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
