#!/usr/bin/env python3
"""P1: variable-token-budget sweep for the symbolic-decoder question types.

The transmitter keeps only the top-t detections (detector-confidence order,
PADC/ADJSCC-V nested-prefix style) before sending the s1 token payload over
the same LDPC link; the symbolic decoders (counting / comparison /
co_presence / threshold) answer from whatever survives the channel.
Per-(t, channel-bin) counting calibration is recomputed, mirroring a system
calibrated at its operating budget.  Pure CPU offline: detections + the
deterministic channel hash + symbolic decode -- no VLM involved.

Reuses the eval script's own functions (degradation, calibration, payload
accounting) so t=full reproduces the logged M3 numbers exactly.
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
import run_v1_detector_eval as ev  # noqa: E402
from vqa_semcom.config import load_config  # noqa: E402
from vqa_semcom.detector.visdrone_yolo import (  # noqa: E402
    DetectionRecord,
    degrade_detections_for_channel,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CH_CFG = {
    "rician": "configs/v2_0_ldpc_channel.yaml",
    "awgn": "configs/v2_0_awgn.json",
    "rayleigh": "configs/v2_0_rayleigh.json",
}
T_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 0]  # 0 = full (no truncation)
QT_SYMBOLIC = ("counting", "comparison", "co_presence", "threshold")


def derive_snr_labels(pred_csv, scan_rows=100_000):
    """Exact snr_bin label strings from a logged predictions CSV -- the
    deterministic degradation hash is seeded by this string, so it must
    match the eval byte-for-byte."""
    labels = set()
    for i, r in enumerate(csv.DictReader(open(pred_csv))):
        labels.add(r["snr_bin"])
        if i >= scan_rows:
            break
    return sorted(labels, key=bc.snr_val)


def load_detections(path):
    """Keep the CSV row order: the eval's degradation hash uses the record's
    enumerate index, so re-sorting would change the noise realization.
    YOLO writes detections confidence-descending per image; we verify."""
    cache = {}
    for r in csv.DictReader(open(path)):
        rec = DetectionRecord(
            category=r["category"],
            bbox_x=int(float(r["bbox_x"])), bbox_y=int(float(r["bbox_y"])),
            bbox_w=int(float(r["bbox_w"])), bbox_h=int(float(r["bbox_h"])),
            confidence=float(r["confidence"]),
        )
        cache.setdefault(r["image_id"], []).append(rec)
    bad = sum(1 for recs in cache.values()
              for a, b in zip(recs, recs[1:]) if a.confidence < b.confidence)
    if bad:
        print(f"note: {bad} adjacent confidence inversions kept in CSV order")
    return cache


def load_tasks():
    tasks = []
    for path, keep in (("outputs/tasks/v1_7_tasks.csv", ("counting",)),
                       ("outputs/tasks/v2_comparison_tasks.csv", ("comparison",)),
                       ("outputs/tasks/v2_extra_tasks.csv", ("co_presence", "threshold"))):
        for r in csv.DictReader(open(path)):
            if r["question_type"] in keep:
                tasks.append(r)
    return tasks


def logged_counting_set(pred_csv):
    """The main eval selects a task subset (max_tasks_per_image cap); restrict
    counting to exactly the logged tasks so t=full reproduces the paper numbers
    and the per-t calibration mirrors the deployed system's own task mix."""
    allowed = set()
    for r in csv.DictReader(open(pred_csv)):
        if r["question_type"] == "counting":
            allowed.add((r["image_id"], r["question"]))
    return allowed


def decode(task, tx, calib, min_raw, cfg):
    qt = task["question_type"]
    tol = float((cfg.get("vlm", {}) or {}).get("count_tolerance_ratio", 0.10))
    ca = sum(1 for rr in tx if rr.category == task.get("target_class", ""))
    if qt == "counting":
        cal = ev._calibrated_count(ca, task.get("target_class", ""), decode.channel,
                                   calib, min_raw_count=min_raw)
        return ev._semantic_token_prediction(task, cal, cfg)[2]
    if qt == "comparison":
        cb = sum(1 for rr in tx if rr.category == task.get("target_class_b", ""))
        pred = "yes" if ca > cb else "no"
    elif qt == "co_presence":
        cb = sum(1 for rr in tx if rr.category == task.get("target_class_b", ""))
        pred = "yes" if (ca > 0 and cb > 0) else "no"
    else:  # threshold
        try:
            thr = int(float(task.get("threshold_n", "1") or 1))
        except ValueError:
            thr = 1
        pred = "yes" if ca >= thr else "no"
    return ev.check_answer(qt, pred, task["answer"], tolerance_ratio=tol).correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="rician,awgn,rayleigh")
    ap.add_argument("--pred-dir", default="outputs/vlm")
    ap.add_argument("--out", default="outputs/reports/token_budget_sweep.csv")
    ap.add_argument("--out-dir", default="outputs/figures/comparison")
    ap.add_argument("--fig-channel", default="rician")
    ap.add_argument("--tag", default="v3")
    args = ap.parse_args()

    all_tasks = load_tasks()
    SNR_LABELS = derive_snr_labels("outputs/vlm/v3_0_rician_predictions.csv")
    print(f"snr labels (exact eval strings): {SNR_LABELS}")

    rows = []
    for ch in args.channels.split(","):
        cfg = load_config(CH_CFG[ch])
        allowed = logged_counting_set(f"{args.pred_dir}/v2_0_{ch}_predictions.csv")
        tasks = [t for t in all_tasks
                 if t["question_type"] != "counting"
                 or (t["image_id"], t["question"]) in allowed]
        counting_tasks = [t for t in tasks if t["question_type"] == "counting"]
        test_tasks = [t for t in tasks if bc.is_test(t["image_id"])]
        print(f"[{ch}] tasks: total={len(tasks)} test={len(test_tasks)} "
              f"({ {q: sum(1 for t in test_tasks if t['question_type']==q) for q in QT_SYMBOLIC} })")
        det_csv = (cfg.get("detector", {}) or {}).get("detections_csv",
                                                      "outputs/detector/v2_0_snr_detections.csv")
        if not os.path.exists(det_csv):
            det_csv = "outputs/detector/v2_0_snr_detections.csv"
        full_cache = load_detections(det_csv)
        min_raw = int((cfg.get("vlm", {}) or {}).get("count_calibration_min_raw_count", 3))
        mean_dets = sum(len(v) for v in full_cache.values()) / max(1, len(full_cache))
        print(f"[{ch}] detections={det_csv} images={len(full_cache)} mean_dets={mean_dets:.1f}")

        for t in T_GRID:
            cache_t = ({img: recs[: t] for img, recs in full_cache.items()}
                       if t else full_cache)
            det_cache_ev = {img: (recs, 0.0, "") for img, recs in cache_t.items()}
            calib = ev._counting_class_channel_calibration(
                counting_tasks, det_cache_ev, SNR_LABELS, cfg)
            agg = defaultdict(lambda: [0, 0, 0, 0.0, 0])  # (snr,qt)->k,n,bytes,uses,tok
            for task in test_tasks:
                recs = cache_t.get(task["image_id"], [])
                for snr in SNR_LABELS:
                    tx = degrade_detections_for_channel(recs, snr, cfg, task["image_id"])
                    decode.channel = snr
                    ok = decode(task, tx, calib, min_raw, cfg)
                    repr_ = ev.build_detector_lightweight_evidence(task, recs, snr, cfg)
                    pb = ev._payload_bytes(1, "lightweight", repr_, "")
                    for qb in ("all", task["question_type"]):
                        a = agg[(snr, qb)]
                        a[0] += int(bool(ok)); a[1] += 1; a[2] += pb
                        a[3] += pb * 8.0 / bc.SE_TOKEN; a[4] += len(recs)
            for (snr, qt), (k, n, b, u, tok) in sorted(agg.items()):
                rows.append([ch, t if t else "full", bc.snr_val(snr), qt,
                             round(k / n, 4), n, round(tok / n, 2),
                             round(b / n, 1), round(u / n, 1)])
            mid = SNR_LABELS[len(SNR_LABELS) // 2]
            a = agg[(mid, "all")]
            if a[1]:
                print(f"  t={t if t else 'full':>4}: acc@{mid}={a[0]/a[1]:.4f} "
                      f"mean_tokens={a[4]/a[1]:.1f} mean_bytes={a[2]/a[1]:.0f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "t_budget", "snr_db", "qtype", "accuracy", "n",
                    "mean_tokens_sent", "mean_payload_bytes", "mean_channel_uses"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")

    # ---- validation: t=full vs logged M3 (symbolic qtypes, rician test) ----
    cmp_csv = "outputs/reports/comparison_v3_5qt.csv"
    if os.path.exists(cmp_csv):
        logged = {(r["qtype"], float(r["snr_db"])): float(r["accuracy"])
                  for r in csv.DictReader(open(cmp_csv))
                  if r["channel"] == "rician" and r["method"] == "M3_token"}
        print("\nvalidation vs logged M3 (rician): qtype snr sweep/logged")
        for r in rows:
            if r[0] == "rician" and r[1] == "full" and r[3] in QT_SYMBOLIC:
                key = (r[3], r[2])
                if key in logged:
                    flag = "" if abs(r[4] - logged[key]) < 0.02 else "  <-- MISMATCH"
                    print(f"  {r[3]:12s} {r[2]:5.0f}dB  {r[4]:.3f}/{logged[key]:.3f}{flag}")

    # ---- F8: budget sweep figure (fig-channel) ----
    ch = args.fig_channel
    sub = [r for r in rows if r[0] == ch]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    colors = {"counting": "#e06c75", "comparison": "#61afef", "co_presence": "#98c379",
              "threshold": "#c678dd", "all": "#444444"}
    for ax, snr in zip(axes[:2], (-5.0, 20.0)):
        for qt in list(QT_SYMBOLIC) + ["all"]:
            pts = sorted([(r[6], r[4]) for r in sub if r[2] == snr and r[3] == qt])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                        color=colors[qt], label=qt, linewidth=2, markersize=4)
        ax.set_xscale("log"); ax.set_xlabel("mean tokens sent (top-t budget)")
        ax.set_ylabel("accuracy"); ax.grid(True, alpha=0.3)
        ax.set_title(f"{ch}, SNR = {snr:g} dB")
    axes[0].legend(fontsize=8)
    ax = axes[2]
    for qt in list(QT_SYMBOLIC) + ["all"]:
        pts = sorted([(r[8], r[4]) for r in sub if r[2] == 5.0 and r[3] == qt])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    color=colors[qt], label=qt, linewidth=2, markersize=4)
    ax.set_xscale("log"); ax.set_xlabel("mean complex channel uses / query")
    ax.set_ylabel("accuracy"); ax.grid(True, alpha=0.3)
    ax.set_title(f"{ch}, SNR = 5 dB (bandwidth axis)")
    fig.suptitle("Variable token budget: nested top-t truncation of s1 evidence "
                 "(symbolic decoders, test set)", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_dir}/F8_token_budget_{args.tag}.{ext}", dpi=140,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote F8_token_budget_{args.tag} -> {args.out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
