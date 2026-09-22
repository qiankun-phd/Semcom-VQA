#!/usr/bin/env python3
"""M2 baseline eval: DeepSC-style ANALOG image transmission + same Qwen backend.

Mirrors the s2 image branch of run_v1_detector_eval, but the image is sent over the
analog link (graceful degradation, no LDPC cliff) instead of the digital
rate-adaptive path.  Same prompt, same Qwen, same scoring -> bandwidth/back-end fair.
"""
from __future__ import annotations

import argparse
import csv
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from vqa_semcom.config import load_config, resolve_path, ensure_parent
from vqa_semcom.evidence.builder import build_vlm_prompt, image_path_for_task, read_tasks_csv, select_vlm_tasks
from vqa_semcom.snr import parse_snr_bins, snr_bins_from_config, snr_bin_label, channel_bin_from_snr
from vqa_semcom.vlm.evaluator import evaluate_prediction, make_evaluator
from vqa_semcom.degradation.analog_link import transmit_image_analog_to_path

FIELDS = ["image_id", "question_type", "question", "ground_truth_answer", "snr_db", "snr_bin",
          "channel_bin", "method", "predicted_answer", "normalized_prediction", "correct",
          "payload_bytes", "channel_uses", "eff_snr_db", "scale", "model_name"]


def _seed(image_id: str, snr: float) -> int:
    return zlib.crc32(f"{image_id}|{snr}".encode()) & 0xFFFFFFFF


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--evaluator", choices=["mock", "qwen"], default="qwen")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--limit-images", type=int, default=None)
    ap.add_argument("--snr-bins", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    vlm_cfg = cfg.get("vlm", {})
    kind = cfg["vlm"]["fading_link"]["fading"]["kind"]
    qtypes = set(vlm_cfg.get("question_types", ["presence", "counting"]))
    tasks = read_tasks_csv(resolve_path(cfg["paths"]["tasks_csv"]))
    selected = select_vlm_tasks(tasks, question_types=qtypes, limit_images=args.limit_images,
                                max_tasks=args.max_tasks,
                                max_tasks_per_image=int(vlm_cfg.get("max_tasks_per_image", 6)))
    if not selected:
        raise RuntimeError("no tasks selected")
    snr_values = parse_snr_bins(args.snr_bins) if args.snr_bins else snr_bins_from_config(cfg)
    if not snr_values:
        snr_values = [-5, 0, 5, 10, 15, 20]
    visdrone_root = resolve_path(cfg["paths"]["visdrone_val"])
    out_path = Path(args.out) if args.out else resolve_path(f"outputs/vlm/m2_analog_{kind}_predictions.csv")
    degraded_dir = resolve_path(f"outputs/vlm/analog_images_{kind}")
    evaluator = make_evaluator(args.evaluator, cfg)

    rows = []
    cache = {}
    for ti, task in enumerate(selected):
        image_id = task["image_id"]
        src = image_path_for_task(visdrone_root, image_id)
        for snr in snr_values:
            key = (image_id, task["question"], float(snr))
            if key not in cache:
                out_img = Path(degraded_dir) / f"{image_id}_snr{snr}.jpg"
                meta = transmit_image_analog_to_path(src, out_img, float(snr), cfg,
                                                     rng=np.random.default_rng(_seed(image_id, snr)))
                prompt = build_vlm_prompt(task)
                prompt = f"service_level=2-analog snr={snr} channel={kind} evidence_source=analog_image\n{prompt}"
                pred = evaluate_prediction(evaluator, task, prompt, out_img, cfg)
                cache[key] = (pred.predicted_answer, pred.normalized_prediction, pred.correct, pred.model_name, meta)
            pa, npd, corr, mn, meta = cache[key]
            rows.append({
                "image_id": image_id, "question_type": task["question_type"], "question": task["question"],
                "ground_truth_answer": task["answer"], "snr_db": f"{float(snr):g}", "snr_bin": snr_bin_label(snr),
                "channel_bin": channel_bin_from_snr(float(snr)), "method": "M2_analog",
                "predicted_answer": pa, "normalized_prediction": npd, "correct": str(corr),
                # analog occupies the whole slot: channel uses ~ B*tau; bytes-equivalent left as channel_uses
                "payload_bytes": meta.get("channel_uses", 0), "channel_uses": meta.get("channel_uses", 0),
                "eff_snr_db": meta.get("eff_snr_db", ""), "scale": meta.get("scale", ""), "model_name": mn,
            })
        if (ti + 1) % 50 == 0:
            print(f"...{ti+1}/{len(selected)} tasks")

    ensure_parent(out_path)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    print(f"method=M2_analog channel={kind} tasks={len(selected)} rows={len(rows)}")
    print(f"out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
