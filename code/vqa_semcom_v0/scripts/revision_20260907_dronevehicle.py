#!/usr/bin/env python3
"""DroneVehicle target refit and separately identified source-frozen transfer.

Read-only cached detector/VLM outcomes. CPU codec identity gate before costing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np

import revision_20260907 as base
import revision_20260907_supplement as supp


def strict_groups(path: Path, repo: Path | None = None) -> tuple[list[dict], dict]:
    groups, audit = base.load_groups(path)
    metadata, image_paths, detector_counts = {}, {}, {}
    with path.open(newline="") as stream:
        for r in csv.DictReader(stream):
            if r["service_level"] not in ("1", "2"):
                continue
            snr = int(float(r["snr_bin"].replace("dB", "")))
            key = (r["image_id"], r["question"], snr)
            meta = tuple(r[k] for k in ("question_type", "target_class", "object_count", "ground_truth_answer"))
            if key in metadata and metadata[key] != meta:
                raise ValueError(f"Conflicting cross-branch metadata: {key}")
            metadata[key] = meta
            if r["service_level"] == "1":
                raw_key = (r["image_id"], r["target_class"])
                raw = int(float(r["raw_detector_count"]))
                if repo is None and raw_key in detector_counts and detector_counts[raw_key] != raw:
                    raise ValueError(f"Raw detector feature changes across SNR/question: {raw_key}")
                detector_counts[raw_key] = raw
            else:
                if "Qwen2-VL-2B-Instruct" not in r["model_name"]:
                    raise ValueError("Unexpected receiver")
                image_paths.setdefault((r["image_id"], snr), set()).add(r["image_path"])
    for g in groups:
        if g["class"] not in base.CLASSES:
            raise ValueError(f"Unknown class: {g['class']}")
        g["all_image_paths"] = sorted(image_paths[(g["image"], g["snr"])])
    audit["cross_branch_metadata_consistent"] = True
    if repo is not None:
        detector_sources = {tag: repo / f"outputs/detector/dv_rician_{tag}_detections.csv" for tag in ("main", "cmp", "extra")}
        detections = {}
        signature_fields = ("image_id", "category", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "confidence", "detector_model")
        for tag, source in detector_sources.items():
            with source.open(newline="") as stream:
                detections[tag] = list(csv.DictReader(stream))
        signatures = {tag: Counter(tuple(r[k] for k in signature_fields) for r in rows) for tag, rows in detections.items()}
        supp.require_same(signatures["main"], signatures["extra"], "main/extra original detector records")
        if signatures["cmp"] - signatures["main"]:
            raise ValueError("Comparison original detector records differ from main")
        counts = Counter((r["image_id"], r["category"]) for r in detections["main"])
        changes = []
        for g in groups:
            actual = counts[(g["image"], g["class"])]
            if g["qt"] in ("presence", "counting") and actual != g["raw"]:
                raise ValueError("Main cached sender count disagrees with original detector records")
            if actual != g["raw"]:
                changes.append({"image": g["image"], "question": g["question"], "snr": g["snr"], "logged_raw": g["raw"], "sender_count": actual})
            g["raw"] = actual
        audit["sender_count_source"] = "original main detector bounding-box rows; counts validated against every main-task raw count; main/extra exact signatures, cmp subset"
        audit["sender_count_source_sha256"] = {str(p): base.sha(p) for p in detector_sources.values()}
        audit["logged_raw_field_corrections"] = changes
    audit["effective_sender_count_invariant_across_snr_and_questions"] = True
    audit["receiver"] = "Qwen2-VL-2B-Instruct"
    audit["classes"] = sorted({g["class"] for g in groups})
    return groups, audit


def reconstruct(groups: list[dict], repo: Path, limit: int | None = None) -> dict:
    import cv2
    cv2.setNumThreads(1)
    sys.path.insert(0, str(repo / "src"))
    from vqa_semcom.degradation.digital_link import (
        build_link_config, link_config_to_dict, _seed_from,
        transmit_image_rate_adaptive, ergodic_spectral_efficiency)
    configs = {tag: supp.read(repo / f"configs/dv_rician_{tag}.json") for tag in ("main", "cmp", "extra")}
    cfg = build_link_config(configs["main"])
    for tag, c in configs.items():
        supp.require_same(link_config_to_dict(build_link_config(c)), link_config_to_dict(cfg), f"DV {tag} channel config")
        supp.require_same(c["paths"]["visdrone_val"], configs["main"]["paths"]["visdrone_val"], "DV source image directory")
    if cfg.channel_mode != "rate_adaptive" or cfg.fading.kind != "rician":
        raise ValueError("Unexpected codec mode or channel")
    source_dir = repo / configs["main"]["paths"]["visdrone_val"] / "images"
    unique = {(g["image"], g["snr"]): g for g in groups}
    records, source_hashes = {}, {}
    previous, im = None, None
    for j, ((iid, snr), g) in enumerate(sorted(unique.items())):
        if limit is not None and j >= limit:
            break
        source = source_dir / f"{iid}.jpg"
        if previous != iid:
            im = cv2.imread(str(source))
            if im is None:
                raise FileNotFoundError(source)
            source_hashes[iid] = base.sha(source)
            previous = iid
        decoded, meta = transmit_image_rate_adaptive(im, snr, cfg, np.random.default_rng(_seed_from(iid, f"{snr}dB")))
        ok, jpeg = cv2.imencode(".jpg", decoded, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise ValueError("JPEG encode failed")
        expected = hashlib.sha256(jpeg.tobytes()).hexdigest()
        matches = {p: Path(p).is_file() and base.sha(Path(p)) == expected for p in g["all_image_paths"]}
        airtime = cfg.tx_time_budget_s if meta["outage"] else meta["bytes"] * 8 / ergodic_spectral_efficiency(snr, cfg.fading) / cfg.bandwidth_hz
        records[f"{iid}|{snr}"] = {"meta": meta, "airtime_s": airtime,
            "saved_jpeg_matches": all(matches.values()), "path_matches": matches,
            "wire_payload_bytes": meta.get("bytes"), "logged_file_bytes": g["2"]["bytes"]}
        if (j + 1) % 100 == 0:
            print(f"[payload DV] {j+1}/{len(unique)}", flush=True)
    return {"records": records, "n": len(records), "matched_saved_images": sum(r["saved_jpeg_matches"] for r in records.values()),
        "source_image_sha256": source_hashes, "source_directory": str(source_dir), "cv2_version": cv2.__version__,
        "link_source_sha256": base.sha(repo / "src/vqa_semcom/degradation/digital_link.py"),
        "config": link_config_to_dict(cfg), "outage_cost": "full 0.3 s slot; original transmitted bytes unrecorded"}


def frozen_transfer(args: argparse.Namespace, parts: dict, x: dict, energy: dict, baseline_energy: np.ndarray) -> dict:
    dest = args.out / "source_frozen_transfer"
    dest.mkdir()
    src = args.formal / "rician/train_only"
    ratios = supp.read(src / "calibration.json")
    y = base.labels(parts["test"], ratios)
    base.dump(dest / "source_calibration.json", ratios)
    with np.load(src / "baseline_outcomes.npz") as old:
        weights = old["linear_weights"].copy()
    base.dump(dest / "calibration_scope.json", {"fit": "VisDrone training images only; no DroneVehicle labels or validation used", "missing_class_snr_ratio": "1.0"})
    np.savez_compressed(dest / "evaluation_inputs.npz", y=y, energy=energy["test"], baseline_energy=baseline_energy, linear_weights=weights)
    baseline = {name: base.summarize(parts["test"], y, pick, baseline_energy)
        for name, pick in {"image": np.ones(len(y), int), "detection": np.zeros(len(y), int),
            "rule": np.array([g["qt"] == "presence" for g in parts["test"]], int)}.items()}
    base.dump(dest / "baselines.json", baseline)
    records = []
    for model_name, seed in [("linear", None)] + [("mlp", s) for s in args.seeds]:
        if model_name == "linear":
            record = supp.read(args.supplement / "rician/train_only/linear.json")
            pt = base.linear_predict(weights, x["test"])
        else:
            record = supp.read(src / f"seed_{seed}.json")
            models = joblib.load(src / f"seed_{seed}_models.joblib")
            pt = np.column_stack([base.probabilities(m, x["test"]) for m in models])
        index = record["validation_selected_index"]
        supp.require_same(record["sweep"][index]["lambda"], base.PRICES[index], "source price")
        pick = base.choose(pt, energy["test"], 0)
        selected = base.choose(pt, energy["test"], base.PRICES[index])
        result = {"model": model_name, "seed": seed, "source_selected_index": index,
            "source_selected_lambda": base.PRICES[index], "main": base.summarize(parts["test"], y, pick, energy["test"]),
            "selected_test": base.metrics(y, selected, energy["test"]),
            "selected_summary": base.summarize(parts["test"], y, selected, energy["test"]),
            "interpretation": "VisDrone-frozen model/calibration/price, DroneVehicle test only; source relative-accuracy criterion does not guarantee target accuracy loss"}
        name = "linear" if seed is None else f"seed_{seed}"
        base.dump(dest / f"{name}.json", result)
        np.savez_compressed(dest / f"{name}_outcomes.npz", probabilities=pt, pick=pick, selected_pick=selected)
        if seed is not None:
            records.append(result)
    return {"mlp": supp.describe(records), "baselines": baseline, "linear": supp.read(dest / "linear.json")}


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("repo", "formal", "supplement", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--target-only", action="store_true", help="Defer source transfer until source feature audit is complete")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--max-epochs", type=int, default=300)
    args = ap.parse_args()
    if not args.smoke and (args.seeds != list(range(10)) or args.max_epochs != 300):
        raise ValueError("Frozen formal training protocol changed")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    import sklearn
    fm = supp.read(args.formal / "manifest.json")
    supp.require_same(supp.read(args.formal / "status.json")["state"], "COMPLETE", "formal source")
    supp.require_same(supp.read(args.supplement / "status.json")["state"], "COMPLETE", "supplement source")
    supp.require_same(base.sha(Path(base.__file__)), fm["script_sha256"], "base code")
    supp.require_same(np.__version__, fm["numpy"], "numpy")
    supp.require_same(sklearn.__version__, fm["sklearn"], "sklearn")
    manifest = {"pid": os.getpid(), "started_unix": started, "smoke": args.smoke, "seeds": args.seeds,
        "max_epochs": args.max_epochs, "features": base.FEATURE_NAMES, "prices": base.PRICES,
        "selection": "minimum validation energy within 0.01 accuracy of each model's unpriced validation result",
        "split": fm["split"], "calibration_modes": ["train_only", "none"],
        "scope": "DroneVehicle target-data refit and separately source-frozen VisDrone transfer; cached Qwen2-VL answers only",
        "source_transfer": "PENDING_SOURCE_FEATURE_AUDIT" if args.target_only else "SOURCE_FROZEN",
        "energy": "same Qwen2-VL 32.31 J proxy, not a DroneVehicle measurement; detector every learned query; same formal exclusions",
        "python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__,
        "script_sha256": base.sha(Path(__file__)), "base_script_sha256": base.sha(Path(base.__file__)),
        "supplement_script_sha256": base.sha(Path(supp.__file__))}
    base.dump(args.out / "manifest.json", manifest)
    try:
        groups, audit = strict_groups(args.repo / "outputs/vlm/dv_rician_predictions.csv", args.repo)
        supp.require_same({s: len(v) for s, v in audit["images_by_split"].items()}, {"train": 275, "validation": 93, "test": 122}, "DV split images")
        supp.require_same(audit["decisions_by_split"], {"train": 9828, "validation": 3414, "test": 4260}, "DV decision counts")
        base.dump(args.out / "data_audit.json", audit)
        payload = reconstruct(groups, args.repo, 6 if args.smoke else None)
        base.dump(args.out / "payload_audit.json", payload)
        if payload["n"] != payload["matched_saved_images"]:
            raise ValueError("Receiver JPEG identity gate failed: cannot assign new wire costs to old answers")
        if args.smoke:
            # Smoke trains on real full split, but no test-energy evaluation with partial payload.
            parts = {s: [g for g in groups if g["split"] == s] for s in ("train", "validation", "test")}
            ratios = base.fit_calibration(parts["train"])
            for mode in ("train_only", "none"):
                x = {s: np.array([base.features(g) for g in rows]) for s, rows in parts.items()}
                y = {s: base.labels(rows, ratios if mode == "train_only" else {}) for s, rows in parts.items()}
                models, hist = base.fit_pair(x["train"], y["train"], x["validation"], y["validation"], 0, 2)
                assert np.isfinite(np.column_stack([base.probabilities(m, x["test"]) for m in models])).all()
                base.dump(args.out / f"smoke_{mode}.json", {"training": hist, "feature_dimension": x["train"].shape[1]})
            base.dump(args.out / "status.json", {"state": "COMPLETE", "smoke": True, "seconds": time.time() - started})
            return
        parts = {s: [g for g in groups if g["split"] == s] for s in ("train", "validation", "test")}
        x = {s: np.array([base.features(g) for g in rows]) for s, rows in parts.items()}
        evlm = supp.read(args.repo / "outputs/energy/gpu_power_phases.json")["phases"]["vlm"]["joule_per_item_incremental"]
        energy = {s: base.energy_matrix(rows, payload, evlm, True) for s, rows in parts.items()}
        baseline_energy = base.energy_matrix(parts["test"], payload, evlm, False)
        for s, rows in parts.items():
            base.dump(args.out / f"{s}_keys.json", [{k: g[k] for k in ("image", "question", "qt", "snr", "class")} for g in rows])
        hashes = {str(p): base.sha(p) for root in (args.formal / "rician/train_only", args.supplement / "rician/train_only") for p in root.glob("*") if p.is_file()}
        for p in [args.repo / "outputs/vlm/dv_rician_predictions.csv", args.repo / "outputs/energy/gpu_power_phases.json"] + [args.repo / f"configs/dv_rician_{t}.json" for t in ("main", "cmp", "extra")]:
            hashes[str(p)] = base.sha(p)
        base.dump(args.out / "source_sha256.json", hashes)
        transfer = {"state": "PENDING_SOURCE_FEATURE_AUDIT"} if args.target_only else frozen_transfer(args, parts, x, energy, baseline_energy)
        summary = {"source_frozen_transfer": transfer, "target_refit": {}}
        base.dump(args.out / "summary.json", summary)
        completed = 0
        for mode in ("train_only", "none"):
            dest = args.out / "target_refit" / mode
            dest.mkdir(parents=True)
            ratios = base.fit_calibration(parts["train"]) if mode == "train_only" else {}
            base.dump(dest / "calibration.json", ratios)
            y = {s: base.labels(rows, ratios) for s, rows in parts.items()}
            policy = base.fit_lut(parts["train"], y["train"])
            base.dump(dest / "lut.json", policy)
            picks = {"image": np.ones(len(y["test"]), int), "detection": np.zeros(len(y["test"]), int),
                "rule": np.array([g["qt"] == "presence" for g in parts["test"]], int),
                "lut": np.array([policy[f'{g["qt"]}|{g["snr"]}'] for g in parts["test"]], int)}
            baselines = {name: base.summarize(parts["test"], y["test"], pick, baseline_energy) for name, pick in picks.items()}
            baselines["oracle_accuracy"] = float(y["test"].max(axis=1).mean())
            base.dump(dest / "baselines.json", baselines)
            weights = base.fit_linear(x["train"], y["train"])
            np.savez_compressed(dest / "evaluation_inputs.npz", test_y=y["test"], validation_y=y["validation"],
                train_y=y["train"], test_energy=energy["test"], validation_energy=energy["validation"],
                baseline_energy=baseline_energy, linear_weights=weights, **picks)
            lr = supp.save_evaluation(dest / "linear", parts, y, energy,
                base.linear_predict(weights, x["validation"]), base.linear_predict(weights, x["test"]),
                {"model": "linear", "features": base.FEATURE_NAMES, "mode": mode, "scope": "target refit"})
            records = []
            for seed in args.seeds:
                start = time.time()
                models, history = base.fit_pair(x["train"], y["train"], x["validation"], y["validation"], seed, args.max_epochs)
                pv = np.column_stack([base.probabilities(m, x["validation"]) for m in models])
                pt = np.column_stack([base.probabilities(m, x["test"]) for m in models])
                record = supp.save_evaluation(dest / f"seed_{seed}", parts, y, energy, pv, pt,
                    {"seed": seed, "mode": mode, "features": base.FEATURE_NAMES, "scope": "target refit", "training": history, "seconds": time.time() - start})
                joblib.dump(models, dest / f"seed_{seed}_models.joblib")
                records.append(record)
                completed += 1
                base.dump(args.out / "status.json", {"state": "RUNNING", "completed_fits": completed, "expected_fits": 20, "mode": mode, "seed": seed, "seconds": time.time() - started})
                print(f"[done DV {mode} seed={seed}] seconds={time.time()-start:.1f}", flush=True)
            summary["target_refit"][mode] = {"mlp": supp.describe(records), "linear": supp.describe([lr]), "baselines": baselines}
            base.dump(args.out / "summary.json", summary)
        base.dump(args.out / "status.json", {"state": "COMPLETE", "scope": "target_refit" if args.target_only else "target_refit_and_source_transfer", "source_transfer": "PENDING_SOURCE_FEATURE_AUDIT" if args.target_only else "COMPLETE", "completed_fits": completed, "linear_sweeps": 2, "source_frozen_mlp_seeds": 0 if args.target_only else len(args.seeds), "seconds": time.time() - started})
    except Exception as exc:
        base.dump(args.out / "status.json", {"state": "FAILED", "error": repr(exc), "seconds": time.time() - started})
        raise


if __name__ == "__main__":
    main()
