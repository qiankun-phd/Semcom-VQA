#!/usr/bin/env python3
"""Matched-cache receiver-specific fitting and frozen-source transfer; CPU only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np

import revision_20260907 as base

RECEIVERS = {
    "qwen2": ("Qwen2-VL-2B-Instruct", ["v3_0_rician"]),
    "qwen25": ("Qwen2.5-VL-3B-Instruct", [f"v25_rician_{p}" for p in ("main", "cmp", "extra")]),
    "smol": ("SmolVLM-Instruct", [f"v26_rician_{p}" for p in ("main", "cmp", "extra")]),
}
SPLITS = ("train", "validation", "test")
BOX_FIELDS = ("image_id", "category", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
              "confidence", "detector_model")


def key(g: dict) -> tuple:
    return g["image"], g["question"], g["snr"]


def check_equal(a: object, b: object, label: str) -> None:
    if a != b:
        raise ValueError(f"Mismatch {label}: {a!r} != {b!r}")


def load_receiver(repo: Path, receiver: str) -> tuple[dict, dict]:
    """Collapse freshness copies only if every prediction/evidence field agrees."""
    expected_model, stems = RECEIVERS[receiver]
    groups, duplicates, models, hashes = {}, 0, Counter(), {}
    csv.field_size_limit(sys.maxsize)
    for stem in stems:
        path = repo / f"outputs/vlm/{stem}_predictions.csv"
        hashes[str(path)] = base.sha(path)
        config = repo / f"configs/{stem}.json"
        if config.is_file():
            hashes[str(config)] = base.sha(config)
            cfg = json.loads(config.read_text())
            if expected_model not in cfg["vlm"]["model_name"]:
                raise ValueError(f"Receiver config identity mismatch: {config}")
        with path.open(newline="") as stream:
            for r in csv.DictReader(stream):
                b = r["service_level"]
                if b not in ("1", "2"):
                    continue
                snr = int(float(r["snr_bin"].replace("dB", "")))
                k = (r["image_id"], r["question"], snr)
                metadata = {"image": k[0], "question": k[1], "snr": snr,
                            "qt": r["question_type"], "class": r["target_class"],
                            "split": base.split_for(k[0]), "gt": int(float(r["object_count"])),
                            "answer": r["ground_truth_answer"]}
                if metadata["qt"] not in base.QTYPES or snr not in base.SNRS:
                    raise ValueError(f"Unexpected question type/SNR: {k}")
                g = groups.setdefault(k, metadata.copy())
                for name, value in metadata.items():
                    check_equal(g[name], value, f"row metadata {k} {name}")
                branch = {"correct": base.truth(r["correct"]), "bytes": int(r["payload_bytes"]),
                          "prediction": r["predicted_answer"], "normalized": r["normalized_prediction"],
                          "model": r["model_name"], "evidence": r["evidence_repr"],
                          "decoder": r["decoder_mode"], "detector": r["detector_model"],
                          "detector_conf": r["detector_conf"]}
                models[branch["model"]] += 1
                if b == "1":
                    if branch["model"] != "semantic-token-decoder":
                        raise ValueError("Detection branch is not the symbolic decoder")
                    branch.update(raw=int(float(r["raw_detector_count"])),
                                  transmitted=int(float(r["transmitted_detector_count"])),
                                  calibrated=int(float(r["calibrated_detector_count"])),
                                  gt=metadata["gt"], answer=metadata["answer"])
                else:
                    if expected_model not in branch["model"]:
                        raise ValueError(f"Cached image receiver mismatch: {branch['model']}")
                    branch["path"] = r["image_path"]
                if b in g:
                    duplicates += 1
                    check_equal(g[b], branch, f"duplicate {k} branch {b}")
                g[b] = branch
    for k, g in groups.items():
        if "1" not in g or "2" not in g:
            raise ValueError(f"Unpaired decision {receiver} {k}")
        g["raw"] = g["1"]["raw"]
    return groups, {"source_sha256": hashes, "models": dict(models), "decisions": len(groups),
                    "identical_duplicates": duplicates}


def compare_sender(a: dict, b: dict) -> dict:
    """Old counting calibration may differ, but is replaced using common train data."""
    for name in ("image", "question", "snr", "qt", "class", "split", "gt", "answer", "raw"):
        check_equal(a[name], b[name], f"sender metadata {key(a)} {name}")
    counts = {}
    ignored = {"correct", "calibrated", "prediction", "normalized"} if a["qt"] == "counting" else set()
    for name in a["1"]:
        if name in ignored:
            if a["1"][name] != b["1"][name]:
                counts[name] = 1
        else:
            check_equal(a["1"][name], b["1"][name], f"detection evidence {key(a)} {name}")
    return counts


def coverage(groups: list[dict]) -> dict:
    result = {}
    for split in SPLITS:
        rows = [g for g in groups if g["split"] == split]
        result[split] = {"decisions": len(rows), "images": sorted({g["image"] for g in rows}),
                         "per_type": {}}
        for qt in base.QTYPES:
            sub = [g for g in rows if g["qt"] == qt]
            result[split]["per_type"][qt] = {"decisions": len(sub), "images": len({g["image"] for g in sub})}
    return result


def load_boxes(path: Path) -> tuple[Counter, dict]:
    counts, boxes = Counter(), defaultdict(Counter)
    with path.open(newline="") as stream:
        for r in csv.DictReader(stream):
            counts[r["image_id"], r["category"]] += 1
            boxes[r["image_id"]][tuple(r[f] for f in BOX_FIELDS)] += 1
    return counts, boxes


def restore_count(g: dict, count: int) -> dict | None:
    if g["qt"] in ("presence", "counting"):
        check_equal(g["raw"], count, f"primary task sender anchor {key(g)}")
    change = None
    if g["raw"] != count:
        change = {"image": g["image"], "question": g["question"], "snr": g["snr"],
                  "qt": g["qt"], "split": g["split"], "logged_raw": g["raw"], "sender_count": count}
    g["raw"] = count
    return change


def restore_sender(repo: Path, all_groups: dict, out: Path) -> None:
    """Recover pre-transmission counts from actual source detector box caches."""
    stems = {"qwen2": ["v2_0_snr", "v2_0_rician_cmp", "v2_0_rician_extra"],
             "qwen25": [f"v25_rician_{p}" for p in ("main", "cmp", "extra")],
             "smol": [f"v26_rician_{p}" for p in ("main", "cmp", "extra")]}
    canonical_path = repo / "outputs/detector/v2_0_snr_detections.csv"
    canonical_counts, canonical_boxes = load_boxes(canonical_path)
    audit = {"source_sha256": {}, "corrections": {},
             "source_bug": "run_v1_detector_eval.py extra types put transmitted counts in raw_detector_count position; recover g.raw only, retain original g['1']['raw'] for audit",
             "formal_v1_readonly_audit": {}}
    for receiver, names in stems.items():
        local_counts = []
        for stem in names:
            path = repo / f"outputs/detector/{stem}_detections.csv"
            count, boxes = load_boxes(path)
            audit["source_sha256"][str(path)] = base.sha(path)
            for iid, signatures in boxes.items():
                check_equal(signatures, canonical_boxes.get(iid, Counter()), f"original detector boxes {stem} {iid}")
            local_counts.append(count)
        changes = []
        for g in all_groups[receiver].values():
            source = 0 if g["qt"] in ("presence", "counting") else (1 if g["qt"] == "comparison" else 2)
            k = (g["image"], g["class"])
            count = canonical_counts[k]
            check_equal(local_counts[source][k], count, f"sender detector cache coverage {receiver} {key(g)}")
            change = restore_count(g, count)
            if change is not None:
                changes.append(change)
        audit["corrections"][receiver] = changes
    for channel, stem in (("awgn", "v2_0_awgn"), ("rayleigh", "v2_0_rayleigh"), ("rician", "v2_0_snr")):
        path = repo / f"outputs/detector/{stem}_detections.csv"
        counts, boxes = load_boxes(path)
        for iid, signatures in boxes.items():
            check_equal(signatures, canonical_boxes.get(iid, Counter()), f"formal sender frames {channel} {iid}")
        prediction_path = repo / f"outputs/vlm/v3_0_{channel}_predictions.csv"
        groups, _ = base.load_groups(prediction_path)
        audit["source_sha256"][str(path)] = base.sha(path)
        audit["source_sha256"][str(prediction_path)] = base.sha(prediction_path)
        changes = [{"image": g["image"], "question": g["question"], "snr": g["snr"],
                    "qt": g["qt"], "split": g["split"], "logged_raw": g["raw"],
                    "sender_count": counts[g["image"], g["class"]]}
                   for g in groups if g["raw"] != counts[g["image"], g["class"]]]
        audit["formal_v1_readonly_audit"][channel] = {"decisions": len(groups), "changed": len(changes),
                    "per_type": dict(Counter(g["qt"] for g in changes)),
                    "per_split": dict(Counter(g["split"] for g in changes)), "records": changes}
    audit["source_sha256"][str(repo / "scripts/run_v1_detector_eval.py")] = base.sha(repo / "scripts/run_v1_detector_eval.py")
    base.dump(out / "sender_count_audit.json", audit)


def prepare(repo: Path, out: Path) -> tuple[dict, dict, list[str]]:
    all_groups, audits = {}, {}
    for receiver in RECEIVERS:
        all_groups[receiver], audits[receiver] = load_receiver(repo, receiver)
    restore_sender(repo, all_groups, out)
    common = sorted(set.intersection(*(set(d) for d in all_groups.values())))
    if not common:
        raise ValueError("Empty common paired-branch subset")
    matched = {r: [d[k] for k in common] for r, d in all_groups.items()}
    mismatch = {r: Counter() for r in RECEIVERS if r != "qwen2"}
    image_hashes = {}
    identity_records = {}
    for i, k in enumerate(common):
        source = matched["qwen2"][i]
        for receiver in RECEIVERS:
            g = matched[receiver][i]
            if receiver != "qwen2":
                mismatch[receiver].update(compare_sender(source, g))
            path = Path(g["2"]["path"])
            if not path.is_absolute():
                path = repo / path
            if str(path) not in image_hashes:
                image_hashes[str(path)] = base.sha(path)
            check_equal(path.stat().st_size, g["2"]["bytes"], "logged receiver JPEG bytes")
            # Match received image bytes, not filenames or VLM preprocessing.
            identity_records.setdefault((g["image"], g["snr"]), set()).add(image_hashes[str(path)])
        paths = []
        for receiver in RECEIVERS:
            p = Path(matched[receiver][i]["2"]["path"])
            paths.append(str(p if p.is_absolute() else repo / p))
        if len({image_hashes[p] for p in paths}) != 1:
            raise ValueError(f"Receiver JPEGs differ on common key {k}")
    rows = matched["qwen2"]
    cov = coverage(rows)
    for split in SPLITS:
        if not cov[split]["decisions"]:
            raise ValueError(f"Empty {split}")
    by_image = defaultdict(set)
    for g in rows:
        by_image[g["image"]].add(g["qt"])
    complete_images = sorted(i for i, types in by_image.items() if types == set(base.QTYPES))
    audit = {"receivers": audits, "common_decisions": len(common), "coverage": cov,
             "old_counting_calibration_differences_replaced": {r: dict(c) for r, c in mismatch.items()},
             "identical_receiver_image_snr_pairs": len(identity_records),
             "common_all_five_types_images": complete_images,
             "comparison_scope": "same sender evidence and received JPEG; receiver-specific internal preprocessing remains part of receiver",
             "pooling": "actual common-key task mix; unequal task-type image coverage, not full five-type benchmark"}
    base.dump(out / "data_audit.json", audit)
    base.dump(out / "receiver_image_sha256.json", image_hashes)
    base.dump(out / "common_keys.json", [{k: g[k] for k in ("image", "question", "snr", "qt", "class", "split")} for g in rows])
    return matched, audit, complete_images


def summarize(rows: list[dict], y: np.ndarray, pick: np.ndarray, complete_images: list[str]) -> dict:
    def metric(mask: np.ndarray) -> dict:
        selected = pick[mask]
        target = y[mask]
        return {"n": len(selected), "accuracy": float(target[np.arange(len(selected)), selected].mean()),
                "image_fraction": float(selected.mean())}
    result = {"pooled": metric(np.ones(len(rows), dtype=bool)), "per_type": {}, "per_snr": {}}
    for field, values, dest in (("qt", base.QTYPES, "per_type"), ("snr", base.SNRS, "per_snr")):
        for value in values:
            mask = np.array([g[field] == value for g in rows])
            if mask.any():
                result[dest][str(value)] = metric(mask)
    mask = np.array([g["image"] in complete_images for g in rows])
    result["all_five_types_image_subset"] = metric(mask) if mask.any() else None
    return result


def describe(records: list[dict]) -> dict:
    return {"n_seeds": len(records), "aggregation": "pooled matched test keys; mean and sample SD across seeds",
            **{name: {"mean": float(np.mean([r["pooled"][name] for r in records])),
                       "sample_sd": float(np.std([r["pooled"][name] for r in records], ddof=1)) if len(records) > 1 else None}
               for name in ("accuracy", "image_fraction")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seeds = [0] if args.smoke else list(range(10))
    epochs = 2 if args.smoke else 300
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    import sklearn
    base.dump(args.out / "manifest.json", {"pid": os.getpid(), "started_unix": started,
              "script_sha256": base.sha(Path(__file__)), "base_script_sha256": base.sha(Path(base.__file__)),
              "smoke": args.smoke, "seeds": seeds, "max_epochs": epochs, "features": base.FEATURE_NAMES,
              "python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__,
              "split": "original crc32(image)%100: test<20, validation>=80, train otherwise",
              "scope": "common-key receiver-specific refit and Qwen2-frozen transfer; lambda=0 only",
              "energy": "not reported: receiver-specific energy measurements unavailable",
              "inference": "no VLM, detector or codec execution; existing cache only"})
    try:
        matched, audit, complete_images = prepare(args.repo, args.out)
        parts = {r: {s: [g for g in rows if g["split"] == s] for s in SPLITS} for r, rows in matched.items()}
        x = {s: np.asarray([base.features(g) for g in parts["qwen2"][s]]) for s in SPLITS}
        ratios = base.fit_calibration(parts["qwen2"]["train"])
        base.dump(args.out / "shared_train_calibration.json", ratios)
        y = {r: {s: base.labels(rows, ratios) for s, rows in split.items()} for r, split in parts.items()}
        for r in RECEIVERS:
            for s in SPLITS:
                np.testing.assert_array_equal(x[s], np.asarray([base.features(g) for g in parts[r][s]]))
                np.testing.assert_array_equal(y[r][s][:, 0], y["qwen2"][s][:, 0])
        print(f"[audit passed] common={len(matched['qwen2'])} splits={[(s,len(x[s])) for s in SPLITS]}", flush=True)
        summary, source_probabilities, completed = {}, {}, 0
        for receiver in RECEIVERS:
            dest = args.out / receiver
            dest.mkdir()
            test = parts[receiver]["test"]
            yt = y[receiver]["test"]
            policy = base.fit_lut(parts[receiver]["train"], y[receiver]["train"])
            w = base.fit_linear(x["train"], y[receiver]["train"])
            lp = base.linear_predict(w, x["test"])
            picks = {"image": np.ones(len(test), dtype=np.int8), "detection": np.zeros(len(test), dtype=np.int8),
                     "rule": np.array([g["qt"] == "presence" for g in test], dtype=np.int8),
                     "lut": np.array([policy.get(f'{g["qt"]}|{g["snr"]}', 0) for g in test]),
                     "linear": (lp[:, 1] > lp[:, 0]).astype(np.int8)}
            baselines = {name: summarize(test, yt, pk, complete_images) for name, pk in picks.items()}
            baselines["oracle_accuracy"] = float(yt.max(axis=1).mean())
            base.dump(dest / "baselines.json", baselines)
            base.dump(dest / "lut.json", policy)
            np.savez_compressed(dest / "evaluation_inputs.npz", **{f"{s}_y": y[receiver][s] for s in SPLITS},
                                **{f"{s}_x": x[s] for s in SPLITS}, linear_weights=w, linear_probabilities=lp, **picks)
            records, transfers = [], []
            for seed in seeds:
                begin = time.time()
                models, history = base.fit_pair(x["train"], y[receiver]["train"], x["validation"],
                                               y[receiver]["validation"], seed, epochs)
                pt = np.column_stack([base.probabilities(m, x["test"]) for m in models])
                pv = np.column_stack([base.probabilities(m, x["validation"]) for m in models])
                pk = (pt[:, 1] > pt[:, 0]).astype(np.int8)
                result = summarize(test, yt, pk, complete_images)
                if receiver == "qwen2":
                    source_probabilities[seed] = pt.copy()
                sp = source_probabilities[seed]
                transfer_pick = (sp[:, 1] > sp[:, 0]).astype(np.int8)
                transfer = summarize(test, yt, transfer_pick, complete_images)
                record = {"seed": seed, "receiver": receiver, "main": result, "source_frozen": transfer,
                          "training": history, "seconds": time.time() - begin}
                base.dump(dest / f"seed_{seed}.json", record)
                np.savez_compressed(dest / f"seed_{seed}_outcomes.npz", probabilities=pt,
                                    validation_probabilities=pv, pick=pk, source_frozen_probabilities=sp,
                                    source_frozen_pick=transfer_pick)
                joblib.dump(models, dest / f"seed_{seed}_models.joblib")
                records.append(result)
                transfers.append(transfer)
                completed += 1
                base.dump(args.out / "status.json", {"state": "RUNNING", "completed_fits": completed,
                          "expected_fits": len(RECEIVERS) * len(seeds), "receiver": receiver, "seed": seed})
                print(f"[done {receiver} seed={seed}] seconds={record['seconds']:.1f}", flush=True)
            summary[receiver] = {"refit": describe(records), "source_frozen": describe(transfers),
                                  "baselines": baselines}
            base.dump(args.out / "summary.json", summary)
        base.dump(args.out / "status.json", {"state": "COMPLETE", "completed_fits": completed,
                  "seconds": time.time() - started, "smoke": args.smoke})
    except Exception as exc:
        base.dump(args.out / "status.json", {"state": "FAILED", "error": repr(exc), "seconds": time.time() - started})
        raise


if __name__ == "__main__":
    main()
