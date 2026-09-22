#!/usr/bin/env python3
"""Repair sender-count features, selectively refit affected routers, preserve V1."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

import joblib
import numpy as np

import revision_20260907 as base
import revision_20260907_supplement as supp
import revision_20260907_crossreceiver as sender

SPLITS = ("train", "validation", "test")
CHANNELS = ("awgn", "rician", "rayleigh")
EXPECTED = {"awgn": (0, 0), "rayleigh": (341, 279), "rician": (44, 33)}


def corrected_parts(parts: dict, counts: Counter) -> tuple[dict, dict]:
    fixed = copy.deepcopy(parts)
    corrections, changed_features = [], []
    for split, rows in fixed.items():
        for g in rows:
            before = base.features(g)
            change = sender.restore_count(g, counts[g["image"], g["class"]])
            if change is not None:
                corrections.append(change)
                if before != base.features(g):
                    changed_features.append(change)
    return fixed, {"raw_field_changes": len(corrections), "feature_changes": len(changed_features),
                   "raw_per_split": dict(Counter(r["split"] for r in corrections)),
                   "feature_per_split": dict(Counter(r["split"] for r in changed_features)),
                   "raw_per_type": dict(Counter(r["qt"] for r in corrections)),
                   "feature_per_type": dict(Counter(r["qt"] for r in changed_features)),
                   "raw_records": corrections, "feature_records": changed_features}


def verify_prediction_record(path: Path, x: dict, y: dict, energy: dict, rows: list[dict],
                             with_models: bool = True) -> dict:
    record = supp.read(path.with_suffix(".json"))
    file = path.parent / f"{path.name}_outcomes.npz"
    with zipfile.ZipFile(file) as z:
        if z.testzip() is not None:
            raise ValueError(f"Corrupt NPZ: {file}")
    with np.load(file) as z:
        if not all(np.isfinite(z[k]).all() for k in z.files):
            raise ValueError(f"Nonfinite values: {file}")
        if with_models:
            models = joblib.load(path.parent / f"{path.name}_models.joblib")
            for split, name in (("test", "probabilities"), ("validation", "validation_probabilities")):
                actual = np.column_stack([base.probabilities(m, x[split]) for m in models])
                np.testing.assert_array_equal(actual, z[name])
            if not all(0 < r["epochs_run"] <= 300 and len(r["validation_bce"]) == r["epochs_run"]
                       for r in record["training"]):
                raise ValueError("Invalid training history")
        ts, tp = supp.sweep(y["test"], z["probabilities"], energy["test"])
        vs, vp = supp.sweep(y["validation"], z["validation_probabilities"], energy["validation"])
        np.testing.assert_array_equal(z["prices"], base.PRICES)
        np.testing.assert_array_equal(z["sweep_picks"], tp)
        np.testing.assert_array_equal(z["pick"], tp[0])
        if "validation_sweep_picks" in z:
            np.testing.assert_array_equal(z["validation_sweep_picks"], vp)
        supp.require_same(ts, record["sweep"], "test sweep replay")
        supp.require_same(vs, record["validation_sweep"], "validation sweep replay")
        selected = supp.select_price(vs)
        supp.require_same(selected, record["validation_selected_index"], "validation-only selection")
        supp.require_same(ts[selected], record["selected_test"], "selected test replay")
        supp.require_same(base.summarize(rows, y["test"], tp[0], energy["test"]), record["main"], "main summary")
        if "detector_free_image_path_sensitivity" in record:
            de = supp.deployment_energy(energy["test"], record["variant"])
            sensitivity = record["detector_free_image_path_sensitivity"]
            supp.require_same(base.metrics(y["test"], tp[0], de), sensitivity["unpriced"], "sensitivity unpriced")
            supp.require_same(base.metrics(y["test"], tp[selected], de), sensitivity["selected"], "sensitivity selected")
    return record


def copy_verified(source: Path, target: Path, provenance: dict) -> None:
    """Byte-preserving copy, recording immutable origin; never replace a target."""
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    digest = base.sha(source)
    supp.require_same(base.sha(target), digest, "copied artifact SHA256")
    provenance[str(target)] = {"source": str(source), "sha256": digest, "operation": "byte-identical reuse"}


def copy_tree_verified(source: Path, target: Path, provenance: dict) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            copy_verified(path, target / path.relative_to(source), provenance)


def prepare(repo: Path, oldformal: Path, oldsupp: Path, channel: str, hashes: dict) -> dict:
    d = supp.prepare_channel(repo, oldformal, channel, hashes)
    stems = {"awgn": ("v2_0_awgn", "v2_0_awgn_cmp", "v2_0_awgn_extra"),
             "rayleigh": ("v2_0_rayleigh", "v2_0_rayleigh_cmp", "v2_0_rayleigh_extra"),
             "rician": ("v2_0_snr", "v2_0_rician_cmp", "v2_0_rician_extra")}[channel]
    main_path = repo / f"outputs/detector/{stems[0]}_detections.csv"
    counts, canonical = sender.load_boxes(main_path)
    caches = []
    for stem in stems:
        path = repo / f"outputs/detector/{stem}_detections.csv"
        count, boxes = sender.load_boxes(path)
        hashes[str(path)] = base.sha(path)
        for iid, signatures in boxes.items():
            sender.check_equal(signatures, canonical.get(iid, Counter()), f"sender boxes {channel} {stem} {iid}")
        caches.append(count)
    fixed, audit = corrected_parts(d["parts"], counts)
    supp.require_same((audit["raw_field_changes"], audit["feature_changes"]), EXPECTED[channel], "expected sender correction counts")
    for rows in fixed.values():
        for g in rows:
            ix = 0 if g["qt"] in ("presence", "counting") else (1 if g["qt"] == "comparison" else 2)
            supp.require_same(caches[ix][g["image"], g["class"]], g["raw"], "per-task sender cache coverage")
    x = {s: np.asarray([base.features(g) for g in rows]) for s, rows in fixed.items()}
    for mode, m in d["modes"].items():
        md = oldformal / channel / mode
        ratios = base.fit_calibration(fixed["train"]) if mode == "train_only" else {}
        supp.require_same(ratios, supp.read(md / "calibration.json"), "calibration unchanged by sender fix")
        for s in SPLITS:
            np.testing.assert_array_equal(base.labels(fixed[s], ratios), m["y"][s])
        supp.require_same(base.fit_lut(fixed["train"], m["y"]["train"]), supp.read(md / "lut.json"), "LUT unchanged")
        with np.load(oldsupp / channel / mode / "evaluation_inputs.npz") as z:
            for s in ("test", "validation"):
                np.testing.assert_array_equal(z[f"{s}_y"], m["y"][s])
                np.testing.assert_array_equal(z[f"{s}_energy"], d["energy"][s])
        if channel == "awgn":
            for s in SPLITS:
                np.testing.assert_array_equal(x[s], d["x"][s])
            for seed in range(10):
                verify_prediction_record(md / f"seed_{seed}", x, m["y"], d["energy"], fixed["test"])
    for variant, inds in supp.VARIANTS.items():
        if channel == "awgn" or variant in ("no_detector", "question_type_only"):
            for s in SPLITS:
                np.testing.assert_array_equal(x[s][:, inds], d["x"][s][:, inds])
            xv = {s: a[:, inds] for s, a in x.items()}
            for seed in range(10):
                verify_prediction_record(oldsupp / channel / "train_only" / variant / f"seed_{seed}",
                                         xv, d["modes"]["train_only"]["y"], d["energy"], fixed["test"])
    # g.raw does not enter energy_matrix. Recompute all split costs to assert that fact.
    payload = supp.read(oldformal / channel / "payload_audit.json")
    evlm = supp.read(repo / "outputs/energy/gpu_power_phases.json")["phases"]["vlm"]["joule_per_item_incremental"]
    for s in SPLITS:
        np.testing.assert_array_equal(base.energy_matrix(fixed[s], payload, evlm, True), d["energy"][s])
    for path in (oldsupp / channel).rglob("*"):
        if path.is_file():
            hashes[str(path)] = base.sha(path)
    audit.update(source_feature_policy="original_sender_detector_boxes", source_sha256={str(repo / f"outputs/detector/{s}_detections.csv"): hashes[str(repo / f"outputs/detector/{s}_detections.csv")] for s in stems},
                 invariant="labels, calibration, LUT, branch costs, image splits/order unchanged; sender count independent of SNR",
                 reused_variants=["no_detector", "question_type_only"] if channel != "awgn" else ["full", *supp.VARIANTS])
    d.update(parts=fixed, x=x, sender_audit=audit)
    return d


def fit_save(dest: Path, d: dict, mode: str, seed: int, epochs: int, inds: list[int], variant: str) -> dict:
    begin = time.time()
    x = {s: a[:, inds] for s, a in d["x"].items()}
    y = d["modes"][mode]["y"]
    models, history = base.fit_pair(x["train"], y["train"], x["validation"], y["validation"], seed, epochs)
    pv = np.column_stack([base.probabilities(m, x["validation"]) for m in models])
    pt = np.column_stack([base.probabilities(m, x["test"]) for m in models])
    result = supp.save_evaluation(dest, d["parts"], y, d["energy"], pv, pt,
                {"seed": seed, "mode": mode, "variant": variant, "features": [base.FEATURE_NAMES[i] for i in inds],
                 "source_feature_policy": "original_sender_detector_boxes", "training": history, "seconds": time.time() - begin}, variant)
    joblib.dump(models, dest.parent / f"{dest.name}_models.joblib")
    verify_prediction_record(dest, x, y, d["energy"], d["parts"]["test"])
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--old-formal", type=Path, required=True)
    ap.add_argument("--old-supplement", type=Path, required=True)
    ap.add_argument("--formal", type=Path, required=True)
    ap.add_argument("--supplement", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    oldmanifest = supp.read(args.old_formal / "manifest.json")
    oldsm = supp.read(args.old_supplement / "manifest.json")
    for source in (args.old_formal, args.old_supplement):
        supp.require_same(supp.read(source / "status.json")["state"], "COMPLETE", "source completion")
    supp.require_same(base.sha(Path(base.__file__)), oldmanifest["script_sha256"], "base source")
    supp.require_same(base.sha(Path(supp.__file__)), oldsm["script_sha256"], "supplement source")
    # Refuse all output collisions before creating any experiment directory.
    for dest in (args.formal, args.supplement, args.out):
        if dest.exists():
            raise FileExistsError(dest)
    for dest in (args.formal, args.supplement, args.out):
        dest.mkdir(parents=True, exist_ok=False)
    started = time.time()
    seeds = [0] if args.smoke else list(range(10))
    epochs = 2 if args.smoke else 300
    import sklearn
    supp.require_same(np.__version__, oldmanifest["numpy"], "numpy version")
    supp.require_same(sklearn.__version__, oldmanifest["sklearn"], "sklearn version")
    manifest = {"pid": os.getpid(), "started_unix": started, "smoke": args.smoke,
                "script_sha256": base.sha(Path(__file__)), "base_script_sha256": base.sha(Path(base.__file__)),
                "supplement_script_sha256": base.sha(Path(supp.__file__)),
                "sender_helper_sha256": base.sha(Path(sender.__file__)),
                "source_feature_policy": "original_sender_detector_boxes", "channels": list(CHANNELS),
                "new_fit_seeds": seeds, "seeds": list(range(10)), "max_epochs": epochs,
                "new_full_fits": 4 * len(seeds), "new_ablation_fits": 2 * len(seeds),
                "new_linear_fits": 4, "reused_full_fits": 20, "reused_ablation_fits": 70,
                "reused_linear_sweeps": 2, "features": base.FEATURE_NAMES, "prices": base.PRICES,
                "selection": oldmanifest["selection"], "split": oldmanifest["split"],
                "old_formal": str(args.old_formal), "old_supplement": str(args.old_supplement),
                "old_formal_manifest_sha256": base.sha(args.old_formal / "manifest.json"),
                "old_supplement_manifest_sha256": base.sha(args.old_supplement / "manifest.json"),
                "python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__,
                "scope": "selective CPU refitting only; no VLM/detector/codec execution, no test selection",
                "reuse": "AWGN all; no_detector/question_type_only all; fixed branches/rule/LUT/calibration after invariance checks",
                "formal_path": str(args.formal), "supplement_path": str(args.supplement)}
    for dest in (args.formal, args.supplement, args.out):
        base.dump(dest / "manifest.json", manifest)
        base.dump(dest / "status.json", {"state": "RUNNING", "completed_new_fits": 0})
    try:
        data, hashes, provenance, summary = {}, {}, {}, {}
        for channel in CHANNELS:
            data[channel] = prepare(args.repo, args.old_formal, args.old_supplement, channel, hashes)
            print(f"[verified {channel}] sender={data[channel]['sender_audit']['raw_field_changes']}/{data[channel]['sender_audit']['feature_changes']}", flush=True)
        base.dump(args.out / "reused_artifact_sha256.json", hashes)
        completed = 0
        for channel, d in data.items():
            fd, sd = args.formal / channel, args.supplement / channel
            of, oldsp = args.old_formal / channel, args.old_supplement / channel
            if channel == "awgn":
                copy_tree_verified(of, fd, provenance)
                copy_tree_verified(oldsp, sd, provenance)
                summary[channel] = supp.read(args.old_supplement / "summary.json")[channel]
            else:
                for name in ("data_audit.json", "payload_audit.json", "test_keys.json"):
                    copy_verified(of / name, fd / name, provenance)
                copy_verified(oldsp / "data_audit.json", sd / "data_audit.json", provenance)
                summary[channel] = {}
                for mode, m in d["modes"].items():
                    fm, sm = fd / mode, sd / mode
                    for name in ("calibration.json", "lut.json"):
                        copy_verified(of / mode / name, fm / name, provenance)
                    sm.mkdir(parents=True, exist_ok=True)
                    y, x, en = m["y"], d["x"], d["energy"]
                    w = base.fit_linear(x["train"], y["train"])
                    lp = base.linear_predict(w, x["test"])
                    pv = base.linear_predict(w, x["validation"])
                    with np.load(of / mode / "baseline_outcomes.npz") as old:
                        arrays = {k: old[k].copy() for k in old.files}
                    oldbase = supp.read(of / mode / "baselines.json")
                    for name in ("image", "detection", "rule", "lut"):
                        supp.require_same(base.summarize(d["parts"]["test"], y["test"], arrays[name], arrays["baseline_energy"]), oldbase[name], f"unchanged baseline {name}")
                    arrays.update(linear_weights=w, linear_probabilities=lp, linear=base.choose(lp, en["test"], 0))
                    np.savez_compressed(fm / "baseline_outcomes.npz", **arrays)
                    oldbase["linear"] = base.summarize(d["parts"]["test"], y["test"], arrays["linear"], en["test"])
                    base.dump(fm / "baselines.json", oldbase)
                    np.savez_compressed(sm / "evaluation_inputs.npz", test_y=y["test"], validation_y=y["validation"],
                                        test_energy=en["test"], validation_energy=en["validation"], linear_weights=w)
                    lr = supp.save_evaluation(sm / "linear", d["parts"], y, en, pv, lp,
                                {"model": "linear", "features": base.FEATURE_NAMES, "source_feature_policy": "original_sender_detector_boxes"})
                    verify_prediction_record(sm / "linear", x, y, en, d["parts"]["test"], False)
                    print(f"[linear done {channel} {mode}]", flush=True)
                    records = []
                    for seed in seeds:
                        records.append(fit_save(fm / f"seed_{seed}", d, mode, seed, epochs, list(range(18)), "full"))
                        completed += 1
                        for dest in (args.formal, args.supplement, args.out):
                            base.dump(dest / "status.json", {"state": "RUNNING", "completed_new_fits": completed, "expected_new_fits": 6 * len(seeds)})
                        print(f"[fit done {channel} {mode} full seed={seed}]", flush=True)
                    summary[channel][mode] = {"linear": supp.describe([lr]), "full": supp.describe(records)}
                for variant in ("no_detector", "question_type_only"):
                    copy_tree_verified(oldsp / "train_only" / variant, sd / "train_only" / variant, provenance)
                    summary[channel]["train_only"][variant] = supp.read(args.old_supplement / "summary.json")[channel]["train_only"][variant]
                records = []
                target = sd / "train_only" / "no_snr"
                target.mkdir(parents=True, exist_ok=False)
                for seed in seeds:
                    records.append(fit_save(target / f"seed_{seed}", d, "train_only", seed, epochs, supp.VARIANTS["no_snr"], "no_snr"))
                    completed += 1
                    for dest in (args.formal, args.supplement, args.out):
                        base.dump(dest / "status.json", {"state": "RUNNING", "completed_new_fits": completed, "expected_new_fits": 6 * len(seeds)})
                    print(f"[fit done {channel} train_only no_snr seed={seed}]", flush=True)
                summary[channel]["train_only"]["no_snr"] = supp.describe(records)
            # Explicit sidecar: unchanged data_audit.json remains byte-identical.
            base.dump(fd / "sender_count_audit.json", d["sender_audit"])
            base.dump(sd / "sender_count_audit.json", d["sender_audit"])
            for mode, m in d["modes"].items():
                np.savez_compressed(fd / mode / "sender_feature_inputs.npz",
                    **{f"{s}_x": d["x"][s] for s in SPLITS}, **{f"{s}_y": m["y"][s] for s in SPLITS})
            for dest in (args.formal, args.supplement, args.out):
                base.dump(dest / "summary.json", summary)
                base.dump(dest / "reuse_provenance.json", provenance)
        for dest in (args.formal, args.supplement, args.out):
            base.dump(dest / "status.json", {"state": "COMPLETE", "completed_new_fits": completed,
                      "new_full_fits": 4 * len(seeds), "new_no_snr_fits": 2 * len(seeds), "new_linear_fits": 4,
                      "reused_full_fits": 20, "reused_ablation_fits": 70, "reused_linear_sweeps": 2,
                      "seconds": time.time() - started, "smoke": args.smoke})
    except Exception as exc:
        for dest in (args.formal, args.supplement, args.out):
            base.dump(dest / "status.json", {"state": "FAILED", "error": repr(exc), "seconds": time.time() - started})
        raise


if __name__ == "__main__":
    main()
