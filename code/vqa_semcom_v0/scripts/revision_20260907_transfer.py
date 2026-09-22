#!/usr/bin/env python3
"""Source-frozen VisDrone->DroneVehicle evaluation; no fit or codec replay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np

import revision_20260907 as base
import revision_20260907_supplement as supp
import revision_20260907_dronevehicle as dv
import revision_20260907_crossreceiver as cross

POLICY = "original_sender_detector_boxes"


def dependency_state(formal: Path, supplement: Path) -> tuple[bool, str]:
    for root in (formal, supplement):
        status = root / "status.json"
        if not status.is_file():
            return False, f"Missing {status}"
        state = supp.read(status).get("state")
        if state != "COMPLETE":
            return False, f"{root.name}: {state}"
        manifest = supp.read(root / "manifest.json")
        if manifest.get("source_feature_policy") != POLICY:
            raise ValueError(f"Rejecting source with unverified sender feature provenance: {root}")
    return True, "Both corrected source bundles COMPLETE"


def selected_source_price(record: dict) -> tuple[int, float]:
    supp.require_same([r["lambda"] for r in record["validation_sweep"]], base.PRICES, "source fixed price grid")
    index = supp.select_price(record["validation_sweep"])
    supp.require_same(index, record["validation_selected_index"], "source validation selection")
    supp.require_same(record["sweep"][index], record["selected_test"], "source selected-test record integrity")
    return index, base.PRICES[index]


def frozen_evaluate(rows: list[dict], y: np.ndarray, probability: np.ndarray,
                    energy: np.ndarray, source_record: dict) -> tuple[dict, dict]:
    # Target labels are used for reporting only, never for choice of price.
    index, price = selected_source_price(source_record)
    p0 = base.choose(probability, energy, 0.0)
    ps = base.choose(probability, energy, price)
    return {"source_selected_index": index, "source_selected_lambda": price,
            "source_unpriced_validation": source_record["validation_sweep"][0],
            "source_selected_validation": source_record["validation_sweep"][index],
            "main": base.summarize(rows, y, p0, energy),
            "selected_test": base.metrics(y, ps, energy),
            "selected_summary": base.summarize(rows, y, ps, energy)}, {
                "probabilities": probability, "pick": p0, "selected_pick": ps}


def verify_target(repo: Path, target: Path) -> dict:
    state = supp.read(target / "status.json")
    supp.require_same(state["state"], "COMPLETE", "target refit complete")
    supp.require_same(state["scope"], "target_refit", "target cached scope")
    groups, audit = dv.strict_groups(repo / "outputs/vlm/dv_rician_predictions.csv", repo)
    supp.require_same(audit, supp.read(target / "data_audit.json"), "DV full source hashes/split/count restore")
    rows = [g for g in groups if g["split"] == "test"]
    keys = [{k: g[k] for k in ("image", "question", "qt", "snr", "class")} for g in rows]
    supp.require_same(keys, supp.read(target / "test_keys.json"), "DV frozen test keys")
    supp.require_same(len(rows), 4260, "DV test decision count")
    payload = supp.read(target / "payload_audit.json")
    supp.require_same(payload["n"], 2940, "DV payload count")
    supp.require_same(payload["matched_saved_images"], payload["n"], "DV payload identity")
    if not all(all(r["path_matches"].values()) for r in payload["records"].values()):
        raise ValueError("Unmatched target payload audit")
    supp.require_same(set(payload["records"]), {f'{g["image"]}|{g["snr"]}' for g in groups}, "DV payload coverage")
    for iid, digest in payload["source_image_sha256"].items():
        supp.require_same(base.sha(Path(payload["source_directory"]) / f"{iid}.jpg"), digest, "DV original image hash")
    for path, digest in supp.read(target / "source_sha256.json").items():
        supp.require_same(base.sha(Path(path)), digest, "DV reused source hashes")
    supp.require_same(base.sha(repo / "src/vqa_semcom/degradation/digital_link.py"), payload["link_source_sha256"], "DV codec source")
    evlm = supp.read(repo / "outputs/energy/gpu_power_phases.json")["phases"]["vlm"]["joule_per_item_incremental"]
    energy = base.energy_matrix(rows, payload, evlm, True)
    baseline_energy = base.energy_matrix(rows, payload, evlm, False)
    target_ratios = supp.read(target / "target_refit/train_only/calibration.json")
    target_y = base.labels(rows, target_ratios)
    with np.load(target / "target_refit/train_only/evaluation_inputs.npz") as old:
        np.testing.assert_array_equal(target_y, old["test_y"])
        np.testing.assert_array_equal(energy, old["test_energy"])
        np.testing.assert_array_equal(baseline_energy, old["baseline_energy"])
    return {"rows": rows, "keys": keys, "audit": audit, "x": np.array([base.features(g) for g in rows]),
            "energy": energy, "baseline_energy": baseline_energy, "target_y": target_y,
            "evlm": evlm, "payload_audit_sha256": base.sha(target / "payload_audit.json")}


def source_validation(repo: Path, formal: Path) -> dict:
    root = formal / "rician"
    groups, audit = base.load_groups(repo / "outputs/vlm/v3_0_rician_predictions.csv")
    path = repo / "outputs/detector/v2_0_snr_detections.csv"
    counts, _ = cross.load_boxes(path)
    changes = []
    for g in groups:
        change = cross.restore_count(g, counts[g["image"], g["class"]])
        if change is not None:
            changes.append(change)
    rows = [g for g in groups if g["split"] == "validation"]
    if len(rows) != 5454:
        raise ValueError("Unexpected source validation coverage")
    ratios = supp.read(root / "train_only/calibration.json")
    y = base.labels(rows, ratios)
    payload = supp.read(root / "payload_audit.json")
    supp.require_same(payload["matched_saved_images"], payload["n"], "source payload audit")
    evlm = supp.read(repo / "outputs/energy/gpu_power_phases.json")["phases"]["vlm"]["joule_per_item_incremental"]
    return {"x": np.array([base.features(g) for g in rows]), "y": y,
            "energy": base.energy_matrix(rows, payload, evlm, True), "ratios": ratios,
            "audit": {"source_prediction_sha256": audit["sha256"], "original_detector_sha256": base.sha(path),
                      "restored_sender_count_decisions": len(changes), "changes": changes,
                      "validation_decisions": len(rows), "source_feature_policy": POLICY}}


def run(args: argparse.Namespace) -> None:
    fm = supp.read(args.formal / "manifest.json")
    import sklearn
    for key, value in (("numpy", np.__version__), ("sklearn", sklearn.__version__)):
        if key in fm:
            supp.require_same(fm[key], value, f"source runtime {key}")
    d = verify_target(args.repo, args.target)
    src = source_validation(args.repo, args.formal)
    root = args.formal / "rician/train_only"
    y = base.labels(d["rows"], src["ratios"])
    base.dump(args.out / "target_data_audit.json", d["audit"])
    base.dump(args.out / "source_feature_audit.json", src["audit"])
    base.dump(args.out / "test_keys.json", d["keys"])
    base.dump(args.out / "source_calibration.json", src["ratios"])
    base.dump(args.out / "label_comparability.json", {
        "source_calibration": "frozen VisDrone train_only; missing class/SNR ratio defaults to 1.0",
        "target_refit_calibration": "unchanged cached DroneVehicle train_only, used only in secondary same-label diagnostics",
        "different_detection_outcomes": int((y[:, 0] != d["target_y"][:, 0]).sum()),
        "different_image_outcomes": int((y[:, 1] != d["target_y"][:, 1]).sum()),
        "direct_headline_comparison": "not an identical decoder protocol if any detection outcomes differ",
        "secondary_common_label_analysis": "apply identical saved picks to DV target-refit labels; diagnostic only, does not replace frozen source decoder"})
    policy = supp.read(root / "lut.json")
    picks = {"image": np.ones(len(y), int), "detection": np.zeros(len(y), int),
             "rule": np.array([g["qt"] == "presence" for g in d["rows"]], int),
             "source_lut": np.array([policy[f'{g["qt"]}|{g["snr"]}'] for g in d["rows"]], int)}
    baseline = {name: base.summarize(d["rows"], y, p, d["baseline_energy"]) for name, p in picks.items()}
    baseline["oracle_accuracy"] = float(y.max(axis=1).mean())
    base.dump(args.out / "baselines.json", baseline)
    base.dump(args.out / "source_lut.json", policy)
    np.savez_compressed(args.out / "evaluation_inputs.npz", source_calibrated_y=y,
                        target_refit_y=d["target_y"], sender_features=d["x"], energy=d["energy"],
                        baseline_energy=d["baseline_energy"], **picks)
    with np.load(root / "baseline_outcomes.npz") as a:
        weights = a["linear_weights"].copy()
    records = []
    for name in ["linear"] + [f"seed_{s}" for s in range(10)]:
        if name == "linear":
            record = supp.read(args.supplement / "rician/train_only/linear.json")
            pv, pt = base.linear_predict(weights, src["x"]), base.linear_predict(weights, d["x"])
            old_path = args.supplement / "rician/train_only/linear_outcomes.npz"
        else:
            record = supp.read(root / f"{name}.json")
            models = joblib.load(root / f"{name}_models.joblib")
            if len(models) != 2 or any(m.n_features_in_ != 18 for m in models):
                raise ValueError("Source model feature dimension mismatch")
            pv = np.column_stack([base.probabilities(m, src["x"]) for m in models])
            pt = np.column_stack([base.probabilities(m, d["x"]) for m in models])
            old_path = root / f"{name}_outcomes.npz"
        supp.require_same(record["features"], base.FEATURE_NAMES, "source model feature list")
        vs, _ = supp.sweep(src["y"], pv, src["energy"])
        supp.require_same(vs, record["validation_sweep"], "sender-corrected source validation replay")
        with np.load(old_path) as old:
            np.testing.assert_allclose(pv, old["validation_probabilities"], rtol=1e-13, atol=1e-13)
        result, arrays = frozen_evaluate(d["rows"], y, pt, d["energy"], record)
        result.update(model="linear" if name == "linear" else "mlp", seed=None if name == "linear" else int(name[5:]))
        result["target_refit_labels_same_picks_diagnostic"] = {
            "default": base.summarize(d["rows"], d["target_y"], arrays["pick"], d["energy"]),
            "selected": base.summarize(d["rows"], d["target_y"], arrays["selected_pick"], d["energy"])}
        base.dump(args.out / f"{name}.json", result)
        np.savez_compressed(args.out / f"{name}_outcomes.npz", **arrays,
                            source_validation_probabilities=pv, source_selected_price=np.array(result["source_selected_lambda"]))
        if name != "linear":
            records.append(result)
        print(f"[transfer done {name}] source_lambda={result['source_selected_lambda']:.8g}", flush=True)
    hashes = {}
    for bundle in (args.formal / "rician", args.supplement / "rician/train_only", args.target):
        for p in bundle.rglob("*"):
            if p.is_file():
                hashes[str(p)] = base.sha(p)
    for p in (args.formal / "manifest.json", args.formal / "status.json", args.supplement / "manifest.json", args.supplement / "status.json"):
        hashes[str(p)] = base.sha(p)
    base.dump(args.out / "reused_artifact_sha256.json", hashes)
    base.dump(args.out / "summary.json", {"mlp": supp.describe(records), "linear": supp.read(args.out / "linear.json"),
              "baselines": baseline, "energy_proxy_j": d["evlm"], "test_images": 122, "test_decisions": 4260,
              "scope": "VisDrone-frozen model, decoder calibration, LUT and source-selected price; no target fitting or selection",
              "statistics": "10 seeds mean and sample SD pooled over 6 SNRs; descriptive, not a significance test",
              "limits": ["same Qwen2-VL energy proxy, not target measurement", "source-validation 1 pp condition does not constrain target test loss"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("repo", "formal", "supplement", "target", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    base.dump(args.out / "manifest.json", {"pid": os.getpid(), "started_unix": started,
        "script_sha256": base.sha(Path(__file__)), "source_feature_policy": POLICY,
        "formal": str(args.formal), "supplement": str(args.supplement), "target": str(args.target),
        "seed_ids": list(range(10)), "feature_names": base.FEATURE_NAMES,
        "source_price_grid": base.PRICES, "no_training": True, "no_target_price_selection": True,
        "scope": "frozen source transfer only; target refit and codec replay are not rerun"})
    try:
        while True:
            ready, detail = dependency_state(args.formal, args.supplement)
            if ready:
                break
            base.dump(args.out / "status.json", {"state": "WAITING_FOR_SOURCE", "detail": detail})
            if not args.wait:
                return
            time.sleep(60)
        base.dump(args.out / "status.json", {"state": "RUNNING", "detail": "Source provenance/COMPLETE gates passed"})
        run(args)
        base.dump(args.out / "status.json", {"state": "COMPLETE", "mlp_seeds": 10, "linear_models": 1,
            "source_feature_policy": POLICY, "seconds_including_dependency_wait": time.time() - started})
    except Exception as exc:
        base.dump(args.out / "status.json", {"state": "FAILED", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
