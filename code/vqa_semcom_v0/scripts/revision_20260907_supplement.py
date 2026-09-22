#!/usr/bin/env python3
"""Frozen supplementary routing tests, reusing verified formal_v1 artifacts.

CPU router fits only. No detector/VLM inference or codec reconstruction.
"""
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

VARIANTS = {
    "no_detector": list(range(16)),
    "no_snr": list(range(15)) + [16, 17],
    "question_type_only": list(range(5)),
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def require_same(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise ValueError(f"Reused artifact mismatch: {name}")


def select_price(validation_sweep: list[dict]) -> int:
    """Only validation metrics are accepted; test labels cannot enter selection."""
    threshold = validation_sweep[0]["accuracy"] - 0.01
    return min((i for i, row in enumerate(validation_sweep) if row["accuracy"] >= threshold),
               key=lambda i: (validation_sweep[i]["energy_j"], validation_sweep[i]["lambda"]))


def sweep(y: np.ndarray, probabilities: np.ndarray, energy: np.ndarray) -> tuple[list[dict], np.ndarray]:
    picks = np.asarray([base.choose(probabilities, energy, price) for price in base.PRICES])
    return [{"lambda": price, **base.metrics(y, pick, energy)}
            for price, pick in zip(base.PRICES, picks)], picks


def deployment_energy(energy: np.ndarray, variant: str) -> np.ndarray:
    """Separate same-pick sensitivity: detector-free image path when possible."""
    result = energy.copy()
    if variant in ("no_detector", "question_type_only"):
        result[:, 1] -= base.ENERGY_DET
    return result


def prepare_channel(repo: Path, formal: Path, channel: str, hashes: dict) -> dict:
    source = formal / channel
    groups, audit = base.load_groups(repo / f"outputs/vlm/v3_0_{channel}_predictions.csv")
    require_same(audit, read(source / "data_audit.json"), f"{channel} source hash, groups and split")
    payload = read(source / "payload_audit.json")
    expected_keys = {f'{g["image"]}|{g["snr"]}' for g in groups}
    require_same(set(payload["records"]), expected_keys, "payload keys")
    require_same(payload["n"], len(expected_keys), "payload size")
    require_same(payload["matched_saved_images"], payload["n"], "payload identity count")
    if not all(r["saved_jpeg_matches"] for r in payload["records"].values()):
        raise ValueError("Unmatched cached payload")
    require_same(base.sha(repo / "src/vqa_semcom/degradation/digital_link.py"),
                 payload["link_source_sha256"], "codec source hash")
    parts = {s: [g for g in groups if g["split"] == s] for s in ("train", "validation", "test")}
    keys = [{k: g[k] for k in ("image", "question", "qt", "snr", "class")} for g in parts["test"]]
    require_same(keys, read(source / "test_keys.json"), "test ordering")
    x = {s: np.asarray([base.features(g) for g in rows]) for s, rows in parts.items()}
    power_path = repo / "outputs/energy/gpu_power_phases.json"
    hashes[str(power_path)] = base.sha(power_path)
    evlm = read(power_path)["phases"]["vlm"]["joule_per_item_incremental"]
    energy = {s: base.energy_matrix(rows, payload, evlm, True) for s, rows in parts.items()}
    modes = {}
    for mode in ("train_only", "none"):
        md = source / mode
        ratios = base.fit_calibration(parts["train"]) if mode == "train_only" else {}
        require_same(ratios, read(md / "calibration.json"), "frozen calibration")
        y = {s: base.labels(rows, ratios) for s, rows in parts.items()}
        with np.load(md / "baseline_outcomes.npz") as old:
            np.testing.assert_array_equal(y["test"], old["y"])
            np.testing.assert_array_equal(energy["test"], old["energy"])
            np.testing.assert_array_equal(base.energy_matrix(parts["test"], payload, evlm, False), old["baseline_energy"])
            w = old["linear_weights"].copy()
            lp = base.linear_predict(w, x["test"])
            np.testing.assert_array_equal(lp, old["linear_probabilities"])
            np.testing.assert_array_equal(base.choose(lp, energy["test"], 0), old["linear"])
        # Replay all saved full-model sweeps; verify selection and per-row decisions.
        full_records = []
        for seed in range(10):
            record = read(md / f"seed_{seed}.json")
            require_same(record["features"], base.FEATURE_NAMES, "full-model feature list")
            with np.load(md / f"seed_{seed}_outcomes.npz") as old:
                ts, picks = sweep(y["test"], old["probabilities"], energy["test"])
                vs, _ = sweep(y["validation"], old["validation_probabilities"], energy["validation"])
                np.testing.assert_array_equal(old["prices"], base.PRICES)
                np.testing.assert_array_equal(old["sweep_picks"], picks)
                np.testing.assert_array_equal(old["pick"], picks[0])
                require_same(ts, record["sweep"], "full test sweep")
                require_same(vs, record["validation_sweep"], "full validation sweep")
                require_same(select_price(vs), record["validation_selected_index"], "full selected price")
                require_same(ts[select_price(vs)], record["selected_test"], "full selected test result")
            full_records.append(record)
        modes[mode] = {"y": y, "weights": w, "full_records": full_records}
    for path in source.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".npz"):
            hashes[str(path)] = base.sha(path)
    return {"parts": parts, "x": x, "energy": energy, "modes": modes, "audit": audit}


def save_evaluation(dest: Path, parts: dict, y: dict, energy: dict, pv: np.ndarray,
                    pt: np.ndarray, extras: dict, variant: str = "full") -> dict:
    test_sweep, test_picks = sweep(y["test"], pt, energy["test"])
    val_sweep, val_picks = sweep(y["validation"], pv, energy["validation"])
    selected = select_price(val_sweep)
    record = {**extras, "main": base.summarize(parts["test"], y["test"], test_picks[0], energy["test"]),
              "sweep": test_sweep, "validation_sweep": val_sweep,
              "validation_selected_index": selected, "selected_test": test_sweep[selected],
              "selected_summary": base.summarize(parts["test"], y["test"], test_picks[selected], energy["test"])}
    if variant in ("no_detector", "question_type_only"):
        de = deployment_energy(energy["test"], variant)
        record["detector_free_image_path_sensitivity"] = {
            "interpretation": "same test decisions and original selected index; no re-selection or re-training",
            "unpriced": base.metrics(y["test"], test_picks[0], de),
            "selected": base.metrics(y["test"], test_picks[selected], de),
            "sweep": [{"lambda": price, **base.metrics(y["test"], pk, de)}
                      for price, pk in zip(base.PRICES, test_picks)]}
    base.dump(dest.with_suffix(".json"), record)
    np.savez_compressed(dest.parent / f"{dest.name}_outcomes.npz", probabilities=pt,
                        validation_probabilities=pv, prices=np.asarray(base.PRICES),
                        pick=test_picks[0], sweep_picks=test_picks, validation_sweep_picks=val_picks)
    return record


def describe(records: list[dict]) -> dict:
    out = {"n_seeds": len(records), "aggregation": "pooled six-SNR metrics; mean and sample SD across all declared seeds"}
    for setting in ("unpriced", "validation_selected"):
        rows = [r["main"]["pooled"] if setting == "unpriced" else r["selected_test"] for r in records]
        out[setting] = {key: {"mean": float(np.mean([r[key] for r in rows])),
                             "sample_sd": float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else None}
                        for key in ("accuracy", "energy_j", "image_fraction")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--formal", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--channels", nargs="+", default=["awgn", "rayleigh", "rician"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    old_manifest = read(args.formal / "manifest.json")
    require_same(read(args.formal / "status.json")["state"], "COMPLETE", "formal completion")
    require_same(base.sha(Path(base.__file__)), old_manifest["script_sha256"], "original runner SHA256")
    require_same(base.PRICES, old_manifest["prices"], "fixed price grid")
    if not args.smoke:
        require_same(args.channels, old_manifest["channels"], "formal channels")
        require_same(args.seeds, old_manifest["seeds"], "formal seeds")
        require_same(args.max_epochs, old_manifest["max_epochs"], "training budget")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    import sklearn
    manifest = {"started_unix": started, "pid": os.getpid(), "smoke": args.smoke,
                "channels": args.channels, "seeds": args.seeds, "max_epochs": args.max_epochs,
                "variants": {v: {"indices": inds, "features": [base.FEATURE_NAMES[i] for i in inds]}
                             for v, inds in VARIANTS.items()},
                "script_sha256": base.sha(Path(__file__)), "base_script_sha256": base.sha(Path(base.__file__)),
                "formal_manifest_sha256": base.sha(args.formal / "manifest.json"),
                "prices": base.PRICES, "selection": old_manifest["selection"],
                "selection_caution": "same relative validation rule, not matched absolute accuracy target; no test selection",
                "linear_modes": ["train_only", "none"], "ablation_modes": ["train_only"],
                "main_energy": "detector charged on every query for all learned variants to isolate feature effects",
                "sensitivity": "no_detector/question_type_only: same picks, image branch minus detector energy, separately reported",
                "full_model": "reuse all formal_v1 seeds, no full-model re-fit",
                "scope": "CPU routing only, original MLP fit_pair unchanged; no VLM/detector inference or codec replay",
                "python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__}
    require_same(manifest["numpy"], old_manifest["numpy"], "numpy version")
    require_same(manifest["sklearn"], old_manifest["sklearn"], "sklearn version")
    base.dump(args.out / "manifest.json", manifest)
    try:
        hashes, data, summary = {}, {}, {}
        for channel in args.channels:
            data[channel] = prepare_channel(args.repo, args.formal, channel, hashes)
            print(f"[verified {channel}] cached labels, features, split, energy, linear predictions and 20 full-model sweeps", flush=True)
        base.dump(args.out / "reused_artifact_sha256.json", hashes)
        for channel, d in data.items():
            dest = args.out / channel
            base.dump(dest / "data_audit.json", d["audit"])
            summary[channel] = {}
            for mode, m in d["modes"].items():
                target = dest / mode
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / "evaluation_inputs.npz", test_y=m["y"]["test"],
                                    validation_y=m["y"]["validation"], test_energy=d["energy"]["test"],
                                    validation_energy=d["energy"]["validation"], linear_weights=m["weights"])
                lr = save_evaluation(target / "linear", d["parts"], m["y"], d["energy"],
                                     base.linear_predict(m["weights"], d["x"]["validation"]),
                                     base.linear_predict(m["weights"], d["x"]["test"]),
                                     {"model": "linear", "features": base.FEATURE_NAMES, "source": str(args.formal / channel / mode)})
                summary[channel][mode] = {"linear": describe([lr]), "full": describe(m["full_records"])}
                print(f"[linear done {channel} {mode}] selected={lr['selected_test']}", flush=True)
        base.dump(args.out / "summary.json", summary)
        completed = 0
        for channel, d in data.items():
            y = d["modes"]["train_only"]["y"]
            for variant, inds in VARIANTS.items():
                dest = args.out / channel / "train_only" / variant
                dest.mkdir(parents=True, exist_ok=False)
                xv = {s: a[:, inds] for s, a in d["x"].items()}
                records = []
                for seed in args.seeds:
                    begin = time.time()
                    models, history = base.fit_pair(xv["train"], y["train"], xv["validation"], y["validation"], seed, args.max_epochs)
                    pv = np.column_stack([base.probabilities(model, xv["validation"]) for model in models])
                    pt = np.column_stack([base.probabilities(model, xv["test"]) for model in models])
                    record = save_evaluation(dest / f"seed_{seed}", d["parts"], y, d["energy"], pv, pt,
                                             {"seed": seed, "mode": "train_only", "variant": variant,
                                              "features": [base.FEATURE_NAMES[i] for i in inds],
                                              "training": history, "seconds": time.time() - begin}, variant)
                    joblib.dump(models, dest / f"seed_{seed}_models.joblib")
                    records.append(record)
                    completed += 1
                    base.dump(args.out / "status.json", {"state": "RUNNING", "completed_fits": completed,
                              "expected_fits": len(args.channels) * len(VARIANTS) * len(args.seeds),
                              "channel": channel, "variant": variant, "seed": seed, "seconds": time.time() - started})
                    print(f"[ablation done {channel} {variant} seed={seed}] seconds={record['seconds']:.1f}", flush=True)
                summary[channel]["train_only"][variant] = describe(records)
                base.dump(args.out / "summary.json", summary)
        base.dump(args.out / "status.json", {"state": "COMPLETE", "completed_fits": completed,
                  "linear_sweeps": len(args.channels) * 2, "seconds": time.time() - started, "smoke": args.smoke})
    except Exception as exc:
        base.dump(args.out / "status.json", {"state": "FAILED", "error": repr(exc), "seconds": time.time() - started})
        raise


if __name__ == "__main__":
    main()
