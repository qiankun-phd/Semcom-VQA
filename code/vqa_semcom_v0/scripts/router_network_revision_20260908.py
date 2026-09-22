#!/usr/bin/env python3
"""Bounded validation-only router network/target development on frozen caches."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zlib

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor

CONFIGS = {
    "original_bce": {"objective": "dual_bce", "hidden": [32, 16], "alpha": 0.0001},
    "wide_bce": {"objective": "dual_bce", "hidden": [64, 32], "alpha": 0.001},
    "advantage_mse": {"objective": "advantage_mse", "hidden": [32, 16], "alpha": 0.001},
    "disagreement_bce": {"objective": "disagreement_bce", "hidden": [32, 16], "alpha": 0.001},
}
RECEIVERS = ("qwen2", "qwen25", "smol")
QTYPES = ("presence", "counting", "comparison", "co_presence", "threshold")


def dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parameter_count(config: dict) -> int:
    widths = [18, *config["hidden"], 1]
    count = sum((a + 1) * b for a, b in zip(widths, widths[1:]))
    return count * (2 if config["objective"] == "dual_bce" else 1)


def probability(model: MLPClassifier, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, list(model.classes_).index(1)]


def route_score(models: list, config: dict, x: np.ndarray) -> np.ndarray:
    if config["objective"] == "dual_bce":
        return probability(models[1], x) - probability(models[0], x)
    if config["objective"] == "advantage_mse":
        return models[0].predict(x)
    # Conditional P(image uniquely correct | branch disagreement,x).
    # Its sign about 0.5 is used only at lambda=0, never as an energy price.
    return probability(models[0], x) - 0.5


def accuracy(y: np.ndarray, score: np.ndarray) -> float:
    pick = (score > 0).astype(np.int8)
    return float(y[np.arange(len(y)), pick].mean())


def targets(y: np.ndarray, objective: str, head: int) -> tuple[np.ndarray, np.ndarray]:
    if objective == "disagreement_bce":
        mask = y[:, 0] != y[:, 1]
        return mask, y[mask, 1]
    mask = np.ones(len(y), dtype=bool)
    return mask, y[:, 1] - y[:, 0] if objective == "advantage_mse" else y[:, head]


def benchmark(models: list, cfg: dict, xval: np.ndarray) -> dict:
    # Same fixed validation batch and first 100 queries for every configuration.
    for _ in range(3):
        route_score(models, cfg, xval)
    elapsed = []
    for _ in range(20):
        start = time.perf_counter()
        route_score(models, cfg, xval)
        elapsed.append(time.perf_counter() - start)
    single = []
    for row in xval[:100]:
        start = time.perf_counter()
        route_score(models, cfg, row[None, :])
        single.append(time.perf_counter() - start)
    return {"batch_size": len(xval), "batch_repeats": 20,
            "batch_median_ms": float(np.median(elapsed) * 1000),
            "batch_us_per_query": float(np.median(elapsed) / len(xval) * 1e6),
            "single_query_median_us": float(np.median(single) * 1e6),
            "caution": "CPU wall-clock model calls only; batch amortization and Python overhead, not end-to-end latency or energy"}


def fit(config: dict, data: dict, seed: int, dest: Path, epochs: int = 300) -> tuple[list, dict]:
    dest.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    models, histories = [], []
    objective = config["objective"]
    for head in range(2 if objective == "dual_bce" else 1):
        tm, yt = targets(data["train_y"], objective, head)
        vm, yv = targets(data["validation_y"], objective, head)
        if not tm.any() or not vm.any():
            raise ValueError("Empty training/validation objective subset")
        cls = MLPRegressor if objective == "advantage_mse" else MLPClassifier
        model = cls(hidden_layer_sizes=tuple(config["hidden"]), activation="relu", solver="adam",
                    alpha=config["alpha"], batch_size=200, learning_rate_init=0.001,
                    early_stopping=False, random_state=seed, shuffle=True)
        best, best_loss, stale, history = None, float("inf"), 0, []
        for epoch in range(epochs):
            if objective == "advantage_mse":
                model.partial_fit(data["train_x"][tm], yt)
                pred = model.predict(data["validation_x"][vm])
                loss = float(np.mean((pred - yv) ** 2))
            else:
                model.partial_fit(data["train_x"][tm], yt, classes=np.array([0, 1]))
                pred = np.clip(probability(model, data["validation_x"][vm]), 1e-12, 1 - 1e-12)
                loss = float(-np.mean(yv * np.log(pred) + (1 - yv) * np.log(1 - pred)))
            if not np.isfinite(loss):
                raise ValueError("Nonfinite validation loss")
            history.append(loss)
            if loss < best_loss - 1e-4:
                best, best_loss, stale = copy.deepcopy(model), loss, 0
                joblib.dump(best, dest / f"best_head_{head}.joblib")
            else:
                stale += 1
            if stale >= 10:
                break
        models.append(best)
        histories.append({"head": head, "epochs": len(history), "best_validation_loss": best_loss,
                          "training_rows": int(tm.sum()), "validation_rows": int(vm.sum()), "loss_history": history})
    seconds = time.perf_counter() - begin
    score = route_score(models, config, data["validation_x"])
    params = sum(a.size for m in models for a in [*m.coefs_, *m.intercepts_])
    assert params == parameter_count(config)
    record = {"seed": seed, "config": config, "parameter_count": params, "training_seconds": seconds,
              "validation_accuracy": accuracy(data["validation_y"], score), "training": histories,
              "inference": benchmark(models, config, data["validation_x"])}
    joblib.dump(models, dest / "models.joblib")
    np.savez_compressed(dest / "validation_outcomes.npz", score=score, pick=(score > 0).astype(np.int8))
    dump(dest / "record.json", record)
    return models, record


def select(validation_records: dict) -> str:
    # Input consists only of validation metrics. Exact ties favor fewer parameters.
    return min(validation_records, key=lambda name: (
        -float(np.mean([r["validation_accuracy"] for r in validation_records[name]])),
        parameter_count(CONFIGS[name]), name))


def load_development(source: Path) -> tuple[dict, dict, list]:
    manifest = read(source / "manifest.json")
    assert read(source / "status.json")["state"] == "COMPLETE"
    assert manifest["script_sha256"] == "7d749d67eec5318229e4ea642a925c117bf5b315c669d1653addba2f95e49b6e"
    assert len(manifest["features"]) == 18
    keys = read(source / "common_keys.json")
    hashes = {str(p): sha(p) for p in (source / "manifest.json", source / "status.json", source / "common_keys.json",
              source / "shared_train_calibration.json", source / "sender_count_audit.json", source / "data_audit.json")}
    data = {}
    for receiver in RECEIVERS:
        path = source / receiver / "evaluation_inputs.npz"
        hashes[str(path)] = sha(path)
        with np.load(path) as z:
            # Test arrays are not read before all configuration selections are frozen.
            data[receiver] = {f"{s}_{kind}": z[f"{s}_{kind}"].copy()
                              for s in ("train", "validation") for kind in ("x", "y")}
        for split, n in (("train", 9150), ("validation", 2646)):
            x, y = data[receiver][f"{split}_x"], data[receiver][f"{split}_y"]
            assert x.shape == (n, 18) and y.shape == (n, 2)
            assert np.isfinite(x).all() and set(np.unique(y)) <= {0, 1}
            np.testing.assert_array_equal(x, data["qwen2"][f"{split}_x"])
            np.testing.assert_array_equal(y[:, 0], data["qwen2"][f"{split}_y"][:, 0])
        data[receiver]["train_y"] = data[receiver]["train_y"].astype(np.int8)
        data[receiver]["validation_y"] = data[receiver]["validation_y"].astype(np.int8)
    image_sets = {}
    for split in ("train", "validation", "test"):
        image_sets[split] = {k["image"] for k in keys if k["split"] == split}
        for iid in image_sets[split]:
            bucket = zlib.crc32(iid.encode()) % 100
            assert split == ("test" if bucket < 20 else "validation" if bucket >= 80 else "train")
    assert not any(image_sets[a] & image_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
    return data, hashes, keys


def test_summary(keys: list, y: np.ndarray, score: np.ndarray) -> dict:
    pick = (score > 0).astype(np.int8)
    out = {"n": len(y), "accuracy": accuracy(y, score), "image_fraction": float(pick.mean()), "per_type": {}}
    for qt in QTYPES:
        mask = np.array([k["qt"] == qt for k in keys])
        out["per_type"][qt] = {"n": int(mask.sum()), "accuracy": accuracy(y[mask], score[mask]),
                                "image_fraction": float(pick[mask].mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    import sklearn
    protocol = {"pid": os.getpid(), "started_unix": started, "smoke": args.smoke,
                "configs": CONFIGS, "screen_seeds": [0, 1, 2], "final_seeds": list(range(10)),
                "max_epochs": 300, "min_delta": 0.0001, "patience": 10,
                "selection": "per receiver: highest mean full validation answer accuracy across seeds0,1,2; exact ties fewer parameters, then ID",
                "test_gate": "write validation_selection.json for all receivers before loading test arrays; no subsequent retuning",
                "scope": "existing VisDrone matched Rician benchmark, lambda=0, same18 pre-transmission features, same frozen image split/calibration/labels",
                "prior_exposure": "test performance has previously been examined; follow-up development evaluation, not a pristine unseen holdout",
                "disagreement_caution": "conditional class probability only for lambda=0; not an energy-priced branch probability difference",
                "source": str(args.source), "script_sha256": sha(Path(__file__)),
                "python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__,
                "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}
    dump(args.out / "protocol.json", protocol)
    dump(args.out / "status.json", {"state": "PREPARING"})
    try:
        data, hashes, allkeys = load_development(args.source)
        dump(args.out / "source_sha256.json", hashes)
        screen, completed = {}, 0
        for receiver in RECEIVERS:
            screen[receiver] = {}
            for name, config in CONFIGS.items():
                screen[receiver][name] = []
                for seed in ([0] if args.smoke else [0, 1, 2]):
                    _, record = fit(config, data[receiver], seed, args.out / "screen" / receiver / name / f"seed_{seed}", 2 if args.smoke else 300)
                    screen[receiver][name].append(record)
                    completed += 1
                    dump(args.out / "status.json", {"state": "SCREENING", "completed_screen_fits": completed, "expected": 12 if args.smoke else 36})
                    print(f"[screen {receiver} {name} seed={seed}] seconds={record['training_seconds']:.3f} validation={record['validation_accuracy']:.6f}", flush=True)
        if args.smoke:
            dump(args.out / "status.json", {"state": "COMPLETE", "scope": "validation-only smoke; no test loaded", "seconds": time.time() - started})
            return
        selected = {receiver: select(records) for receiver, records in screen.items()}
        selection = {"frozen_unix": time.time(), "selected": selected,
                     "validation_candidates": {r: {name: {"mean": float(np.mean([v["validation_accuracy"] for v in records])),
                        "sample_sd": float(np.std([v["validation_accuracy"] for v in records], ddof=1)),
                        "all_seeds": [v["validation_accuracy"] for v in records], "parameters": parameter_count(CONFIGS[name])}
                         for name, records in methods.items()} for r, methods in screen.items()},
                     "test_used": False, "protocol_sha256": sha(args.out / "protocol.json")}
        dump(args.out / "validation_selection.json", selection)
        selection_hash = sha(args.out / "validation_selection.json")
        print(f"[selection frozen] {selected} sha256={selection_hash}", flush=True)
        summary, completed = {}, 0
        keys = [k for k in allkeys if k["split"] == "test"]
        for receiver in RECEIVERS:
            with np.load(args.source / receiver / "evaluation_inputs.npz") as z:
                xtest, ytest = z["test_x"].copy(), z["test_y"].astype(np.int8)
                linear = z["linear"].copy()
            assert len(xtest) == len(keys) == 2808
            config = CONFIGS[selected[receiver]]
            records = []
            for seed in range(10):
                dest = args.out / "final" / receiver / f"seed_{seed}"
                models, record = fit(config, data[receiver], seed, dest)
                assert sha(args.out / "validation_selection.json") == selection_hash
                score = route_score(models, config, xtest)
                result = test_summary(keys, ytest, score)
                record.update(test=result, selection_sha256=selection_hash)
                dump(dest / "record.json", record)
                np.savez_compressed(dest / "test_outcomes.npz", score=score, pick=(score > 0).astype(np.int8), y=ytest)
                records.append(record)
                completed += 1
                dump(args.out / "status.json", {"state": "FINAL_EVALUATION", "completed_final_fits": completed, "expected": 30})
                print(f"[final {receiver} seed={seed}] seconds={record['training_seconds']:.3f} test={result['accuracy']:.6f}", flush=True)
            baseline = read(args.source / receiver / "baselines.json")
            old = [read(args.source / receiver / f"seed_{s}.json")["main"] for s in range(10)]
            summary[receiver] = {"selected_config": selected[receiver], "parameters": parameter_count(config),
                "accuracy_mean": float(np.mean([v["test"]["accuracy"] for v in records])),
                "accuracy_sample_sd": float(np.std([v["test"]["accuracy"] for v in records], ddof=1)),
                "image_fraction_mean": float(np.mean([v["test"]["image_fraction"] for v in records])),
                "training_seconds_mean": float(np.mean([v["training_seconds"] for v in records])),
                "inference_single_us_mean": float(np.mean([v["inference"]["single_query_median_us"] for v in records])),
                "inference_batch_us_per_query_mean": float(np.mean([v["inference"]["batch_us_per_query"] for v in records])),
                "all_test_seeds": [v["test"] for v in records], "baselines": baseline,
                "original_mlp_mean": float(np.mean([v["pooled"]["accuracy"] for v in old])),
                "original_mlp_sample_sd": float(np.std([v["pooled"]["accuracy"] for v in old], ddof=1)),
                "linear_accuracy": float(ytest[np.arange(len(ytest)), linear].mean())}
            dump(args.out / "summary.json", summary)
        dump(args.out / "status.json", {"state": "COMPLETE", "screen_fits": 36, "final_fits": 30,
                  "selection_sha256": selection_hash, "seconds": time.time() - started})
    except Exception as exc:
        dump(args.out / "status.json", {"state": "FAILED", "error": repr(exc), "seconds": time.time() - started})
        raise


if __name__ == "__main__":
    main()
