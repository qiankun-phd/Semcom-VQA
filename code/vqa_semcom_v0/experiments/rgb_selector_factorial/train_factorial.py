"""Bounded EXP-013 2x2 selector refit on cached EXP-012 train/validation data.

This module never reads historical-development outcomes or sealed-test data.
All twelve checkpoints are frozen before development feature scores are saved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import signal
import sys
import time
from typing import Any, Iterator


GROUPS = {
    "original_absolute": {"balanced": False, "target": "absolute"},
    "balanced_absolute": {"balanced": True, "target": "absolute"},
    "original_gain": {"balanced": False, "target": "gain"},
    "balanced_gain": {"balanced": True, "target": "gain"},
}
SEEDS = [7, 17, 27]
CELLS = [(budget, tier) for budget in (2000, 4000, 8000) for tier in ("low", "medium", "high")]
PIXELS = {"low": 50176, "medium": 100352, "high": 200704}
EXPECTED_SPLITS = {"train": 480, "validation": 120, "legacy_dev": 120}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_hashes(mapping: dict[str, str]) -> None:
    for name, expected in mapping.items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")


def validate_protocol(protocol: dict) -> None:
    expected = {
        "experiment_id": "EXP-013",
        "groups": GROUPS,
        "data": {"source_experiment": "EXP-012", "train": 480, "validation": 120, "legacy_dev": 120,
                 "test_access": False, "new_vlm_or_codec_calls": 0},
        "features": {"question_dimensions": 256, "image_dimensions": 83, "image_scaler_fit": "train_only",
                     "balanced_image_multiplier": "1/sqrt(83)", "question_transform": "unchanged", "per_example_normalization": False},
        "network": {"dimensions": [339, 128, 64, 9], "activation": "ReLU", "dropout": .1},
        "absolute": {"output": "sigmoid(z_a)", "target": "correct_a", "loss": "mean BCEWithLogits across nine heads",
                     "checkpoint": "minimum validation BCE"},
        "gain": {"reference_action": 0, "output": "tanh(z_a-z_0); reference exactly zero",
                 "target": "correct_a-correct_0 in {-1,0,1}",
                 "loss": "mean MSE over eight non-reference heads, without reweighting or resampling",
                 "checkpoint": "minimum validation gain MSE"},
        "training": {"seeds": SEEDS, "optimizer": "AdamW", "learning_rate": .001, "weight_decay": .0001,
                     "batch_size": 32, "max_epochs": 50, "min_epochs": 10, "patience": 8, "cpu_threads": 2,
                     "maximum_fits": 12, "wall_clock_limit_seconds": 1800},
        "selection": {"lambda": .05, "nominal_cost": "(raw_cap+1)/8001 + pixel_target/200704",
                      "absolute_score": "p_a-lambda*cost_a", "gain_score": "gain_a-lambda*(cost_a-cost_0)",
                      "tie_break": "lower nominal cost, then lower action index",
                      "ensemble": "arithmetic mean of three seed outputs before choosing action", "lambda_search": False},
        "actions": [list(cell) for cell in CELLS],
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"EXP-013 protocol mismatch: {key}")
    if protocol["evaluation"].get("freeze_all_checkpoints_before_legacy_scoring") is not True:
        raise ValueError("All checkpoints must freeze before historical scoring")


@contextmanager
def wall_clock_budget(seconds: float) -> Iterator[None]:
    """Interrupt even an inherited train_one call at the total experiment deadline."""
    if seconds <= 0:
        raise TimeoutError("EXP-013 total wall-clock budget exhausted")

    def expired(_signal: int, _frame: Any) -> None:
        raise TimeoutError("EXP-013 total wall-clock budget exhausted; no further fits permitted")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] != 0:
        raise RuntimeError("Another real-time alarm is active")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def load_source_module(source_code: Path) -> Any:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source_code))
    spec = importlib.util.spec_from_file_location("exp012_factorial_source", source_code / "train_selector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def feature_tensor(frozen: Any, rows: list[dict], scaler: dict, balanced: bool) -> Any:
    import torch
    values = torch.tensor([frozen.vector(row, scaler) for row in rows], dtype=torch.float32)
    if balanced:
        values[:, 256:] *= 1 / math.sqrt(83)
    return values


def gain_targets(correctness: Any) -> Any:
    return correctness - correctness[:, :1]


def gain_outputs(logits: Any) -> Any:
    import torch
    # The shared reference logit receives gradients from all eight contrasts.
    return torch.tanh(logits - logits[:, :1])


def fit_gain(frozen: Any, x_train: Any, y_train: Any, x_validation: Any,
             y_validation: Any, seed: int, settings: dict) -> tuple[Any, list[dict], dict]:
    """Original architecture/optimizer/order/stopping; signed contrast MSE only."""
    import torch
    from torch.nn import functional as F

    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = frozen.make_model(9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    target_train, target_validation = gain_targets(y_train), gain_targets(y_validation)
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    history = []
    start = time.perf_counter()
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        permutation = torch.randperm(len(x_train), generator=generator)
        total = 0.
        for batch in permutation.split(settings["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(gain_outputs(model(x_train[batch]))[:, 1:], target_train[batch, 1:])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite gain MSE")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite gain gradient")
            optimizer.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.inference_mode():
            val_loss = F.mse_loss(gain_outputs(model(x_validation))[:, 1:], target_validation[:, 1:]).item()
        if not math.isfinite(val_loss):
            raise ValueError("Nonfinite validation gain MSE")
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch, "train_gain_mse": total / len(x_train),
                        "validation_gain_mse": val_loss, "checkpoint_improved": improved})
        if epoch >= settings["min_epochs"] and stale >= settings["patience"]:
            break
    if best_state is None:
        raise ValueError("No finite gain checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, {"best_epoch": best_epoch, "best_validation_gain_mse": best_loss,
                           "epochs_run": len(history), "training_seconds": time.perf_counter() - start}


def nominal_cost(action: int) -> float:
    budget, tier = CELLS[action]
    return (budget + 1) / 8001 + PIXELS[tier] / 200704


def choose_action(scores: list[float], target: str = "absolute") -> int:
    if target not in ("absolute", "gain") or len(scores) != 9:
        raise ValueError("Unknown target or wrong score width")
    lower = 0 if target == "absolute" else -1
    if any(not math.isfinite(score) or not lower <= score <= 1 for score in scores):
        raise ValueError("Invalid selector scores")
    if target == "gain" and scores[0] != 0:
        raise ValueError("Reference gain must be exactly zero")
    baseline = nominal_cost(0) if target == "gain" else 0
    return max(range(9), key=lambda a: (scores[a] - .05 * (nominal_cost(a) - baseline), -nominal_cost(a), -a))


def reproduction_check(predictions: list[dict], baseline: dict) -> dict:
    reference = {row["id"]: row for row in baseline["rows"]}
    if len(reference) != len(baseline["rows"]) or len({row["id"] for row in predictions}) != len(predictions) or set(reference) != {row["id"] for row in predictions}:
        raise ValueError("Original reproduction requires identical unique image identities")
    if baseline["actions"]["joint9"] != list(range(9)):
        raise ValueError("Original joint action mapping changed")
    details, maximum, mismatches, checked = [], 0., 0, 0
    for split in EXPECTED_SPLITS:
        rows = [row for row in predictions if row["split"] == split]
        for replicate in (*map(str, SEEDS), "ensemble"):
            local_maximum, local_mismatches = 0., 0
            for row in rows:
                previous = reference[row["id"]]
                if previous["split"] != split:
                    raise ValueError("Original reproduction split changed")
                observed = row["scores"]["original_absolute"][replicate]
                expected = previous["probabilities"]["joint9"][replicate]
                if len(expected) != 9:
                    raise ValueError("Original probability width mismatch")
                local_maximum = max(local_maximum, max(abs(a - b) for a, b in zip(observed, expected)))
                local_mismatches += choose_action(observed) != choose_action(expected)
                checked += 9
            details.append({"split": split, "replicate": replicate, "n": len(rows),
                            "max_abs_difference": local_maximum, "action_mismatches": local_mismatches})
            maximum = max(maximum, local_maximum)
            mismatches += local_mismatches
    return {"schema_version": 1, "passed": maximum <= 1e-6 and mismatches == 0,
            "max_abs_difference": maximum, "tolerance": 1e-6, "action_mismatches": mismatches,
            "checked_scores": checked, "checked_actions": checked // 9, "details": details}


def prepare(source: Path, source_code: Path, protocol_path: Path, baseline_path: Path) -> tuple:
    """Allowlist train/validation labels; no legacy outcome path is opened or hashed."""
    allowed_paths = [source / "features.json", source / "supervision_records.json", source / "policy.json",
                     source / "controller_frozen.json", source / "training_frozen_inputs.json",
                     source_code / "train_selector.py", source_code / "prepare_data.py", source_code / "protocol.json",
                     protocol_path, baseline_path, Path(__file__).resolve()]
    allowed_paths += [source / f"{split}_{kind}.json" for split in ("train", "validation") for kind in ("manifest", "truth")]
    for path in allowed_paths:
        if any(re.search(r"(^|[^a-z])test([^a-z]|$)", part.lower()) for part in path.resolve().parts):
            raise ValueError("Test data paths are prohibited")
    controller = read(source / "controller_frozen.json")
    if controller.get("state") != "FROZEN_BEFORE_LEGACY_DEV_LABELS":
        raise ValueError("EXP-012 source controller is not frozen")
    # This old controller covers training metadata, validation report and models;
    # its training-input JSON is hashed, never recursively opened for legacy data.
    for name in controller["sha256"]:
        path = Path(name).resolve()
        if source not in path.parents or "legacy" in path.name or "test" in path.name:
            raise ValueError("Unexpected outcome path in original controller fingerprint")
    verify_hashes(controller["sha256"])
    allowed_paths += [Path(path) for path in controller["sha256"]]
    frozen_input = read(source / "training_frozen_inputs.json")
    for path in allowed_paths:
        expected = frozen_input["sha256"].get(str(path.resolve()))
        if expected is not None and sha(path) != expected:
            raise ValueError(f"Changed EXP-012 source input: {path}")
    fingerprints = {str(path.resolve()): sha(path) for path in allowed_paths}
    frozen = load_source_module(source_code)
    old_protocol = read(source_code / "protocol.json")
    frozen.validate_protocol(old_protocol)
    features = read(source / "features.json")
    if dict(Counter(row["split"] for row in features)) != EXPECTED_SPLITS:
        raise ValueError("Registered 480/120/120 feature inventory changed")
    manifests = {split: read(source / f"{split}_manifest.json") for split in ("train", "validation")}
    if [len(manifests[split]) for split in ("train", "validation")] != [480, 120]:
        raise ValueError("Registered training or validation sizes changed")
    expected = {split: {row["id"] for row in manifest} for split, manifest in manifests.items()}
    expected["legacy_dev"] = {row["id"] for row in features if row["split"] == "legacy_dev"}
    mapped = frozen.validate_features(features, expected)
    truths = {split: read(source / f"{split}_truth.json") for split in ("train", "validation")}
    records = read(source / "supervision_records.json")
    if len(records) != 5400 or {row["split"] for row in records} != {"train", "validation"}:
        raise ValueError("Only the existing train/validation nine-action outcomes are allowed")
    grids = {split: frozen.score_grid(records, truths[split], [row["id"] for row in manifest],
                                      old_protocol["receiver_sha256"], sha(source_code / "protocol.json"))
             for split, manifest in manifests.items()}
    rows = {split: [mapped[row["id"]] for row in manifest] for split, manifest in manifests.items()}
    scaler = frozen.fit_scaler(rows["train"])
    if scaler != read(source / "policy.json")["scaler"]:
        raise ValueError("Train-only scaler did not reproduce the original scaler")
    return frozen, features, rows, grids, scaler, fingerprints, frozen_input["torch_version"]


def run(source: Path, output: Path, protocol_path: Path, source_code: Path, baseline_path: Path) -> dict:
    import torch

    started = time.perf_counter()
    source, output, protocol_path, source_code, baseline_path = [path.resolve() for path in (source, output, protocol_path, source_code, baseline_path)]
    if source == output or source in output.parents:
        raise ValueError("EXP-013 output must be separate from frozen EXP-012")
    protocol = read(protocol_path)
    validate_protocol(protocol)
    reserved = ("frozen_inputs.json", "controller_frozen.json", "training_complete.json", "training_history.json",
                "predictions.json", "reproduction_check.json", "scaler.json", "checkpoints")
    if any((output / name).exists() for name in reserved):
        raise ValueError("Output contains experiment artifacts; no overwrite or additional fits are permitted")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(protocol["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    histories, checkpoints, models = {}, {}, {}
    completed_fits = 0
    try:
        with wall_clock_budget(protocol["training"]["wall_clock_limit_seconds"] - (time.perf_counter() - started)):
            frozen, features, rows, grids, scaler, source_hashes, original_torch = prepare(source, source_code, protocol_path, baseline_path)
            if torch.__version__ != original_torch:
                raise ValueError("Torch version must match the original reproducibility environment")
            fingerprint_path = output / "frozen_inputs.json"
            save(fingerprint_path, {"schema_version": 1, "sha256": source_hashes, "torch_version": torch.__version__,
                                    "cpu_threads": 2, "legacy_labels_opened": False, "sealed_test_opened": False,
                                    "source_run": str(source), "source_code": str(source_code),
                                    "protocol_path": str(protocol_path), "baseline_scores_path": str(baseline_path),
                                    "legacy_features_available": True, "input_scope": "cached train/validation outcomes and 720 cached feature vectors only"})
            save(output / "scaler.json", {"schema_version": 1, "base_scaler": scaler,
                                          "balanced_image_multiplier": 1 / math.sqrt(83), "question_transform": "unchanged"})
            labels = {split: torch.tensor([[cell["correct"] for cell in grids[split][row["id"]]] for row in values], dtype=torch.float32)
                      for split, values in rows.items()}
            for group, config in GROUPS.items():
                histories[group], checkpoints[group], models[group] = {}, {}, {}
                x = {split: feature_tensor(frozen, values, scaler, config["balanced"]) for split, values in rows.items()}
                for seed in SEEDS:
                    if completed_fits >= protocol["training"]["maximum_fits"]:
                        raise RuntimeError("Maximum fit count exhausted")
                    save(output / "training_status.json", {"state": "TRAINING", "group": group, "seed": seed,
                                                            "completed_fits": completed_fits, "maximum_fits": 12})
                    if config["target"] == "absolute":
                        # Reuse the exact original training implementation.
                        model, history, stats = frozen.train_one(x["train"], labels["train"], x["validation"], labels["validation"],
                                                                 list(range(9)), seed, protocol["training"])
                    else:
                        model, history, stats = fit_gain(frozen, x["train"], labels["train"], x["validation"], labels["validation"],
                                                         seed, protocol["training"])
                    completed_fits += 1
                    checkpoint = output / "checkpoints" / f"{group}-seed{seed}.pt"
                    frozen.torch_save_atomic(checkpoint, model.state_dict())
                    checkpoints[group][str(seed)] = {"path": str(checkpoint), "relative_path": str(checkpoint.relative_to(output)),
                                                    "sha256": sha(checkpoint)}
                    histories[group][str(seed)] = {"seed": seed, "group": group, "target": config["target"],
                                                  "balanced": config["balanced"], "history": history, **stats,
                                                  "input_sha256": sha(fingerprint_path), "checkpoint_sha256": sha(checkpoint)}
                    models[group][str(seed)] = model
                    save(output / "training_history.json", histories)
            if completed_fits != 12:
                raise RuntimeError("All twelve fits must complete before freezing")
            verify_hashes(source_hashes)
            freeze_files = [protocol_path, Path(__file__).resolve(), fingerprint_path, output / "scaler.json", output / "training_history.json"]
            freeze_files += [Path(meta["path"]) for family in checkpoints.values() for meta in family.values()]
            controller_path = output / "controller_frozen.json"
            save(controller_path, {"schema_version": 1, "state": "FROZEN_BEFORE_LEGACY_LABELS",
                                   "created_utc": datetime.now(timezone.utc).isoformat(),
                                   "sha256": {str(path): sha(path) for path in freeze_files}, "checkpoints": checkpoints,
                                   "artifact_sha256": {str(path.relative_to(output)): sha(path) for path in freeze_files if output in path.parents},
                                   "scaler_sha256": sha(output / "scaler.json"), "frozen_inputs_sha256": sha(fingerprint_path),
                                   "groups": list(GROUPS), "seeds": SEEDS, "completed_fits": completed_fits,
                                   "protocol_sha256": sha(protocol_path), "legacy_labels_opened": False,
                                   "sealed_test_opened": False, "legacy_outcome_scoring_performed": False})
            verify_hashes(read(controller_path)["sha256"])
            predictions = [{"id": row["id"], "split": row["split"], "scores": {}} for row in features]
            for group, config in GROUPS.items():
                values = feature_tensor(frozen, features, scaler, config["balanced"])
                scores = {}
                for seed in SEEDS:
                    model = models[group][str(seed)]
                    model.eval()
                    with torch.inference_mode():
                        logits = model(values)
                        scores[str(seed)] = logits.sigmoid() if config["target"] == "absolute" else gain_outputs(logits)
                scores["ensemble"] = torch.stack(list(scores.values())).mean(0)
                for replicate, output_values in scores.items():
                    for row, score in zip(predictions, output_values.tolist()):
                        choose_action(score, config["target"])
                        row["scores"].setdefault(group, {})[replicate] = score
            reproduction = reproduction_check(predictions, read(baseline_path))
            reproduction.update(controller_sha256=sha(controller_path), baseline_scores_sha256=sha(baseline_path))
            save(output / "reproduction_check.json", reproduction)
            if not reproduction["passed"]:
                raise ValueError("Original absolute baseline failed reproduction; stop before interpreting interventions")
            save(output / "predictions.json", {"schema_version": 1, "rows": predictions,
                                               "controller_sha256": sha(controller_path), "protocol_sha256": sha(protocol_path),
                                               "score_semantics": {name: config["target"] for name, config in GROUPS.items()},
                                               "legacy_labels_opened": False, "sealed_test_opened": False})
            verify_hashes(source_hashes)
            verify_hashes(read(controller_path)["sha256"])
            complete = {"schema_version": 1, "state": "COMPLETE", "completed_fits": completed_fits,
                        "maximum_fits": 12, "elapsed_seconds": time.perf_counter() - started,
                        "wall_clock_limit_seconds": 1800, "split_counts": EXPECTED_SPLITS,
                        "controller_sha256": sha(controller_path), "protocol_sha256": sha(protocol_path),
                        "scaler_sha256": sha(output / "scaler.json"), "frozen_inputs_sha256": sha(fingerprint_path),
                        "training_history_sha256": sha(output / "training_history.json"),
                        "predictions_sha256": sha(output / "predictions.json"),
                        "reproduction_check_sha256": sha(output / "reproduction_check.json"),
                        "reproduction_passed": True, "source_unchanged": True, "legacy_labels_opened": False,
                        "sealed_test_opened": False, "new_vlm_or_codec_calls": 0, "full_cost_gate": "PENDING"}
            save(output / "training_complete.json", complete)
            save(output / "training_status.json", complete)
            return complete
    except BaseException as error:
        save(output / "training_status.json", {"state": "FAILED", "completed_fits": completed_fits,
                                                "elapsed_seconds": time.perf_counter() - started,
                                                "error_type": type(error).__name__, "error": str(error),
                                                "legacy_labels_opened": False, "sealed_test_opened": False,
                                                "automatic_retry_allowed": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-code", type=Path)
    parser.add_argument("--baseline-scores", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.source_run, arguments.output, arguments.protocol,
                         arguments.source_code or arguments.source_run / "code",
                         arguments.baseline_scores or arguments.source_run / "diagnostics" / "scored_inputs.json")))
