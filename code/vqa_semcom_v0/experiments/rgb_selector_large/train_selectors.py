"""EXP-014: 84 large selectors and 12 nested joint selectors, train/val only.

No test manifest, feature, prediction, image or answer is read by this trainer.
Selection uses validation realized utility; every comparison freezes before test.
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
import signal
import statistics
import sys
import time
from typing import Any, Iterator

RECIPES = {
    "original_absolute": {"balanced": False, "target": "absolute"},
    "balanced_absolute": {"balanced": True, "target": "absolute"},
    "original_gain": {"balanced": False, "target": "gain"},
    "balanced_gain": {"balanced": True, "target": "gain"},
}
FAMILIES = {"joint9": list(range(9)), "rate_at_low": [0, 3, 6], "rate_at_medium": [1, 4, 7],
            "rate_at_high": [2, 5, 8], "compute_at_2000": [0, 1, 2],
            "compute_at_4000": [3, 4, 5], "compute_at_8000": [6, 7, 8]}
CELLS = [(budget, tier) for budget in (2000, 4000, 8000) for tier in ("low", "medium", "high")]
PIXELS = {"low": 50176, "medium": 100352, "high": 200704}
SEEDS = [7, 17, 27]
TASKS = ["object_presence", "counting", "color", "positional_reasoning", "scene_recognition", "activity_recognition"]


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
            raise ValueError(f"Frozen dependency changed: {name}")


def validate_protocol(protocol: dict) -> None:
    expected_training = {"seeds": SEEDS, "hidden": [128, 64], "dropout": .1, "max_epochs": 50,
                         "min_epochs": 10, "patience": 8, "batch_size": 32, "learning_rate": .001,
                         "weight_decay": .0001, "cpu_threads": 2, "large_fits": 84, "nested_joint_fits": 12,
                         "maximum_fits": 96, "wallclock_limit_seconds": 3600}
    if protocol.get("experiment_id") != "EXP-014" or protocol.get("recipes") != RECIPES or protocol.get("families") != FAMILIES:
        raise ValueError("EXP-014 recipe/family contract differs")
    if protocol["training"] != expected_training:
        raise ValueError("EXP-014 bounded training settings differ")
    if protocol["data"]["tasks"] != TASKS or protocol["data"]["splits"] != {"train": 4800, "validation": 1200, "test": 2400}:
        raise ValueError("Registered task/split inventory differs")
    if protocol["data"]["nested_train_per_type"] != 80 or protocol["data"]["per_type"] != {"train": 800, "validation": 200, "test": 400}:
        raise ValueError("Registered class-balanced quotas differ")
    if protocol["budgets"] != [2000, 4000, 8000] or protocol["frame_header_bytes"] != 1 or protocol["tiers"] != [{"name": name, "pixels": count} for name, count in PIXELS.items()]:
        raise ValueError("Frozen RGB action/framing contract differs")
    if protocol["features"]["question_dimensions"] != 256 or protocol["features"]["image_dimensions"] != 83 or protocol["features"]["balanced_multiplier"] != "1/sqrt(83)":
        raise ValueError("Feature contract differs")
    if protocol["selection"]["lambda"] != .05 or protocol["selection"]["lambda_search"] or not protocol["selection"]["no_test_selection"]:
        raise ValueError("Fixed cost and validation-only selection are required")
    if protocol["test"]["old_test300_opened"] or not protocol["test"]["new_test_authorized"]:
        raise ValueError("Only the new frozen test protocol is authorized")


@contextmanager
def wall_clock_budget(seconds: float) -> Iterator[None]:
    if seconds <= 0:
        raise TimeoutError("EXP-014 one-hour training budget exhausted")
    previous = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError("An existing real-time alarm prevents bounded fitting")

    def expired(_signal: int, _frame: Any) -> None:
        raise TimeoutError("EXP-014 one-hour training budget exhausted; no automatic new fits")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def load_source_module(source_code: Path) -> Any:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source_code))
    spec = importlib.util.spec_from_file_location("exp014_frozen_exp012_helpers", source_code / "train_selector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nominal_cost(action: int) -> float:
    budget, tier = CELLS[action]
    return (budget + 1) / 8001 + PIXELS[tier] / 200704


def reference_action(actions: list[int]) -> int:
    if not actions or len(set(actions)) != len(actions) or any(a not in range(9) for a in actions):
        raise ValueError("Invalid candidate action mapping")
    return min(actions, key=lambda a: (nominal_cost(a), a))


def choose_action(scores: list[float], actions: list[int], target: str) -> int:
    reference = reference_action(actions)
    if target not in ("absolute", "gain") or len(scores) != len(actions):
        raise ValueError("Target/score width differs from family")
    lower = 0 if target == "absolute" else -1
    if any(not math.isfinite(p) or not lower <= p <= 1 for p in scores):
        raise ValueError("Invalid selector score")
    if target == "gain" and scores[actions.index(reference)] != 0:
        raise ValueError("Family reference gain must be exactly zero")
    offset = nominal_cost(reference) if target == "gain" else 0
    return max(zip(actions, scores), key=lambda ap: (ap[1] - .05 * (nominal_cost(ap[0]) - offset), -nominal_cost(ap[0]), -ap[0]))[0]


def feature_tensor(frozen: Any, rows: list[dict], scaler: dict, balanced: bool) -> Any:
    import torch
    values = torch.tensor([frozen.vector(row, scaler) for row in rows], dtype=torch.float32)
    if balanced:
        values[:, 256:] *= 1 / math.sqrt(83)
    return values


def gain_outputs(logits: Any, reference_column: int) -> Any:
    import torch
    return torch.tanh(logits - logits[:, reference_column:reference_column + 1])


def gain_targets(correctness: Any, actions: list[int]) -> tuple[Any, int]:
    reference = reference_action(actions)
    return correctness[:, actions] - correctness[:, reference:reference + 1], actions.index(reference)


def fit_gain(frozen: Any, x_train: Any, y_train: Any, x_validation: Any, y_validation: Any,
             actions: list[int], seed: int, settings: dict) -> tuple[Any, list[dict], dict]:
    import torch
    from torch.nn import functional as F
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = frozen.make_model(len(actions))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    target_train, reference_column = gain_targets(y_train, actions)
    target_validation, _ = gain_targets(y_validation, actions)
    nonreference = [column for column in range(len(actions)) if column != reference_column]
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    history, started = [], time.perf_counter()
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        permutation = torch.randperm(len(x_train), generator=generator)
        total = 0.
        for batch in permutation.split(settings["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(gain_outputs(model(x_train[batch]), reference_column)[:, nonreference], target_train[batch][:, nonreference])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite gain MSE")
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise ValueError("Nonfinite gain gradient")
            optimizer.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.inference_mode():
            loss = F.mse_loss(gain_outputs(model(x_validation), reference_column)[:, nonreference], target_validation[:, nonreference]).item()
        if not math.isfinite(loss):
            raise ValueError("Nonfinite validation gain MSE")
        improved = loss < best_loss
        if improved:
            best_loss, best_epoch, stale = loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch, "train_gain_mse": total / len(x_train), "validation_gain_mse": loss,
                        "checkpoint_improved": improved})
        if epoch >= settings["min_epochs"] and stale >= settings["patience"]:
            break
    if best_state is None:
        raise ValueError("No finite gain checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, {"best_epoch": best_epoch, "best_validation_gain_mse": best_loss,
                           "epochs_run": len(history), "training_seconds": time.perf_counter() - started}


def summarize_choices(ids: list[str], grid: dict[str, list[dict]], choices: list[int]) -> dict:
    selected = [grid[identity][action] for identity, action in zip(ids, choices)]
    return {"n": len(ids), "correct": sum(row["correct"] for row in selected),
            "accuracy": statistics.mean(row["correct"] for row in selected),
            "utility": statistics.mean(row["utility"] for row in selected),
            "mean_nominal_cost": statistics.mean(nominal_cost(a) for a in choices),
            "mean_framed_image_bytes": statistics.mean(row["image_bytes"] for row in selected),
            "mean_codec_image_bytes": statistics.mean(row["codec_image_bytes"] for row in selected),
            "mean_ldpc_complex_symbols": statistics.mean(row["ldpc_complex_symbols"] for row in selected),
            "mean_actual_visual_tokens": statistics.mean(row["actual_visual_tokens"] for row in selected),
            "routing_histogram": {str(a): choices.count(a) for a in range(9)}, "selected_actions": choices}


def select_candidate(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("No validation candidates")
    return min(candidates, key=lambda row: (-row["summary"]["utility"], row["summary"]["mean_nominal_cost"], row["recipe"], row["family"]))


def select_fixed(fixed: dict[str, dict]) -> int:
    return min(range(9), key=lambda action: (-fixed[str(action)]["utility"], nominal_cost(action), action))


def validation_selection(report: dict) -> dict:
    large = report["evaluations"]["large"]
    output = {}
    for label, prefix in (("primary_joint", "joint9"), ("strong_rate", "rate_at_"), ("strong_compute", "compute_at_")):
        candidates = [{"arm": "large", "recipe": recipe, "family": family, "summary": replicates["ensemble"]}
                      for recipe, families in large.items() for family, replicates in families.items() if family.startswith(prefix)]
        selected = select_candidate(candidates)
        output[label] = {key: selected[key] for key in ("arm", "recipe", "family")}
        output[label]["validation_utility"] = selected["summary"]["utility"]
    output["strong_fixed_action"] = select_fixed(report["fixed"])
    output["primary_fixed_action"] = 4
    output["selection_split"] = "validation"
    output["seed_selection"] = "Same ensemble-selected recipe/family for every seed; no per-seed reselection"
    return output


def development_screens(report: dict, selection: dict, settings: dict) -> dict:
    evaluations = report["evaluations"]["large"]
    primary_fixed = report["fixed"]["4"]
    strong_fixed = report["fixed"][str(selection["strong_fixed_action"])]
    result = {}
    for recipe in RECIPES:
        joint = evaluations[recipe]["joint9"]
        consistent = []
        for seed in map(str, SEEDS):
            controls = [strong_fixed] + [evaluations[selection[key]["recipe"]][selection[key]["family"]][seed] for key in ("strong_rate", "strong_compute")]
            if all(joint[seed]["utility"] - baseline["utility"] >= settings["minimum_utility_gain_over_strong_fixed_and_single_axes"] for baseline in controls):
                consistent.append(int(seed))
        ensemble = joint["ensemble"]
        controls = {"strong_fixed": strong_fixed, **{key: evaluations[selection[key]["recipe"]][selection[key]["family"]]["ensemble"] for key in ("strong_rate", "strong_compute")}}
        gains = {name: ensemble["utility"] - baseline["utility"] for name, baseline in controls.items()}
        checks = {"accuracy_margin": ensemble["accuracy"] >= primary_fixed["accuracy"] - settings["maximum_accuracy_loss_vs_4000_medium"] - 1e-12,
                  "bytes_ratio": ensemble["mean_framed_image_bytes"] <= primary_fixed["mean_framed_image_bytes"] * settings["maximum_mean_image_byte_ratio_vs_4000_medium"],
                  "utility_over_strong_controls": all(gain >= settings["minimum_utility_gain_over_strong_fixed_and_single_axes"] for gain in gains.values()),
                  "seed_consistency": len(consistent) >= settings["minimum_consistent_seeds"]}
        result[recipe] = {"checks": checks, "point_screen_pass": all(checks.values()), "consistent_seeds": consistent,
                          "utility_gains": gains, "full_cost_gate": "PENDING", "test_runs_regardless_of_screen": True}
    return result


def prepare_data(data: Path, protocol_path: Path, frozen: Any) -> tuple:
    protocol = read(protocol_path)
    completion = read(data / "frozen_data.json")
    if completion["state"] != "COMPLETE" or completion["protocol_sha256"] != sha(protocol_path):
        raise ValueError("Data are not frozen under EXP-014 protocol")
    # Explicit allowlist. The preparer freezes the test manifest indirectly in
    # metadata; this function deliberately never opens or hashes any test-data file.
    names = ["frozen_data.json", "selection_frozen.json", "features.json", "features_complete.json",
             "supervision_records.json", "supervision_complete.json",
             "train_manifest.json", "validation_manifest.json", "nested_train_manifest.json",
             "train_truth.json", "validation_truth.json", "nested_train_truth.json"]
    paths = [data / name for name in names]
    paths += [data / "features_frozen.json"] if (data / "features_frozen.json").exists() else []
    manifest = {split: read(data / f"{split}_manifest.json") for split in ("train", "validation", "nested_train")}
    truth = {split: read(data / f"{split}_truth.json") for split in manifest}
    for split, quota in (("train", 800), ("validation", 200), ("nested_train", 80)):
        if dict(Counter(row["question_type"] for row in manifest[split])) != {task: quota for task in TASKS}:
            raise ValueError(f"Unbalanced or incomplete {split} manifest")
        if len({row["id"] for row in manifest[split]}) != 6 * quota or len({row["image_id"] for row in manifest[split]}) != 6 * quota:
            raise ValueError("Manifest identities/images repeat")
        if len(truth[split]) != 6 * quota or {row["id"] for row in truth[split]} != {row["id"] for row in manifest[split]}:
            raise ValueError("Truth identities differ from the registered manifest")
    train_ids, val_ids = ({row["image_id"] for row in manifest[split]} for split in ("train", "validation"))
    if train_ids & val_ids:
        raise ValueError("Training and validation images overlap")
    by_train = {row["id"]: row for row in manifest["train"]}
    by_truth = {row["id"]: row["answer"] for row in truth["train"]}
    if any(row["id"] not in by_train or row["image_id"] != by_train[row["id"]]["image_id"] for row in manifest["nested_train"]):
        raise ValueError("Nested manifests are not a subset of the new source training arm")
    if any(by_truth.get(row["id"]) != row["answer"] for row in truth["nested_train"]):
        raise ValueError("Nested truths differ from their training counterparts")
    expected = {split: {row["id"] for row in manifest[split]} for split in ("train", "validation")}
    features = read(data / "features.json")
    if dict(Counter(row["split"] for row in features)) != {"train": 4800, "validation": 1200}:
        raise ValueError("Trainer requires only the 6000 train/validation cached features")
    if any(set(row) & set(protocol["features"]["forbidden_inputs"]) for row in features):
        raise ValueError("Forbidden metadata in deployment feature records")
    mapped = frozen.validate_features(features, expected)
    feature_complete = read(data / "features_complete.json")
    if feature_complete.get("features_sha256") != sha(data / "features.json") or feature_complete.get("protocol_sha256") not in (None, sha(protocol_path)):
        raise ValueError("Feature inventory changed")
    records = read(data / "supervision_records.json")
    complete = read(data / "supervision_complete.json")
    if (len(records) != 54000 or complete.get("records_sha256") != sha(data / "supervision_records.json")
            or complete.get("state") != "SUPERVISION_COMPLETE" or complete.get("records") != 54000
            or complete.get("protocol_sha256") != sha(protocol_path)
            or complete.get("receiver_sha256") != protocol["receiver_sha256"]):
        raise ValueError("Complete unchanged 6000x9 supervision is required")
    if {row["split"] for row in records} != {"train", "validation"}:
        raise ValueError("Test or unknown outcomes in training supervision")
    allowed_record_ids = expected["train"] | expected["validation"]
    if {row["id"] for row in records} != allowed_record_ids:
        raise ValueError("Supervision identities differ from manifests")
    registered = {row["id"]: (split, row["image_id"]) for split in ("train", "validation") for row in manifest[split]}
    if any((row["split"], row["image_id"]) != registered[row["id"]] for row in records):
        raise ValueError("Supervision image/split differs from registered identity")
    grids = {split: frozen.score_grid(records, truth[split], [row["id"] for row in manifest[split]],
                                      protocol["receiver_sha256"], sha(protocol_path)) for split in ("train", "validation")}
    for path in paths:
        expected_hash = completion.get("sha256", {}).get(str(path.resolve()))
        if path.name.endswith("_truth.json"):
            expected_hash = completion["truth_sha256"].get(path.name)
        if expected_hash is not None and sha(path) != expected_hash:
            raise ValueError(f"Frozen data artifact changed: {path.name}")
    rows = {split: [mapped[row["id"]] for row in manifest[split]] for split in manifest}
    return rows, grids, paths, completion


def training_grid_dependencies(data: Path, protocol_path: Path, receiver_hash: str) -> dict[str, str]:
    """Pin completed trainval codec/receiver provenance and every downstream input."""
    data, protocol_path = data.resolve(), protocol_path.resolve()
    directory = data / "grid_trainval"
    protocol_hash = sha(protocol_path)
    combined = {}
    fingerprints = [directory / "codec_frozen.json", directory / "inference_frozen.json"]
    for path in fingerprints:
        frozen = read(path)
        if frozen.get("phase") != "trainval" or frozen.get("controller_sha256") is not None or not frozen.get("sha256"):
            raise ValueError("Grid fingerprint is not a pre-controller trainval freeze")
        dependencies = frozen["sha256"]
        if dependencies.get(str(protocol_path)) != protocol_hash:
            raise ValueError("Grid fingerprint does not pin the current protocol")
        for name, digest in dependencies.items():
            dependency = Path(name).resolve()
            if ("truth" in dependency.name.lower() or dependency.name.startswith("test_") and dependency.suffix != ".py"
                    or "grid_test" in dependency.parts or any("test300" in part.lower() for part in dependency.parts)):
                raise ValueError("Test/answer data are forbidden in label-free grid dependencies")
            if str(dependency) in combined and combined[str(dependency)] != digest:
                raise ValueError("Codec and receiver freezes disagree on a shared dependency")
            combined[str(dependency)] = digest
        combined[str(path)] = sha(path)
    # Verification uses the existing stage hashes, never a newly accepted hash of
    # changed historical codec/model code or weights.
    verify_hashes(combined)
    encoding_path = directory / "encoding_complete.json"
    inference_path = data / "supervision_complete.json"
    encoding, inference = read(encoding_path), read(inference_path)
    for complete in (encoding, inference):
        if (complete.get("phase") != "trainval" or complete.get("protocol_sha256") != protocol_hash
                or complete.get("controller_sha256") is not None):
            raise ValueError("Completed grid phase/protocol/controller differs from training")
    representation_path, records_path = directory / "representations.json", data / "supervision_records.json"
    if (encoding.get("state") != "COMPLETE" or encoding.get("representations") != 18000
            or encoding.get("representations_sha256") != sha(representation_path)):
        raise ValueError("Complete unchanged 6000x3 codec inventory is required")
    if (inference.get("state") != "SUPERVISION_COMPLETE" or inference.get("records") != 54000
            or inference.get("records_sha256") != sha(records_path)
            or inference.get("receiver_sha256") != receiver_hash or inference.get("labels_loaded") is not False
            or inference.get("old_test300_opened") is not False):
        raise ValueError("Complete frozen label-free 6000x9 receiver grid is required")
    for path in (encoding_path, inference_path, representation_path, records_path, directory / "model_runtime.json"):
        digest = sha(path)
        if str(path) in combined and combined[str(path)] != digest:
            raise ValueError("Completed stage artifact conflicts with a grid fingerprint")
        combined[str(path)] = digest
    return combined


def run(data: Path, output: Path, protocol_path: Path, source_code: Path, evaluator_path: Path) -> dict:
    import torch
    started = time.perf_counter()
    data, output, protocol_path, source_code, evaluator_path = [path.resolve() for path in (data, output, protocol_path, source_code, evaluator_path)]
    protocol = read(protocol_path)
    validate_protocol(protocol)
    if not evaluator_path.is_file():
        raise ValueError("Evaluation implementation must exist and be frozen before fitting")
    if (output / "training_complete.json").exists():
        existing = read(output / "training_complete.json")
        controller = read(output / "controller_frozen.json")
        verify_hashes(controller["sha256"])
        verify_hashes(controller["source_inputs_sha256"])
        if existing.get("state") != "COMPLETE" or existing.get("completed_fits") != 96 or existing["controller_sha256"] != sha(output / "controller_frozen.json") or controller["protocol_sha256"] != sha(protocol_path):
            raise ValueError("Existing completion is invalid; no additional fitting budget")
        return existing
    if any((output / name).exists() for name in ("training_frozen_inputs.json", "training_history.json", "controller_frozen.json", "checkpoints")):
        raise ValueError("Partial fitting artifacts exist; automatic refitting is prohibited")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    frozen = load_source_module(source_code)
    histories, checkpoints, models, completed = {}, {}, {}, 0
    try:
        with wall_clock_budget(3600 - (time.perf_counter() - started)):
            rows, grids, inputs, data_completion = prepare_data(data, protocol_path, frozen)
            grid_dependencies = training_grid_dependencies(data, protocol_path, protocol["receiver_sha256"])
            runtime_code = sorted(Path(__file__).resolve().parent.glob("*.py"))
            runtime_code += sorted(source_code.glob("*.py"))
            inputs += [protocol_path, evaluator_path, *runtime_code]
            source_hashes = {str(path): sha(path) for path in inputs}
            for name, expected in grid_dependencies.items():
                if name in source_hashes and source_hashes[name] != expected:
                    raise ValueError("Training and completed grid fingerprints disagree")
                source_hashes[name] = expected
            save(output / "training_frozen_inputs.json", {"schema_version": 1, "sha256": source_hashes,
                "data_root": str(data), "source_code": str(source_code), "protocol_sha256": sha(protocol_path),
                "torch_version": torch.__version__, "cpu_threads": 2, "test_data_read": False, "test_labels_opened": False})
            training_rows = {"large": rows["train"], "nested": rows["nested_train"]}
            scalers = {arm: frozen.fit_scaler(values) for arm, values in training_rows.items()}
            save(output / "scalers.json", {"schema_version": 1, "scalers": scalers, "balanced_multiplier": 1 / math.sqrt(83)})
            val_ids = [row["id"] for row in rows["validation"]]
            y_val = torch.tensor([[cell["correct"] for cell in grids["validation"][identity]] for identity in val_ids], dtype=torch.float32)
            report = {"schema_version": 1, "validation_ids": val_ids, "evaluations": {}, "fixed": {
                str(action): summarize_choices(val_ids, grids["validation"], [action] * len(val_ids)) for action in range(9)}}
            predictions = [{"id": identity, "split": "validation", "scores": {}} for identity in val_ids]
            for arm, train_rows in training_rows.items():
                histories[arm], checkpoints[arm], models[arm], report["evaluations"][arm] = {}, {}, {}, {}
                arm_families = FAMILIES if arm == "large" else {"joint9": FAMILIES["joint9"]}
                y_train = torch.tensor([[cell["correct"] for cell in grids["train"][row["id"]]] for row in train_rows], dtype=torch.float32)
                for recipe, config in RECIPES.items():
                    histories[arm][recipe], checkpoints[arm][recipe], models[arm][recipe], report["evaluations"][arm][recipe] = {}, {}, {}, {}
                    x_train = feature_tensor(frozen, train_rows, scalers[arm], config["balanced"])
                    x_val = feature_tensor(frozen, rows["validation"], scalers[arm], config["balanced"])
                    for family, actions in arm_families.items():
                        histories[arm][recipe][family], checkpoints[arm][recipe][family], models[arm][recipe][family] = {}, {}, {}
                        scores = {}
                        for seed in SEEDS:
                            if completed >= 96:
                                raise RuntimeError("96-fit budget exhausted")
                            save(output / "training_status.json", {"state": "TRAINING", "arm": arm, "recipe": recipe, "family": family,
                                "seed": seed, "completed_fits": completed, "maximum_fits": 96})
                            if config["target"] == "absolute":
                                model, history, stats = frozen.train_one(x_train, y_train, x_val, y_val, actions, seed, protocol["training"])
                            else:
                                model, history, stats = fit_gain(frozen, x_train, y_train, x_val, y_val, actions, seed, protocol["training"])
                            completed += 1
                            path = output / "checkpoints" / f"{arm}-{recipe}-{family}-seed{seed}.pt"
                            frozen.torch_save_atomic(path, model.state_dict())
                            checkpoints[arm][recipe][family][str(seed)] = {"path": str(path), "relative_path": str(path.relative_to(output)),
                                "sha256": sha(path), "actions": actions, "reference_action": reference_action(actions)}
                            histories[arm][recipe][family][str(seed)] = {"history": history, **stats, "checkpoint_sha256": sha(path)}
                            models[arm][recipe][family][str(seed)] = model
                            model.eval()
                            with torch.inference_mode():
                                logits = model(x_val)
                                scores[str(seed)] = logits.sigmoid() if config["target"] == "absolute" else gain_outputs(logits, actions.index(reference_action(actions)))
                            save(output / "training_history.json", histories)
                        scores["ensemble"] = torch.stack(list(scores.values())).mean(0)
                        evaluations = {}
                        for replicate, tensor in scores.items():
                            values = tensor.tolist()
                            choices = [choose_action(value, actions, config["target"]) for value in values]
                            evaluations[replicate] = summarize_choices(val_ids, grids["validation"], choices)
                            for row, value in zip(predictions, values):
                                row["scores"].setdefault(arm, {}).setdefault(recipe, {}).setdefault(family, {})[replicate] = value
                        report["evaluations"][arm][recipe][family] = evaluations
            if completed != 96:
                raise RuntimeError("All 96 fits are required")
            selection = validation_selection(report)
            report["selection"] = selection
            report["development_screen"] = development_screens(report, selection, protocol["screen"])
            save(output / "selection.json", selection)
            save(output / "validation_report.json", report)
            save(output / "validation_predictions.json", {"schema_version": 1, "rows": predictions, "families": FAMILIES,
                "score_semantics": {recipe: config["target"] for recipe, config in RECIPES.items()}, "protocol_sha256": sha(protocol_path)})
            verify_hashes(source_hashes)
            artifacts = [output / name for name in ("training_frozen_inputs.json", "training_history.json", "scalers.json", "selection.json", "validation_report.json", "validation_predictions.json")]
            artifacts += [Path(meta["path"]) for recipes in checkpoints.values() for families in recipes.values() for seeds in families.values() for meta in seeds.values()]
            fingerprints = {str(path): sha(path) for path in artifacts + [protocol_path, evaluator_path, *runtime_code]}
            fingerprints.update(grid_dependencies)
            controller = {"schema_version": 1, "state": "FROZEN_BEFORE_TEST", "created_utc": datetime.now(timezone.utc).isoformat(),
                "completed_fits": completed, "seeds": SEEDS, "recipes": RECIPES, "families": FAMILIES, "checkpoints": checkpoints,
                "selection": selection, "sha256": fingerprints, "artifact_sha256": {str(path.relative_to(output)): sha(path) for path in artifacts},
                "source_inputs_sha256": source_hashes, "evaluation_code_sha256": {str(evaluator_path): sha(evaluator_path)},
                "protocol_sha256": sha(protocol_path), "frozen_data_sha256": sha(data / "frozen_data.json"),
                "test_manifest_sha256": data_completion["test_manifest_sha256"],
                "sealed_test_truth_sha256": data_completion["sealed_test_truth_sha256"],
                "test_labels_opened": False, "test_features_opened": False, "test_records_opened": False,
                "source_code": str(source_code), "data_root": str(data), "full_cost_gate": "PENDING"}
            save(output / "controller_frozen.json", controller)
            verify_hashes(fingerprints)
            complete = {"schema_version": 1, "state": "COMPLETE", "completed_fits": completed,
                "large_fits": 84, "nested_joint_fits": 12, "elapsed_seconds": time.perf_counter() - started,
                "controller_sha256": sha(output / "controller_frozen.json"), "protocol_sha256": sha(protocol_path),
                "selection": selection, "test_labels_opened": False, "test_data_read": False, "source_unchanged": True}
            save(output / "training_complete.json", complete)
            save(output / "training_status.json", complete)
            return complete
    except BaseException as error:
        save(output / "training_status.json", {"state": "FAILED", "completed_fits": completed,
            "elapsed_seconds": time.perf_counter() - started, "error_type": type(error).__name__, "error": str(error),
            "automatic_retry_allowed": False, "test_data_read": False, "test_labels_opened": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-code", type=Path, required=True)
    parser.add_argument("--source-factorial-code", type=Path, help="Compatibility argument; generalized gain trainer is defined here")
    parser.add_argument("--evaluation-code", type=Path, default=Path(__file__).with_name("evaluate_test.py"))
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.data_root, arguments.output, arguments.protocol, arguments.source_code, arguments.evaluation_code)))
