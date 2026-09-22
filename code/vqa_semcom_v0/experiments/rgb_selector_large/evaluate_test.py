"""One frozen EXP-014 test evaluation; no fitting, selection, or receiver calls.

Every controller dependency is checked before opening new test inputs. A started
receipt prevents concurrent or implicit repeated scoring. A completed report is
returned only after its frozen controller, input hashes, and receipt verify.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

SEEDS = (7, 17, 27)
REPLICATES = ("7", "17", "27", "ensemble")
RECIPES = ("original_absolute", "balanced_absolute", "original_gain", "balanced_gain")
FAMILIES = {"joint9": list(range(9)), "rate_at_low": [0, 3, 6], "rate_at_medium": [1, 4, 7],
            "rate_at_high": [2, 5, 8], "compute_at_2000": [0, 1, 2], "compute_at_4000": [3, 4, 5],
            "compute_at_8000": [6, 7, 8]}
CELLS = [(budget, tier) for budget in (2000, 4000, 8000) for tier in ("low", "medium", "high")]
BOOTSTRAPS, SIGN_FLIPS, RANDOM_SEED = 2000, 10000, 20260923
PRIMARY_CONTROLS = ("strong_fixed", "strong_rate", "strong_compute")
TEST_INPUT_NAMES = ("test_manifest.json", "test_features.json", "test_features_complete.json",
                    "test_records.json", "test_inference_complete.json", "test_truth.sealed.json")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path: Path, value: Any) -> None:
    """Write once. An interrupted publication is never silently overwritten."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path.name}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError(f"Concurrent output appeared: {path.name}")
    temporary.replace(path)


def load_module(path: Path, name: str) -> Any:
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load frozen module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nominal_cost(action: int) -> float:
    budget, tier = CELLS[action]
    return (budget + 1) / 8001 + {"low": .25, "medium": .5, "high": 1.}[tier]


def select_action(scores: list[float], actions: list[int], target: str) -> int:
    if target not in ("absolute", "gain") or not actions or len(scores) != len(actions):
        raise ValueError("Invalid frozen score/family contract")
    if len(set(actions)) != len(actions) or any(a not in range(9) for a in actions):
        raise ValueError("Invalid action indices")
    minimum = -1 if target == "gain" else 0
    if any(not math.isfinite(v) or not minimum <= v <= 1 for v in scores):
        raise ValueError("Invalid score range")
    reference = min(actions, key=lambda a: (nominal_cost(a), a))
    if target == "gain" and scores[actions.index(reference)] != 0:
        raise ValueError("Family reference gain must be exactly zero")
    offset = nominal_cost(reference) if target == "gain" else 0
    return max(zip(actions, scores), key=lambda ap: (
        ap[1] - .05 * (nominal_cost(ap[0]) - offset), -nominal_cost(ap[0]), -ap[0]))[0]


def require_pinned(path: Path, hashes: dict[str, str]) -> None:
    if hashes.get(str(path.resolve())) != sha(path):
        raise ValueError(f"Actual file is not a frozen dependency: {path.name}")


def verify_non_test_hashes(hashes: dict[str, str]) -> None:
    if not hashes:
        raise ValueError("Empty controller dependency inventory")
    for name, expected in hashes.items():
        path = Path(name)
        # Test input hashes belong to the post-controller gate, not this loop.
        if (path.name in TEST_INPUT_NAMES or path.name == "dev_truth.sealed.json"
                or any("test300" in part.lower() for part in path.parts)):
            raise ValueError("Forbidden premature or historical test read in controller inventory")
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Frozen controller dependency changed: {path.name}")


def verify_controller(output: Path, protocol_path: Path, source_code: Path, trainer_path: Path) -> tuple[dict, dict]:
    """Do not open test manifest/features/records/truth anywhere in this gate."""
    controller_path = output / "controller_frozen.json"
    if not controller_path.is_file():
        raise ValueError("All controllers must freeze before any new test input is read")
    controller = read(controller_path)
    if controller.get("state") != "FROZEN_BEFORE_TEST" or controller.get("completed_fits") != 96:
        raise ValueError("The 96-fit controller freeze is incomplete")
    if controller.get("seeds") != list(SEEDS) or any(controller.get(name) is not False for name in
            ("test_labels_opened", "test_features_opened", "test_records_opened")):
        raise ValueError("Invalid frozen seed inventory or test-access chronology")
    verify_non_test_hashes(controller["sha256"])
    verify_non_test_hashes(controller["source_inputs_sha256"])
    require_pinned(protocol_path, controller["sha256"])
    require_pinned(Path(__file__).resolve(), controller["evaluation_code_sha256"])
    require_pinned(trainer_path, controller["source_inputs_sha256"])
    require_pinned(source_code / "train_selector.py", controller["source_inputs_sha256"])
    require_pinned(source_code / "prepare_data.py", controller["source_inputs_sha256"])
    if controller["protocol_sha256"] != sha(protocol_path):
        raise ValueError("Frozen protocol fingerprint differs")
    for name in ("selection.json", "scalers.json", "validation_predictions.json"):
        require_pinned(output / name, controller["sha256"])
    complete = read(output / "training_complete.json")
    if (complete.get("state") != "COMPLETE" or complete.get("completed_fits") != 96
            or complete.get("controller_sha256") != sha(controller_path)):
        raise ValueError("Training completion does not pin this complete controller")
    selection = read(output / "selection.json")
    if selection != controller["selection"] or selection.get("selection_split") != "validation":
        raise ValueError("Selections changed after validation")
    for name, prefix in (("primary_joint", "joint9"), ("strong_rate", "rate_at_"), ("strong_compute", "compute_at_")):
        chosen = selection[name]
        if (chosen["arm"] != "large" or chosen["recipe"] not in RECIPES
                or chosen["family"] not in FAMILIES or not chosen["family"].startswith(prefix)):
            raise ValueError("Primary/control selection violates its frozen candidate family")
    if selection.get("primary_fixed_action") != 4 or selection.get("strong_fixed_action") not in range(9):
        raise ValueError("Invalid fixed controls")
    count = 0
    if set(controller["checkpoints"]) != {"large", "nested"}:
        raise ValueError("Both large and nested arms must freeze")
    for arm, recipes in controller["checkpoints"].items():
        if set(recipes) != set(RECIPES):
            raise ValueError("Incomplete frozen recipes")
        for families in recipes.values():
            expected = FAMILIES if arm == "large" else {"joint9": FAMILIES["joint9"]}
            if set(families) != set(expected):
                raise ValueError("Incomplete frozen families")
            for family, seeds in families.items():
                if set(seeds) != set(map(str, SEEDS)):
                    raise ValueError("Every family must contain all three seeds")
                for meta in seeds.values():
                    require_pinned(Path(meta["path"]), controller["sha256"])
                    if meta["sha256"] != sha(Path(meta["path"])) or meta["actions"] != expected[family]:
                        raise ValueError("Checkpoint actions or content changed")
                    if meta["reference_action"] != min(meta["actions"], key=lambda a: (nominal_cost(a), a)):
                        raise ValueError("Checkpoint gain reference differs from the family minimum")
                    count += 1
    if count != 96:
        raise ValueError("Expected exactly 96 frozen checkpoints")
    return controller, selection


def snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha(path) for path in paths}


def verify_snapshot(hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Test input changed: {Path(name).name}")


def verify_test_stage_hashes(hashes: dict[str, str], output: Path) -> None:
    """Validate post-freeze, label-free stage dependencies without old test reads."""
    output = output.resolve()
    if not hashes:
        raise ValueError("Missing test-stage dependency inventory")
    for name in hashes:
        path = Path(name).resolve()
        if ("truth" in path.name or any("test300" in part.lower() for part in path.parts)
                or (path.name.startswith("test_") and path.suffix != ".py" and output not in path.parents)):
            raise ValueError("Test-stage provenance points at labels or an external test artifact")
    verify_snapshot(hashes)


def load_test_inputs(output: Path, protocol: dict, protocol_path: Path, controller: dict,
                     frozen: Any) -> tuple[list[dict], list[dict], dict, dict]:
    """Called only after the complete controller gate and one-attempt lock."""
    paths = [output / name for name in TEST_INPUT_NAMES]
    before = snapshot(paths)
    if sha(output / "test_manifest.json") != controller["test_manifest_sha256"]:
        raise ValueError("New test manifest differs from the preselected holdout")
    if sha(output / "test_truth.sealed.json") != controller["sealed_test_truth_sha256"]:
        raise ValueError("New test truth differs from its pre-training fingerprint")
    features_complete = read(output / "test_features_complete.json")
    inference_complete = read(output / "test_inference_complete.json")
    controller_hash = sha(output / "controller_frozen.json")
    for complete in (features_complete, inference_complete):
        if (complete.get("phase") != "test" or complete.get("protocol_sha256") != sha(protocol_path)
                or complete.get("controller_sha256") != controller_hash or complete.get("labels_loaded") is not False):
            raise ValueError("Test preprocessing/inference lacks the frozen label-free provenance")
    n = protocol["data"]["splits"]["test"]
    if (features_complete.get("state") != "COMPLETE" or features_complete.get("records") != n
            or features_complete.get("features_sha256") != sha(output / "test_features.json")):
        raise ValueError("Test features are incomplete or changed")
    if (inference_complete.get("state") != "SUPERVISION_COMPLETE" or inference_complete.get("records") != 9 * n
            or inference_complete.get("records_sha256") != sha(output / "test_records.json")
            or inference_complete.get("receiver_sha256") != protocol["receiver_sha256"]
            or inference_complete.get("old_test300_opened") is not False):
        raise ValueError("Test receiver grid is incomplete or uses a changed receiver")
    feature_frozen_path = output / "test_features_frozen.json"
    if sha(feature_frozen_path) != features_complete.get("frozen_sha256"):
        raise ValueError("Test feature provenance changed")
    feature_frozen = read(feature_frozen_path)
    if feature_frozen.get("phase") != "test" or feature_frozen.get("labels_loaded") is not False:
        raise ValueError("Feature-stage provenance is not label-free test preprocessing")
    verify_test_stage_hashes(feature_frozen["sha256"], output)
    data_frozen_path = output / "test_data_frozen.json"
    require_pinned(data_frozen_path, feature_frozen["sha256"])
    test_data = read(data_frozen_path)
    if (test_data.get("state") != "COMPLETE" or test_data.get("test") != n
            or test_data.get("protocol_sha256") != sha(protocol_path)
            or test_data.get("controller_sha256") != controller_hash or test_data.get("test_labels_opened") is not False):
        raise ValueError("Test data were not frozen after this controller")
    verify_test_stage_hashes(test_data["sha256"], output)
    before.update(feature_frozen["sha256"])
    before.update(test_data["sha256"])
    before.update(snapshot([feature_frozen_path, data_frozen_path]))
    manifest, features = read(output / "test_manifest.json"), read(output / "test_features.json")
    if (len(manifest) != n or len({r["id"] for r in manifest}) != n or len({r["image_id"] for r in manifest}) != n
            or dict(Counter(r["question_type"] for r in manifest)) !=
            {task: protocol["data"]["per_type"]["test"] for task in protocol["data"]["tasks"]}
            or any(r.get("split") != "test" or r.get("source_split") != "val2014" for r in manifest)):
        raise ValueError("New test manifest violates image uniqueness, source, or balanced quotas")
    image_ids = {r["image_id"] for r in manifest}
    data_root = Path(controller["data_root"])
    for split in ("train", "validation"):
        path = data_root / f"{split}_manifest.json"
        require_pinned(path, controller["source_inputs_sha256"])
        if image_ids & {r["image_id"] for r in read(path)}:
            raise ValueError("New test images overlap training or validation")
    test_hash_path = output / "test_image_hashes.json"
    require_pinned(test_hash_path, feature_frozen["sha256"])
    image_hashes = read(test_hash_path)
    if set(image_hashes) != {r["id"] for r in manifest} or len(set(image_hashes.values())) != n:
        raise ValueError("Test content hashes are duplicated or incomplete")
    source_data_path = data_root / "frozen_data.json"
    require_pinned(source_data_path, controller["source_inputs_sha256"])
    source_data = read(source_data_path)
    forbidden_hashes = set()
    for filename in ("image_hashes.json", "historical_available_image_hashes.json"):
        path = data_root / filename
        require_pinned(path, source_data["sha256"])
        forbidden_hashes.update(read(path).values())
        before.update(snapshot([path]))
    if set(image_hashes.values()) & forbidden_hashes:
        raise ValueError("New test content overlaps training/validation or enumerated historical images")
    if any(row.get("source_image_sha256") != image_hashes.get(row["id"]) for row in features):
        raise ValueError("Test feature source image differs from its frozen content hash")
    forbidden_fields = {"answer", "question_type", "image_id", "question_id", "prediction"}
    if any(set(row) & forbidden_fields for row in features):
        raise ValueError("Forbidden annotation fields in deployment features")
    mapped = frozen.validate_features(features, {"test": {r["id"] for r in manifest}})
    rows = [mapped[r["id"]] for r in manifest]
    truth, records = read(output / "test_truth.sealed.json"), read(output / "test_records.json")
    identities = [r["id"] for r in manifest]
    if (len(truth) != n or len(records) != 9 * n or {r["id"] for r in records} != set(identities)
            or any(r.get("split") != "test" for r in records)):
        raise ValueError("New test outcomes have unknown, duplicated, or missing units")
    metadata = {r["id"]: r for r in manifest}
    for record in records:
        expected = metadata[record["id"]]
        if record["image_id"] != expected["image_id"] or record["question_type"] != expected["question_type"]:
            raise ValueError("Receiver result image/type differs from its frozen manifest")
    # New records already contain the one-byte frame. Never apply legacy +1 here.
    grid = frozen.score_grid(records, truth, identities, protocol["receiver_sha256"], sha(protocol_path), legacy=False)
    verify_snapshot(before)
    return manifest, rows, grid, before


def required_variants(selection: dict) -> list[tuple[str, str, str]]:
    variants = [(arm, recipe, "joint9") for arm in ("large", "nested") for recipe in RECIPES]
    for name in ("strong_rate", "strong_compute"):
        chosen = selection[name]
        variants.append((chosen["arm"], chosen["recipe"], chosen["family"]))
    return list(dict.fromkeys(variants))


def forward_frozen(rows: list[dict], output: Path, controller: dict, selection: dict,
                   trainer: Any, frozen: Any) -> dict:
    """CPU forward only, reusing exactly the trainer's feature/output functions."""
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    scalers = read(output / "scalers.json")
    if scalers["balanced_multiplier"] != 1 / math.sqrt(83):
        raise ValueError("Frozen block multiplier changed")
    results = {}
    for arm, recipe, family in required_variants(selection):
        actions = FAMILIES[family]
        config = trainer.RECIPES[recipe]
        x = trainer.feature_tensor(frozen, rows, scalers["scalers"][arm], config["balanced"])
        predictions = {}
        for seed in SEEDS:
            meta = controller["checkpoints"][arm][recipe][family][str(seed)]
            model = frozen.load_checkpoint(Path(meta["path"]), actions)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            with torch.inference_mode():
                logits = model(x)
                predictions[str(seed)] = (logits.sigmoid() if config["target"] == "absolute" else
                    trainer.gain_outputs(logits, actions.index(trainer.reference_action(actions))))
        predictions["ensemble"] = torch.stack(list(predictions.values())).mean(0)
        results[(arm, recipe, family)] = {label: values.tolist() for label, values in predictions.items()}
    return results


def within_image_auc(scores: list[float], labels: list[int]) -> float | None:
    if len(scores) != len(labels):
        raise ValueError("AUC labels and scores are not paired")
    positive = [p for p, y in zip(scores, labels) if y]
    negative = [p for p, y in zip(scores, labels) if not y]
    if not positive or not negative:
        return None
    return sum(float(a > b) + .5 * float(a == b) for a in positive for b in negative) / (len(positive) * len(negative))


def bootstrap_mean(values: list[float], repeats: int = BOOTSTRAPS) -> dict:
    if not values:
        return {"n_images": 0, "mean": None, "ci95_unadjusted": None}
    values_array = np.asarray(values, dtype=float)
    if not np.isfinite(values_array).all():
        raise ValueError("Nonfinite paired-image statistics")
    rng = np.random.default_rng(RANDOM_SEED)
    samples = []
    for start in range(0, repeats, 128):
        indices = rng.integers(0, len(values), (min(128, repeats - start), len(values)))
        samples.extend(np.mean(values_array[indices], axis=1).tolist())
    return {"n_images": len(values), "mean": float(values_array.mean()),
            "ci95_unadjusted": np.quantile(samples, [.025, .975]).tolist(), "draws": repeats,
            "conditional_on_frozen_models": True}


def paired_sign_flip(differences: list[float], draws: int = SIGN_FLIPS) -> dict:
    result = bootstrap_mean(differences)
    if not differences:
        raise ValueError("Primary utility comparisons require nonempty paired images")
    array = np.asarray(differences, dtype=float)
    observed, extreme = abs(float(array.mean())), 0
    rng = np.random.default_rng(RANDOM_SEED)
    for start in range(0, draws, 128):
        signs = rng.choice([-1., 1.], size=(min(128, draws - start), len(array)))
        extreme += int(np.count_nonzero(np.abs(np.mean(signs * array, axis=1)) >= observed - 1e-14))
    result.update(sign_flip_p=(extreme + 1) / (draws + 1), sign_flip_draws=draws,
                  null_mean_difference=0., mean_difference=result["mean"], extreme_draws=extreme)
    return result


def holm_three(comparisons: dict[str, dict]) -> None:
    if set(comparisons) != set(PRIMARY_CONTROLS):
        raise ValueError("Primary multiplicity family must contain exactly the three frozen controls")
    previous = 0.
    for rank, result in enumerate(sorted(comparisons.values(), key=lambda r: r["sign_flip_p"])):
        previous = max(previous, min(1., (3 - rank) * result["sign_flip_p"]))
        result["holm_p"] = previous
        result["holm_family_size"] = 3


def selected_metrics(selected: list[dict], baseline: list[dict], auc: list[float] | None = None) -> dict:
    if not selected or [r["id"] for r in selected] != [r["id"] for r in baseline]:
        raise ValueError("Selected outcomes are not paired with the low-resource baseline")
    result = {"n": len(selected), "correct": sum(r["correct"] for r in selected),
              "accuracy": statistics.mean(r["correct"] for r in selected),
              "utility": statistics.mean(r["utility"] for r in selected),
              "rescue_vs_2000_low": sum(not b["correct"] and r["correct"] for r, b in zip(selected, baseline)),
              "harm_vs_2000_low": sum(b["correct"] and not r["correct"] for r, b in zip(selected, baseline)),
              "routing_histogram": dict(sorted(Counter(f"{r['budget']}_{r['tier']}" for r in selected).items())),
              "energy_j": None, "wireless_delivery_rate": None, "live_end_to_end_seconds": None}
    for field in ("codec_image_bytes", "image_bytes", "ldpc_complex_symbols", "actual_visual_tokens"):
        result[f"mean_{'framed_image_bytes' if field == 'image_bytes' else field}"] = statistics.mean(r[field] for r in selected)
    result["within_image_auc"] = {"n_images": len(auc), "mean": statistics.mean(auc) if auc else None} if auc is not None else None
    result["cached_component_timing"] = {}
    for field in ("encode_seconds", "decode_seconds", "preprocessing_seconds", "receiver_seconds"):
        values = [r[field] for r in selected if isinstance(r.get(field), (int, float)) and math.isfinite(r[field])]
        result["cached_component_timing"][field] = {"observed_n": len(values), "mean": statistics.mean(values) if values else None}
    return result


def method_report(rows: list[dict], baseline: list[dict], grid: dict, *, scores: list[list[float]] | None = None,
                  actions: list[int] | None = None, intervals: bool = False) -> dict:
    auc_by_id = {}
    if scores is not None:
        if len(scores) != len(rows) or actions != list(range(9)):
            raise ValueError("Reported comparative AUC requires the common nine-action image set")
        auc_by_id = {row["id"]: value for row, score in zip(rows, scores)
                     if (value := within_image_auc(score, [r["correct"] for r in grid[row["id"]]])) is not None}
    result = selected_metrics(rows, baseline, list(auc_by_id.values()) if scores is not None else None)
    result["by_type"] = {}
    for question_type in sorted({r["question_type"] for r in rows}):
        positions = [i for i, r in enumerate(rows) if r["question_type"] == question_type]
        auc = [auc_by_id[rows[i]["id"]] for i in positions if rows[i]["id"] in auc_by_id]
        result["by_type"][question_type] = selected_metrics([rows[i] for i in positions], [baseline[i] for i in positions],
                                                           auc if scores is not None else None)
    if intervals:
        result["uncertainty"] = {field: bootstrap_mean([r[field] for r in rows]) for field in
                                 ("correct", "utility", "image_bytes", "ldpc_complex_symbols", "actual_visual_tokens")}
        if scores is not None:
            result["within_image_auc"] = bootstrap_mean(list(auc_by_id.values()))
    return result


def point_screen(joint: dict, controls: dict, settings: dict) -> dict:
    ensemble = joint["ensemble"]
    reference = controls["fixed_4000_medium"]["ensemble"]
    gains = {name: ensemble["utility"] - controls[name]["ensemble"]["utility"] for name in PRIMARY_CONTROLS}
    consistent = [seed for seed in SEEDS if all(
        joint[str(seed)]["utility"] - controls[name][str(seed)]["utility"] >=
        settings["minimum_utility_gain_over_strong_fixed_and_single_axes"] for name in PRIMARY_CONTROLS)]
    checks = {
        "accuracy_margin": ensemble["correct"] >= reference["correct"] - ensemble["n"] * settings["maximum_accuracy_loss_vs_4000_medium"] - 1e-12,
        "bytes_ratio": ensemble["mean_framed_image_bytes"] <= reference["mean_framed_image_bytes"] * settings["maximum_mean_image_byte_ratio_vs_4000_medium"],
        "utility_over_strong_controls": all(value >= settings["minimum_utility_gain_over_strong_fixed_and_single_axes"] for value in gains.values()),
        "seed_consistency": len(consistent) >= settings["minimum_consistent_seeds"],
    }
    return {"checks": checks, "point_screen_pass": all(checks.values()), "consistent_seeds": consistent,
            "utility_gains": gains, "accuracy_margin": settings["maximum_accuracy_loss_vs_4000_medium"],
            "equivalent_net_lost_questions": ensemble["n"] * settings["maximum_accuracy_loss_vs_4000_medium"],
            "full_cost_gate": "PENDING", "statistical_noninferiority_established_by_screen": False}


def evaluate_predictions(protocol: dict, manifest: list[dict], grid: dict, scores: dict, selection: dict) -> dict:
    identities = [r["id"] for r in manifest]
    baseline = [grid[identity][0] for identity in identities]
    variants, chosen_rows = {}, {}
    for key in required_variants(selection):
        arm, recipe, family = key
        actions = FAMILIES[family]
        variants[key], chosen_rows[key] = {}, {}
        if set(scores[key]) != set(REPLICATES):
            raise ValueError("Frozen forward output is missing seeds or ensemble")
        expected = np.mean(np.asarray([scores[key][str(seed)] for seed in SEEDS], dtype=np.float32), axis=0)
        if not np.allclose(expected, scores[key]["ensemble"], rtol=0, atol=1e-7):
            raise ValueError("Ensemble differs from mean seed outputs")
        for replicate in REPLICATES:
            values = scores[key][replicate]
            if len(values) != len(identities):
                raise ValueError("Frozen scores do not match the paired test images")
            rows = [grid[identity][select_action(p, actions, recipe.rsplit("_", 1)[1])] for identity, p in zip(identities, values)]
            chosen_rows[key][replicate] = rows
            variants[key][replicate] = method_report(rows, baseline, grid,
                scores=values if family == "joint9" else None, actions=actions if family == "joint9" else None,
                intervals=replicate == "ensemble")
        variants[key]["seed_summary"] = {metric: {
            "mean": statistics.mean(variants[key][str(seed)][metric] for seed in SEEDS),
            "sample_sd": statistics.stdev(variants[key][str(seed)][metric] for seed in SEEDS)}
            for metric in ("accuracy", "utility", "mean_framed_image_bytes")}
    primary = selection["primary_joint"]
    primary_key = primary["arm"], primary["recipe"], primary["family"]
    controls, control_rows = {}, {}
    for name, action in (("strong_fixed", selection["strong_fixed_action"]), ("fixed_4000_medium", 4), ("fixed_2000_low", 0)):
        rows = [grid[identity][action] for identity in identities]
        control_rows[name] = {seed: rows for seed in REPLICATES}
        controls[name] = {seed: method_report(rows, baseline, grid, intervals=seed == "ensemble") for seed in REPLICATES}
    for name in ("strong_rate", "strong_compute"):
        selected = selection[name]
        key = selected["arm"], selected["recipe"], selected["family"]
        controls[name], control_rows[name] = variants[key], chosen_rows[key]
    main_rows = chosen_rows[primary_key]["ensemble"]
    utility = {name: paired_sign_flip([a["utility"] - b["utility"] for a, b in zip(main_rows, control_rows[name]["ensemble"])])
               for name in PRIMARY_CONTROLS}
    holm_three(utility)
    paired = {}
    for name in (*PRIMARY_CONTROLS, "fixed_4000_medium"):
        comparison = control_rows[name]["ensemble"]
        if [r["id"] for r in comparison] != identities:
            raise ValueError("A primary comparison lost image pairing")
        paired[name] = {field: bootstrap_mean([a[field] - b[field] for a, b in zip(main_rows, comparison)]) for field in
                        ("correct", "utility", "image_bytes", "ldpc_complex_symbols", "actual_visual_tokens")}
        paired[name]["gained_answers"] = sum(a["correct"] > b["correct"] for a, b in zip(main_rows, comparison))
        paired[name]["lost_answers"] = sum(a["correct"] < b["correct"] for a, b in zip(main_rows, comparison))
    mixed = [identity for identity in identities if 0 < sum(r["correct"] for r in grid[identity]) < 9]
    return {
        "n_test_images": len(identities), "mixed_image_n": len(mixed),
        "mixed_image_ids_sha256": hashlib.sha256(json.dumps(sorted(mixed)).encode()).hexdigest(),
        "selection": selection, "primary_joint": variants[primary_key], "controls": controls,
        "primary_utility_comparisons": utility, "paired_primary_differences": paired,
        "primary_point_screen": point_screen(variants[primary_key], controls, protocol["screen"]),
        "all_large_joint_point_screens_descriptive": {
            recipe: point_screen(variants[("large", recipe, "joint9")], controls, protocol["screen"]) for recipe in RECIPES},
        "secondary_joint_variants": {arm: {recipe: variants[(arm, recipe, "joint9")] for recipe in RECIPES} for arm in ("large", "nested")},
        "statistical_scope": {
            "primary": "Three frozen joint-minus-control utility contrasts against zero; two-sided sign flips; Holm family 3.",
            "primary_predictor": "Arithmetic mean of three seed outputs before routing; no seed or strategy selected on test.",
            "assumptions": "Distinct paired images; sign flips assume contrast signs are exchangeable/symmetric under the null, not random treatment assignment.",
            "intervals": "Unadjusted 95% image bootstrap intervals, 2000 draws; conditional on fitted models and validation selections.",
            "auc": "Equal-weight within-image AUC on the identical nine-action mixed-image subset for all large/nested joints; excludes all-correct/all-wrong images.",
            "secondary": "Other recipes, nested fits, seeds, per-type results and point screens are descriptive; no additional confirmatory tests.",
            "limits": "Significance against zero does not establish a gain above 0.005. The point accuracy margin does not establish statistical noninferiority. Three seeds are not independent datasets.",
            "random_seed": RANDOM_SEED, "bootstrap_draws": BOOTSTRAPS, "sign_flip_draws": SIGN_FLIPS,
        },
        "full_cost_gate": "PENDING", "unmeasured": protocol["unmeasured"], "limitations": protocol["limitations"],
    }


def cached_report(output: Path, controller_hash: str) -> dict | None:
    report_path, complete_path = output / "test_report.json", output / "test_evaluation_complete.json"
    if not report_path.exists() and not complete_path.exists():
        return None
    if not report_path.is_file() or not complete_path.is_file():
        raise ValueError("Incomplete test publication; automatic re-evaluation is prohibited")
    complete = read(complete_path)
    if (complete.get("state") != "COMPLETE" or complete.get("controller_sha256") != controller_hash
            or complete.get("report_sha256") != sha(report_path)
            or complete.get("started_sha256") != sha(output / "test_evaluation_started.json")):
        raise ValueError("Existing test report or completion receipt is invalid")
    verify_snapshot(complete["input_sha256"])
    report = read(report_path)
    if report["provenance"]["controller_sha256"] != controller_hash or report["provenance"]["input_sha256"] != complete["input_sha256"]:
        raise ValueError("Existing test report provenance differs from its receipt")
    return report


def run(output: Path, protocol_path: Path, source_code: Path, trainer_code: Path) -> dict:
    output, protocol_path, source_code, trainer_code = [p.resolve() for p in (output, protocol_path, source_code, trainer_code)]
    trainer_path = trainer_code / "train_selectors.py"
    controller, selection = verify_controller(output, protocol_path, source_code, trainer_path)
    trainer = load_module(trainer_path, "exp014_verified_trainer")
    protocol = read(protocol_path)
    trainer.validate_protocol(protocol)
    controller_hash = sha(output / "controller_frozen.json")
    previous = cached_report(output, controller_hash)
    if previous is not None:
        return previous
    started = output / "test_evaluation_started.json"
    # O_EXCL is the process lock. It stays after failure to prevent an implicit retry.
    with started.open("x", encoding="utf-8") as handle:
        json.dump({"state": "STARTED", "controller_sha256": controller_hash,
                   "created_utc": datetime.now(timezone.utc).isoformat(), "automatic_retry_allowed": False}, handle)
        handle.write("\n")
    frozen = trainer.load_source_module(source_code)
    manifest, features, grid, inputs = load_test_inputs(output, protocol, protocol_path, controller, frozen)
    scores = forward_frozen(features, output, controller, selection, trainer, frozen)
    report = evaluate_predictions(protocol, manifest, grid, scores, selection)
    verify_snapshot(inputs)
    verify_controller(output, protocol_path, source_code, trainer_path)
    report.update(schema_version=1, experiment_id="EXP-014", split="new_independent_test",
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  provenance={"controller_sha256": controller_hash, "protocol_sha256": sha(protocol_path),
                              "input_sha256": inputs, "evaluation_code_sha256": sha(Path(__file__).resolve()),
                              "trainer_code_sha256": sha(trainer_path), "source_code_sha256": sha(source_code / "train_selector.py")},
                  execution={"optimizer_updates": 0, "vlm_or_codec_calls": 0, "old_test300_opened": False,
                             "new_test_labels_opened_after_controller_verification": True, "test_selection_performed": False,
                             "device": "cpu", "cpu_threads": 2, "evaluation_attempts": 1})
    report_path = output / "test_report.json"
    save_new(report_path, report)
    save_new(output / "test_evaluation_complete.json", {
        "state": "COMPLETE", "controller_sha256": controller_hash, "report_sha256": sha(report_path),
        "started_sha256": sha(started), "input_sha256": inputs, "old_test300_opened": False,
        "new_test_scored": True, "policy_changed": False, "automatic_re_evaluation_allowed": False})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-code", type=Path, required=True)
    parser.add_argument("--trainer-code", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report = run(args.output, args.protocol, args.source_code, args.trainer_code)
    print(json.dumps({"state": "COMPLETE", "n": report["n_test_images"],
                      "correct": report["primary_joint"]["ensemble"]["correct"],
                      "point_screen_pass": report["primary_point_screen"]["point_screen_pass"],
                      "full_cost_gate": "PENDING", "old_test300_opened": False}))


if __name__ == "__main__":
    main()
