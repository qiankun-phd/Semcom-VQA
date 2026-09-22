"""Frozen EXP-013 four-group evaluation from cached receiver results only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any

import numpy as np

_source_parser = argparse.ArgumentParser(add_help=False)
_source_parser.add_argument("--source-code", type=Path,
                            default=Path(__file__).resolve().parents[1] / "rgb_joint_selector")
SOURCE_CODE = _source_parser.parse_known_args()[0].source_code.resolve()
SCORING_SOURCE = SOURCE_CODE / "train_selector.py"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE_CODE))
_spec = importlib.util.spec_from_file_location("exp012_frozen_scoring", SCORING_SOURCE)
_source = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_source)
CELLS, SEEDS, WEIGHT = _source.CELLS, _source.SEEDS, _source.WEIGHT
gate, nominal_cost, score_grid, summarize = _source.gate, _source.nominal_cost, _source.score_grid, _source.summarize

GROUPS = ("original_absolute", "balanced_absolute", "original_gain", "balanced_gain")
SPLITS = ("validation", "legacy_dev")
LABELS = ("7", "17", "27", "ensemble")
CONTROLS = ("fixed_2000_low", "fixed_4000_medium", "fixed_4000_low", "frozen_rate_at_low", "frozen_compute_at_4000")
CONTRASTS = {"scale": (-.5, .5, -.5, .5), "target_loss": (-.5, -.5, .5, .5), "interaction": (1., -1., -1., 1.)}
BOOTSTRAPS, SIGN_FLIPS, RANDOM_SEED = 2000, 10000, 20260922


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def resolve_pinned_file(name: str, expected: str, roots: tuple[Path, ...]) -> Path:
    """Allow a copied run on another host, but only with matching file content."""
    original = Path(name)
    candidates = [original]
    for root in roots:
        marker = f"/{root.name}/"
        if marker in name:
            candidates.append(root / name.rsplit(marker, 1)[1])
        candidates.append(root / original.name)
        if "checkpoints" in original.parts:
            candidates.append(root / "checkpoints" / original.name)
    for path in dict.fromkeys(candidates):
        if path.is_file() and sha(path) == expected:
            return path
    raise ValueError(f"Frozen input/checkpoint hash mismatch: {original.name}")


def verify_hashes(hashes: dict[str, str], roots: tuple[Path, ...] = ()) -> None:
    if not hashes:
        raise ValueError("Empty frozen hash inventory")
    for name, expected in hashes.items():
        path = Path(name)
        if re.search(r"(?:legacy.*truth|dev_truth|test.*(?:truth|manifest|records)|test300)", path.name.lower()):
            raise ValueError("Premature legacy or forbidden test label access through a hash inventory")
        resolve_pinned_file(name, expected, roots)


def validate_protocol(protocol: dict) -> None:
    if protocol["experiment_id"] != "EXP-013" or tuple(protocol["groups"]) != GROUPS:
        raise ValueError("Unexpected four-group factorial protocol")
    if protocol["training"]["seeds"] != SEEDS or protocol["selection"]["lambda"] != WEIGHT:
        raise ValueError("Seed or lambda contract changed")
    if protocol["actions"] != [list(cell) for cell in CELLS]:
        raise ValueError("Action contract changed")
    if protocol["data"]["test_access"] or protocol["data"]["new_vlm_or_codec_calls"] != 0:
        raise ValueError("Only cached development evaluation is permitted")
    for group in GROUPS:
        if protocol["groups"][group] != {"balanced": group.startswith("balanced"), "target": group.rsplit("_", 1)[1]}:
            raise ValueError("Factor definitions changed")


def require_pinned_input(path: Path, hashes: dict[str, str]) -> None:
    if not any(Path(name).name == path.name and digest == sha(path) for name, digest in hashes.items()):
        raise ValueError(f"Supplied source does not match a frozen input: {path.name}")


def verify_frozen(output: Path, protocol_path: Path, source_dir: Path | None = None) -> tuple[dict, dict, dict]:
    """This entire check finishes before any historical truth or report is read."""
    frozen_path = output / "controller_frozen.json"
    if not frozen_path.is_file():
        raise ValueError("Freeze all controllers before historical labels are opened")
    frozen = read(frozen_path)
    if frozen.get("state") != "FROZEN_BEFORE_LEGACY_LABELS" or frozen.get("completed_fits") != 12:
        raise ValueError("Controller freeze is incomplete")
    if set(frozen.get("groups", [])) != set(GROUPS) or frozen.get("seeds") != SEEDS:
        raise ValueError("Controller freeze does not contain all groups and seeds")
    if frozen.get("legacy_labels_opened") is not False or frozen.get("sealed_test_opened") is not False:
        raise ValueError("Invalid label-access chronology in controller freeze")
    roots = (output, protocol_path.parent, SOURCE_CODE)
    if source_dir is not None:
        roots += (source_dir,)
    verify_hashes(frozen["sha256"], roots)
    for group in GROUPS:
        for seed in map(str, SEEDS):
            checkpoint = frozen["checkpoints"][group][seed]
            resolve_pinned_file(checkpoint["path"], checkpoint["sha256"], roots)
    inputs_path = output / "frozen_inputs.json"
    if not any(Path(name).name == inputs_path.name and value == sha(inputs_path) for name, value in frozen["sha256"].items()):
        raise ValueError("Training input manifest is not pinned by controller freeze")
    verify_hashes(read(inputs_path)["sha256"], roots)
    if not any(Path(name).name == protocol_path.name and value == sha(protocol_path) for name, value in frozen["sha256"].items()):
        raise ValueError("Evaluation protocol is not the frozen training protocol")
    complete = read(output / "training_complete.json")
    if complete.get("controller_sha256") != sha(frozen_path):
        raise ValueError("Training completion does not pin this controller freeze")
    for filename, field in (("predictions.json", "predictions_sha256"), ("reproduction_check.json", "reproduction_check_sha256")):
        if complete.get(field) != sha(output / filename):
            raise ValueError(f"Training completion artifact mismatch: {filename}")
    reproduction = read(output / "reproduction_check.json")
    if (reproduction.get("passed") is not True or reproduction.get("max_abs_difference", math.inf) > 1e-6
            or reproduction.get("action_mismatches") != 0):
        raise ValueError("Original absolute refit failed the required reproduction check")
    predictions = read(output / "predictions.json")
    if predictions.get("controller_sha256") != sha(frozen_path) or predictions.get("protocol_sha256") != sha(protocol_path):
        raise ValueError("Predictions are not linked to this frozen controller and protocol")
    return frozen, predictions, reproduction


def select_action(scores: list[float], target: str) -> int:
    if len(scores) != 9 or target not in ("absolute", "gain"):
        raise ValueError("Invalid score/action contract")
    lower = -1 if target == "gain" else 0
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not lower <= v <= 1 for v in scores):
        raise ValueError("Scores violate the target output range")
    if target == "gain" and scores[0] != 0:
        raise ValueError("Gain reference must be exactly zero")
    reference_cost = nominal_cost(0) if target == "gain" else 0
    return max(range(9), key=lambda action: (
        scores[action] - WEIGHT * (nominal_cost(action) - reference_cost), -nominal_cost(action), -action))


def validate_predictions(predictions: dict, features: list[dict]) -> dict[str, dict]:
    if predictions.get("schema_version") != 1:
        raise ValueError("Unsupported predictions schema")
    rows = predictions["rows"]
    expected = {row["id"]: row["split"] for row in features}
    mapped = {row["id"]: row for row in rows}
    if len(mapped) != len(rows) or len(expected) != len(features) or set(mapped) != set(expected):
        raise ValueError("Predicted feature identities are missing or duplicated")
    for identity, row in mapped.items():
        if row["split"] != expected[identity] or set(row["scores"]) != set(GROUPS):
            raise ValueError("Prediction split or group inventory mismatch")
        for group, scores in row["scores"].items():
            if set(scores) != set(LABELS):
                raise ValueError("Report every seed and ensemble")
            for label in LABELS:
                select_action(scores[label], group.rsplit("_", 1)[1])
            calculated = np.mean(np.asarray([scores[str(seed)] for seed in SEEDS], dtype=np.float32), axis=0)
            if not np.allclose(calculated, scores["ensemble"], rtol=0, atol=1e-7):
                raise ValueError("Ensemble must average all three seed outputs before routing")
    return mapped


def within_image_auc(scores: list[float], outcomes: list[int]) -> float | None:
    correct = [score for score, outcome in zip(scores, outcomes) if outcome]
    wrong = [score for score, outcome in zip(scores, outcomes) if not outcome]
    if not correct or not wrong:
        return None
    return sum(float(a > b) + .5 * float(a == b) for a in correct for b in wrong) / (len(correct) * len(wrong))


def bootstrap_summary(values: list[float], repeats: int = BOOTSTRAPS) -> dict:
    if not values:
        return {"n": 0, "mean": None, "ci95": None, "sample_sd": None}
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite image statistics")
    indices = np.random.default_rng(RANDOM_SEED).integers(0, len(array), (repeats, len(array)))
    boot = np.mean(array[indices], axis=1)
    return {"n": len(values), "mean": float(np.mean(array)), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "sample_sd": float(np.std(array, ddof=1)) if len(array) > 1 else None}


def paired_contrast(values: dict[str, dict[str, float]], weights: tuple[float, ...], *, draws: int = SIGN_FLIPS) -> dict:
    identities = list(values[GROUPS[0]])
    if not identities or any(set(values[group]) != set(identities) for group in GROUPS):
        raise ValueError("Factorial comparisons require the exact same paired image set")
    differences = [sum(weight * values[group][identity] for weight, group in zip(weights, GROUPS)) for identity in identities]
    result = bootstrap_summary(differences)
    array = np.asarray(differences)
    observed = abs(float(array.mean()))
    signs = np.random.default_rng(RANDOM_SEED).choice([-1., 1.], size=(draws, len(array)))
    randomized = np.abs(np.mean(signs * array, axis=1))
    extreme = int(np.count_nonzero(randomized >= observed - 1e-14))
    result.update(mean_difference=result["mean"], sign_flip_p=(extreme + 1) / (draws + 1),
                  sign_flip_draws=draws, nonzero_images=int(np.count_nonzero(array)),
                  positive_images=int(np.count_nonzero(array > 0)), negative_images=int(np.count_nonzero(array < 0)))
    return result


def holm_adjust(results: list[dict]) -> None:
    previous = 0.
    for rank, result in enumerate(sorted(results, key=lambda row: row["sign_flip_p"])):
        previous = max(previous, min(1., (len(results) - rank) * result["sign_flip_p"]))
        result["holm_p"] = previous
        result["holm_family_size"] = len(results)


def selected_summary(selected: list[dict], baseline: list[dict], auc: dict[str, float] | None = None) -> dict:
    if [r["id"] for r in selected] != [r["id"] for r in baseline]:
        raise ValueError("Selected and baseline images are not paired")
    base = summarize(selected)
    vectors = {metric: [float(row[metric]) for row in selected] for metric in
               ("correct", "utility", "image_bytes", "ldpc_complex_symbols", "actual_visual_tokens")}
    rescue = [float(not b["correct"] and a["correct"]) for a, b in zip(selected, baseline)]
    harm = [float(b["correct"] and not a["correct"]) for a, b in zip(selected, baseline)]
    vectors.update(rescue=rescue, harm=harm)
    return {"summary": base, "rescue": int(sum(rescue)), "harm": int(sum(harm)),
            "uncertainty": {metric: bootstrap_summary(values) for metric, values in vectors.items()},
            "within_image_auc": bootstrap_summary(list(auc.values())) if auc is not None else None}


def report_control_rows(report: dict, grid: dict[str, list[dict]], variant: str, seed: str, identities: list[str]) -> list[dict]:
    stored = {row["id"]: row for row in report["variants"][variant][seed]["selected"]}
    if set(stored) != set(identities):
        raise ValueError("Frozen control uses different images")
    selected = []
    for identity in identities:
        row = stored[identity]
        cached = grid[identity][row["action"]]
        for field in ("correct", "image_bytes", "actual_visual_tokens", "utility"):
            if not math.isclose(row[field], cached[field], abs_tol=1e-12):
                raise ValueError("Frozen control differs from the receiver cache")
        selected.append(cached)
    return selected


def evaluate_split(grid: dict[str, list[dict]], predictions: dict[str, dict], report: dict, screen: dict) -> tuple[dict, dict]:
    identities = list(grid)
    mixed = [identity for identity in identities if 0 < sum(r["correct"] for r in grid[identity]) < 9]
    base_rows = [grid[identity][0] for identity in identities]
    controls, control_rows = {}, {}
    for name in CONTROLS:
        controls[name], control_rows[name] = {}, {}
        for seed in LABELS:
            if name.startswith("fixed_"):
                budget, tier = name.removeprefix("fixed_").split("_", 1)
                action = CELLS.index((int(budget), tier))
                rows = [grid[identity][action] for identity in identities]
            else:
                variant = name.removeprefix("frozen_")
                rows = report_control_rows(report, grid, variant, seed, identities)
            control_rows[name][seed] = rows
            controls[name][seed] = selected_summary(rows, base_rows)
    if report["selection"] != {"strong_fixed_action": 3, "rate_variant": "rate_at_low", "compute_variant": "compute_at_4000"}:
        raise ValueError("Original frozen strong controls differ from EXP-013 protocol")
    groups, unit_values, group_rows = {}, {}, {}
    for group in GROUPS:
        groups[group], unit_values[group], group_rows[group] = {}, {}, {}
        for seed in LABELS:
            scores = {identity: predictions[identity]["scores"][group][seed] for identity in identities}
            rows = [grid[identity][select_action(scores[identity], group.rsplit("_", 1)[1])] for identity in identities]
            auc = {identity: within_image_auc(scores[identity], [row["correct"] for row in grid[identity]]) for identity in mixed}
            if any(value is None for value in auc.values()):
                raise ValueError("Mixed AUC set changed across groups")
            groups[group][seed] = selected_summary(rows, base_rows, auc)
            unit_values[group][seed] = {"auc": auc, "utility": {row["id"]: row["utility"] for row in rows}}
            group_rows[group][seed] = rows
        groups[group]["seed_mean_sd"] = {
            metric: {"mean": statistics.mean(groups[group][str(seed)]["summary"][metric] for seed in SEEDS),
                     "sample_sd": statistics.stdev(groups[group][str(seed)]["summary"][metric] for seed in SEEDS)}
            for metric in ("accuracy", "utility")}
        groups[group]["seed_mean_sd"]["auc"] = {
            "mean": statistics.mean(groups[group][str(seed)]["within_image_auc"]["mean"] for seed in SEEDS),
            "sample_sd": statistics.stdev(groups[group][str(seed)]["within_image_auc"]["mean"] for seed in SEEDS)}
        methods = {"joint": groups[group]["ensemble"], "primary_fixed": controls["fixed_4000_medium"]["ensemble"],
                   "strong_fixed": controls["fixed_4000_low"]["ensemble"], "rate_only": controls["frozen_rate_at_low"]["ensemble"],
                   "compute_only": controls["frozen_compute_at_4000"]["ensemble"]}
        seed_methods = {seed: {"joint": groups[group][seed], "strong_fixed": controls["fixed_4000_low"][seed],
                              "rate_only": controls["frozen_rate_at_low"][seed], "compute_only": controls["frozen_compute_at_4000"][seed]}
                        for seed in map(str, SEEDS)}
        groups[group]["system_screen"] = gate(methods, seed_methods, screen)
        groups[group]["paired_controls"] = {}
        for control in CONTROLS:
            a, b = group_rows[group]["ensemble"], control_rows[control]["ensemble"]
            groups[group]["paired_controls"][control] = {
                metric: bootstrap_summary([float(x[metric] - y[metric]) for x, y in zip(a, b)])
                for metric in ("correct", "utility", "image_bytes", "ldpc_complex_symbols", "actual_visual_tokens")}
    statistics_output = {}
    for metric in ("auc", "utility"):
        statistics_output[metric] = {}
        values = {group: unit_values[group]["ensemble"][metric] for group in GROUPS}
        for name, weights in CONTRASTS.items():
            result = paired_contrast(values, weights)
            result["seed_effects"] = {seed: statistics.mean(
                sum(weight * unit_values[group][seed][metric][identity] for weight, group in zip(weights, GROUPS))
                for identity in values[GROUPS[0]]) for seed in map(str, SEEDS)}
            result["seed_effect_mean"] = statistics.mean(result["seed_effects"].values())
            result["seed_effect_sample_sd"] = statistics.stdev(result["seed_effects"].values())
            statistics_output[metric][name] = result
    return {"n": len(identities), "mixed_outcome_n": len(mixed),
            "mixed_set_identical_for_all_groups_and_seeds": True,
            "mixed_set_sha256": hashlib.sha256(json.dumps(sorted(mixed)).encode()).hexdigest(),
            "groups": groups, "controls": controls, "factorial_contrasts": statistics_output}, unit_values


def run(source_dir: Path, output: Path, protocol_path: Path, source_protocol_path: Path,
        legacy_grid: Path | None = None, legacy_truth: Path | None = None) -> dict:
    protocol = read(protocol_path)
    validate_protocol(protocol)
    frozen, predictions, reproduction = verify_frozen(output, protocol_path, source_dir)
    input_hashes = read(output / "frozen_inputs.json")["sha256"]
    for path in (source_protocol_path, SCORING_SOURCE, source_dir / "features.json",
                 source_dir / "supervision_records.json", source_dir / "validation_truth.json",
                 source_dir / "validation_report.json"):
        require_pinned_input(path, input_hashes)
    source_protocol = read(source_protocol_path)
    features = read(source_dir / "features.json")
    predicted = validate_predictions(predictions, features)
    expected_sizes = {"train": 480, "validation": 120, "legacy_dev": 120}
    if {split: sum(row["split"] == split for row in features) for split in expected_sizes} != expected_sizes:
        raise ValueError("Frozen source feature split counts changed")
    # Historical truth is first opened below, after every freeze/input/prediction check.
    summaries, provenance = {}, {str(protocol_path): sha(protocol_path), str(source_protocol_path): sha(source_protocol_path),
                                 str(output / "controller_frozen.json"): sha(output / "controller_frozen.json"),
                                 str(output / "predictions.json"): sha(output / "predictions.json")}
    for split in SPLITS:
        legacy = split == "legacy_dev"
        records_path = source_dir / ("legacy_grid_records.json" if legacy else "supervision_records.json")
        complete_path = source_dir / ("legacy_grid_complete.json" if legacy else "supervision_complete.json")
        if legacy and legacy_grid is not None:
            records_path, complete_path = legacy_grid / "records.json", legacy_grid / "complete.json"
        complete = read(complete_path)
        if complete["records_sha256"] != sha(records_path):
            raise ValueError("Cached receiver records differ from completion fingerprint")
        truth_path = legacy_truth if legacy and legacy_truth is not None else source_dir / f"{split}_truth.json"
        report_path = source_dir / f"{split}_report.json"
        report = read(report_path)
        if legacy and report["provenance"]["legacy_truth_sha256"] != sha(truth_path):
            raise ValueError("Historical development truth changed")
        if legacy and report["provenance"]["legacy_records_sha256"] != sha(records_path):
            raise ValueError("Historical receiver grid differs from original report provenance")
        truth = read(truth_path)
        identities = [row["id"] for row in truth]
        if len(truth) != 120 or set(identities) != {identity for identity, row in predicted.items() if row["split"] == split}:
            raise ValueError("Development labels and predictions do not describe identical images")
        grid = score_grid(read(records_path), truth, identities, source_protocol["receiver_sha256"], complete["protocol_sha256"], legacy=legacy)
        if len({cells[0]["image_id"] for cells in grid.values()}) != len(grid):
            raise ValueError("Repeated source images violate the paired image analysis unit")
        summaries[split], _ = evaluate_split(grid, predicted, report, source_protocol["screen"])
        for path in (records_path, complete_path, truth_path, report_path):
            provenance[str(path)] = sha(path)
    tests = [summaries[split]["factorial_contrasts"][metric][contrast]
             for split in SPLITS for metric in ("auc", "utility") for contrast in CONTRASTS]
    if len(tests) != 12:
        raise ValueError("Multiplicity family changed")
    holm_adjust(tests)
    result = {
        "schema_version": 1, "experiment_id": "EXP-013", "groups": list(GROUPS), "seeds": SEEDS,
        "splits": summaries, "reproduction": reproduction, "screen": source_protocol["screen"],
        "statistics": {"bootstrap_draws": BOOTSTRAPS, "sign_flip_draws": SIGN_FLIPS, "seed": RANDOM_SEED,
                       "contrasts": {name: dict(zip(GROUPS, weights)) for name, weights in CONTRASTS.items()},
                       "holm_family_size": 12, "primary_predictor": "ensemble before selection",
                       "auc": "Mean of per-image probability(correct score > wrong score), ties 0.5, nine actions; identical mixed-label subset for every group/seed.",
                       "p_value": "Two-sided paired image sign flips: (1 + count(abs(flipped mean) >= abs(observed mean)))/(10000+1).",
                       "assumptions": "Images are distinct paired units. Sign flips assume contrast signs are exchangeable/symmetric under the null; this is not random treatment assignment. No parametric normality assumption or test was used.",
                       "conditional_scope": "Intervals condition on fitted checkpoints; three training seeds are not independent datasets. Validation selection and reused-development uncertainty are not captured."},
        "limitations": protocol["limitations"], "full_cost_gate": "PENDING", "test_or_snr_started": False,
        "provenance": {"sha256": provenance, "evaluation_source_sha256": sha(Path(__file__)),
                       "original_scoring_source_sha256": sha(SCORING_SOURCE)},
        "execution": {"training_run": False, "vlm_or_codec_calls": 0, "sealed_test_opened": False,
                      "controller_and_inputs_verified_before_legacy_truth": True},
    }
    save(output / "evaluation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--source-code", type=Path, default=SOURCE_CODE)
    parser.add_argument("--source-protocol", type=Path, default=SOURCE_CODE / "protocol.json")
    parser.add_argument("--legacy-grid", type=Path, help="Original historical directory containing records.json and complete.json")
    parser.add_argument("--legacy-truth", type=Path, help="Original historical dev truth JSON; never a test truth file")
    args = parser.parse_args()
    if args.legacy_truth and re.search(r"test", args.legacy_truth.name.lower()):
        parser.error("Historical development truth only; test truth is forbidden")
    result = run(args.source_dir.resolve(), args.output.resolve(), args.protocol.resolve(), args.source_protocol.resolve(),
                 args.legacy_grid.resolve() if args.legacy_grid else None,
                 args.legacy_truth.resolve() if args.legacy_truth else None)
    print(json.dumps({split: {group: {"correct": result["splits"][split]["groups"][group]["ensemble"]["summary"]["correct"],
                                    "pass": result["splits"][split]["groups"][group]["system_screen"]["preliminary_pass"]}
                             for group in GROUPS} for split in SPLITS}))


if __name__ == "__main__":
    main()
