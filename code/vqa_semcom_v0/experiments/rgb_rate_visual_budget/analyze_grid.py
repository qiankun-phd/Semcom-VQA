"""Paired, development-only analysis for the frozen RGB rate/compute grid.

No test labels, model inference, policy training, or wireless evaluation occurs
here. Answer-informed selections are explicitly oracle upper bounds, not a
deployable controller. Only Python's standard library is required.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Any

BUDGETS = (2000, 4000, 8000)
TIERS = ("low", "medium", "high")
CELLS = tuple((budget, tier) for budget in BUDGETS for tier in TIERS)
LAMBDAS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4)
PRIMARY_LAMBDA = 0.05
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20260922
Row = dict[str, Any]


def normalize(text: str) -> str:
    """Exact existing evaluate_codec.py:169 normalization, not VQA soft score."""
    numbers = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    value = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", " ".join(str(text).lower().split()))
    return numbers[int(value)] if value.isdigit() and int(value) <= 20 else value


def cell_name(cell: tuple[int, str]) -> str:
    return f"{cell[0]}_{cell[1]}"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_dev_truth(path: Path) -> list[Row]:
    """Fail before opening a supplied sealed-test or ambiguous label path."""
    resolved = path.resolve()
    if "dev" not in path.name.lower() or any(
        re.search(r"(^|[^a-z])test([^a-z]|$)", part.lower())
        for part in (*path.parts, *resolved.parts)
    ):
        raise ValueError("Only explicitly named development truth paths may be read")
    result = read_json(path)
    if not isinstance(result, list):
        raise ValueError("Development truth must be a list")
    return result


def read_records(path: Path) -> list[Row]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        value = json.loads(text)
    else:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(value, list):
        raise ValueError("Records must be a JSON list or JSONL rows")
    return value


def finite_number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{field} is outside its permitted range")
    return float(value)


def positive_integer(value: Any, field: str) -> int:
    finite_number(value, field, positive=True)
    if not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def validate(
    records: list[Row], truth: list[Row], protocol: Row,
    protocol_sha256: str, *, allow_partial: bool = False,
) -> tuple[list[Row], Row]:
    if protocol.get("split") != "dev":
        raise ValueError("Only a development protocol is permitted")
    frozen_contract = {
        "budgets": list(BUDGETS), "lambda_grid": list(LAMBDAS), "primary_lambda": PRIMARY_LAMBDA,
        "primary_reference": {"budget": 4000, "tier": "high"},
        "bootstrap_replicates": BOOTSTRAP_REPEATS,
        "screening_thresholds": {"fixed_max_lost_questions": 1,
            "fixed_min_median_latency_reduction": .2, "fixed_max_image_byte_ratio_vs_reference": 1.0,
            "joint_min_utility_over_best_fixed": .025,
            "joint_min_utility_over_each_one_axis_oracle": .01, "joint_min_correct_gain_over_best_fixed": 3},
    }
    for field, expected_value in frozen_contract.items():
        if field in protocol and protocol[field] != expected_value:
            raise ValueError(f"Protocol {field} differs from preregistered analysis implementation")
    expected = positive_integer(protocol.get("expected_questions", 120), "expected_questions")
    receiver = protocol.get("receiver_sha256")
    if not isinstance(receiver, str) or not re.fullmatch(r"[0-9a-f]{64}", receiver):
        raise ValueError("Protocol needs the frozen receiver_sha256")
    answers: dict[str, str] = {}
    for item in truth:
        identity = item.get("id")
        if not isinstance(identity, str) or not identity or re.search(r"(^|-)test(-|$)", identity):
            raise ValueError("Truth IDs must be nonempty development IDs")
        if identity in answers or not isinstance(item.get("answer"), str):
            raise ValueError("Duplicate truth ID or non-string answer")
        answers[identity] = item["answer"]
    if len(answers) != expected:
        raise ValueError(f"Truth has {len(answers)} questions; protocol expects {expected}")
    if not records:
        raise ValueError("No records to analyze")
    per_id: dict[str, dict[tuple[int, str], Row]] = defaultdict(dict)
    metadata: dict[str, tuple[str, str]] = {}
    byte_identity: dict[tuple[str, int], int] = {}
    token_identity: dict[tuple[str, str], int] = {}
    for source in records:
        row = dict(source)
        identity = row.get("id")
        if identity not in answers:
            raise ValueError(f"Prediction ID is outside development truth: {identity!r}")
        if row.get("split", "dev") != "dev":
            raise ValueError("Prediction split must be dev")
        budget, tier = row.get("budget"), row.get("tier")
        if isinstance(budget, bool) or not isinstance(budget, int) or (budget, tier) not in CELLS:
            raise ValueError(f"Invalid grid cell: {budget!r}, {tier!r}")
        if (budget, tier) in per_id[identity]:
            raise ValueError(f"Duplicate prediction for {identity}, {budget}, {tier}")
        if row.get("protocol_sha256") != protocol_sha256 or row.get("receiver_sha256") != receiver:
            raise ValueError("Mixed or mismatched frozen protocol/receiver hashes")
        if not isinstance(row.get("prediction"), str):
            raise ValueError("Every row needs a string prediction, including empty failures")
        if not isinstance(row.get("question_type"), str) or not row["question_type"]:
            raise ValueError("Missing question_type")
        if isinstance(row.get("image_id"), bool) or not isinstance(row.get("image_id"), (str, int)):
            raise ValueError("Missing image_id")
        image = str(row["image_id"])
        if not image:
            raise ValueError("Empty image_id")
        meta = image, row["question_type"]
        if identity in metadata and metadata[identity] != meta:
            raise ValueError("Question metadata differs across grid cells")
        metadata[identity] = meta
        size = positive_integer(row.get("image_bytes"), "image_bytes")
        if size > budget:
            raise ValueError("Actual bitstream exceeds its frozen byte budget")
        tokens = positive_integer(row.get("actual_visual_tokens"), "actual_visual_tokens")
        finite_number(row.get("receiver_seconds"), "receiver_seconds", positive=True)
        if "energy_j" not in row:
            raise ValueError("energy_j must explicitly be measured or null")
        if row["energy_j"] is not None:
            finite_number(row["energy_j"], "energy_j")
        for mapping, key, value, label in (
            (byte_identity, (image, budget), size, "bitstream bytes"),
            (token_identity, (image, tier), tokens, "actual visual tokens"),
        ):
            if key in mapping and mapping[key] != value:
                raise ValueError(f"Same image/config has inconsistent {label}")
            mapping[key] = value
        row.update(answer=answers[identity], correct=normalize(row["prediction"]) == normalize(answers[identity]),
                   ldpc_complex_symbols=510 * math.ceil(size / 48))
        per_id[identity][budget, tier] = row
    complete = sorted(identity for identity, cells in per_id.items() if set(cells) == set(CELLS))
    incomplete = sorted(set(answers) - set(complete))
    if incomplete and not allow_partial:
        raise ValueError(f"Incomplete paired grid: {len(complete)}/{expected} questions complete")
    if not complete:
        raise ValueError("No complete 9-cell pairs; wait for the smoke subset to finish")
    for identity in complete:
        for budget in BUDGETS:
            levels = [per_id[identity][budget, tier]["actual_visual_tokens"] for tier in TIERS]
            if not levels[0] < levels[1] < levels[2]:
                raise ValueError("Visual tiers are not actually distinct and ordered")
    types = Counter(metadata[identity][1] for identity in complete)
    expected_types = protocol.get("expected_question_type_counts")
    if not allow_partial and expected_types is not None and dict(types) != expected_types:
        raise ValueError("Question-type counts differ from frozen protocol")
    scored = []
    for identity in complete:
        high_tokens = per_id[identity][4000, "high"]["actual_visual_tokens"]
        for cell in CELLS:
            row = per_id[identity][cell]
            row["normalized_resource_cost"] = row["image_bytes"] / 8000 + row["actual_visual_tokens"] / high_tokens
            scored.append(row)
    return scored, {
        "scope": "development", "n_questions": len(complete),
        "n_images": len({metadata[identity][0] for identity in complete}),
        "n_input_records": len(records), "n_paired_records": len(scored),
        "n_excluded_incomplete_questions": len(incomplete), "excluded_incomplete_ids": incomplete,
        "question_type_counts": dict(sorted(types.items())), "expected_questions": expected,
        "decision_eligible": not allow_partial and len(complete) == expected,
        "mode": "smoke_partial_nondecision" if allow_partial else "complete_development",
        "protocol_sha256": protocol_sha256, "receiver_sha256": receiver,
    }


def quantile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] * (hi - position) + values[hi] * (position - lo) if hi != lo else values[lo]


def distribution(values: list[float]) -> Row:
    if not values:
        return {"n": 0, "mean": None, "median": None, "sd": None, "min": None, "max": None,
                "q25": None, "q75": None, "p90": None, "p95": None}
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values),
            **{label: quantile(values, fraction) for label, fraction in
               (("q25", .25), ("q75", .75), ("p90", .90), ("p95", .95))}}


def aggregate(rows: list[Row], weight: float = PRIMARY_LAMBDA) -> Row:
    energy = [row["energy_j"] for row in rows if row["energy_j"] is not None]
    correct = sum(row["correct"] for row in rows)
    return {"n": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "image_bytes": distribution([row["image_bytes"] for row in rows]),
            "ldpc_complex_symbols": distribution([row["ldpc_complex_symbols"] for row in rows]),
            "actual_visual_tokens": distribution([row["actual_visual_tokens"] for row in rows]),
            "receiver_seconds": distribution([row["receiver_seconds"] for row in rows]),
            "energy_j_observed_only": distribution(energy), "energy_missing_count": len(rows) - len(energy),
            "energy_j_all_rows_mean": statistics.mean(energy) if len(energy) == len(rows) else None,
            "normalized_resource_cost": statistics.mean(row["normalized_resource_cost"] for row in rows),
            "utility_lambda": weight,
            "utility_mean": statistics.mean(row["correct"] - weight * row["normalized_resource_cost"] for row in rows),
            "selection_counts": dict(sorted(Counter(cell_name((row["budget"], row["tier"])) for row in rows).items()))}


def oracle(rows: list[Row], weight: float) -> list[Row]:
    per_id: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        per_id[row["id"]].append(row)
    return [max(options, key=lambda row: (
        row["correct"] - weight * row["normalized_resource_cost"],
        -row["normalized_resource_cost"], -row["image_bytes"], -row["actual_visual_tokens"],
        -row["budget"], -TIERS.index(row["tier"]),
    )) for _, options in sorted(per_id.items())]


def paired_comparison(candidate: list[Row], reference: list[Row], weight: float = PRIMARY_LAMBDA,
                      repeats: int = BOOTSTRAP_REPEATS) -> Row:
    ref = {row["id"]: row for row in reference}
    if set(ref) != {row["id"] for row in candidate}:
        raise ValueError("Paired comparison requires identical question IDs")
    by_image: dict[str, list[tuple[float, float]]] = defaultdict(list)
    gained = lost = 0
    for row in candidate:
        old = ref[row["id"]]
        difference = int(row["correct"]) - int(old["correct"])
        gained += difference > 0
        lost += difference < 0
        by_image[str(row["image_id"])].append((difference, difference - weight * (
            row["normalized_resource_cost"] - old["normalized_resource_cost"])))
    clusters = list(by_image.values())
    rng = random.Random(BOOTSTRAP_SEED)
    accuracy_samples, utility_samples = [], []
    for _ in range(repeats):
        sampled = [value for _ in clusters for value in rng.choice(clusters)]
        accuracy_samples.append(statistics.mean(value[0] for value in sampled))
        utility_samples.append(statistics.mean(value[1] for value in sampled))
    return {"n_questions": len(candidate), "n_image_clusters": len(clusters),
            "gained_correct": gained, "lost_correct": lost, "net_correct": gained - lost,
            "accuracy_difference": (gained - lost) / len(candidate),
            "utility_difference": aggregate(candidate, weight)["utility_mean"] - aggregate(reference, weight)["utility_mean"],
            "paired_image_bootstrap": {"repeats": repeats, "seed": BOOTSTRAP_SEED,
                "accuracy_difference_percentile_95": [quantile(accuracy_samples, .025), quantile(accuracy_samples, .975)],
                "utility_difference_percentile_95": [quantile(utility_samples, .025), quantile(utility_samples, .975)],
                "interpretation": "Descriptive development resampling; no independent-test significance claim; selections held fixed."}}


def analyze(scored: list[Row], audit: Row, *, bootstrap_repeats: int = BOOTSTRAP_REPEATS) -> tuple[Row, Row]:
    cells = {cell: [row for row in scored if (row["budget"], row["tier"]) == cell] for cell in CELLS}
    metrics = {cell: aggregate(rows) for cell, rows in cells.items()}
    primary = cells[4000, "high"]
    primary_metric = metrics[4000, "high"]
    best_fixed = max(CELLS, key=lambda cell: (metrics[cell]["utility_mean"],
        -metrics[cell]["normalized_resource_cost"], -cell[0], -TIERS.index(cell[1])))
    pareto = []
    for cell in CELLS:
        def dominates(other: tuple[int, str]) -> bool:
            a, b = metrics[other], metrics[cell]
            comparisons = (a["accuracy"] >= b["accuracy"], a["image_bytes"]["mean"] <= b["image_bytes"]["mean"],
                           a["actual_visual_tokens"]["mean"] <= b["actual_visual_tokens"]["mean"])
            strict = (a["accuracy"] > b["accuracy"] or a["image_bytes"]["mean"] < b["image_bytes"]["mean"] or
                      a["actual_visual_tokens"]["mean"] < b["actual_visual_tokens"]["mean"])
            return all(comparisons) and strict
        if not any(dominates(other) for other in CELLS if other != cell):
            pareto.append(cell_name(cell))
    selected_rate = max(BUDGETS, key=lambda budget: (metrics[budget, "high"]["correct"],
        -metrics[budget, "high"]["image_bytes"]["mean"], -budget))
    independent = min((cell for cell in CELLS if cell[0] == selected_rate and
                       metrics[cell]["correct"] >= metrics[selected_rate, "high"]["correct"] - 1),
                      key=lambda cell: (metrics[cell]["actual_visual_tokens"]["mean"], -metrics[cell]["correct"]))
    lambda_results: dict[str, Row] = {}
    primary_oracles: dict[str, list[Row]] = {}
    for weight in LAMBDAS:
        rate_options = {tier: oracle([row for row in scored if row["tier"] == tier], weight) for tier in TIERS}
        compute_options = {budget: oracle([row for row in scored if row["budget"] == budget], weight) for budget in BUDGETS}
        best_tier = max(TIERS, key=lambda tier: (
            aggregate(rate_options[tier], weight)["utility_mean"],
            -aggregate(rate_options[tier], weight)["normalized_resource_cost"]))
        best_rate = max(BUDGETS, key=lambda budget: (
            aggregate(compute_options[budget], weight)["utility_mean"],
            -aggregate(compute_options[budget], weight)["normalized_resource_cost"]))
        choices = {
            "joint_oracle": oracle(scored, weight),
            "rate_only_oracle_best_fixed_tier": rate_options[best_tier],
            "compute_only_oracle_best_fixed_rate": compute_options[best_rate],
            "rate_only_oracle_high_auxiliary": rate_options["high"],
            "compute_only_oracle_4000_auxiliary": compute_options[4000],
        }
        fixed = {cell: aggregate(rows, weight) for cell, rows in cells.items()}
        best = max(CELLS, key=lambda cell: (fixed[cell]["utility_mean"], -fixed[cell]["normalized_resource_cost"]))
        lambda_results[str(weight)] = {"weight": weight, "best_fixed_cell": cell_name(best), "best_fixed": fixed[best],
                                     "rate_only_selected_fixed_tier": best_tier,
                                     "compute_only_selected_fixed_rate": best_rate,
                                     **{name: aggregate(rows, weight) for name, rows in choices.items()}}
        if weight == PRIMARY_LAMBDA:
            primary_oracles = choices
    joint = primary_oracles["joint_oracle"]
    comparisons = {
        "best_fixed_vs_primary": paired_comparison(cells[best_fixed], primary, repeats=bootstrap_repeats),
        "independent_fixed_vs_primary": paired_comparison(cells[independent], primary, repeats=bootstrap_repeats),
        "joint_oracle_vs_best_fixed": paired_comparison(joint, cells[best_fixed], repeats=bootstrap_repeats),
        "joint_oracle_vs_rate_only_oracle": paired_comparison(joint, primary_oracles["rate_only_oracle_best_fixed_tier"], repeats=bootstrap_repeats),
        "joint_oracle_vs_compute_only_oracle": paired_comparison(joint, primary_oracles["compute_only_oracle_best_fixed_rate"], repeats=bootstrap_repeats),
    }
    candidates = []
    for cell in CELLS:
        saving = 1 - metrics[cell]["receiver_seconds"]["median"] / primary_metric["receiver_seconds"]["median"]
        byte_ratio = metrics[cell]["image_bytes"]["mean"] / primary_metric["image_bytes"]["mean"]
        if (cell[1] != "high" and metrics[cell]["correct"] >= primary_metric["correct"] - 1
                and saving >= .20 and byte_ratio <= 1.0):
            candidates.append({"cell": cell_name(cell), "median_latency_saving_fraction": saving,
                               "mean_image_byte_ratio_vs_reference": byte_ratio,
                               "correct_difference": metrics[cell]["correct"] - primary_metric["correct"]})
    routing_checks = {
        "joint_utility_gain_vs_best_fixed_at_least_0_025": comparisons["joint_oracle_vs_best_fixed"]["utility_difference"] >= .025 - 1e-12,
        "joint_utility_gain_vs_rate_oracle_at_least_0_01": comparisons["joint_oracle_vs_rate_only_oracle"]["utility_difference"] >= .01 - 1e-12,
        "joint_utility_gain_vs_compute_oracle_at_least_0_01": comparisons["joint_oracle_vs_compute_only_oracle"]["utility_difference"] >= .01 - 1e-12,
        "joint_net_correct_gain_vs_best_fixed_at_least_3": comparisons["joint_oracle_vs_best_fixed"]["net_correct"] >= 3,
    }
    eligible = audit["decision_eligible"]
    decision = {
        "development_screen_only": True, "decision_eligible": eligible,
        "efficient_fixed_candidate": bool(candidates) if eligible else None,
        "efficient_fixed_candidates": candidates,
        "potential_routing_headroom": all(routing_checks.values()) if eligible else None,
        "routing_checks": routing_checks, "primary_lambda": PRIMARY_LAMBDA,
        "next_stage_requires_frozen_deployable_policy": True,
        "independent_test_authorized_by_this_analysis": False,
        "recommendation": "smoke_nondecision" if not eligible else (
            "consider_bounded_deployable_controller_training_then_freeze" if all(routing_checks.values()) else
            "freeze_and_validate_efficient_fixed_configuration_before_wireless_test" if candidates else
            "stop_expansion_no_prespecified_efficiency_or_routing_signal"),
        "warning": "Oracle uses development answers; it is not a trained/deployable policy. No test or wireless claim follows from this screen.",
    }
    category = {}
    for cell, rows in cells.items():
        category[cell_name(cell)] = {kind: aggregate([row for row in rows if row["question_type"] == kind])
                                   for kind in sorted({row["question_type"] for row in rows})}
    summary = {"audit": audit, "primary": "4000_high", "primary_lambda": PRIMARY_LAMBDA,
               "metric": "Exact match under existing project normalization; not VQA soft accuracy",
               "resource_definition": "Actual image bitstream bytes only; question/downlink overhead excluded. LDPC symbols = 510 * ceil(bytes/48), accounting only, not simulated delivery.",
               "utility_definition": "correct - lambda * (image_bytes/8000 + actual_visual_tokens/high_tokens_same_image)",
               "one_axis_oracle_definition": "Rate-only maximizes per-question utility over budgets, then selects the best single fixed tier by mean utility; compute-only maximizes over tiers, then selects the best single fixed budget. These strongest one-axis upper bounds, not the auxiliary high/4k slices, define the gate.",
               "energy_warning": "Missing energy stays null. Tokens and latency are not energy measurements.",
               "inference_warning": "One frozen receiver run. Reused development set; no confirmatory p-values. Bootstrap intervals are unadjusted descriptive resampling, not evidence of generalization; model-selection uncertainty is not included.",
               "cells": {cell_name(cell): value for cell, value in metrics.items()},
               "question_types": category, "pareto_axes": ["accuracy_max", "mean_image_bytes_min", "mean_actual_visual_tokens_min"],
               "pareto_fixed_cells": pareto, "best_fixed_primary_utility": cell_name(best_fixed),
               "independent_fixed_selection": {"cell": cell_name(independent), "selected_rate": selected_rate,
                   "rule": "Choose rate by high-tier development correct count, tie lower mean bytes; then smallest actual-token tier losing at most one answer versus high at this rate."},
               "oracle_upper_bounds": lambda_results, "paired_comparisons_primary_lambda": comparisons,
               "oracle_primary_assignments": {name: [{"id": row["id"], "cell": cell_name((row["budget"], row["tier"]))} for row in rows]
                                              for name, rows in primary_oracles.items()}}
    return summary, decision


def markdown_report(summary: Row, decision: Row) -> str:
    audit = summary["audit"]
    lines = ["# RGB rate × receiver visual-budget development screen", "",
             f"Mode: `{audit['mode']}`. {audit['n_questions']} paired questions / {audit['n_images']} images; one frozen receiver.",
             "", "Primary reference: 4,000-byte / high tier. Image bytes exclude question/downlink overhead.",
             "", "| Cell | Correct / N | Accuracy | Mean bytes | Mean LDPC complex symbols | Mean visual tokens | Median receiver seconds | Mean energy J |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell, row in summary["cells"].items():
        energy = row["energy_j_all_rows_mean"]
        lines.append(f"| {cell} | {row['correct']}/{row['n']} | {row['accuracy']:.2%} | {row['image_bytes']['mean']:.2f} | {row['ldpc_complex_symbols']['mean']:.2f} | {row['actual_visual_tokens']['mean']:.2f} | {row['receiver_seconds']['median']:.4f} | {'unavailable/incomplete' if energy is None else f'{energy:.4f}'} |")
    lines.extend(["", f"Best fixed at λ=0.05: `{summary['best_fixed_primary_utility']}`.",
                  f"Sequential independently selected fixed setting: `{summary['independent_fixed_selection']['cell']}`.",
                  f"Pareto fixed cells (accuracy/bytes/tokens): {', '.join(summary['pareto_fixed_cells'])}.",
                  "", "## Answer-informed upper bound, not deployable performance", "",
                  "Utility: correctness − λ × (bytes/8000 + visual tokens / same-image high-tier tokens).",
                  "", "| Primary λ=0.05 selection | Correct / N | Mean bytes | Mean tokens | Mean utility |",
                  "|---|---:|---:|---:|---:|"])
    for name in ("joint_oracle", "rate_only_oracle_best_fixed_tier", "compute_only_oracle_best_fixed_rate"):
        row = summary["oracle_upper_bounds"]["0.05"][name]
        lines.append(f"| {name} | {row['correct']}/{row['n']} | {row['image_bytes']['mean']:.2f} | {row['actual_visual_tokens']['mean']:.2f} | {row['utility_mean']:.5f} |")
    lines.extend(["", summary["one_axis_oracle_definition"], "", "## Decision", "", f"Screen recommendation: `{decision['recommendation']}`.",
                  f"Efficient fixed candidate: {decision['efficient_fixed_candidate']}; potential routing headroom: {decision['potential_routing_headroom']}.",
                  "A frozen deployable configuration/policy is still required before wireless and independent-test evaluation. This script never authorizes opening sealed test labels.",
                  "", "## Evidence limits", "", summary["inference_warning"], "",
                  summary["energy_warning"], "", "LDPC symbol counts are exact payload accounting, not a measured delivery rate. No channel was simulated here.",
                  "", "Paired gained/lost answers, image-cluster bootstrap intervals (2,000 resamples, seed 20260922), full latency distributions, per-type results, missing-energy counts, and all λ sensitivity points are in `summary.json`. Selections are held fixed during bootstrap; this is an exploratory screen only.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="Analyze complete subsets descriptively; disable every decision gate")
    args = parser.parse_args()
    scored, audit = validate(read_records(args.records), read_dev_truth(args.truth), read_json(args.protocol),
                             digest(args.protocol), allow_partial=args.allow_partial)
    summary, decision = analyze(scored, audit)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary.json", summary), ("decision.json", decision), ("scored.json", scored)):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(markdown_report(summary, decision), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "questions": audit["n_questions"], "recommendation": decision["recommendation"]}))


if __name__ == "__main__":
    main()
