"""Descriptive EXP-012 frozen-score diagnosis; no fitting or policy selection.

The only prespecified score intervention is lambda=0 versus frozen lambda=.05.
Nine outcomes from one image are paired, and image-macro quantities use images
as units. Pooled head AUC is descriptive, never nine independent observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics


CELLS = [(budget, tier) for budget in (2000, 4000, 8000) for tier in ("low", "medium", "high")]
PIXELS = {"low": 50176, "medium": 100352, "high": 200704}
VARIANTS = {"joint9": list(range(9)), "question_only9": list(range(9)),
            "rate_at_low": [0, 3, 6], "compute_at_4000": [3, 4, 5]}
REPLICATES = ("7", "17", "27", "ensemble")
SPLITS = {"train": 480, "validation": 120, "legacy_dev": 120}


def read(path: Path) -> object:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    numbers = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    value = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", " ".join(str(text).lower().split()))
    return numbers[int(value)] if value.isdigit() and int(value) <= 20 else value


def nominal_cost(action: int) -> float:
    budget, tier = CELLS[action]
    return (budget + 1) / 8001 + PIXELS[tier] / 200704


def choose_action(probabilities: list[float], actions: list[int], weight: float = .05) -> int:
    if weight not in (0, .05):
        raise ValueError("Only the prespecified lambda=0 and frozen lambda=.05 are permitted")
    if len(probabilities) != len(actions) or not actions or len(set(actions)) != len(actions):
        raise ValueError("Invalid probability/action dimensions")
    if any(a not in range(9) for a in actions):
        raise ValueError("Unknown action")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid correctness probabilities")
    return max(zip(actions, probabilities), key=lambda ap: (
        ap[1] - weight * nominal_cost(ap[0]), -nominal_cost(ap[0]), -ap[0]))[0]


def auc(labels: list[int], scores: list[float]) -> float | None:
    """Mann-Whitney AUC with exact probability ties receiving half credit."""
    if len(labels) != len(scores) or any(y not in (0, 1) for y in labels):
        raise ValueError("Invalid AUC inputs")
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    ordered = sorted(zip(scores, labels))
    negative_below, wins, cursor = 0, 0., 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        group_positive = sum(y for _, y in ordered[cursor:end])
        group_negative = end - cursor - group_positive
        wins += group_positive * (negative_below + .5 * group_negative)
        negative_below += group_negative
        cursor = end
    return wins / (positive * negative)


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    ordered = sorted(values)
    return {"n": len(values), "mean": mean(values), "median": statistics.median(values),
            "p10": ordered[round(.1 * (len(values) - 1))], "p90": ordered[round(.9 * (len(values) - 1))]}


def image_mean_interval(values: list[float], repeats: int = 2000) -> dict:
    """The values supplied are already one scalar per distinct image."""
    if not values:
        return {"n_images": 0, "mean": None, "image_bootstrap_95_percentile_interval": None}
    rng = random.Random(20260922)
    sampled = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values)
                     for _ in range(repeats))
    return {"n_images": len(values), "mean": mean(values),
            "image_bootstrap_95_percentile_interval": [sampled[int(.025 * (repeats - 1))], sampled[int(.975 * (repeats - 1))]],
            "replicates": repeats, "conditional_on_frozen_model": True}


def correlation(x: list[float], y: list[float]) -> float | None:
    mx, my = mean(x), mean(y)
    covariance = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return covariance / denominator if denominator else None


def calibration(labels: list[int], scores: list[float]) -> dict:
    bins = []
    for index in range(10):
        members = [(y, p) for y, p in zip(labels, scores) if min(int(p * 10), 9) == index]
        bins.append({"lower": index / 10, "upper": (index + 1) / 10, "n": len(members),
                     "observed_correctness": mean([y for y, _ in members]),
                     "mean_predicted_correctness": mean([p for _, p in members])})
    return {"brier": mean([(p - y) ** 2 for y, p in zip(labels, scores)]),
            "mean_predicted_correctness": mean(scores), "observed_correctness": mean(labels),
            "ece_10_fixed_equal_width_bins": sum(b["n"] * abs(b["mean_predicted_correctness"] - b["observed_correctness"])
                                                 for b in bins if b["n"]) / len(labels), "bins": bins}


def action_summary(rows: list[dict], actions: list[int], probabilities: list[list[float]], weight: float) -> dict:
    choices = [choose_action(p, actions, weight) for p in probabilities]
    correct = [r["correct"][a] for r, a in zip(rows, choices)]
    scores = [p[actions.index(a)] for p, a in zip(probabilities, choices)]
    eligible = [i for i, row in enumerate(rows) if row["correct"][0] == 0 and max(row["correct"]) == 1]
    recoverable = [i for i in eligible if max(rows[i]["correct"][a] for a in actions) == 1]
    rescued = [i for i in eligible if correct[i] == 1]
    harmed = [i for i, row in enumerate(rows) if row["correct"][0] == 1 and correct[i] == 0]
    return {"lambda": weight, "n": len(rows), "correct": sum(correct), "accuracy": mean(correct),
            "utility_at_original_lambda_005": mean([y - .05 * r["actual_cost"][a] for r, a, y in zip(rows, choices, correct)]),
            "mean_framed_image_bytes": mean([r["image_bytes"][a] for r, a in zip(rows, choices)]),
            "mean_visual_tokens": mean([r["visual_tokens"][a] for r, a in zip(rows, choices)]),
            "action_histogram": {str(a): choices.count(a) for a in range(9)},
            "action_rates": {str(a): choices.count(a) / len(rows) for a in range(9)},
            "fixed_2000_low_correct": sum(r["correct"][0] for r in rows),
            "global_oracle_correct": sum(max(r["correct"]) for r in rows),
            "candidate_oracle_correct": sum(max(r["correct"][a] for a in actions) for r in rows),
            "eligible_low_wrong_any_other_correct": len(eligible), "eligible_with_correct_in_candidate_set": len(recoverable),
            "rescued_vs_2000_low": len(rescued), "harmed_vs_2000_low": len(harmed),
            "rescue_fraction_global_eligible": len(rescued) / len(eligible) if eligible else None,
            "rescue_fraction_candidate_eligible": len(rescued) / len(recoverable) if recoverable else None,
            "eligible_upgraded_from_2000_low": sum(choices[i] != 0 for i in eligible),
            "eligible_upgraded_but_still_wrong": sum(choices[i] != 0 and correct[i] == 0 for i in eligible),
            "selected_calibration": calibration(correct, scores)}


def penalty_decomposition(rows: list[dict], probabilities: list[list[float]], actions: list[int]) -> dict:
    no_penalty = [choose_action(p, actions, 0) for p in probabilities]
    original = [choose_action(p, actions, .05) for p in probabilities]
    strata = {"all_images": list(range(len(rows))),
              "candidate_mixed_outcomes": [i for i, r in enumerate(rows) if 0 < sum(r["correct"][a] for a in actions) < len(actions)],
              "low_wrong_but_global_oracle_correct": [i for i, r in enumerate(rows) if r["correct"][0] == 0 and max(r["correct"]) == 1]}
    result = {}
    for stratum, positions in strata.items():
        categories = Counter()
        for i in positions:
            y0, yp = rows[i]["correct"][no_penalty[i]], rows[i]["correct"][original[i]]
            categories[{(0, 0): "wrong_without_penalty_and_with_penalty", (1, 1): "correct_without_penalty_and_with_penalty",
                        (1, 0): "penalty_changed_correct_to_wrong", (0, 1): "penalty_changed_wrong_to_correct"}[(y0, yp)]] += 1
        result[stratum] = {"n": len(positions), "actions_changed": sum(no_penalty[i] != original[i] for i in positions),
                           "wrong_without_penalty_and_with_penalty": categories["wrong_without_penalty_and_with_penalty"],
                           "correct_without_penalty_and_with_penalty": categories["correct_without_penalty_and_with_penalty"],
                           "penalty_changed_correct_to_wrong": categories["penalty_changed_correct_to_wrong"],
                           "penalty_changed_wrong_to_correct": categories["penalty_changed_wrong_to_correct"],
                           "lambda0_correct": sum(rows[i]["correct"][no_penalty[i]] for i in positions),
                           "lambda005_correct": sum(rows[i]["correct"][original[i]] for i in positions)}
    result["candidate_oracle_errors_without_penalty"] = sum(max(r["correct"][a] for a in actions) - r["correct"][a0]
                                                           for r, a0 in zip(rows, no_penalty))
    result["candidate_oracle_errors_original"] = sum(max(r["correct"][a] for a in actions) - r["correct"][ap]
                                                    for r, ap in zip(rows, original))
    result["interpretation"] = "On candidate-mixed images, lambda0 errors are probability-ranking failures; lambda005 errors = persistent lambda0 errors + penalty-introduced errors. Penalty-repaired errors are shown separately."
    return result


def scoring_metrics(rows: list[dict], actions: list[int], probabilities: list[list[float]], intervals: bool) -> dict:
    labels = [[row["correct"][a] for a in actions] for row in rows]
    macro_auc = [value for y, p in zip(labels, probabilities) if (value := auc(y, p)) is not None]
    image_brier = [mean([(v - y) ** 2 for y, v in zip(label, p)]) for label, p in zip(labels, probabilities)]
    flat_y = [y for row in labels for y in row]
    flat_p = [p for row in probabilities for p in row]
    image_p_means = [mean(p) for p in probabilities]
    image_y_means = [mean(y) for y in labels]
    between_variance = statistics.pvariance(image_p_means)
    within_variance = mean([statistics.pvariance(p) for p in probabilities])
    per_action = {}
    reference = min(actions, key=lambda a: (nominal_cost(a), a))
    reference_column = actions.index(reference)
    for column, action in enumerate(actions):
        ys, ps = [row[column] for row in labels], [row[column] for row in probabilities]
        predicted_gains = [p[column] - p[reference_column] for p in probabilities]
        observed_gains = [y[column] - y[reference_column] for y in labels]
        penalty_gap = .05 * (nominal_cost(action) - nominal_cost(reference))
        per_action[str(action)] = {"brier": mean([(p - y) ** 2 for p, y in zip(ps, ys)]),
                                   "auc_across_images": auc(ys, ps), "mean_predicted_correctness": mean(ps),
                                   "observed_correctness": mean(ys), "lowest_cost_reference_action": reference,
                                   "mean_probability_gain_over_reference": mean(predicted_gains),
                                   "mean_observed_correctness_gain_over_reference": mean(observed_gains),
                                   "nominal_penalty_gap_over_reference": penalty_gap,
                                   "fraction_probability_gain_strictly_exceeds_penalty_gap": mean([g > penalty_gap for g in predicted_gains])}
    pairwise = []
    for left in range(len(actions)):
        for right in range(left + 1, len(actions)):
            gaps = [p[right] - p[left] for p in probabilities]
            true_gaps = [y[right] - y[left] for y in labels]
            discordant = [i for i, gap in enumerate(true_gaps) if gap]
            pairwise.append({"action_a": actions[left], "action_b": actions[right],
                             "mean_predicted_b_minus_a": mean(gaps), "mean_observed_b_minus_a": mean(true_gaps),
                             "mean_abs_predicted_difference": mean([abs(g) for g in gaps]),
                             "mean_abs_observed_difference": mean([abs(g) for g in true_gaps]),
                             "discordant_images": len(discordant),
                             "ranking_accuracy_on_discordant": mean([1. if gaps[i] * true_gaps[i] > 0 else .5 if gaps[i] == 0 else 0. for i in discordant])})
    result = {"n_images": len(rows), "candidate_heads": len(actions),
              "all_correct_candidate_images": sum(all(y) for y in labels),
              "all_wrong_candidate_images": sum(not any(y) for y in labels),
              "mixed_candidate_images": len(macro_auc), "within_image_macro_auc_mixed_only": mean(macro_auc),
              "within_image_auc_distribution": describe(macro_auc),
              "pooled_all_candidate_heads_auc_descriptive": auc(flat_y, flat_p),
              "brier_all_candidate_heads_image_mean": mean(image_brier),
              "probability_range_within_image": describe([max(p) - min(p) for p in probabilities]),
              "image_mean_probability_vs_fraction_correct_pearson": correlation(image_p_means, image_y_means),
              "prediction_variance_between_image_means": between_variance,
              "prediction_variance_within_image_actions": within_variance,
              "between_image_share_of_probability_variance": between_variance / (between_variance + within_variance) if between_variance + within_variance else None,
              "per_action": per_action, "action_pair_differences": pairwise,
              "selected_lambda0": action_summary(rows, actions, probabilities, 0),
              "selected_lambda005": action_summary(rows, actions, probabilities, .05),
              "penalty_decomposition": penalty_decomposition(rows, probabilities, actions)}
    if intervals:
        result["within_image_auc_image_bootstrap"] = image_mean_interval(macro_auc)
        result["brier_image_bootstrap"] = image_mean_interval(image_brier)
    return result


def stability(rows: list[dict], variant: str, actions: list[int]) -> dict:
    result = {}
    for label, weight in (("lambda0", 0), ("lambda005", .05)):
        choices = [[choose_action(row["probabilities"][variant][seed], actions, weight) for seed in REPLICATES[:3]] for row in rows]
        pairs = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            pairs.append({"seeds": [REPLICATES[left], REPLICATES[right]],
                          "action_disagreement": mean([a[left] != a[right] for a in choices]),
                          "correctness_disagreement": mean([r["correct"][a[left]] != r["correct"][a[right]] for r, a in zip(rows, choices)])})
        result[label] = {"n_images": len(rows), "all_three_seeds_same_action": sum(len(set(a)) == 1 for a in choices),
                         "any_seed_action_disagreement_fraction": mean([len(set(a)) > 1 for a in choices]), "seed_pairs": pairs}
    result["mean_sample_sd_probability_per_image_action"] = mean([
        statistics.stdev(row["probabilities"][variant][seed][column] for seed in REPLICATES[:3])
        for row in rows for column in range(len(actions))])
    return result


def prepare_rows(root: Path, scored: dict) -> list[dict]:
    if scored["actions"] != VARIANTS:
        raise ValueError("Frozen action mapping differs from diagnostic mapping")
    if dict(Counter(row["split"] for row in scored["rows"])) != SPLITS:
        raise ValueError("Unexpected split inventory")
    rows = {row["id"]: {**row, "correct": [None] * 9, "image_bytes": [None] * 9,
                        "visual_tokens": [None] * 9} for row in scored["rows"]}
    if len(rows) != 720:
        raise ValueError("Duplicate or missing image identities")
    truths = {}
    image_ids = set()
    for split in SPLITS:
        labels = read(root / f"{split}_truth.json")
        manifest = read(root / f"{split}_manifest.json")
        if len(labels) != SPLITS[split] or len(manifest) != SPLITS[split]:
            raise ValueError("Truth or manifest length mismatch")
        split_ids = {r["id"] for r in scored["rows"] if r["split"] == split}
        if set(r["id"] for r in labels) != split_ids or set(r["id"] for r in manifest) != split_ids:
            raise ValueError("Truth/manifest identities differ from exported features")
        truths.update({r["id"]: r["answer"] for r in labels})
        for row in manifest:
            if row["image_id"] in image_ids:
                raise ValueError("Images repeat across diagnostic units or splits")
            image_ids.add(row["image_id"])
            rows[row["id"]]["image_id"] = row["image_id"]
    sources = [(read(root / "supervision_records.json"), False), (read(root / "legacy_grid_records.json"), True)]
    if [len(records) for records, _ in sources] != [5400, 1080]:
        raise ValueError("Unexpected outcome record counts")
    for records, legacy in sources:
        for record in records:
            row = rows[record["id"]]
            if (row["split"] == "legacy_dev") != legacy or row["image_id"] != record["image_id"]:
                raise ValueError("Record/image/split mismatch")
            if not legacy and row["split"] != record["split"]:
                raise ValueError("Record split mismatch")
            action = CELLS.index((record["budget"], record["tier"]))
            if row["correct"][action] is not None:
                raise ValueError("Duplicate outcome cell")
            row["correct"][action] = int(normalize(record["prediction"]) == normalize(truths[record["id"]]))
            row["image_bytes"][action] = record["image_bytes"] + int(legacy)
            row["visual_tokens"][action] = record["actual_visual_tokens"]
    for row in rows.values():
        if any(y is None for y in row["correct"]):
            raise ValueError("Incomplete nine-action outcome grid")
        high = row["visual_tokens"][5]
        row["actual_cost"] = [b / 8001 + t / high for b, t in zip(row["image_bytes"], row["visual_tokens"])]
    for split in ("validation", "legacy_dev"):
        report = read(root / f"{split}_report.json")
        for variant, actions in VARIANTS.items():
            for replicate in REPLICATES:
                for previous in report["variants"][variant][replicate]["selected"]:
                    row = rows[previous["id"]]
                    choice = choose_action(row["probabilities"][variant][replicate], actions)
                    if choice != previous["action"] or row["correct"][choice] != previous["correct"]:
                        raise ValueError("Existing report action/correctness mismatch")
    return list(rows.values())


def run(root: Path, scored_path: Path, output: Path) -> dict:
    source_files = [scored_path, root / "policy.json", root / "supervision_records.json", root / "legacy_grid_records.json"]
    source_files += [root / f"{split}_{kind}.json" for split in SPLITS for kind in ("truth", "manifest")]
    source_files += [root / f"{split}_report.json" for split in ("validation", "legacy_dev")]
    before = {str(path.resolve()): sha(path) for path in source_files}
    scored = read(scored_path)
    rows = prepare_rows(root, scored)
    result = {"schema": "exp012-frozen-score-diagnostics-v1", "actions": VARIANTS,
              "scope": "Descriptive mechanism diagnosis conditional on frozen models; train fit, reused internal validation, and reused historical development; no independent test claim.",
              "uncertainty": "Bootstrap resamples distinct images. Within-image macro AUC excludes all-correct/all-wrong images and weights every mixed image equally. Pooled AUC is descriptive. Three seeds are model replicates, not independent datasets.",
              "prespecified_lambda_probe": [0, .05], "lambda_probe_used_for_policy_update": False,
              "splits": {}}
    for split in SPLITS:
        selected = [r for r in rows if r["split"] == split]
        result["splits"][split] = {"n_images": len(selected), "variants": {}}
        for variant, actions in VARIANTS.items():
            result["splits"][split]["variants"][variant] = {
                "replicates": {replicate: scoring_metrics(selected, actions,
                    [r["probabilities"][variant][replicate] for r in selected], replicate == "ensemble") for replicate in REPLICATES},
                "seed_stability": stability(selected, variant, actions)}
    after = {str(path.resolve()): sha(path) for path in source_files}
    if before != after:
        raise ValueError("Diagnostic source changed")
    result["provenance"] = {"source_sha256": before, "source_unchanged": True,
                             "script_sha256": sha(Path(__file__).resolve()), "no_training_or_policy_update": True,
                             "sealed_test_opened": False}
    output.mkdir(parents=True, exist_ok=True)
    (output / "score_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    # These rows contain outcomes and scores only; no image, question, or answer text.
    compact = []
    for row in rows:
        compact.append({"id": row["id"], "image_id": row["image_id"], "split": row["split"],
                        "correct": row["correct"], "joint_probabilities": row["probabilities"]["joint9"],
                        "joint_selected_actions": {replicate: {"lambda0": choose_action(row["probabilities"]["joint9"][replicate], list(range(9)), 0),
                                                               "lambda005": choose_action(row["probabilities"]["joint9"][replicate], list(range(9)), .05)} for replicate in REPLICATES},
                        "within_image_auc_joint_ensemble": auc(row["correct"], row["probabilities"]["joint9"]["ensemble"])})
    (output / "score_diagnostic_rows.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n")
    return {split: {"joint_within_auc": result["splits"][split]["variants"]["joint9"]["replicates"]["ensemble"]["within_image_macro_auc_mixed_only"],
                    "joint_lambda0_correct": result["splits"][split]["variants"]["joint9"]["replicates"]["ensemble"]["selected_lambda0"]["correct"],
                    "joint_lambda005_correct": result["splits"][split]["variants"]["joint9"]["replicates"]["ensemble"]["selected_lambda005"]["correct"]} for split in SPLITS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scored-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.root, arguments.scored_inputs, arguments.output)))
