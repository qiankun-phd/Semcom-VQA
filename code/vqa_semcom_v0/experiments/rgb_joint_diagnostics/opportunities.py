"""Answer-assisted opportunity analysis of the frozen nine-cell caches only.

No model is loaded, no policy is fitted or selected for deployment, and no
sealed test input is accepted.  Per-sample oracles and retrospective choices of
a fixed other axis are descriptive upper bounds, never deployed policies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rgb_joint_selector.train_selector import (  # noqa: E402
    CELLS, VARIANTS, WEIGHT, normalize, score_grid,
)

TIERS = ("low", "medium", "high")
EXPECTED_SIZES = {"train": 480, "validation": 120, "legacy_dev": 120}
RATE_UP = (3, 6)
COMPUTE_UP = (1, 2)
BOTH_UP = (4, 5, 7, 8)


def label(action: int) -> str:
    budget, tier = CELLS[action]
    return f"{budget}_{tier}"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    if private:
        temporary.chmod(0o600)
    temporary.replace(path)


def cost(row: dict, cells: list[dict]) -> float:
    high_tokens = cells[CELLS.index((4000, "high"))]["actual_visual_tokens"]
    return row["image_bytes"] / 8001 + row["actual_visual_tokens"] / high_tokens


def oracle_action(cells: list[dict], actions: list[int] | tuple[int, ...]) -> int:
    return max(actions, key=lambda action: (
        cells[action]["utility"], -cost(cells[action], cells), -action))


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "correct": sum(row["correct"] for row in rows),
        "accuracy": statistics.mean(row["correct"] for row in rows) if n else None,
        "utility": statistics.mean(row["utility"] for row in rows) if n else None,
        "mean_image_bytes": statistics.mean(row["image_bytes"] for row in rows) if n else None,
        "mean_actual_visual_tokens": statistics.mean(row["actual_visual_tokens"] for row in rows) if n else None,
        "routing_histogram": dict(sorted(Counter(label(row["action"]) for row in rows).items())),
    }


def by_type(rows: list[dict]) -> dict:
    return {question_type: summarize([r for r in rows if r["question_type"] == question_type])
            for question_type in sorted({r["question_type"] for r in rows})}


def opportunity_flags(cells: list[dict]) -> dict[str, bool]:
    wrong = not cells[0]["correct"]
    rate = wrong and any(cells[a]["correct"] for a in RATE_UP)
    compute = wrong and any(cells[a]["correct"] for a in COMPUTE_UP)
    both = wrong and any(cells[a]["correct"] for a in BOTH_UP)
    return {
        "baseline_correct": not wrong,
        "baseline_wrong": wrong,
        "rate_only_rescuable": rate,
        "compute_only_rescuable": compute,
        "both_axes_increased_rescuable": both,
        "rate_or_compute_union": rate or compute,
        "rate_and_compute_intersection": rate and compute,
        "rate_without_compute": rate and not compute,
        "compute_without_rate": compute and not rate,
        "rate_and_both_intersection": rate and both,
        "compute_and_both_intersection": compute and both,
        "all_three_intersection": rate and compute and both,
        "both_axes_required": both and not (rate or compute),
        "any_grid_rescuable": rate or compute or both,
        "unrescuable_any_grid": wrong and not (rate or compute or both),
    }


def comparable_pairs() -> list[tuple[int, int, str]]:
    pairs = []
    for a, (budget_a, tier_a) in enumerate(CELLS):
        for b, (budget_b, tier_b) in enumerate(CELLS):
            rate_up = budget_b > budget_a
            compute_up = TIERS.index(tier_b) > TIERS.index(tier_a)
            if budget_b >= budget_a and TIERS.index(tier_b) >= TIERS.index(tier_a) and a != b:
                axis = "both_axes" if rate_up and compute_up else "rate_only" if rate_up else "compute_only"
                pairs.append((a, b, axis))
    return pairs


def nonmonotonic_summary(grid: dict[str, list[dict]]) -> tuple[dict, dict[str, list[dict]]]:
    transitions, sample_losses = {}, {identity: [] for identity in grid}
    for source, target, axis in comparable_pairs():
        gains = losses = unchanged_correct = unchanged_wrong = 0
        for identity, cells in grid.items():
            before, after = cells[source]["correct"], cells[target]["correct"]
            gains += after > before
            losses += after < before
            unchanged_correct += bool(before and after)
            unchanged_wrong += not (before or after)
            if before and not after:
                sample_losses[identity].append({"from": label(source), "to": label(target), "axis": axis})
        transitions[f"{label(source)} -> {label(target)}"] = {
            "axis": axis, "gained_answers": gains, "lost_answers": losses,
            "unchanged_correct": unchanged_correct, "unchanged_wrong": unchanged_wrong,
        }
    summary = {"comparable_pair_count": len(transitions), "transitions": transitions,
               "images_with_any_correct_to_wrong_increase": sum(bool(v) for v in sample_losses.values())}
    for axis in ("rate_only", "compute_only", "both_axes"):
        summary[f"images_with_{axis}_loss"] = sum(any(p["axis"] == axis for p in v) for v in sample_losses.values())
    for name, actions in (("rate_only", RATE_UP), ("compute_only", COMPUTE_UP), ("both_axes", BOTH_UP), ("any", range(1, 9))):
        summary[f"baseline_correct_images_lost_by_{name}_increase"] = sum(
            bool(cells[0]["correct"]) and any(not cells[a]["correct"] for a in actions)
            for cells in grid.values())
    return summary, sample_losses


def resource_delta(oracle: dict, deployed: dict) -> dict:
    return {
        "correct_gain": oracle["correct"] - deployed["correct"],
        "accuracy_gain": oracle["accuracy"] - deployed["accuracy"],
        "mean_utility_regret": oracle["utility"] - deployed["utility"],
        "oracle_minus_deployed_mean_image_bytes": oracle["mean_image_bytes"] - deployed["mean_image_bytes"],
        "oracle_minus_deployed_mean_actual_visual_tokens": oracle["mean_actual_visual_tokens"] - deployed["mean_actual_visual_tokens"],
    }


def summarize_oracles(grid: dict[str, list[dict]], selection: dict | None) -> tuple[dict, dict[str, dict[str, int]]]:
    families = {"nine_way": list(range(9)), **{name: actions for name, actions in VARIANTS.items()
                if name.startswith(("rate_at_", "compute_at_"))}}
    choices = {name: {identity: oracle_action(cells, actions) for identity, cells in grid.items()}
               for name, actions in families.items()}
    summaries = {name: {**summarize([grid[identity][action] for identity, action in selected.items()]),
                        "by_type": by_type([grid[identity][action] for identity, action in selected.items()])}
                 for name, selected in choices.items()}
    strongest = {}
    for axis, prefix in (("rate_only", "rate_at_"), ("compute_only", "compute_at_")):
        candidates = [name for name in families if name.startswith(prefix)]
        best = max(candidates, key=lambda name: (summaries[name]["utility"], -candidates.index(name)))
        strongest[axis] = {"selected_retrospectively_by_mean_utility": best, "summary": summaries[best],
                           "nine_way_minus_axis_oracle": resource_delta(summaries["nine_way"], summaries[best])}
    frozen = None
    if selection is not None:
        frozen = {axis: {"variant": selection[key], "summary": summaries[selection[key]],
                        "nine_way_minus_axis_oracle": resource_delta(summaries["nine_way"], summaries[selection[key]])}
                  for axis, key in (("rate_only", "rate_variant"), ("compute_only", "compute_variant"))}
    return {"scope": "Answer-assisted retrospective upper bounds; not deployable routing or new frozen selection.",
            "all_families": summaries, "strongest_fixed_other_axis": strongest,
            "frozen_deployed_axis_family_oracles": frozen}, choices


def check_deployed(report: dict, grid: dict[str, list[dict]]) -> dict[str, dict[str, dict]]:
    methods = {}
    for name, method in report["methods"].items():
        selected = method["selected"]
        if len(selected) != len(grid) or {r["id"] for r in selected} != set(grid):
            raise ValueError(f"Deployed report identity mismatch: {name}")
        mapped = {}
        for row in selected:
            action = row["action"]
            if (row["budget"], row["tier"]) != CELLS[action]:
                raise ValueError("Deployed action descriptor mismatch")
            cached = grid[row["id"]][action]
            for field in ("correct", "image_bytes", "actual_visual_tokens", "utility"):
                if not math.isclose(row[field], cached[field], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"Deployed cached score mismatch: {name}/{field}")
            if normalize(row["prediction"]) != normalize(cached["prediction"]):
                raise ValueError("Deployed prediction differs from frozen cache")
            mapped[row["id"]] = cached
        recalculated = summarize(list(mapped.values()))
        for field in ("n", "correct", "accuracy", "utility"):
            if not math.isclose(recalculated[field], method["summary"][field], abs_tol=1e-12):
                raise ValueError(f"Deployed summary mismatch: {name}/{field}")
        methods[name] = mapped
    return methods


def method_misses(name: str, selected: dict[str, dict], grid: dict[str, list[dict]],
                  oracle_choices: dict[str, int], selection: dict) -> tuple[dict, dict[str, dict]]:
    allowed = (VARIANTS[selection["rate_variant"]] if name == "rate_only" else
               VARIANTS[selection["compute_variant"]] if name == "compute_only" else
               [CELLS.index((4000, "medium"))] if name == "primary_fixed" else
               [selection["strong_fixed_action"]] if name == "strong_fixed" else list(range(9)))
    details = {}
    for identity, chosen in selected.items():
        if chosen["action"] not in allowed:
            raise ValueError(f"Deployed action outside its frozen action family: {name}")
        cells = grid[identity]
        oracle = cells[oracle_choices[identity]]
        correct_actions = [a for a in range(9) if cells[a]["correct"]]
        missed = not chosen["correct"] and bool(correct_actions)
        flags = opportunity_flags(cells)
        details[identity] = {
            "selected_action": label(chosen["action"]), "correct": chosen["correct"],
            "missed_rescuable_answer": missed,
            "missed_within_deployed_action_family": missed and any(a in allowed for a in correct_actions),
            "missed_only_outside_deployed_action_family": missed and not any(a in allowed for a in correct_actions),
            "wrong_and_unrescuable_any_grid": not chosen["correct"] and not correct_actions,
            "baseline_correct_lost": bool(cells[0]["correct"]) and not chosen["correct"],
            "baseline_wrong_rescued": not cells[0]["correct"] and bool(chosen["correct"]),
            "baseline_wrong_rescuable_missed": not cells[0]["correct"] and missed,
            "missed_requires_both_axes_from_baseline": missed and flags["both_axes_required"],
            "oracle_utility_regret": oracle["utility"] - chosen["utility"],
            "correctness_regret": oracle["correct"] - chosen["correct"],
            "resource_cost_regret_component": WEIGHT * (cost(chosen, cells) - cost(oracle, cells)),
            "oracle_minus_selected_image_bytes": oracle["image_bytes"] - chosen["image_bytes"],
            "oracle_minus_selected_actual_visual_tokens": oracle["actual_visual_tokens"] - chosen["actual_visual_tokens"],
            "correct_alternative_actions": [label(a) for a in correct_actions] if missed else [],
        }
    boolean_fields = [key for key, value in next(iter(details.values())).items() if isinstance(value, bool)]
    summary = summarize(list(selected.values()))
    oracle_summary = summarize([grid[identity][action] for identity, action in oracle_choices.items()])
    counts = {key: sum(row[key] for row in details.values()) for key in boolean_fields}
    counts["mean_correctness_regret"] = statistics.mean(row["correctness_regret"] for row in details.values())
    counts["mean_resource_cost_regret_component"] = statistics.mean(row["resource_cost_regret_component"] for row in details.values())
    type_misses = {qtype: {"n": sum(grid[i][0]["question_type"] == qtype for i in grid),
                           **{key: sum(row[key] for i, row in details.items() if grid[i][0]["question_type"] == qtype)
                              for key in boolean_fields}}
                   for qtype in sorted({cells[0]["question_type"] for cells in grid.values()})}
    return {"summary": summary, "by_type": by_type(list(selected.values())), "misses": counts,
            "misses_by_type": type_misses, "nine_way_oracle_regret": resource_delta(oracle_summary, summary)}, details


def analyze_split(grid: dict[str, list[dict]], report: dict | None = None) -> tuple[dict, list[dict]]:
    if not grid:
        raise ValueError("Cannot analyze an empty grid")
    cell_summaries = {label(a): {**summarize([cells[a] for cells in grid.values()]),
                               "by_type": by_type([cells[a] for cells in grid.values()])} for a in range(9)}
    flags = {identity: opportunity_flags(cells) for identity, cells in grid.items()}
    flag_keys = list(next(iter(flags.values())))
    flag_counts = {key: sum(row[key] for row in flags.values()) for key in flag_keys}
    counts_by_type = {qtype: {"n": sum(cells[0]["question_type"] == qtype for cells in grid.values()),
                            **{key: sum(row[key] for i, row in flags.items() if grid[i][0]["question_type"] == qtype)
                               for key in flag_keys}}
                     for qtype in sorted({cells[0]["question_type"] for cells in grid.values()})}
    histogram = Counter(sum(row["correct"] for row in cells) for cells in grid.values())
    nonmonotonic, sample_losses = nonmonotonic_summary(grid)
    selection = report["selection"] if report else None
    oracles, oracle_choices = summarize_oracles(grid, selection)
    cheapest_correct = {identity: oracle_action(cells, [a for a in range(9) if cells[a]["correct"]])
                        for identity, cells in grid.items() if any(row["correct"] for row in cells)}
    methods, method_details = {}, {}
    if report:
        for name, selected in check_deployed(report, grid).items():
            methods[name], method_details[name] = method_misses(name, selected, grid, oracle_choices["nine_way"], selection)
    summary = {
        "n": len(grid), "cells": cell_summaries,
        "grid_agreement": {"all_correct": histogram[9], "all_wrong": histogram[0],
                           "disagreement": len(grid) - histogram[9] - histogram[0],
                           "correct_cell_count_histogram": {str(k): histogram[k] for k in range(10)}},
        "opportunities_from_2000_low": {"counts": flag_counts, "by_type": counts_by_type},
        "nonmonotonicity": nonmonotonic, "oracles": oracles,
        "cheapest_correct": {"definition": "Minimum actual framed bytes/8001 + visual tokens/(4000_high visual tokens); ties use action order.",
                             "unrescuable_n": len(grid) - len(cheapest_correct),
                             **summarize([grid[i][a] for i, a in cheapest_correct.items()]),
                             "by_type": by_type([grid[i][a] for i, a in cheapest_correct.items()])},
        "deployed_selection_unchanged": selection,
        "deployed_methods": methods,
        "deployed_evaluation_available": bool(report),
        "deployed_evaluation_note": ("Existing frozen report selections checked against the nine-cell cache."
                                     if report else "No existing train deployment report supplied; no selector inference was run."),
    }
    per_image = [{"id": identity, "image_id": cells[0].get("image_id"),
                  "question_type": cells[0]["question_type"],
                  "cells": [{"action": label(a), "correct": row["correct"], "utility": row["utility"],
                             "image_bytes": row["image_bytes"], "actual_visual_tokens": row["actual_visual_tokens"]}
                            for a, row in enumerate(cells)],
                  "opportunities_from_2000_low": flags[identity],
                  "cheapest_correct_action": label(cheapest_correct[identity]) if identity in cheapest_correct else None,
                  "oracle_actions": {name: label(choices[identity]) for name, choices in oracle_choices.items()},
                  "nonmonotonic_losses": sample_losses[identity],
                  "deployed": {name: details[identity] for name, details in method_details.items()}}
                 for identity, cells in grid.items()]
    return summary, per_image


def verify_inputs(input_dir: Path) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict]:
    """Read the explicit three authorized splits and verify available provenance."""
    records, truths, provenance = {}, {}, {}
    for kind, filename, complete_name, count in (
        ("supervision", "supervision_records.json", "supervision_complete.json", 5400),
        ("legacy", "legacy_grid_records.json", "legacy_grid_complete.json", 1080),
    ):
        path = input_dir / filename
        rows, complete = read(path), read(input_dir / complete_name)
        if len(rows) != count or complete["records"] != count or complete["records_sha256"] != sha(path):
            raise ValueError(f"Incomplete or altered cached records: {kind}")
        records[kind] = rows
        provenance[filename] = sha(path)
        provenance[complete_name] = sha(input_dir / complete_name)
    frozen_data = read(input_dir / "frozen_data.json")
    truth_hashes = {Path(name).name: value for name, value in frozen_data["sha256"].items()}
    for split, expected_size in EXPECTED_SIZES.items():
        path = input_dir / f"{split}_truth.json"
        truths[split] = read(path)
        if len(truths[split]) != expected_size:
            raise ValueError(f"Unexpected split size: {split}")
        expected_hash = (read(input_dir / "legacy_dev_report.json")["provenance"]["legacy_truth_sha256"]
                         if split == "legacy_dev" else truth_hashes[path.name])
        if sha(path) != expected_hash:
            raise ValueError(f"Truth differs from the existing frozen provenance: {split}")
        provenance[path.name] = sha(path)
    seen = set()
    for split, rows in truths.items():
        identities = {row["id"] for row in rows}
        if len(identities) != len(rows) or seen & identities:
            raise ValueError(f"Overlapping or duplicate truth identities: {split}")
        seen.update(identities)
    if {r["id"] for r in records["supervision"]} != {r["id"] for s in ("train", "validation") for r in truths[s]}:
        raise ValueError("Supervision identities differ from train + validation truth")
    if {r["id"] for r in records["legacy"]} != {r["id"] for r in truths["legacy_dev"]}:
        raise ValueError("Legacy record identities differ from historical development truth")
    return records, truths, provenance


def run(input_dir: Path, output_dir: Path) -> dict:
    records, truths, provenance = verify_inputs(input_dir)
    summaries, per_image = {}, {}
    for split in EXPECTED_SIZES:
        legacy = split == "legacy_dev"
        complete = read(input_dir / ("legacy_grid_complete.json" if legacy else "supervision_complete.json"))
        grid = score_grid(records["legacy" if legacy else "supervision"], truths[split],
                          [row["id"] for row in truths[split]], complete["receiver_sha256"],
                          complete["protocol_sha256"], legacy=legacy)
        report_path = input_dir / f"{split}_report.json"
        report = read(report_path) if split != "train" else None
        if report:
            provenance[report_path.name] = sha(report_path)
        summaries[split], per_image[split] = analyze_split(grid, report)
    if summaries["validation"]["deployed_selection_unchanged"] != summaries["legacy_dev"]["deployed_selection_unchanged"]:
        raise ValueError("Historical development report changed the frozen selection")
    source = Path(__file__).resolve().parents[1] / "rgb_joint_selector" / "train_selector.py"
    result = {
        "schema": "rgb-joint-opportunities-v1", "splits": summaries,
        "scope": "Cached train, internal validation, and reused historical development only; exploratory descriptive diagnosis.",
        "oracle_warning": "Answers select oracle actions. All oracle numbers are hindsight upper bounds, not measured routing performance. Strongest fixed-other-axis oracles are chosen separately on each displayed split for bounding only.",
        "definitions": {
            "utility": "correct - 0.05 * (framed_image_bytes/8001 + actual_visual_tokens/(same image's 4000_high tokens))",
            "framing": "Legacy raw image_bytes receive exactly one route byte through original score_grid; supervision bytes already include it.",
            "R": "2000_low wrong; correct at 4000_low or 8000_low",
            "C": "2000_low wrong; correct at 2000_medium or 2000_high",
            "B": "2000_low wrong; correct at an action with both budget >2000 and tier >low",
            "both_axes_required": "B minus (R union C)",
            "any_grid_rescuable": "R union C union B; all such images start wrong at 2000_low",
            "nonmonotonic_loss": "Correct at a lower ordered resource action and wrong after increasing one or both action axes; one image may have multiple transitions.",
            "strong_axis_oracle": "Enumerate every globally fixed other-axis value, choose each image's maximum-utility action inside that family, then select the family with maximum mean utility. This retrospective family selection is not the deployed frozen baseline.",
            "regret_decomposition": "oracle utility - deployed utility = (oracle correct - deployed correct) + 0.05 * (deployed normalized cost - oracle normalized cost)",
            "resource_scope": "Framed bytes and actual visual tokens only; tokens proxy visual workload. No measured energy or live end-to-end cost claim.",
        },
        "execution": {"training_run": False, "vlm_inference_run": False, "codec_inference_run": False,
                      "selector_inference_run": False, "sealed_test_opened": False, "frozen_protocol_changed": False},
        "provenance": {"input_sha256": provenance, "original_scoring_module_sha256": sha(source),
                       "diagnostic_script_sha256": sha(Path(__file__))},
        "per_image_local_artifact": "opportunities_per_image.local.json",
    }
    write(output_dir / "opportunities_per_image.local.json", {"scope": "LOCAL ONLY: includes sample identities; do not publish.", "splits": per_image}, private=True)
    write(output_dir / "opportunities.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run(args.input_dir.resolve(), (args.output_dir or args.input_dir / "diagnostics").resolve())
    for split, summary in result["splits"].items():
        print(json.dumps({"split": split, "n": summary["n"],
                          "2000_low_correct": summary["cells"]["2000_low"]["correct"],
                          "nine_way_oracle_correct": summary["oracles"]["all_families"]["nine_way"]["correct"],
                          "rescuable_from_2000_low": summary["opportunities_from_2000_low"]["counts"]["any_grid_rescuable"],
                          "joint_correct": summary["deployed_methods"].get("joint", {}).get("summary", {}).get("correct")}))


if __name__ == "__main__":
    main()
