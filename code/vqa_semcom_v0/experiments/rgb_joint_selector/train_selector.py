"""Train bounded, answer-blind-at-deployment EXP-012 selectors.

Historical development labels may be opened only after controller_frozen.json
pins all checkpoints, axis selections and strong fixed baseline. This program
never reads the sealed test or runs a wireless experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any

try:
    from .prepare_data import check_hashes, read, save, sha
except ImportError:
    from prepare_data import check_hashes, read, save, sha

CELLS = [(b, t) for b in (2000, 4000, 8000) for t in ("low", "medium", "high")]
PIXELS = {"low": 50176, "medium": 100352, "high": 200704}
VARIANTS = {"joint9": list(range(9)), **{
    f"rate_at_{tier}": [i for i, (_, t) in enumerate(CELLS) if t == tier]
    for tier in PIXELS}, **{
    f"compute_at_{budget}": [i for i, (b, _) in enumerate(CELLS) if b == budget]
    for budget in (2000, 4000, 8000)}, "question_only9": list(range(9))}
SEEDS = [7, 17, 27]
WEIGHT = .05
REFERENCE = CELLS.index((4000, "medium"))


def normalize(text: str) -> str:
    numbers = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    value = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", " ".join(str(text).lower().split()))
    return numbers[int(value)] if value.isdigit() and int(value) <= 20 else value


def nominal_cost(action: int) -> float:
    budget, tier = CELLS[action]
    return (budget + 1) / 8001 + PIXELS[tier] / 200704


def action_descriptor(action: int) -> dict:
    """The wire byte identifies visual tier, never the nine-way policy action.

    The unchanged codec packet already contains its decoding information; its
    source-byte cap is a sender policy decision and does not need a new header.
    """
    budget, tier = CELLS[action]
    return {"index": action, "budget": budget, "tier": tier, "pixel_target": PIXELS[tier],
            "nominal_cost": nominal_cost(action), "route_byte": tuple(PIXELS).index(tier)}


def choose_action(probabilities: list[float], actions: list[int]) -> int:
    if len(probabilities) != len(actions) or not actions:
        raise ValueError("Probability/action dimensions disagree")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid predicted probabilities")
    return max(zip(actions, probabilities), key=lambda ap: (
        ap[1] - WEIGHT * nominal_cost(ap[0]), -nominal_cost(ap[0]), -ap[0]))[0]


def validate_protocol(protocol: dict) -> None:
    expected_training = {"seeds": SEEDS, "hidden": [128, 64], "dropout": .1, "max_epochs": 50,
                         "patience": 8, "min_epochs": 10, "batch_size": 32,
                         "learning_rate": .001, "weight_decay": .0001}
    if protocol["budgets"] != [2000, 4000, 8000] or protocol["frame_header_bytes"] != 1:
        raise ValueError("Protocol differs from registered action/framing contract")
    if protocol["tiers"] != [{"name": name, "pixels": pixels} for name, pixels in PIXELS.items()]:
        raise ValueError("Protocol visual tiers changed")
    if protocol["selection"]["lambda"] != WEIGHT or protocol["selection"]["primary_fixed_reference"] != "4000_medium":
        raise ValueError("Protocol utility/primary fixed reference changed")
    for key, value in expected_training.items():
        if protocol["training"][key] != value:
            raise ValueError(f"Training protocol changed: {key}")


def validate_features(features: list[dict], expected: dict[str, set[str]]) -> dict[str, dict]:
    mapped = {}
    for row in features:
        identity, split = row.get("id"), row.get("split")
        if identity in mapped or split not in expected or identity not in expected[split]:
            raise ValueError("Duplicate or unregistered feature identity/split")
        for field, width in (("question_features", 256), ("image_features", 83)):
            values = row.get(field)
            if not isinstance(values, list) or len(values) != width or any(
                    not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError(f"Invalid {field}")
        for field in ("question_feature_seconds", "image_feature_seconds"):
            if not isinstance(row.get(field), (int, float)) or not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError(f"Invalid feature timing: {field}")
        mapped[identity] = row
    if set(mapped) != set().union(*expected.values()):
        raise ValueError("Incomplete feature inventory")
    return mapped


def fit_scaler(rows: list[dict]) -> dict:
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("Only selector training rows may fit the scaler")
    values = [row["image_features"] for row in rows]
    mean = [statistics.mean(column) for column in zip(*values)]
    std = [max(math.sqrt(statistics.mean((row[j] - mean[j]) ** 2 for row in values)), 1e-6)
           for j in range(83)]
    return {"mean": mean, "std": std, "fit_split": "train", "n": len(rows), "question_standardized": False}


def vector(row: dict, scaler: dict, question_only: bool = False) -> list[float]:
    image = ([0.] * 83 if question_only else
             [(v - m) / s for v, m, s in zip(row["image_features"], scaler["mean"], scaler["std"])])
    result = list(row["question_features"]) + image
    if len(result) != 339 or not all(math.isfinite(v) for v in result):
        raise ValueError("Invalid normalized deployment feature vector")
    return result


def score_grid(records: list[dict], truth: list[dict], identities: list[str],
               receiver_hash: str, protocol_hash: str | None, legacy: bool = False) -> dict[str, list[dict]]:
    answers = {row["id"]: row["answer"] for row in truth}
    if len(answers) != len(truth) or set(answers) != set(identities):
        raise ValueError("Truth identities differ from selected split")
    grid: dict[str, dict[int, dict]] = {identity: {} for identity in identities}
    for source in records:
        identity = source["id"]
        if identity not in grid:
            continue
        if source["receiver_sha256"] != receiver_hash or (
                protocol_hash is not None and source["protocol_sha256"] != protocol_hash):
            raise ValueError("Mixed receiver or protocol in supervision")
        cell = source["budget"], source["tier"]
        if cell not in CELLS:
            raise ValueError("Unregistered supervision action")
        action = CELLS.index(cell)
        if action in grid[identity]:
            raise ValueError("Duplicate supervision cell")
        row = dict(source)
        if legacy:
            row["codec_image_bytes"] = source["image_bytes"]
            row["image_bytes"] = source["image_bytes"] + 1
        if not (0 < row["codec_image_bytes"] <= row["budget"] and
                row["image_bytes"] == row["codec_image_bytes"] + 1):
            raise ValueError("Invalid raw/framed byte accounting")
        if row["actual_visual_tokens"] <= 0:
            raise ValueError("Invalid visual token count")
        row.update(action=action, correct=int(normalize(row["prediction"]) == normalize(answers[identity])),
                   ldpc_complex_symbols=510 * math.ceil(row["image_bytes"] / 48))
        grid[identity][action] = row
    output = {}
    for identity, cells in grid.items():
        if set(cells) != set(range(9)):
            raise ValueError(f"Incomplete supervision grid: {identity}")
        high = cells[CELLS.index((4000, "high"))]["actual_visual_tokens"]
        rows = [cells[action] for action in range(9)]
        for row in rows:
            row["utility"] = row["correct"] - WEIGHT * (row["image_bytes"] / 8001 + row["actual_visual_tokens"] / high)
        output[identity] = rows
    return output


def make_model(heads: int) -> Any:
    from torch import nn
    return nn.Sequential(nn.Linear(339, 128), nn.ReLU(), nn.Dropout(.1),
                         nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64, heads))


def train_one(x_train: Any, y_train: Any, x_validation: Any, y_validation: Any,
              actions: list[int], seed: int, settings: dict) -> tuple[Any, list[dict], dict]:
    import torch
    from torch.nn import functional as F
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = make_model(len(actions))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    target_train, target_val = y_train[:, actions], y_validation[:, actions]
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    history = []
    start = time.perf_counter()
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        permutation = torch.randperm(len(x_train), generator=generator)
        total = 0.
        for batch in permutation.split(settings["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x_train[batch]), target_train[batch])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training BCE")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite selector gradient")
            optimizer.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.inference_mode():
            val_loss = F.binary_cross_entropy_with_logits(model(x_validation), target_val).item()
        if not math.isfinite(val_loss):
            raise ValueError("Nonfinite validation BCE")
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch, "train_bce": total / len(x_train), "validation_bce": val_loss,
                        "checkpoint_improved": improved})
        if epoch >= settings["min_epochs"] and stale >= settings["patience"]:
            break
    if best_state is None:
        raise ValueError("No finite selector checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, {"best_epoch": best_epoch, "best_validation_bce": best_loss,
                           "epochs_run": len(history), "training_seconds": time.perf_counter() - start}


def torch_save_atomic(path: Path, value: Any) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, actions: list[int]) -> Any:
    import torch
    model = make_model(len(actions))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict_rows(models: list[Any], actions: list[int], rows: list[dict], scaler: dict,
                 question_only: bool = False, *, warmup: bool = True) -> tuple[list[list[float]], list[float]]:
    """Single-example profiling includes normalization, CPU tensor and ensemble."""
    import torch
    predictions, durations = [], []
    if rows and warmup:
        with torch.inference_mode():
            warmup = torch.tensor([vector(rows[0], scaler, question_only)], dtype=torch.float32)
            for model in models:
                model(warmup)
    for row in rows:
        start = time.perf_counter()
        with torch.inference_mode():
            inputs = torch.tensor([vector(row, scaler, question_only)], dtype=torch.float32)
            p = torch.stack([model(inputs).sigmoid()[0] for model in models]).mean(0).tolist()
        choose_action(p, actions)  # Include final decision in timing.
        durations.append(time.perf_counter() - start)
        predictions.append(p)
    return predictions, durations


def evaluate_policy(rows: list[dict], grid: dict[str, list[dict]], probabilities: list[list[float]],
                    actions: list[int], router_seconds: list[float], *, question_only: bool = False) -> dict:
    selected, brier = [], []
    for feature, probs, latency in zip(rows, probabilities, router_seconds):
        identity = feature["id"]
        action = choose_action(probs, actions)
        chosen = dict(grid[identity][action])
        brier.extend((p - grid[identity][a]["correct"]) ** 2 for a, p in zip(actions, probs))
        feature_seconds = feature["question_feature_seconds"] + (0 if question_only else feature["image_feature_seconds"])
        chosen.update(router_seconds=latency, feature_seconds=feature_seconds,
                      decision_seconds=feature_seconds + latency)
        components = [chosen.get(field) for field in ("encode_seconds", "decode_seconds", "preprocessing_seconds", "receiver_seconds")]
        chosen["component_profile_sum_seconds"] = (sum(components) + feature_seconds + latency
                                                   if all(isinstance(v, (float, int)) for v in components) else None)
        selected.append(chosen)
    return {"summary": summarize(selected), "brier_all_candidate_heads": statistics.mean(brier),
            "selected": selected}


def summarize(selected: list[dict]) -> dict:
    fields = ("image_bytes", "ldpc_complex_symbols", "actual_visual_tokens", "receiver_seconds",
              "preprocessing_seconds", "encode_seconds", "decode_seconds", "decision_seconds", "component_profile_sum_seconds")
    result = {"n": len(selected), "correct": sum(r["correct"] for r in selected),
              "accuracy": statistics.mean(r["correct"] for r in selected),
              "utility": statistics.mean(r["utility"] for r in selected),
              "routing_histogram": dict(sorted(Counter(f"{CELLS[r['action']][0]}_{CELLS[r['action']][1]}" for r in selected).items())),
              "energy_j": None, "full_cost_gate": "PENDING", "timing_scope": "component profile; not a paired live selected-path measurement"}
    for field in fields:
        observed = [r[field] for r in selected if isinstance(r.get(field), (int, float)) and math.isfinite(r[field])]
        result[field] = {"mean": statistics.mean(observed) if len(observed) == len(selected) else None,
                         "observed_n": len(observed), "missing_n": len(selected) - len(observed)}
    return result


def fixed_policy(rows: list[dict], grid: dict[str, list[dict]], action: int) -> dict:
    selected = []
    for feature in rows:
        row = dict(grid[feature["id"]][action])
        row.update(router_seconds=0., feature_seconds=0., decision_seconds=0.)
        components = [row.get(f) for f in ("encode_seconds", "decode_seconds", "preprocessing_seconds", "receiver_seconds")]
        row["component_profile_sum_seconds"] = sum(components) if all(isinstance(v, (float, int)) for v in components) else None
        selected.append(row)
    return {"summary": summarize(selected), "selected": selected}


def choose_best_fixed(rows: list[dict], grid: dict[str, list[dict]]) -> int:
    return max(range(9), key=lambda action: (
        statistics.mean(grid[row["id"]][action]["utility"] for row in rows), -nominal_cost(action), -action))


def choose_best_axis(evaluations: dict[str, dict], prefix: str) -> str:
    candidates = [name for name in VARIANTS if name.startswith(prefix)]
    return max(candidates, key=lambda name: (evaluations[name]["ensemble"]["summary"]["utility"],
        -statistics.mean(nominal_cost(a) for a in VARIANTS[name]), -list(VARIANTS).index(name)))


def bootstrap_difference(a: list[float], b: list[float], repeats: int = 2000) -> dict:
    if len(a) != len(b) or not a:
        raise ValueError("Paired bootstrap requires identical nonempty units")
    differences = [x - y for x, y in zip(a, b)]
    rng = random.Random(20260922)
    sampled = sorted(statistics.mean(differences[rng.randrange(len(a))] for _ in a) for _ in range(repeats))
    return {"paired_mean_difference": statistics.mean(differences),
            "image_bootstrap_95_percentile_interval": [sampled[int(.025 * (repeats - 1))], sampled[int(.975 * (repeats - 1))]],
            "replicates": repeats, "limitations": "conditional on frozen fitted models; ignores checkpoint/axis selection and training uncertainty"}


def comparisons(methods: dict[str, dict]) -> dict:
    joint = methods["joint"]["selected"]
    output = {}
    for name in ("primary_fixed", "strong_fixed", "rate_only", "compute_only", "question_only", "image_shuffle"):
        baseline = methods[name]["selected"]
        if [r["id"] for r in joint] != [r["id"] for r in baseline]:
            raise ValueError("Paired report unit order mismatch")
        gains = sum(a["correct"] > b["correct"] for a, b in zip(joint, baseline))
        losses = sum(a["correct"] < b["correct"] for a, b in zip(joint, baseline))
        discordant = gains + losses
        p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / 2 ** discordant) if discordant else 1.
        output[name] = {"gained_answers": gains, "lost_answers": losses,
                        "accuracy_difference": bootstrap_difference([r["correct"] for r in joint], [r["correct"] for r in baseline]),
                        "utility_difference": bootstrap_difference([r["utility"] for r in joint], [r["utility"] for r in baseline]),
                        "paired_accuracy_exact_mcnemar_p": p}
    previous = 0.
    ordered = sorted(output, key=lambda name: output[name]["paired_accuracy_exact_mcnemar_p"])
    for rank, name in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered) - rank) * output[name]["paired_accuracy_exact_mcnemar_p"]))
        output[name]["paired_accuracy_holm_adjusted_p"] = previous
    return output


def gate(methods: dict[str, dict], seed_methods: dict[str, dict], screen: dict) -> dict:
    joint = methods["joint"]["summary"]
    ref = methods["primary_fixed"]["summary"]
    others = ("strong_fixed", "rate_only", "compute_only")
    gains = {name: joint["utility"] - methods[name]["summary"]["utility"] for name in others}
    consistent = [seed for seed, groups in seed_methods.items() if all(
        groups["joint"]["summary"]["utility"] - groups[name]["summary"]["utility"] >= screen["minimum_utility_gain_over_strong_fixed_and_learned_single_axes"] for name in others)]
    checks = {"accuracy_within_one_question": joint["correct"] >= ref["correct"] - screen["maximum_lost_questions_vs_primary_fixed"],
              "bytes_at_least_ten_percent_lower": joint["image_bytes"]["mean"] <= ref["image_bytes"]["mean"] * screen["maximum_mean_image_byte_ratio_vs_primary_fixed"],
              "utility_over_all_strong_controls": all(gain >= screen["minimum_utility_gain_over_strong_fixed_and_learned_single_axes"] for gain in gains.values()),
              "seed_consistency": len(consistent) >= screen["minimum_consistent_seeds"]}
    return {"preliminary_checks": checks, "preliminary_pass": all(checks.values()),
            "utility_gains": gains, "consistent_seeds": consistent, "full_cost_gate": "PENDING",
            "expand_to_test_or_snr": False}


def evaluate_split(rows: list[dict], grid: dict[str, list[dict]], models: dict[str, dict[int, Any]],
                   scaler: dict, selection: dict | None, screen: dict) -> tuple[dict, dict]:
    variants = {}
    for name, actions in VARIANTS.items():
        question_only = name == "question_only9"
        variants[name] = {}
        for label, chosen_models in [(str(s), [models[name][s]]) for s in SEEDS] + [("ensemble", list(models[name].values()))]:
            probabilities, times = predict_rows(chosen_models, actions, rows, scaler, question_only)
            variants[name][label] = evaluate_policy(rows, grid, probabilities, actions, times, question_only=question_only)
    if selection is None:
        selection = {"strong_fixed_action": choose_best_fixed(rows, grid),
                     "rate_variant": choose_best_axis(variants, "rate_at_"),
                     "compute_variant": choose_best_axis(variants, "compute_at_")}
    shuffled = [{**row, "image_features": rows[(index + 1) % len(rows)]["image_features"]} for index, row in enumerate(rows)]
    shuffle_probs, shuffle_times = predict_rows(list(models["joint9"].values()), VARIANTS["joint9"], shuffled, scaler)
    methods = {"joint": variants["joint9"]["ensemble"], "question_only": variants["question_only9"]["ensemble"],
               "rate_only": variants[selection["rate_variant"]]["ensemble"], "compute_only": variants[selection["compute_variant"]]["ensemble"],
               "primary_fixed": fixed_policy(rows, grid, REFERENCE),
               "strong_fixed": fixed_policy(rows, grid, selection["strong_fixed_action"]),
               "image_shuffle": evaluate_policy(rows, grid, shuffle_probs, VARIANTS["joint9"], shuffle_times)}
    seed_methods = {str(seed): {"joint": variants["joint9"][str(seed)],
                    "rate_only": variants[selection["rate_variant"]][str(seed)],
                    "compute_only": variants[selection["compute_variant"]][str(seed)],
                    "strong_fixed": methods["strong_fixed"]} for seed in SEEDS}
    summary = {"variants": variants, "methods": methods, "seed_methods": seed_methods,
               "paired_comparisons": comparisons(methods), "gate": gate(methods, seed_methods, screen),
               "selection": selection, "seeds": SEEDS, "seed_summary": {metric: {
                   "mean": statistics.mean(seed_methods[str(seed)]["joint"]["summary"][metric] for seed in SEEDS),
                   "sample_sd": statistics.stdev(seed_methods[str(seed)]["joint"]["summary"][metric] for seed in SEEDS)}
                   for metric in ("accuracy", "utility")},
               "statistical_scope": "paired images; 3 training seeds are not 3 independent datasets; selection/reused-dev bias precludes confirmatory claims"}
    return summary, selection


def load_frozen_policy(output: Path, variant_name: str = "joint9", seed: int | None = None) -> dict:
    """Label-free deployment loader; returns model objects and frozen feature scale."""
    frozen = read(output / "controller_frozen.json")
    check_hashes(frozen["sha256"])
    policy = read(output / "policy.json")
    if variant_name not in VARIANTS or (seed is not None and seed not in SEEDS):
        raise ValueError("Unregistered deployment variant or seed")
    selected_seeds = SEEDS if seed is None else [seed]
    models = [load_checkpoint(Path(policy["checkpoints"][variant_name][str(s)]["path"]), VARIANTS[variant_name]) for s in selected_seeds]
    return {"models": models, "actions": VARIANTS[variant_name], "scaler": policy["scaler"],
            "question_only": variant_name == "question_only9", "controller_sha256": sha(output / "controller_frozen.json")}


def predict_deployment(bundle: dict, question_features: list[float], image_features: list[float]) -> dict:
    row = {"question_features": question_features, "image_features": image_features}
    probabilities, duration = predict_rows(bundle["models"], bundle["actions"], [row], bundle["scaler"], bundle["question_only"], warmup=False)
    action = choose_action(probabilities[0], bundle["actions"])
    descriptor = action_descriptor(action)
    return {"action": action, "budget": descriptor["budget"], "tier": descriptor["tier"],
            "route_byte": descriptor["route_byte"],
            "probabilities": probabilities[0], "router_seconds": duration[0]}


def read_legacy_truth(path: Path, output: Path) -> list[dict]:
    """Enforce the chronology at the actual legacy label access boundary."""
    frozen_path = output / "controller_frozen.json"
    if not frozen_path.is_file():
        raise ValueError("Controller must be frozen before historical labels are accessed")
    frozen = read(frozen_path)
    if frozen.get("state") != "FROZEN_BEFORE_LEGACY_DEV_LABELS":
        raise ValueError("Controller is not a finalized pre-legacy selection")
    check_hashes(frozen["sha256"])
    if path.name != "dev_truth.sealed.json" or any(re.search(r"(^|[^a-z])test([^a-z]|$)", part.lower()) for part in path.resolve().parts):
        raise ValueError("Only explicitly named legacy development truth is allowed")
    return read(path)


def verify_inputs(output: Path, protocol_path: Path, legacy_stage1: Path | None) -> tuple[dict, dict, dict, dict]:
    protocol = read(protocol_path)
    validate_protocol(protocol)
    frozen_data = read(output / "frozen_data.json")
    if frozen_data["state"] != "COMPLETE" or frozen_data["protocol_sha256"] != sha(protocol_path):
        raise ValueError("Data not frozen under current protocol")
    check_hashes(frozen_data["sha256"])
    feature_complete = read(output / "features_complete.json")
    if (feature_complete.get("state") != "COMPLETE" or
            feature_complete.get("features_sha256") != sha(output / "features.json") or
            feature_complete.get("frozen_sha256") != sha(output / "features_frozen.json")):
        raise ValueError("Complete unchanged feature inventory is required")
    check_hashes(read(output / "features_frozen.json")["sha256"])
    inference_frozen = read(output / "inference_frozen.json")
    if (inference_frozen.get("receiver_sha256") != protocol["receiver_sha256"] or
            inference_frozen.get("protocol_sha256") != sha(protocol_path)):
        raise ValueError("Inference provenance uses a different model or protocol")
    check_hashes(inference_frozen["sha256"])
    complete = read(output / "supervision_complete.json")
    records_path = output / "supervision_records.json"
    if (complete.get("state") != "SUPERVISION_COMPLETE" or complete.get("records") != 5400 or
            complete.get("records_sha256") != sha(records_path) or complete.get("protocol_sha256") != sha(protocol_path) or
            complete.get("receiver_sha256") != protocol["receiver_sha256"]):
        raise ValueError("Complete unchanged 600x9 supervision is required")
    manifests = {split: read(output / f"{split}_manifest.json") for split in ("train", "validation")}
    if [len(manifests[s]) for s in ("train", "validation")] != [480, 120]:
        raise ValueError("Registered train/validation sizes changed")
    if legacy_stage1 is not None:
        manifests["legacy_dev"] = read(legacy_stage1 / "dev_manifest.json")
        if len(manifests["legacy_dev"]) != 120:
            raise ValueError("Legacy development size changed")
    expected = {split: {r["id"] for r in rows} for split, rows in manifests.items()}
    features = validate_features(read(output / "features.json"), expected)
    truth = {split: read(output / f"{split}_truth.json") for split in ("train", "validation")}
    records = read(records_path)
    if len(records) != 5400:
        raise ValueError("Supervision is growing or contains an unexpected record count")
    grids = {split: score_grid(records, truth[split], [r["id"] for r in manifests[split]],
                              protocol["receiver_sha256"], sha(protocol_path)) for split in ("train", "validation")}
    return protocol, manifests, features, grids


def run(output: Path, protocol_path: Path, legacy_stage1: Path | None, legacy_grid: Path | None) -> dict:
    import torch
    torch.set_num_threads(2)
    if (legacy_stage1 is None) != (legacy_grid is None):
        raise ValueError("Both legacy development roots are required together")
    protocol, manifests, features, grids = verify_inputs(output, protocol_path, legacy_stage1)
    paths = [protocol_path, Path(__file__).resolve(), output / "frozen_data.json", output / "features.json",
             output / "features_frozen.json", output / "features_complete.json",
             output / "supervision_records.json", output / "supervision_complete.json", output / "inference_frozen.json"]
    paths += [output / f"{split}_{kind}.json" for split in ("train", "validation") for kind in ("manifest", "truth")]
    if legacy_stage1 is not None:
        paths += [legacy_stage1 / "dev_manifest.json", legacy_grid / "records.json", legacy_grid / "complete.json"]
    fingerprint = {"sha256": {str(p): sha(p) for p in paths}, "torch_version": torch.__version__,
                   "cpu_threads": 2, "sealed_test_opened": False, "legacy_dev_labels_opened": False}
    inputs_path = output / "training_frozen_inputs.json"
    if inputs_path.exists():
        if read(inputs_path) != fingerprint:
            raise ValueError("Training inputs/code/runtime changed on resume")
    else:
        save(inputs_path, fingerprint)
    train_rows = [features[r["id"]] for r in manifests["train"]]
    val_rows = [features[r["id"]] for r in manifests["validation"]]
    scaler = fit_scaler(train_rows)
    y_train = torch.tensor([[r["correct"] for r in grids["train"][f["id"]]] for f in train_rows], dtype=torch.float32)
    y_val = torch.tensor([[r["correct"] for r in grids["validation"][f["id"]]] for f in val_rows], dtype=torch.float32)
    models, checkpoints, histories = {}, {}, {}
    for name, actions in VARIANTS.items():
        models[name], checkpoints[name], histories[name] = {}, {}, {}
        x_train = torch.tensor([vector(row, scaler, name == "question_only9") for row in train_rows], dtype=torch.float32)
        x_val = torch.tensor([vector(row, scaler, name == "question_only9") for row in val_rows], dtype=torch.float32)
        for seed in SEEDS:
            checkpoint = output / "checkpoints" / f"{name}-seed{seed}.pt"
            job_path = output / "training_jobs" / f"{name}-seed{seed}.json"
            if job_path.exists():
                job = read(job_path)
                if job["input_sha256"] != sha(inputs_path) or job["checkpoint_sha256"] != sha(checkpoint):
                    raise ValueError("Completed training job provenance changed")
                model = load_checkpoint(checkpoint, actions)
            else:
                model, history, stats = train_one(x_train, y_train, x_val, y_val, actions, seed, protocol["training"])
                torch_save_atomic(checkpoint, model.state_dict())
                job = {"variant": name, "seed": seed, "actions": actions, "history": history, **stats,
                       "input_sha256": sha(inputs_path), "checkpoint_sha256": sha(checkpoint)}
                save(job_path, job)
            models[name][seed] = model
            checkpoints[name][str(seed)] = {"path": str(checkpoint), "sha256": sha(checkpoint)}
            histories[name][str(seed)] = job
            save(output / "training_status.json", {"state": "TRAINING", "completed_jobs": sum(len(v) for v in histories.values()), "total_jobs": 24})
    save(output / "training_history.json", histories)
    frozen_path = output / "controller_frozen.json"
    if frozen_path.exists():
        frozen = read(frozen_path)
        check_hashes(frozen["sha256"])
        selection = read(output / "policy.json")["selection"]
        validation_report = read(output / "validation_report.json")
    else:
        validation_report, selection = evaluate_split(val_rows, grids["validation"], models, scaler, None, protocol["screen"])
        validation_report["scope"] = "new internal validation; checkpoint and axis selection data, not an independent test"
        save(output / "validation_report.json", validation_report)
        save(output / "policy.json", {"schema": "rgb-joint-selector-v1", "selection": selection, "scaler": scaler,
             "checkpoints": checkpoints, "actions": [action_descriptor(i) for i in range(len(CELLS))],
             "probability_ensemble": "arithmetic mean over all three seeds", "lambda": WEIGHT,
             "feature_dimensions": {"question": 256, "image": 83}, "train_image_scaler_only": True,
             "inference_requires_answers": False, "question_type_is_input": False})
        freeze_files = [inputs_path, output / "training_history.json", output / "validation_report.json", output / "policy.json"]
        freeze_files += [Path(meta["path"]) for family in checkpoints.values() for meta in family.values()]
        save(frozen_path, {"state": "FROZEN_BEFORE_LEGACY_DEV_LABELS", "sha256": {str(p): sha(p) for p in freeze_files},
             "legacy_dev_labels_opened": False, "sealed_test_opened": False, "full_cost_gate": "PENDING"})
    check_hashes(read(frozen_path)["sha256"])
    legacy_report = None
    if legacy_stage1 is not None:
        # This is deliberately the first and only legacy truth read in the pipeline.
        truth_path = legacy_stage1 / "dev_truth.sealed.json"
        legacy_records = read(legacy_grid / "records.json")
        legacy_complete = read(legacy_grid / "complete.json")
        if legacy_complete.get("records_sha256") != sha(legacy_grid / "records.json"):
            raise ValueError("Legacy grid completion hash mismatch")
        legacy_truth = read_legacy_truth(truth_path, output)
        legacy_ids = [r["id"] for r in manifests["legacy_dev"]]
        legacy_protocol_hash = legacy_complete.get("protocol_sha256")
        if not isinstance(legacy_protocol_hash, str):
            raise ValueError("Missing legacy grid protocol fingerprint")
        legacy_scored = score_grid(legacy_records, legacy_truth, legacy_ids, protocol["receiver_sha256"], legacy_protocol_hash, legacy=True)
        legacy_rows = [features[identity] for identity in legacy_ids]
        report_path = output / "legacy_dev_report.json"
        legacy_provenance = {"controller_sha256": sha(frozen_path), "legacy_truth_sha256": sha(truth_path),
                             "legacy_records_sha256": sha(legacy_grid / "records.json")}
        if report_path.exists():
            legacy_report = read(report_path)
            if legacy_report["provenance"] != legacy_provenance:
                raise ValueError("Legacy development result provenance changed")
        else:
            legacy_report, _ = evaluate_split(legacy_rows, legacy_scored, models, scaler, selection, protocol["screen"])
            legacy_report.update(provenance=legacy_provenance, scope="reused historical development; exploratory only; no selection on this split")
            save(report_path, legacy_report)
    result = {"state": "COMPLETE", "jobs": 24, "train_images": 480, "validation_images": 120,
              "validation_preliminary_pass": validation_report["gate"]["preliminary_pass"],
              "legacy_dev_preliminary_pass": legacy_report["gate"]["preliminary_pass"] if legacy_report else None,
              "full_cost_gate": "PENDING", "test_or_snr_started": False, "controller_sha256": sha(frozen_path)}
    save(output / "training_complete.json", result)
    save(output / "training_status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--legacy-stage1", type=Path)
    parser.add_argument("--legacy-grid", type=Path)
    args = parser.parse_args()
    result = run(args.output.resolve(), args.protocol.resolve(),
                 args.legacy_stage1.resolve() if args.legacy_stage1 else None,
                 args.legacy_grid.resolve() if args.legacy_grid else None)
    print(result)


if __name__ == "__main__":
    main()
