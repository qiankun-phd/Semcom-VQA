"""Export frozen EXP-012 CPU forward scores, without training or policy changes."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import statistics
import sys


VARIANTS = ("joint9", "question_only9", "rate_at_low", "compute_at_4000")
SEEDS = (7, 17, 27)
SPLITS = {"train": 480, "validation": 120, "legacy_dev": 120}


def read(path: Path) -> object:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def checked_fingerprints(expected: dict[str, str]) -> dict[str, str]:
    result = {}
    for name, digest in expected.items():
        path = Path(name).resolve()
        if any(re.search(r"(^|[^a-z])test([^a-z]|$)", p.lower()) for p in path.parts):
            raise ValueError("Sealed-test paths are prohibited, including hash reads")
        result[str(path)] = sha(path)
        if result[str(path)] != digest:
            raise ValueError(f"Frozen source hash mismatch: {path}")
    return result


def first_layer_contributions(source: Path, destination: Path) -> dict:
    """Read-only decomposition of existing first-layer preactivations, not an intervention."""
    import torch

    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Diagnostics must not write under the frozen source run")
    export = read(destination / "export_manifest.json")
    before = checked_fingerprints(export["source_sha256_after"])
    torch.set_num_threads(2)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "code"))
    spec = importlib.util.spec_from_file_location("exp012_first_layer_forward", source / "code" / "train_selector.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    features, policy = read(source / "features.json"), read(source / "policy.json")
    inputs = torch.tensor([frozen.vector(row, policy["scaler"]) for row in features], dtype=torch.float32)

    def describe(values: list[float]) -> dict:
        ordered = sorted(values)
        return {"n": len(values), "mean": statistics.mean(values), "min": ordered[0],
                "p10": ordered[round(.1 * (len(ordered) - 1))], "median": statistics.median(values),
                "p90": ordered[round(.9 * (len(ordered) - 1))], "max": ordered[-1]}

    result = {}
    for seed in SEEDS:
        model = frozen.load_checkpoint(Path(policy["checkpoints"]["joint9"][str(seed)]["path"]), list(range(9)))
        model.eval()
        with torch.inference_mode():
            question = inputs[:, :256] @ model[0].weight[:, :256].T
            image = inputs[:, 256:] @ model[0].weight[:, 256:].T
            qnorm, inorm = question.norm(dim=1), image.norm(dim=1)
            if bool(torch.any(qnorm == 0)):
                raise ValueError("Zero question contribution prevents a finite ratio")
            biasnorm = model[0].bias.norm().item()
        result[str(seed)] = {"bias_l2": biasnorm, "splits": {}}
        for split in SPLITS:
            positions = [i for i, row in enumerate(features) if row["split"] == split]
            result[str(seed)]["splits"][split] = {
                "question_Wx_l2": describe(qnorm[positions].tolist()),
                "image_Wx_l2": describe(inorm[positions].tolist()),
                "image_to_question_l2_ratio": describe((inorm[positions] / qnorm[positions]).tolist()),
                "image_to_question_rms_ratio": describe((inorm[positions] / qnorm[positions]).tolist()),
                "image_share_separate_squared_signals": describe(
                    (inorm[positions].square() / (inorm[positions].square() + qnorm[positions].square())).tolist())}
    after = checked_fingerprints(before)
    payload = {"schema": "exp012-first-layer-contributions-v1", "joint9": result,
               "definition": "First Linear(339,128), q=W[:,:256]x_q; i=W[:,256:]x_i; bias excluded and reported separately. RMS ratios equal L2 ratios because both outputs have width 128.",
               "interpretation_limit": "Descriptive signal sizes only; no causal intervention, retraining, or policy change; nonlinear later layers can change relative importance.",
               "source_sha256_before": before, "source_sha256_after": after, "source_unchanged": before == after,
               "script_sha256": sha(Path(__file__).resolve()), "training_performed": False,
               "sealed_test_opened": False, "device": "cpu", "cpu_threads": 2}
    save_new(destination / "first_layer_contributions.json", payload)
    return {"first_layer_contributions": result, "source_unchanged": before == after}


def run(source: Path, destination: Path) -> dict:
    import torch

    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Diagnostics must not write under the frozen source run")
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in ("scored_inputs.json", "export_manifest.json")):
        raise ValueError("Export already exists; do not overwrite frozen diagnostic output")
    controller = read(source / "controller_frozen.json")
    training = read(source / "training_frozen_inputs.json")
    if controller["state"] != "FROZEN_BEFORE_LEGACY_DEV_LABELS":
        raise ValueError("Controller is not frozen")
    expected = dict(training["sha256"])
    expected.update(controller["sha256"])
    for path in [source / "controller_frozen.json", source / "legacy_dev_report.json",
                 source / "code" / "train_selector.py", source / "code" / "prepare_data.py"]:
        expected.setdefault(str(path), sha(path))
    before = checked_fingerprints(expected)

    # Import definitions only; do not run the frozen module's training entrypoint.
    # Suppressing bytecode also prevents writes into the frozen code directory.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "code"))
    spec = importlib.util.spec_from_file_location("exp012_frozen_forward", source / "code" / "train_selector.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    policy = read(source / "policy.json")
    if policy["lambda"] != .05 or policy["selection"]["rate_variant"] != "rate_at_low" or policy["selection"]["compute_variant"] != "compute_at_4000":
        raise ValueError("Unexpected registered frozen policy")
    features = read(source / "features.json")
    if dict(Counter(row["split"] for row in features)) != SPLITS or len({row["id"] for row in features}) != 720:
        raise ValueError("Expected exactly the registered 720 unique feature rows")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    output = [{"id": row["id"], "split": row["split"], "probabilities": {}, "selected_actions": {}}
              for row in features]
    actions = {name: frozen.VARIANTS[name] for name in VARIANTS}
    loaded = []
    for name in VARIANTS:
        inputs = torch.tensor([frozen.vector(row, policy["scaler"], name == "question_only9")
                               for row in features], dtype=torch.float32)
        probabilities = {}
        for seed in SEEDS:
            meta = policy["checkpoints"][name][str(seed)]
            checkpoint = Path(meta["path"])
            if sha(checkpoint) != meta["sha256"]:
                raise ValueError("Checkpoint differs from frozen policy")
            model = frozen.load_checkpoint(checkpoint, actions[name])
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            with torch.inference_mode():
                probabilities[str(seed)] = model(inputs).sigmoid()
            loaded.append(str(checkpoint))
        probabilities["ensemble"] = torch.stack(list(probabilities.values())).mean(0)
        for label, values in probabilities.items():
            for target, values_row in zip(output, values.tolist()):
                target["probabilities"].setdefault(name, {})[label] = values_row
                target["selected_actions"].setdefault(name, {})[label] = frozen.choose_action(values_row, actions[name])

    indexed = {row["id"]: row for row in output}
    checks = []
    for split in ("validation", "legacy_dev"):
        report = read(source / f"{split}_report.json")
        for name in VARIANTS:
            for label in (*map(str, SEEDS), "ensemble"):
                previous = report["variants"][name][label]["selected"]
                if len(previous) != SPLITS[split]:
                    raise ValueError("Report length differs from registered split")
                mismatches = [row["id"] for row in previous
                              if indexed[row["id"]]["split"] != split or
                              indexed[row["id"]]["selected_actions"][name][label] != row["action"]]
                if mismatches:
                    raise ValueError(f"Frozen report decision mismatch: {split}/{name}/{label}: {mismatches}")
                checks.append({"split": split, "variant": name, "replicate": label,
                               "checked": len(previous), "mismatches": 0})
    after = checked_fingerprints(before)
    if before != after:
        raise ValueError("Source files changed during forward export")
    payload = {"schema": "exp012-frozen-forward-scores-v1", "actions": actions,
               "action_descriptors": [frozen.action_descriptor(a) for a in range(9)],
               "probability_semantics": "independent sigmoid correctness heads; not a softmax distribution",
               "rows": output}
    save_new(destination / "scored_inputs.json", payload)
    manifest = {"schema": "exp012-frozen-forward-manifest-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_run": str(source), "source_sha256_before": before, "source_sha256_after": after,
                "source_unchanged": before == after, "export_script_sha256": sha(Path(__file__).resolve()),
                "scored_inputs_sha256": sha(destination / "scored_inputs.json"), "split_counts": SPLITS,
                "torch_version": torch.__version__, "python_version": platform.python_version(),
                "device": "cpu", "cpu_threads": 2, "model_mode": "eval", "autograd_mode": "inference_mode",
                "loaded_checkpoints": loaded, "verified_checkpoints": len(policy["checkpoints"]) * len(SEEDS),
                "batch_rows": 720, "report_action_checks": checks,
                "training_performed": False, "policy_modified": False, "vlm_inference_performed": False,
                "codec_run": False, "sealed_test_opened": False, "labels_used_for_forward": False,
                "diagnostic_only": True, "lambda_probe": [0, .05],
                "restrictions": "No refit, retuning, threshold or lambda search, test access, or policy update"}
    save_new(destination / "export_manifest.json", manifest)
    return {"rows": len(output), "report_action_checks": sum(r["checked"] for r in checks),
            "source_unchanged": True, "loaded_checkpoints": len(loaded)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--first-layer-only", action="store_true")
    arguments = parser.parse_args()
    action = first_layer_contributions if arguments.first_layer_only else run
    print(json.dumps(action(arguments.source, arguments.destination)))
