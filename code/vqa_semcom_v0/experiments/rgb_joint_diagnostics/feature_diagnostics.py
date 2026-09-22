"""Describe frozen feature scales and question overlap, without fitting models."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def read(path: Path) -> object:
    return json.loads(path.read_text())


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def distribution(values: np.ndarray) -> dict:
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("A finite nonempty distribution is required")
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.,
            "min": float(values.min()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def describe(rows: list[dict], scaler: dict) -> dict:
    q = np.asarray([r["question_features"] for r in rows], dtype=float)
    image = np.asarray([r["image_features"] for r in rows], dtype=float)
    mean, std = np.asarray(scaler["mean"]), np.asarray(scaler["std"])
    if q.shape != (len(rows), 256) or image.shape != (len(rows), 83):
        raise ValueError("Unexpected frozen feature dimensions")
    if not np.isfinite(q).all() or not np.isfinite(image).all() or not np.all(std > 0):
        raise ValueError("Invalid inputs or scaler")
    z = (image - mean) / std
    q_power = np.square(q).sum(axis=1)
    image_power = np.square(z).sum(axis=1)
    if np.any(q_power <= 0):
        raise ValueError("Empty question vector")
    return {"n": len(rows), "question_l2": distribution(np.sqrt(q_power)),
            "standardized_image_l2": distribution(np.sqrt(image_power)),
            "image_over_question_l2": distribution(np.sqrt(image_power / q_power)),
            "image_power_share": distribution(image_power / (image_power + q_power)),
            "image_abs_z_max": float(np.abs(z).max()),
            "images_with_any_abs_z_above_5": int(np.any(np.abs(z) > 5, axis=1).sum()),
            "question_nonzero_bins": distribution((q != 0).sum(axis=1)),
            "zero_variance_scaler_dimensions": int((std <= 1e-6).sum())}


def question_audit(rows: list[dict], train_questions: set[str]) -> dict:
    normalized = [" ".join(row["question"].lower().split()) for row in rows]
    collision_fractions = []
    for row in rows:
        words = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", row["question"].lower())
        terms = set(["u:" + w for w in words] + ["b:" + a + " " + b for a, b in zip(words, words[1:])])
        bins = {int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8, person=b"exp012q").digest(), "little") % 256 for t in terms}
        collision_fractions.append(1 - len(bins) / len(terms) if terms else 0.)
    return {"n": len(rows), "unique_normalized_question_strings": len(set(normalized)),
            "questions_exactly_seen_in_train": sum(q in train_questions for q in normalized),
            "within_question_hash_collision_fraction": distribution(np.asarray(collision_fractions)),
            "question_types_for_analysis_only": dict(Counter(row["question_type"] for row in rows))}


def run(root: Path, output: Path) -> dict:
    features = read(root / "features.json")
    policy = read(root / "policy.json")
    train = [r for r in features if r["split"] == "train"]
    scaler = policy["scaler"]
    raw = np.asarray([r["image_features"] for r in train])
    assert len(train) == scaler["n"] == 480 and scaler["fit_split"] == "train"
    np.testing.assert_allclose(raw.mean(axis=0), scaler["mean"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.maximum(raw.std(axis=0), 1e-6), scaler["std"], rtol=1e-9, atol=1e-9)
    manifests = {split: read(root / file) for split, file in (
        ("train", "train_manifest.json"), ("validation", "validation_manifest.json"), ("legacy_dev", "legacy_dev_manifest.json"))}
    question_set = {" ".join(r["question"].lower().split()) for r in manifests["train"]}
    result = {"scaler_reproduced_from_train_only": True, "splits": {}, "training_history": {},
              "limitation": "Input power imbalance is descriptive, not measured neuron influence or proof of a causal failure. No rescaling/ablation fitting was done."}
    for split, manifest in manifests.items():
        rows = [r for r in features if r["split"] == split]
        assert {r["id"] for r in rows} == {r["id"] for r in manifest}
        result["splits"][split] = {**describe(rows, scaler), "question_audit": question_audit(manifest, question_set)}
    histories = read(root / "training_history.json")
    for variant in ("joint9", "question_only9", "rate_at_low", "compute_at_4000"):
        result["training_history"][variant] = {}
        for seed, job in histories[variant].items():
            start, end = job["history"][0], job["history"][-1]
            result["training_history"][variant][seed] = {
                "epochs": job["epochs_run"], "best_epoch": job["best_epoch"], "best_validation_bce": job["best_validation_bce"],
                "train_bce_first": start["train_bce"], "train_bce_last": end["train_bce"],
                "validation_bce_first": start["validation_bce"], "validation_bce_last": end["validation_bce"]}
    files = ["features.json", "policy.json", "training_history.json", "train_manifest.json", "validation_manifest.json", "legacy_dev_manifest.json"]
    result["source_sha256"] = {file: hashlib.sha256((root / file).read_bytes()).hexdigest() for file in files}
    save(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.output)
    print(json.dumps({"scaler_ok": result["scaler_reproduced_from_train_only"], "splits": {k: {"n": v["n"], "image_power_share": v["image_power_share"]["mean"]} for k, v in result["splits"].items()}}))
