"""Cheap, pre-decision features: no VLM, labels, category metadata or answer cues."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

QUESTION_DIM = 256
IMAGE_DIM = 83


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def question_features(question: str) -> list[float]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A nonempty question string is required")
    words = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", question.lower())
    phrases = ["u:" + word for word in words]
    phrases.extend("b:" + left + " " + right for left, right in zip(words, words[1:]))
    vector = [0.0] * QUESTION_DIM
    for phrase in phrases:
        code = hashlib.blake2b(phrase.encode(), digest_size=8, person=b"exp012q").digest()
        vector[int.from_bytes(code, "little") % QUESTION_DIM] += 1.0
    length = math.sqrt(sum(value * value for value in vector))
    return [value / length for value in vector] if length else vector


def image_features(path: Path) -> list[float]:
    """Pure image statistics; all operations are available before source encoding."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    if min(width, height) <= 0:
        raise ValueError("Image has no pixels")
    tiny = image.copy()
    tiny.thumbnail((64, 64), Image.Resampling.BILINEAR)
    rgb = np.asarray(tiny, dtype=np.float32) / 255.0
    gray = np.asarray(tiny.convert("L"), dtype=np.float32) / 255.0
    grid = np.asarray(tiny.convert("L").resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    histogram = np.histogram(gray, bins=8, range=(0, 1))[0].astype(float)
    histogram /= histogram.sum()
    dx = float(np.abs(np.diff(gray, axis=1)).mean()) if gray.shape[1] > 1 else 0.0
    dy = float(np.abs(np.diff(gray, axis=0)).mean()) if gray.shape[0] > 1 else 0.0
    vector = [*grid.reshape(-1).tolist(), *rgb.mean(axis=(0, 1)).tolist(),
              *rgb.std(axis=(0, 1)).tolist(), *histogram.tolist(), dx, dy,
              math.log1p(width), math.log1p(height), width / height]
    if len(vector) != IMAGE_DIM or not all(math.isfinite(value) for value in vector):
        raise ValueError("Invalid image feature vector")
    return vector


def extract(question: str, image_path: Path) -> dict[str, Any]:
    """Only these two fields enter feature extraction; no question-type label."""
    start = time.perf_counter()
    text = question_features(question)
    text_seconds = time.perf_counter() - start
    start = time.perf_counter()
    visual = image_features(image_path)
    return {"question_features": text, "image_features": visual,
            "question_feature_seconds": text_seconds,
            "image_feature_seconds": time.perf_counter() - start}


def build(output: Path, protocol_path: Path, legacy: Path | None) -> None:
    frozen = read(output / "frozen_data.json")
    if frozen.get("state") != "COMPLETE" or frozen["protocol_sha256"] != sha(protocol_path):
        raise ValueError("Features require completed, matching frozen data")
    protocol = read(protocol_path)
    if protocol["experiment_id"] != "EXP-012":
        raise ValueError("Unexpected feature contract")
    paths = [output / "train_manifest.json", output / "validation_manifest.json",
             output / "image_hashes.json"]
    for path in paths:
        if frozen["sha256"].get(str(path)) != sha(path):
            raise ValueError("Fresh manifest or source hash inventory changed")
    image_hashes = read(output / "image_hashes.json")
    rows = [{**row, "split": split} for split in ("train", "validation")
            for row in read(output / f"{split}_manifest.json")]
    if legacy is not None:
        legacy_manifest, legacy_hashes = legacy / "dev_manifest.json", legacy / "image_hashes.json"
        paths.extend([legacy_manifest, legacy_hashes])
        historical = read(legacy_hashes)
        for row in read(legacy_manifest):
            if not row["id"].startswith("joint-dev-"):
                raise ValueError("Only legacy DEVELOPMENT image features may be cached")
            rows.append({**row, "split": "legacy_dev"})
            image_hashes[row["id"]] = historical[row["id"]]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Feature IDs are not unique")
    fingerprint = {"sha256": {str(path): sha(path) for path in
                   [*paths, protocol_path, Path(__file__).resolve(), output / "frozen_data.json"]},
                   "schema": "hashed-question256-full-view-image83", "legacy_labels_loaded": False}
    snapshot = output / "features_frozen.json"
    if snapshot.exists() and read(snapshot) != fingerprint:
        raise ValueError("Feature code/data/protocol changed; refusing stale feature reuse")
    save(snapshot, fingerprint)
    target = output / "features.json"
    records = read(target) if target.exists() else []
    by_id = {record["id"]: record for record in records}
    if len(by_id) != len(records) or not set(by_id) <= {row["id"] for row in rows}:
        raise ValueError("Duplicate or unknown cached features")
    # Import libraries before timing individual image operations.
    import numpy  # noqa: F401
    from PIL import Image  # noqa: F401

    for index, row in enumerate(rows):
        source = Path(row["file"])
        if sha(source) != image_hashes[row["id"]]:
            raise ValueError("Original image differs from audited bytes")
        if row["id"] in by_id:
            previous = by_id[row["id"]]
            if previous.get("source_image_sha256") != image_hashes[row["id"]] or previous["split"] != row["split"]:
                raise ValueError("Cached features have wrong image provenance")
            continue
        record = {"id": row["id"], "split": row["split"],
                  "source_image_sha256": image_hashes[row["id"]],
                  **extract(row["question"], source)}
        records.append(record)
        by_id[row["id"]] = record
        save(target, records)
        if (index + 1) % 60 == 0:
            save(output / "features_status.json", {"state": "EXTRACTING", "done": len(records), "total": len(rows)})
    save(output / "features_complete.json", {"state": "COMPLETE", "records": len(records),
         "features_sha256": sha(target), "frozen_sha256": sha(snapshot),
         "question_dim": QUESTION_DIM, "image_dim": IMAGE_DIM, "labels_loaded": False,
         "timing_scope": "question hashing; image JPEG load+RGB+thumbnail/statistics; no neural inference"})
    print(json.dumps({"state": "FEATURES_COMPLETE", "records": len(records)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--legacy-stage1", type=Path)
    args = parser.parse_args()
    build(args.output.resolve(), args.protocol.resolve(), args.legacy_stage1.resolve() if args.legacy_stage1 else None)
