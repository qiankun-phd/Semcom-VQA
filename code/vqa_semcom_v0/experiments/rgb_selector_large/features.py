"""Label-free EXP-012 feature functions with EXP-014 staged access and journals."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

if __package__:
    from .prepare_data import check_hashes, load_legacy, read, save, sha, validate_protocol, verify_controller, verify_data
else:
    from prepare_data import check_hashes, load_legacy, read, save, sha, validate_protocol, verify_controller, verify_data


def validate_features(row: dict) -> None:
    for field, width in (("question_features", 256), ("image_features", 83)):
        values = row.get(field)
        if not isinstance(values, list) or len(values) != width or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("Malformed cached feature vector")
    for field in ("question_feature_seconds", "image_feature_seconds"):
        if not isinstance(row.get(field), (int, float)) or not math.isfinite(row[field]) or row[field] < 0:
            raise ValueError("Malformed feature timing")
    if set(row) & {"answer", "question_type", "image_id", "question_id", "prediction"}:
        raise ValueError("Forbidden annotation/answer metadata in feature records")


def build(output: Path, protocol_path: Path, source_code: Path, phase: str = "trainval") -> dict:
    validate_protocol(read(protocol_path))
    if phase not in ("trainval", "test"):
        raise ValueError("Unsupported feature phase")
    if phase == "test":
        verify_controller(output, protocol_path)  # Before test metadata/image access.
    verify_data(output, protocol_path, phase)
    splits = ("train", "validation") if phase == "trainval" else ("test",)
    prefix = "" if phase == "trainval" else "test_"
    image_hash_path = output / ("image_hashes.json" if phase == "trainval" else "test_image_hashes.json")
    paths = [output / f"{split}_manifest.json" for split in splits]
    paths += [image_hash_path, protocol_path, Path(__file__).resolve(), Path(__file__).with_name("prepare_data.py"),
              source_code / "features.py", output / ("frozen_data.json" if phase == "trainval" else "test_data_frozen.json")]
    if phase == "test":
        paths.append(output / "controller_frozen.json")
    frozen = {"schema_version": 1, "phase": phase, "sha256": {str(path): sha(path) for path in paths},
              "schema": "hashed-question256-full-view-image83", "labels_loaded": False}
    snapshot = output / f"{prefix}features_frozen.json"
    save(snapshot, frozen, immutable=True)
    check_hashes(frozen["sha256"], label_free=True)
    target = output / f"{prefix}features.json"
    complete_path = output / f"{prefix}features_complete.json"
    if complete_path.exists():
        complete = read(complete_path)
        if complete["features_sha256"] != sha(target) or complete["frozen_sha256"] != sha(snapshot):
            raise ValueError("Completed feature cache changed")
        return complete
    records, seen = [], set()
    rows = [{**row, "split": split} for split in splits for row in read(output / f"{split}_manifest.json")]
    hashes = read(image_hash_path)
    if len(rows) != (6000 if phase == "trainval" else 2400) or {row["id"] for row in rows} != set(hashes):
        raise ValueError("Feature/image inventory differs from the registered phase")
    legacy = load_legacy(source_code, "features")
    # Resolve imports outside per-image timings.
    import numpy  # noqa: F401
    from PIL import Image  # noqa: F401
    for index, row in enumerate(rows, 1):
        identity = row["id"]
        if identity in seen:
            raise ValueError("Duplicate feature identity")
        seen.add(identity)
        source = Path(row["file"])
        if sha(source) != hashes[identity]:
            raise ValueError("Image changed before feature extraction")
        journal = output / f"feature_records_{phase}" / f"{identity}.json"
        if journal.exists():
            record = read(journal)
            if (record["id"] != identity or record["split"] != row["split"]
                    or record["source_image_sha256"] != hashes[identity]):
                raise ValueError("Feature journal provenance changed")
        else:
            record = {"id": identity, "split": row["split"], "source_image_sha256": hashes[identity],
                      **legacy.extract(row["question"], source)}
            validate_features(record)
            save(journal, record, immutable=True)
        validate_features(record)
        records.append(record)
        if index % 50 == 0 or index == len(rows):
            save(output / f"{prefix}features_status.json", {"state": "EXTRACTING", "done": index, "total": len(rows)})
    save(target, records, immutable=True)
    complete = {"state": "COMPLETE", "records": len(records), "features_sha256": sha(target),
                "frozen_sha256": sha(snapshot), "question_dim": 256, "image_dim": 83, "labels_loaded": False,
                "phase": phase, "test_labels_opened": False,
                "protocol_sha256": sha(protocol_path),
                "controller_sha256": sha(output / "controller_frozen.json") if phase == "test" else None,
                "timing_scope": "Question hashing and JPEG load/statistics; no neural inference"}
    save(complete_path, complete, immutable=True)
    save(output / f"{prefix}features_status.json", {"state": "COMPLETE", "done": len(records), "total": len(rows)})
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-code", type=Path, required=True)
    parser.add_argument("--phase", choices=("trainval", "test"), default="trainval")
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve(), args.protocol.resolve(), args.source_code.resolve(), args.phase)))
