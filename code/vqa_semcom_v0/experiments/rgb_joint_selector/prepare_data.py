"""Freeze fresh TDIUC selector data without inspecting sealed test labels.

Run on the experiment server. Source annotations are used only after image-ID
exclusion. Sampling is deterministic and never depends on receiver predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time
from typing import Any
import urllib.request

TASKS = ("object_presence", "counting", "color", "positional_reasoning",
         "scene_recognition", "activity_recognition")
SEED = 20260922
SPLITS = {"train": 80, "validation": 20}
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_hashes(hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        if not Path(name).is_file() or sha(Path(name)) != expected:
            raise ValueError(f"Frozen dependency changed or missing: {name}")


def image_ids(value: Any) -> set[int]:
    """Read identity fields only; do not access answer fields."""
    if isinstance(value, dict):
        result = {int(value["image_id"])} if "image_id" in value else set()
        for item in value.values():
            if isinstance(item, (dict, list)):
                result.update(image_ids(item))
        return result
    if isinstance(value, list):
        return set().union(*(image_ids(item) for item in value)) if value else set()
    return set()


def image_files(value: Any) -> set[Path]:
    if isinstance(value, dict):
        result = set()
        if "image_id" in value:
            for key in ("file", "image_path", "image_file"):
                item = value.get(key)
                if isinstance(item, str) and Path(item).is_file():
                    result.add(Path(item).resolve())
        for item in value.values():
            if isinstance(item, (dict, list)):
                result.update(image_files(item))
        return result
    if isinstance(value, list):
        return set().union(*(image_files(item) for item in value)) if value else set()
    return set()


def exclusion_sources(project: Path, stage1: Path, output: Path) -> list[Path]:
    """Enumerate manifest identities, explicitly excluding label/result files."""
    required = [project / "outputs/tdiuc_qlora_pilot_20260916/train.json",
                stage1 / "train_manifest.json", stage1 / "dev_manifest.json",
                stage1 / "test_manifest.json", stage1 / "exclusion_audit.json"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required exclusion source missing: {path}")
    found = set(required)
    skip = {"venv", ".venv", "site-packages", "deps", "source", "node_modules", "images",
            "streams", ".git", "weights", "__pycache__", "runs", "decoded", "models"}
    for folder, dirs, files in os.walk(project / "outputs"):
        dirs[:] = sorted(d for d in dirs if d not in skip and
                         (Path(folder) / d).resolve() != output.resolve())
        for name in sorted(files):
            lower = name.lower()
            if lower.endswith(".json") and "manifest" in lower and not any(
                    token in lower for token in ("truth", "prediction", "score", "label")):
                found.add(Path(folder) / name)
    return sorted(found)


def normalized_single_answer(annotation: dict[str, Any]) -> str | None:
    answers = {" ".join(str(item["answer"]).lower().strip().split())
               for item in annotation["answers"]}
    if len(answers) != 1:
        return None
    answer = next(iter(answers))
    return answer or None


def select_rows(annotations: list[dict], questions: dict[int, dict], excluded: set[int],
                output: Path, per_type: dict[str, int] | None = None) -> tuple[dict, dict]:
    """ID-seeded order + agreed single-answer rule; exclusion precedes answers."""
    per_type = SPLITS if per_type is None else per_type
    used = set(excluded)
    manifests, truths = {}, {}
    for split, count in per_type.items():
        rows, truth = [], []
        for task in TASKS:
            candidates = [a for a in annotations if a["question_type"] == task and
                          0 < int(a["image_id"]) < 80000000 and int(a["image_id"]) not in used]
            candidates.sort(key=lambda a: hashlib.sha256(
                f"joint-selector-{split}|{SEED}|{int(a['question_id'])}".encode()).hexdigest())
            selected = 0
            for annotation in candidates:
                iid, qid = int(annotation["image_id"]), int(annotation["question_id"])
                if iid in used:
                    continue
                answer = normalized_single_answer(annotation)
                if answer is None:
                    continue
                question = questions[qid]
                if int(question["image_id"]) != iid:
                    raise ValueError(f"Question/annotation identity disagreement: {qid}")
                filename = f"COCO_val2014_{iid:012d}.jpg"
                row = {"id": f"joint-selector-{split}-{qid}", "image_id": iid,
                       "question_id": qid, "question_type": task, "question": question["question"],
                       "file": str(output / "images" / filename),
                       "url": f"https://s3.amazonaws.com/images.cocodataset.org/val2014/{filename}"}
                rows.append(row)
                truth.append({"id": row["id"], "answer": answer})
                used.add(iid)
                selected += 1
                if selected == count:
                    break
            if selected != count:
                raise ValueError(f"Insufficient fresh eligible data: {split}/{task}: {selected}/{count}")
        random.Random(SEED).shuffle(rows)
        manifests[split], truths[split] = rows, truth
    return manifests, truths


def validate_image(path_or_buffer: Any) -> None:
    from PIL import Image
    with Image.open(path_or_buffer) as image:
        if image.format != "JPEG" or min(image.size) <= 0:
            raise ValueError("Expected a nonempty COCO JPEG")
        image.verify()


def fetch_image(row: dict, expected_hash: str | None = None) -> str:
    path = Path(row["file"])
    url = f"https://s3.amazonaws.com/images.cocodataset.org/val2014/COCO_val2014_{row['image_id']:012d}.jpg"
    if row["url"] != url:
        raise ValueError("Unregistered download URL")
    if path.exists():
        validate_image(path)
        actual = sha(path)
        if expected_hash is not None and actual != expected_hash:
            raise ValueError(f"Cached source image changed: {path}")
        return actual
    if expected_hash is not None:
        raise ValueError(f"Frozen source image disappeared: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                blob = response.read(MAX_IMAGE_BYTES + 1)
            if not 1000 < len(blob) <= MAX_IMAGE_BYTES:
                raise ValueError("Invalid image response size")
            validate_image(io.BytesIO(blob))
            part = path.with_suffix(".part")
            part.write_bytes(blob)
            part.replace(path)
            return sha(path)
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable download state")


def verify_data(output: Path, protocol: Path) -> dict:
    frozen = read(output / "frozen_data.json")
    if frozen["protocol_sha256"] != sha(protocol):
        raise ValueError("Protocol changed after dataset freeze")
    check_hashes(frozen["sha256"])
    check_hashes(read(output / "frozen_inputs.json")["sha256"])
    hashes = read(output / "image_hashes.json")
    for split in SPLITS:
        rows = read(output / f"{split}_manifest.json")
        if Counter(r["question_type"] for r in rows) != Counter({t: SPLITS[split] for t in TASKS}):
            raise ValueError("Frozen split balance changed")
        for row in rows:
            if sha(Path(row["file"])) != hashes[row["id"]]:
                raise ValueError(f"Frozen image changed: {row['id']}")
    return frozen


def prepare(project: Path, stage1: Path, protocol: Path, output: Path) -> None:
    if (output / "frozen_data.json").exists():
        verify_data(output, protocol)
        return
    output.mkdir(parents=True, exist_ok=True)
    sources = exclusion_sources(project, stage1, output)
    annotations_path = project / "data/raw/tdiuc/TDIUC/Annotations/mscoco_val2014_annotations.json"
    questions_path = project / "data/raw/tdiuc/TDIUC/Questions/OpenEnded_mscoco_val2014_questions.json"
    dependencies = sources + [annotations_path, questions_path, protocol, Path(__file__).resolve()]
    snapshot = {str(path.resolve()): sha(path) for path in dependencies}
    inputs_path = output / "frozen_inputs.json"
    if inputs_path.exists():
        if read(inputs_path)["sha256"] != snapshot:
            raise ValueError("Source/exclusion/protocol/code inventory changed during preparation")
    else:
        save(inputs_path, {"sha256": snapshot, "sealed_test_truth_opened": False})
    excluded, available_files = set(), set()
    test_manifest = stage1 / "test_manifest.json"
    for path in sources:
        value = read(path)
        excluded.update(image_ids(value))
        if path.name == "exclusion_audit.json":
            excluded.update(int(v) for v in value.get("historical_image_ids", []))
        # Only already-available Qwen training/dev image files are hashed.
        if path in (project / "outputs/tdiuc_qlora_pilot_20260916/train.json",
                    stage1 / "train_manifest.json", stage1 / "dev_manifest.json"):
            available_files.update(image_files(value))
        if path == test_manifest:
            continue  # Identity exclusion only; no test image loading/downloading.
    historical_hashes = {str(path): sha(path) for path in sorted(available_files)}
    historical_path = output / "historical_available_image_hashes.json"
    if historical_path.exists() and read(historical_path) != historical_hashes:
        raise ValueError("Available historical images changed during preparation")
    save(historical_path, historical_hashes)
    if not (output / "selection_frozen.json").exists():
        annotations = read(annotations_path)["annotations"]
        questions = {int(q["question_id"]): q for q in read(questions_path)["questions"]}
        manifests, truths = select_rows(annotations, questions, excluded, output)
        paths = []
        for split in SPLITS:
            for kind, value in (("manifest", manifests[split]), ("truth", truths[split])):
                path = output / f"{split}_{kind}.json"
                if path.exists() and read(path) != value:
                    raise ValueError(f"Partial selection changed: {path}")
                save(path, value)
                paths.append(path)
        save(output / "selection_frozen.json", {"sha256": {str(p): sha(p) for p in paths}})
    check_hashes(read(output / "selection_frozen.json")["sha256"])
    rows = sum((read(output / f"{s}_manifest.json") for s in SPLITS), [])
    ids = [int(r["image_id"]) for r in rows]
    if len(ids) != 600 or len(set(ids)) != 600 or set(ids) & excluded:
        raise ValueError("Data exclusion/uniqueness invariant failed")
    save(output / "exclusion_audit.json", {"historical_image_ids": sorted(excluded),
         "historical_sources": {str(p): snapshot[str(p.resolve())] for p in sources},
         "selected_overlap": 0, "sealed_test_truth_opened": False,
         "scope": "Enumerable project manifests and available Qwen train/dev file hashes; no base-model pretraining claim",
         "selection": "SHA256 seeded question-ID order after image-ID exclusion; one unique lowercased/whitespace-normalized nonempty answer"})
    partial = output / "image_hashes.partial.json"
    hashes = read(partial) if partial.exists() else {}
    historical_digest_set = set(historical_hashes.values())
    with ThreadPoolExecutor(max_workers=6) as pool:
        for count, (row, digest) in enumerate(zip(rows, pool.map(
                lambda r: fetch_image(r, hashes.get(r["id"])), rows)), 1):
            if digest in historical_digest_set:
                raise ValueError(f"Exact image duplicate with previous training/development: {row['id']}")
            if digest in {v for key, v in hashes.items() if key != row["id"]}:
                raise ValueError(f"Exact source-image duplicate within new dataset: {row['id']}")
            hashes[row["id"]] = digest
            save(partial, hashes)
            if count % 20 == 0 or count == len(rows):
                save(output / "data_status.json", {"state": "DOWNLOADING", "done": count, "total": len(rows)})
    save(output / "image_hashes.json", hashes)
    frozen_files = [inputs_path, historical_path, output / "selection_frozen.json",
                    output / "image_hashes.json", output / "exclusion_audit.json"]
    frozen_files += [output / f"{s}_{kind}.json" for s in SPLITS for kind in ("manifest", "truth")]
    save(output / "frozen_data.json", {"state": "COMPLETE", "seed": SEED, "train": 480,
         "validation": 120, "protocol_sha256": sha(protocol), "test_labels_opened": False,
         "test_images_downloaded": False, "sha256": {str(p): sha(p) for p in frozen_files}})
    verify_data(output, protocol)
    save(output / "data_status.json", {"state": "COMPLETE", "done": 600, "total": 600})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.project_root.resolve(), args.stage1_root.resolve(), args.protocol.resolve(), args.output.resolve())
    print(json.dumps({"state": "COMPLETE", "train": 480, "validation": 120, "output": str(args.output)}))


if __name__ == "__main__":
    main()
