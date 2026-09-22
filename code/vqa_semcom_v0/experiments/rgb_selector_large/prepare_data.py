"""EXP-014 deterministic data freeze and staged downloads; no model inference.

Training data use COCO train2014; validation and a newly sealed test use
COCO val2014. Historical test truth and image files are never opened.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
from typing import Any
import urllib.request

TASKS = ("object_presence", "counting", "color", "positional_reasoning", "scene_recognition", "activity_recognition")
SEED = 20260923
COUNTS = {"train": 800, "validation": 200, "test": 400}
SOURCE_SPLITS = {"train": "train2014", "validation": "val2014", "test": "val2014"}
SKIP_DIRS = {"venv", ".venv", "site-packages", "deps", "source", "node_modules", ".git", "weights", "__pycache__", "runs", "decoded", "models", "streams"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value: Any, *, immutable: bool = False, private: bool = False) -> None:
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(f"Immutable artifact differs: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if private:
        temporary.chmod(0o600)
    temporary.replace(path)


def check_hashes(hashes: dict[str, str], *, label_free: bool = False) -> None:
    for name, expected in hashes.items():
        path = Path(name)
        if label_free and "truth" in path.name.lower():
            raise ValueError("Truth file in a label-free verification inventory")
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Frozen dependency changed or missing: {path.name}")


def load_legacy(source_code: Path, name: str = "prepare_data") -> Any:
    sys.dont_write_bytecode = True
    path = source_code / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"exp012_readonly_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def order_key(domain: str, value: Any, seed: int = SEED) -> str:
    return hashlib.sha256(f"exp014|{domain}|{seed}|{value}".encode()).hexdigest()


def validate_protocol(protocol: dict) -> None:
    expected = {"train": 4800, "validation": 1200, "test": 2400}
    if protocol.get("experiment_id") != "EXP-014" or protocol.get("seed") != SEED:
        raise ValueError("Unexpected EXP-014 protocol/seed")
    data = protocol["data"]
    if (data["tasks"] != list(TASKS) or data["splits"] != expected or data["per_type"] != COUNTS
            or data["source"] != SOURCE_SPLITS or data["nested_train_per_type"] != 80):
        raise ValueError("Dataset source, quotas, or nested control changed")


def exclusion_sources(project: Path, stage1: Path, output: Path, legacy: Any) -> list[Path]:
    found = set(legacy.exclusion_sources(project, stage1, output))
    required = [project / "outputs/rgb_joint_selector_20260922" / f"{s}_manifest.json" for s in ("train", "validation")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing EXP-012 exclusion: {path}")
        found.add(path)
    for folder, dirs, files in os.walk(project / "outputs"):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and d != "images" and
                         (Path(folder) / d).resolve() != output.resolve())
        for name in files:
            lower = name.lower()
            if lower.endswith(".json") and "exclusion" in lower and "audit" in lower and not any(
                    token in lower for token in ("truth", "prediction", "label")):
                found.add(Path(folder) / name)
    return sorted(found)


def collect_exclusions(sources: list[Path], legacy: Any) -> set[int]:
    excluded = set()
    for path in sources:
        value = read(path)
        excluded.update(legacy.image_ids(value))
        if isinstance(value, dict):
            for key in ("historical_image_ids", "excluded_image_ids"):
                excluded.update(int(v) for v in value.get(key, []))
    return excluded


def eligible_questions(annotations: list[dict], questions: dict[int, dict], excluded: set[int],
                       source: str, tasks: tuple[str, ...] = TASKS) -> dict[str, dict[int, dict]]:
    """Keep one deterministic eligible question per (type,image), after exclusion."""
    candidates: dict[str, dict[int, dict]] = {task: {} for task in tasks}
    for annotation in annotations:
        task, identity = annotation["question_type"], int(annotation["image_id"])
        if task not in candidates or not 0 < identity < 80000000 or identity in excluded:
            continue
        # Do not touch answer fields of excluded images.
        answers = {" ".join(str(item["answer"]).lower().split()) for item in annotation["answers"]}
        if len(answers) != 1 or not next(iter(answers)):
            continue
        qid = int(annotation["question_id"])
        question = questions[qid]
        if int(question["image_id"]) != identity or not str(question["question"]).strip():
            raise ValueError("Question/annotation identity disagreement or empty question")
        row = {"image_id": identity, "question_id": qid, "question_type": task,
               "question": question["question"], "answer": next(iter(answers))}
        previous = candidates[task].get(identity)
        if previous is None or order_key(f"question:{source}", qid) < order_key(f"question:{source}", previous["question_id"]):
            candidates[task][identity] = row
    return candidates


def capacity_report(candidates: dict[str, dict[int, dict]], quota: int) -> dict:
    tasks = tuple(candidates)
    checks = []
    for size in range(1, len(tasks) + 1):
        for subset in itertools.combinations(tasks, size):
            available = len(set().union(*(set(candidates[task]) for task in subset)))
            checks.append({"types": list(subset), "available_unique_images": available,
                           "required": size * quota, "slack": available - size * quota,
                           "equal_quota_capacity": available // size})
    return {"quota_per_type": quota, "available_per_type": {task: len(rows) for task, rows in candidates.items()},
            "all_types_union": len(set().union(*(set(rows) for rows in candidates.values()))),
            "jointly_feasible": all(row["slack"] >= 0 for row in checks),
            "hall_subset_count": len(checks), "minimum_slack": min(checks, key=lambda row: row["slack"]),
            "maximum_equal_quota": min(row["equal_quota_capacity"] for row in checks)}


def allocate_images(candidates: dict[str, dict[int, dict]], quota: int, source: str) -> dict[str, list[int]]:
    """Exact capacitated matching through the at-most-63 image membership masks."""
    capacity = capacity_report(candidates, quota)
    if not capacity["jointly_feasible"]:
        raise ValueError("Insufficient independent images; no resampling: " + json.dumps(capacity))
    tasks = tuple(candidates)
    memberships: dict[int, int] = defaultdict(int)
    for index, task in enumerate(tasks):
        for identity in candidates[task]:
            memberships[identity] |= 1 << index
    bins: dict[int, list[int]] = defaultdict(list)
    for identity, mask in memberships.items():
        bins[mask].append(identity)
    masks = sorted(bins)
    source_node, sink = 0, 1 + len(tasks) + len(masks)
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def edge(start: int, end: int, cap: int) -> list[int]:
        forward = [end, len(graph[end]), cap, cap]
        backward = [start, len(graph[start]), 0, 0]
        graph[start].append(forward)
        graph[end].append(backward)
        return forward

    allocation_edges = {}
    for index, task in enumerate(tasks):
        edge(source_node, 1 + index, quota)
        for j, mask in enumerate(masks):
            if mask & (1 << index):
                allocation_edges[task, mask] = edge(1 + index, 1 + len(tasks) + j, quota)
    for j, mask in enumerate(masks):
        edge(1 + len(tasks) + j, sink, len(bins[mask]))
    total = 0
    while True:
        levels = [-1] * len(graph)
        levels[0] = 0
        queue = deque([0])
        while queue:
            node = queue.popleft()
            for target, _, cap, _ in graph[node]:
                if cap and levels[target] < 0:
                    levels[target] = levels[node] + 1
                    queue.append(target)
        if levels[sink] < 0:
            break
        cursors = [0] * len(graph)

        def push(node: int, amount: int) -> int:
            if node == sink:
                return amount
            while cursors[node] < len(graph[node]):
                item = graph[node][cursors[node]]
                target, reverse, cap, _ = item
                if cap and levels[target] == levels[node] + 1:
                    sent = push(target, min(amount, cap))
                    if sent:
                        item[2] -= sent
                        graph[target][reverse][2] += sent
                        return sent
                cursors[node] += 1
            return 0

        while sent := push(0, quota * len(tasks)):
            total += sent
    if total != quota * len(tasks):
        raise ValueError("Exact image allocation failed despite capacity inventory")
    chosen = {task: [] for task in tasks}
    for mask in masks:
        available = sorted(bins[mask], key=lambda identity: order_key(f"image:{source}", identity))
        cursor = 0
        for task in tasks:
            item = allocation_edges.get((task, mask))
            count = item[3] - item[2] if item else 0
            chosen[task].extend(available[cursor:cursor + count])
            cursor += count
    return chosen


def select_source(candidates: dict[str, dict[int, dict]], source: str, split_counts: dict[str, int],
                  output: Path) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    selected = allocate_images(candidates, sum(split_counts.values()), source)
    manifests = {split: [] for split in split_counts}
    truths = {split: [] for split in split_counts}
    for task, identities in selected.items():
        identities.sort(key=lambda identity: order_key(f"split:{source}:{task}", identity))
        cursor = 0
        for split, count in split_counts.items():
            for iid in identities[cursor:cursor + count]:
                item = candidates[task][iid]
                filename = f"COCO_{source}_{iid:012d}.jpg"
                identity = f"large-selector-{split}-{item['question_id']}"
                manifests[split].append({key: item[key] for key in ("image_id", "question_id", "question_type", "question")} |
                                       {"id": identity, "source_split": source, "split": split,
                                        "file": str(output / "images" / source / filename),
                                        "url": f"https://s3.amazonaws.com/images.cocodataset.org/{source}/{filename}"})
                truths[split].append({"id": identity, "answer": item["answer"]})
            cursor += count
    for split in manifests:
        random.Random(SEED).shuffle(manifests[split])
    return manifests, truths


def nested_rows(train: list[dict], truth: list[dict], per_type: int = 80) -> tuple[list[dict], list[dict]]:
    selected = []
    for task in TASKS:
        eligible = sorted((row for row in train if row["question_type"] == task), key=lambda row: order_key("nested", row["id"]))
        if len(eligible) < per_type:
            raise ValueError("Nested subset exceeds the large training inventory")
        selected.extend(eligible[:per_type])
    random.Random(SEED).shuffle(selected)
    answers = {row["id"]: row for row in truth}
    return selected, [answers[row["id"]] for row in selected]


def verify_controller(output: Path, protocol_path: Path) -> dict:
    """Only a complete frozen 96-fit controller permits new test image access."""
    path = output / "controller_frozen.json"
    if not path.is_file():
        raise ValueError("Controller must be frozen before new test image access")
    controller = read(path)
    if (controller.get("state") != "FROZEN_BEFORE_TEST" or controller.get("completed_fits") != 96
            or controller.get("test_labels_opened") is not False or controller.get("test_features_opened") is not False
            or controller.get("protocol_sha256") != sha(protocol_path)):
        raise ValueError("Controller is not the complete pre-test freeze")
    if not controller.get("evaluation_code_sha256") or not controller.get("selection"):
        raise ValueError("All comparisons and evaluation code must freeze before test")
    hashes = controller.get("sha256", {})
    if not hashes or any("test_truth" in Path(name).name for name in hashes):
        raise ValueError("Invalid pre-test controller input hash inventory")
    check_hashes(hashes)
    check_hashes(controller["evaluation_code_sha256"])
    checkpoints = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "path" in value and "sha256" in value and isinstance(value["sha256"], str):
                checkpoints.append(value)
            else:
                for item in value.values():
                    visit(item)

    visit(controller.get("checkpoints", {}))
    if len(checkpoints) != 96:
        raise ValueError("Expected exactly 96 frozen checkpoints")
    check_hashes({row["path"]: row["sha256"] for row in checkpoints})
    if controller.get("frozen_data_sha256") != sha(output / "frozen_data.json"):
        raise ValueError("Controller pins a different prepared dataset")
    return controller


def validate_image(value: Any) -> None:
    from PIL import Image
    with Image.open(value) as image:
        if image.format != "JPEG" or min(image.size) <= 0:
            raise ValueError("Expected nonempty COCO JPEG")
        image.verify()


def cached_images(project: Path, rows: list[dict], output: Path) -> dict[tuple[str, int], Path]:
    """Inventory filenames only; never open an excluded/historical test JPEG."""
    wanted = {(row["source_split"], int(row["image_id"])) for row in rows}
    result = {}
    pattern = re.compile(r"COCO_(train2014|val2014)_(\d{12})\.jpg$")
    for base in (project / "data", project / "outputs"):
        for folder, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and (Path(folder) / d).resolve() != output.resolve())
            for name in sorted(files):
                match = pattern.fullmatch(name)
                if match:
                    key = match[1], int(match[2])
                    if key in wanted:
                        result.setdefault(key, Path(folder) / name)
    return result


def fetch_image(row: dict, expected: str | None, reuse: Path | None, attempts: int) -> tuple[str, str]:
    path = Path(row["file"])
    source, iid = row["source_split"], int(row["image_id"])
    filename = f"COCO_{source}_{iid:012d}.jpg"
    url = f"https://s3.amazonaws.com/images.cocodataset.org/{source}/{filename}"
    if source not in ("train2014", "val2014") or row["url"] != url or path.name != filename:
        raise ValueError("Unregistered image identity or download URL")
    if path.exists():
        validate_image(path)
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError("Cached image bytes changed")
        return digest, "verified_existing"
    if expected is not None:
        raise ValueError("Previously journaled image disappeared")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    if reuse is not None:
        validate_image(reuse)
        shutil.copyfile(reuse, temporary)
        temporary.replace(path)
        return sha(path), "reused_same_image_id"
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                blob = response.read(MAX_IMAGE_BYTES + 1)
            if not 1000 < len(blob) <= MAX_IMAGE_BYTES:
                raise ValueError("Invalid image response size")
            validate_image(io.BytesIO(blob))
            temporary.write_bytes(blob)
            temporary.replace(path)
            return sha(path), "downloaded"
        except (OSError, ValueError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("No configured download attempts")


def download_phase(project: Path, output: Path, rows: list[dict], phase: str, protocol: dict,
                   forbidden_hashes: set[str]) -> dict[str, str]:
    reuse = cached_images(project, rows, output)
    journal_dir = output / f"image_download_records_{phase}"
    known = {}
    for row in rows:
        path = journal_dir / f"{row['id']}.json"
        if path.exists():
            record = read(path)
            if record["id"] != row["id"] or record["file"] != row["file"]:
                raise ValueError("Download journal identity mismatch")
            known[row["id"]] = record["sha256"]
    hashes, owners, methods = {}, {}, Counter()

    def fetch(row: dict) -> tuple[str, str]:
        return fetch_image(row, known.get(row["id"]), reuse.get((row["source_split"], row["image_id"])),
                           protocol["resources"]["download_attempts_per_image"])

    with ThreadPoolExecutor(max_workers=protocol["resources"]["download_workers"]) as pool:
        for index, (row, (digest, method)) in enumerate(zip(rows, pool.map(fetch, rows)), 1):
            if digest in forbidden_hashes or digest in owners:
                raise ValueError("Exact content duplicate detected; frozen selection must not be replaced")
            hashes[row["id"]] = digest
            owners[digest] = row["id"]
            methods[method] += 1
            path = journal_dir / f"{row['id']}.json"
            if not path.exists():
                save(path, {"id": row["id"], "file": row["file"], "sha256": digest, "method": method}, immutable=True)
            if index % 25 == 0 or index == len(rows):
                save(output / f"data_status_{phase}.json", {"state": "DOWNLOADING", "done": index, "total": len(rows), "methods": dict(methods)})
    return hashes


def verify_data(output: Path, protocol_path: Path, phase: str = "trainval", *, verify_images: bool = False) -> dict:
    frozen = read(output / ("frozen_data.json" if phase == "trainval" else "test_data_frozen.json"))
    if frozen.get("state") != "COMPLETE" or frozen["protocol_sha256"] != sha(protocol_path):
        raise ValueError("Incomplete or changed frozen dataset")
    check_hashes(frozen["sha256"], label_free=True)
    if phase == "test":
        verify_controller(output, protocol_path)
    if verify_images:
        hashes = read(output / ("image_hashes.json" if phase == "trainval" else "test_image_hashes.json"))
        for split in (("train", "validation") if phase == "trainval" else ("test",)):
            for row in read(output / f"{split}_manifest.json"):
                if sha(Path(row["file"])) != hashes[row["id"]]:
                    raise ValueError("Frozen source image changed")
    return frozen


def prepare(project: Path, stage1: Path, output: Path, protocol_path: Path, source_code: Path,
            phase: str = "trainval") -> dict:
    protocol = read(protocol_path)
    validate_protocol(protocol)
    if phase not in ("trainval", "test"):
        raise ValueError("Unsupported preparation phase")
    if phase == "test":
        verify_controller(output, protocol_path)  # Before test manifest/JPEG access.
        verify_data(output, protocol_path)
        if (output / "test_data_frozen.json").exists():
            return verify_data(output, protocol_path, "test", verify_images=True)
        selection = read(output / "selection_frozen.json")
        test_manifest = output / "test_manifest.json"
        if selection["sha256"].get(str(test_manifest)) != sha(test_manifest):
            raise ValueError("New test manifest changed after selection freeze")
        rows = read(test_manifest)
        forbidden = set(read(output / "historical_available_image_hashes.json").values()) | set(read(output / "image_hashes.json").values())
        hashes = download_phase(project, output, rows, "test", protocol, forbidden)
        save(output / "test_image_hashes.json", hashes, immutable=True)
        paths = [test_manifest, output / "test_image_hashes.json", output / "controller_frozen.json", output / "frozen_data.json", output / "selection_frozen.json"]
        frozen = {"state": "COMPLETE", "test": len(rows), "protocol_sha256": sha(protocol_path),
                  "controller_sha256": sha(output / "controller_frozen.json"), "test_labels_opened": False,
                  "sha256": {str(path): sha(path) for path in paths}}
        save(output / "test_data_frozen.json", frozen, immutable=True)
        save(output / "data_status_test.json", {"state": "COMPLETE", "done": len(rows), "total": len(rows)})
        return frozen
    if (output / "frozen_data.json").exists():
        return verify_data(output, protocol_path, verify_images=True)
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < protocol["resources"]["minimum_free_disk_gb"] * 1024 ** 3:
        raise ValueError("Insufficient free disk for the registered preparation")
    legacy = load_legacy(source_code)
    sources = exclusion_sources(project, stage1, output, legacy)
    dependencies = [*sources, protocol_path, Path(__file__).resolve(), source_code / "prepare_data.py"]
    dataset_paths = {}
    for source in ("train2014", "val2014"):
        dataset_paths[source] = (project / f"data/raw/tdiuc/TDIUC/Annotations/mscoco_{source}_annotations.json",
                                 project / f"data/raw/tdiuc/TDIUC/Questions/OpenEnded_mscoco_{source}_questions.json")
        dependencies.extend(dataset_paths[source])
    inputs = {"schema_version": 1, "sha256": {str(path): sha(path) for path in dependencies},
              "old_test_truth_or_images_opened": False}
    save(output / "frozen_inputs.json", inputs, immutable=True)
    excluded = collect_exclusions(sources, legacy)
    forbidden_test_ids = legacy.image_ids(read(stage1 / "test_manifest.json"))
    historical_files = set()
    safe_manifests = [project / "outputs/tdiuc_qlora_pilot_20260916/train.json", stage1 / "train_manifest.json", stage1 / "dev_manifest.json",
                      project / "outputs/rgb_joint_selector_20260922/train_manifest.json", project / "outputs/rgb_joint_selector_20260922/validation_manifest.json"]
    for path in safe_manifests:
        value = read(path)
        if legacy.image_ids(value) & forbidden_test_ids:
            raise ValueError("Historical image hash sources overlap old sealed test IDs")
        historical_files.update(legacy.image_files(value))
    historical_hashes = {str(path): sha(path) for path in sorted(historical_files)}
    save(output / "historical_available_image_hashes.json", historical_hashes, immutable=True)
    selection_path = output / "selection_frozen.json"
    if not selection_path.exists():
        manifests, truths, capacities = {}, {}, {}
        for source, paths in dataset_paths.items():
            annotations = read(paths[0])["annotations"]
            questions = {int(row["question_id"]): row for row in read(paths[1])["questions"]}
            candidates = eligible_questions(annotations, questions, excluded, source)
            split_counts = {split: count for split, count in COUNTS.items() if SOURCE_SPLITS[split] == source}
            capacities[source] = capacity_report(candidates, sum(split_counts.values()))
            save(output / "capacity_report.json", {"state": "INVENTORY_BEFORE_SELECTION", "excluded_images": len(excluded),
                 "exclusion_sources": len(sources), "sources": capacities, "initial_readonly_inventory_did_not_select_or_download": True})
            selected, answers = select_source(candidates, source, split_counts, output)
            manifests.update(selected)
            truths.update(answers)
        rows = [row for values in manifests.values() for row in values]
        identities = [row["image_id"] for row in rows]
        if len(identities) != 8400 or len(set(identities)) != 8400 or set(identities) & excluded:
            raise ValueError("Global 8400-image disjointness/exclusion invariant failed")
        nested, nested_truth = nested_rows(manifests["train"], truths["train"])
        for split in COUNTS:
            save(output / f"{split}_manifest.json", manifests[split], immutable=True)
            filename = "test_truth.sealed.json" if split == "test" else f"{split}_truth.json"
            save(output / filename, truths[split], immutable=True, private=split == "test")
        save(output / "nested_train_manifest.json", nested, immutable=True)
        save(output / "nested_train_truth.json", nested_truth, immutable=True)
        manifest_paths = [output / f"{split}_manifest.json" for split in COUNTS] + [output / "nested_train_manifest.json"]
        truth_paths = [output / filename for filename in ("train_truth.json", "validation_truth.json", "nested_train_truth.json")]
        selection = {"schema_version": 1, "seed": SEED, "sha256": {str(path): sha(path) for path in manifest_paths},
                     "truth_sha256": {path.name: sha(path) for path in truth_paths},
                     "sealed_test_truth_sha256": sha(output / "test_truth.sealed.json"),
                     "protocol_sha256": sha(protocol_path), "selection_uses_model_outputs": False}
        save(selection_path, selection, immutable=True)
    selection = read(selection_path)
    check_hashes(selection["sha256"], label_free=True)
    for filename, expected in selection["truth_sha256"].items():
        if sha(output / filename) != expected:
            raise ValueError("Training/validation truth changed")
    # The sealed test truth is not read again during preparation/resumption.
    save(output / "exclusion_audit.json", {"historical_image_ids": sorted(excluded),
         "historical_sources": {str(path): sha(path) for path in sources}, "selected_overlap": 0,
         "all_selected_images": 8400, "old_test300_images": len(forbidden_test_ids),
         "old_test_truth_or_images_opened": False, "test_images_downloaded": False,
         "scope": "Enumerable project exposure; no foundation-pretraining decontamination claim"}, immutable=True)
    trainval_rows = [row for split in ("train", "validation") for row in read(output / f"{split}_manifest.json")]
    hashes = download_phase(project, output, trainval_rows, "trainval", protocol, set(historical_hashes.values()))
    save(output / "image_hashes.json", hashes, immutable=True)
    paths = [output / name for name in ("frozen_inputs.json", "historical_available_image_hashes.json", "selection_frozen.json", "capacity_report.json",
             "exclusion_audit.json", "image_hashes.json", "train_manifest.json", "validation_manifest.json", "nested_train_manifest.json")]
    frozen = {"schema_version": 1, "state": "COMPLETE", "seed": SEED, "train": 4800, "validation": 1200, "test": 2400,
              "nested_train": 480, "protocol_sha256": sha(protocol_path), "sha256": {str(path): sha(path) for path in paths},
              "truth_sha256": selection["truth_sha256"], "sealed_test_truth_sha256": selection["sealed_test_truth_sha256"],
              "test_manifest_sha256": selection["sha256"][str(output / "test_manifest.json")],
              "test_labels_opened_by_training": False, "test_images_downloaded": False}
    save(output / "frozen_data.json", frozen, immutable=True)
    save(output / "data_status_trainval.json", {"state": "COMPLETE", "done": 6000, "total": 6000})
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-code", type=Path, required=True)
    parser.add_argument("--phase", choices=("trainval", "test"), default="trainval")
    args = parser.parse_args()
    try:
        result = prepare(args.project_root.resolve(), args.stage1_root.resolve(), args.output.resolve(),
                         args.protocol.resolve(), args.source_code.resolve(), args.phase)
    except Exception as error:
        if args.output.is_dir():
            save(args.output / f"data_status_{args.phase}.json", {"state": "FAILED", "error": str(error), "automatic_retry": False})
        raise
    print(json.dumps({"state": result["state"], "phase": args.phase, "output": str(args.output)}))


if __name__ == "__main__":
    main()
