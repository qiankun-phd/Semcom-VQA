"""Dependency-light integrity and measurement helpers for the development grid."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    """Atomic checkpoint replacement; a killed write leaves the old file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage1(path: Path) -> ModuleType:
    # Loading the module does not call prepare/verify/evaluate or inspect labels.
    name = "rgb_visual_budget_stage1"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import stage 1 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_protocol(protocol: dict[str, Any]) -> tuple[list[int], list[dict[str, Any]]]:
    if protocol.get("split") != "dev" or protocol.get("expected_questions") != 120:
        raise ValueError("This experiment accepts only the frozen 120-question development split")
    budgets = protocol.get("budgets", protocol.get("image_caps_bytes"))
    tiers = protocol.get("tiers")
    if budgets != [2000, 4000, 8000]:
        raise ValueError("This bounded development grid requires budgets [2000, 4000, 8000]")
    if not isinstance(tiers, list) or len(tiers) != 3:
        raise ValueError("Exactly three pixel-target tiers are required")
    if len({t["name"] for t in tiers}) != 3:
        raise ValueError("Tier names must be distinct")
    pixels = [t["pixels"] for t in tiers]
    if any(not isinstance(p, int) or isinstance(p, bool) or p <= 0 for p in pixels):
        raise ValueError("Tier pixel targets must be positive integers")
    if pixels != sorted(set(pixels)):
        raise ValueError("Tiers must have strictly increasing pixel targets")
    if not isinstance(protocol.get("seed"), int) or protocol.get("max_new_tokens") != 16:
        raise ValueError("An integer seed and frozen max_new_tokens=16 are required")
    return budgets, tiers


def development_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Return a label-free, type-balanced deterministic order; never accept test IDs."""
    if not rows:
        raise ValueError("Development manifest is empty")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identifiers: set[str] = set()
    image_ids: set[int] = set()
    for row in rows:
        identifier = row.get("id", "")
        if not identifier.startswith("joint-dev-"):
            raise ValueError("Only the historical joint-dev manifest is supported; test input is forbidden")
        if any(key in row for key in ("answer", "answers", "label", "labels", "correct", "ground_truth")):
            raise ValueError("Inference manifest must not contain labels")
        if identifier in identifiers or row["image_id"] in image_ids:
            raise ValueError("Development questions and images must be unique")
        identifiers.add(identifier)
        image_ids.add(row["image_id"])
        groups[row["question_type"]].append(row)
    if len(groups) != 6 or len({len(group) for group in groups.values()}) != 1:
        raise ValueError("Development manifest must contain six equally sized question types")
    for name, group in groups.items():
        group.sort(key=lambda r: hashlib.sha256(f"{seed}|{name}|{r['id']}".encode()).hexdigest())
    names = sorted(groups)
    return [groups[name][i] for i in range(len(groups[names[0]])) for name in names]


def configuration_order(budgets: list[int], tiers: list[dict[str, Any]], seed: int, image_index: int) -> list[tuple[int, str]]:
    """Seeded, counterbalanced Latin rotations avoid tying a tier to warm GPU order."""
    configurations = [(budget, tier["name"]) for budget in budgets for tier in tiers]
    random.Random(seed).shuffle(configurations)
    offset = image_index % len(configurations)
    return configurations[offset:] + configurations[:offset]


def verify_representations(records: list[dict[str, Any]], rows: list[dict[str, Any]], budgets: list[int]) -> dict[tuple[str, int], dict[str, Any]]:
    needed = {(r["id"], b) for r in rows for b in budgets}
    selected = {}
    for record in records:
        key = (record["id"], record["budget"])
        if key not in needed:
            continue
        if key in selected:
            raise ValueError(f"Duplicate cached representation: {key}")
        if record.get("roundtrip") is not True or not 0 < record["image_bytes"] <= record["budget"]:
            raise ValueError(f"Invalid packet budget/roundtrip: {key}")
        packet, decoded = Path(record["packet"]), Path(record["decoded"])
        if packet.stat().st_size != record["image_bytes"]:
            raise ValueError(f"Packet size changed: {key}")
        for path, expected in ((packet, record["packet_sha256"]), (decoded, record["decoded_sha256"])):
            if sha256(path) != expected:
                raise ValueError(f"Cached representation hash mismatch: {path}")
        selected[key] = record
    if set(selected) != needed:
        raise ValueError(f"Missing {len(needed - set(selected))} required cached representations")
    return selected


def set_pixel_target(processor: Any, pixels: int) -> None:
    """Set old and current Transformers controls; count observed tokens separately."""
    image_processor = processor.image_processor
    image_processor.min_pixels = pixels
    image_processor.max_pixels = pixels
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict) and ("shortest_edge" in size or "longest_edge" in size):
        image_processor.size = {**size, "shortest_edge": pixels, "longest_edge": pixels}


def visual_measurements(inputs: Any, processor: Any, image_token_id: int) -> dict[str, Any]:
    grid_value = inputs["image_grid_thw"]
    grids = grid_value.tolist() if hasattr(grid_value, "tolist") else grid_value
    ids_value = inputs["input_ids"]
    ids = ids_value.tolist() if hasattr(ids_value, "tolist") else ids_value
    if len(grids) != 1 or len(grids[0]) != 3 or len(ids) != 1:
        raise ValueError("Only single-image, single-question inference is supported")
    merge = int(processor.image_processor.merge_size)
    raw_patches = math.prod(int(v) for v in grids[0])
    if merge <= 0 or raw_patches % (merge * merge):
        raise ValueError("Visual grid cannot be divided by spatial merge size")
    visual_tokens = raw_patches // (merge * merge)
    placeholders = sum(token == image_token_id for token in ids[0])
    if placeholders != visual_tokens or visual_tokens <= 0:
        raise ValueError(f"Image placeholder/grid mismatch: {placeholders} versus {visual_tokens}")
    patch_size = int(processor.image_processor.patch_size)
    return {"actual_visual_tokens": visual_tokens,
            "image_grid_thw": [int(v) for v in grids[0]],
            "input_tokens": len(ids[0]), "image_placeholder_tokens": placeholders,
            "resized_image_hw": [int(grids[0][1]) * patch_size, int(grids[0][2]) * patch_size],
            "spatial_merge_size": merge}


def verify_tier_separation(records: list[dict[str, Any]], tiers: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    across_budgets: dict[tuple[str, str], set[tuple[int, ...]]] = defaultdict(set)
    for record in records:
        groups[(record["id"], record["budget"])][record["tier"]] = record["actual_visual_tokens"]
        across_budgets[(record["id"], record["tier"])].add(tuple(record.get("image_grid_thw", [record["actual_visual_tokens"]])))
    for key, values in groups.items():
        if len(values) != len(tiers):
            raise ValueError(f"Incomplete three-tier comparison: {key}")
        counts = [values[t["name"]] for t in tiers]
        if counts != sorted(set(counts)):
            raise ValueError(f"Pixel tiers did not produce strictly increasing actual visual tokens: {key}: {counts}")
    if any(len(grids) != 1 for grids in across_budgets.values()):
        raise ValueError("For a given image/tier, visual grid changed across byte budgets")


def freeze_or_verify(path: Path, fingerprint: dict[str, Any]) -> None:
    # JSON normalizes integer dictionary keys and tuples in Transformers configs.
    canonical = json.loads(json.dumps(fingerprint, ensure_ascii=False, allow_nan=False))
    if path.exists():
        if read_json(path) != canonical:
            raise ValueError("Resume refused: protocol, receiver, runtime, code or input hashes changed")
    else:
        save_json(path, canonical)


def record_key(record: dict[str, Any]) -> tuple[str, int, str]:
    return record["id"], record["budget"], record["tier"]


def validate_resume(records: list[dict[str, Any]], allowed: set[tuple[str, int, str]], receiver_hash: str, protocol_hash: str) -> set[tuple[str, int, str]]:
    seen = set()
    for record in records:
        key = record_key(record)
        if key not in allowed or key in seen:
            raise ValueError(f"Unexpected or duplicate cached inference: {key}")
        if record.get("receiver_sha256") != receiver_hash or record.get("protocol_sha256") != protocol_hash:
            raise ValueError("Cached inference provenance mismatch")
        seen.add(key)
    return seen
