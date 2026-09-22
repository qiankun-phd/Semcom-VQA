"""Produce frozen-Qwen nine-configuration supervision without loading answers.

Only the fresh training and internal-validation manifests are accepted. The
historical stage-1 module supplies a model loader, not data or verification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

from route_frame import pack as pack_frame, unpack as unpack_frame


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ordered_rows(manifests: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    """Deterministic, balanced train-first order; explicitly reject answer fields."""
    rows: list[dict[str, Any]] = []
    identifiers, images = set(), set()
    forbidden = {"answer", "answers", "label", "labels", "correct", "ground_truth"}
    if set(manifests) != {"train", "validation"}:
        raise ValueError("Only fresh train and internal validation may be inferred")
    for split in ("train", "validation"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in manifests[split]:
            identifier = row.get("id", "")
            if not identifier.startswith(f"joint-selector-{split}-"):
                raise ValueError("Historical development and sealed-test rows are forbidden")
            if forbidden.intersection(row):
                raise ValueError("Inference manifests must not contain answer labels")
            if identifier in identifiers or row["image_id"] in images:
                raise ValueError("Training and internal validation require unique images and IDs")
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError("Missing question text")
            if row.get("split", split) != split:
                raise ValueError("Row split conflicts with manifest")
            identifiers.add(identifier)
            images.add(row["image_id"])
            groups[row["question_type"]].append({**row, "split": split})
        if len(groups) != 6 or len({len(group) for group in groups.values()}) != 1:
            raise ValueError("Each split must contain six equally sized question types")
        for name, group in groups.items():
            group.sort(key=lambda row: hashlib.sha256(f"{seed}|{name}|{row['id']}".encode()).hexdigest())
        names = sorted(groups)
        rows.extend(groups[name][index] for index in range(len(groups[names[0]])) for name in names)
    return rows


def verify_packets(records: list[dict[str, Any]], rows: list[dict[str, Any]],
                   budgets: list[int], sha: Any) -> dict[tuple[str, int], dict[str, Any]]:
    """Audit raw codec packet plus the one-byte universal framing accounting."""
    needed = {(row["id"], budget) for row in rows for budget in budgets}
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = record["id"], record["budget"]
        if key not in needed:
            continue
        if key in selected:
            raise ValueError(f"Duplicate representation {key}")
        packet, decoded = Path(record["packet"]), Path(record["decoded"])
        actual = packet.stat().st_size
        if record.get("roundtrip") is not True or actual <= 0:
            raise ValueError(f"Missing verified roundtrip {key}")
        if actual != record["codec_image_bytes"] or record["image_bytes"] != actual + 1:
            raise ValueError(f"Raw packet/framing byte accounting mismatch {key}")
        if record["codec_image_bytes"] > record["budget"] or record["image_bytes"] > record["budget"] + 1:
            raise ValueError(f"Packet exceeds raw codec cap plus universal one-byte framing {key}")
        for path, expected in ((packet, record["packet_sha256"]), (decoded, record["decoded_sha256"])):
            if sha(path) != expected:
                raise ValueError(f"Cached representation changed: {path}")
        selected[key] = record
    if set(selected) != needed:
        raise ValueError(f"Missing {len(needed - set(selected))} selected representations")
    return selected


def verify_frozen_data(frozen: dict[str, Any], protocol_path: Path,
                       manifest_paths: dict[str, Path], rows: list[dict[str, Any]], sha: Any) -> None:
    """Verify frozen label-free inputs, without opening even the truth hash targets."""
    if frozen.get("state") != "COMPLETE" or frozen.get("train") != 480 or frozen.get("validation") != 120:
        raise ValueError("Fresh selector data preparation is not complete")
    if frozen.get("protocol_sha256") != sha(protocol_path):
        raise ValueError("Data and inference protocols differ")
    hashes = frozen["sha256"]
    for path in manifest_paths.values():
        if hashes.get(str(path.resolve())) != sha(path):
            raise ValueError(f"Manifest differs from frozen selection: {path}")
    # Original-image identity is covered by preparation's image_hashes manifest.
    # Do not enumerate/open all frozen paths: some are intentionally sealed labels.
    image_hashes_path = next((path.parent / "image_hashes.json" for path in manifest_paths.values()), None)
    if image_hashes_path is None or hashes.get(str(image_hashes_path.resolve())) != sha(image_hashes_path):
        raise ValueError("Original-image hash inventory differs from frozen data")


def verify_cached_records(records: list[dict[str, Any]], allowed: set[tuple[str, int, str]],
                          representations: dict[tuple[str, int], dict[str, Any]],
                          receiver_hash: str, protocol_hash: str) -> set[tuple[str, int, str]]:
    seen: set[tuple[str, int, str]] = set()
    for record in records:
        key = record["id"], record["budget"], record["tier"]
        if key not in allowed or key in seen:
            raise ValueError("Unexpected or duplicate inference cache entry")
        if record.get("receiver_sha256") != receiver_hash or record.get("protocol_sha256") != protocol_hash:
            raise ValueError("Cached inference model/protocol provenance mismatch")
        representation = representations[(key[0], key[1])]
        for field in ("packet_sha256", "decoded_sha256", "image_bytes", "codec_image_bytes"):
            if record.get(field) != representation[field]:
                raise ValueError(f"Cached inference representation changed: {key}: {field}")
        for field in ("receiver_seconds", "preprocessing_seconds"):
            value = record.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid cached inference timing")
        if not isinstance(record.get("prediction"), str) or record.get("actual_visual_tokens", 0) <= 0:
            raise ValueError("Invalid cached prediction/token measurement")
        seen.add(key)
    return seen


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New data/codec root containing the fresh manifests and representations")
    parser.add_argument("--grid-code", type=Path, required=True,
                        help="EXP-011 code directory containing audited runtime.py")
    parser.add_argument("--smoke", action="store_true", help="First six balanced training images (54 calls)")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    root, output = args.stage1_root.resolve(), args.output.resolve()
    adapter, protocol_path = args.adapter.resolve(), args.protocol.resolve()
    runtime_path = args.grid_code.resolve() / "runtime.py"
    runtime = load_module("joint_selector_grid_runtime", runtime_path)
    read, save, sha = runtime.read_json, runtime.save_json, runtime.sha256
    protocol = read(protocol_path)
    budgets, tiers = protocol["budgets"], protocol["tiers"]
    if budgets != [2000, 4000, 8000] or [(tier["name"], tier["pixels"]) for tier in tiers] != [
            ("low", 50176), ("medium", 100352), ("high", 200704)]:
        raise ValueError("Only the frozen EXP-011 nine configuration interface is supported")
    if protocol["max_new_tokens"] != 16:
        raise ValueError("Receiver generation must remain at 16 maximum new tokens")
    if protocol.get("frame_header_bytes") != 1:
        raise ValueError("The receiver tier requires exactly one universally charged frame byte")
    seed = protocol["seed"]
    manifest_paths = {split: output / f"{split}_manifest.json" for split in ("train", "validation")}
    manifests = {split: read(path) for split, path in manifest_paths.items()}
    if len(manifests["train"]) != 480 or len(manifests["validation"]) != 120:
        raise ValueError("Expected the frozen 480 training / 120 internal-validation images")
    all_rows = ordered_rows(manifests, seed)
    rows = all_rows[:6] if args.smoke else all_rows
    frozen_data_path = output / "frozen_data.json"
    if not frozen_data_path.exists():
        raise ValueError("Fresh data must be frozen before receiver inference")
    frozen_data = read(frozen_data_path)
    verify_frozen_data(frozen_data, protocol_path, manifest_paths, rows, sha)
    original_hashes = read(output / "image_hashes.json")
    for row in rows:
        if sha(Path(row["file"])) != original_hashes[row["id"]]:
            raise ValueError("An original selected image differs from its frozen identity")
    representations_path = output / "representations.json"
    if not args.smoke:
        completion = read(output / "encoding_complete.json")
        if completion.get("representations_sha256") != sha(representations_path):
            raise ValueError("Full inference requires completed, unchanged codec representations")
    adapter_path = adapter / "adapter_model.safetensors"
    receiver_hash, protocol_hash = sha(adapter_path), sha(protocol_path)
    if receiver_hash != protocol["receiver_sha256"]:
        raise ValueError("The receiver is not the frozen standard adapter")
    records_path, frozen_path = output / "supervision_records.json", output / "inference_frozen.json"
    records = read(records_path) if records_path.exists() else []
    # Smoke resume may be requested after a full run: audit every cached packet,
    # while allowing the producer's representations list to grow between runs.
    cached_ids = {record["id"] for record in records}
    audit_rows = [row for row in all_rows if row["id"] in cached_ids or row in rows]
    representations = verify_packets(read(representations_path), audit_rows, budgets, sha)
    stage1 = runtime.load_stage1(root / "run_stage1.py")
    _, qwen = stage1.context()
    files = {Path(__file__).resolve(), Path(__file__).with_name("route_frame.py").resolve(), runtime_path, protocol_path, frozen_data_path,
             *manifest_paths.values(), adapter_path, adapter / "adapter_config.json", root / "run_stage1.py"}
    for module in list(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source:
            path = Path(source).resolve()
            if path.is_file() and path.suffix == ".py" and root.parent in path.parents and not any(
                    part in ("venv", ".venv", "site-packages") for part in path.parts):
                files.add(path)
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "bitsandbytes", "Pillow")}
    fingerprint = {"schema": "joint-selector-supervision-v1", "runtime": versions,
                   "sha256": {str(path): sha(path) for path in sorted(files)},
                   "receiver_sha256": receiver_hash, "protocol_sha256": protocol_hash,
                   "data_frozen": frozen_data}
    if records and not frozen_path.exists():
        raise ValueError("Inference cache has no frozen provenance")
    runtime.freeze_or_verify(frozen_path, fingerprint)
    allowed = {(row["id"], budget, tier["name"]) for row in all_rows for budget in budgets for tier in tiers}
    seen = verify_cached_records(records, allowed, representations, receiver_hash, protocol_hash)
    selected = {(row["id"], budget, tier["name"]) for row in rows for budget in budgets for tier in tiers}
    pending = selected - seen
    started = time.monotonic()

    if pending:
        import torch
        from PIL import Image

        torch.set_num_threads(2)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU benchmark substitution")
        save(output / "inference_status.json", {"stage": "LOADING_RECEIVER", "pending": len(pending)})
        model, processor = stage1.load_receiver(adapter, False)
        random.seed(seed)
        torch.manual_seed(seed)
        runtime.freeze_or_verify(output / "inference_model_runtime.json", {
            "model_config": model.config.to_dict(), "gpu": torch.cuda.get_device_name(),
            "image_processor_class": type(processor.image_processor).__name__,
            "image_processor_initial_config": processor.image_processor.to_dict(),
            "runtime": versions, "torch_num_threads": torch.get_num_threads(),
        })
        image_token_id = getattr(model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)

        def infer(row: dict[str, Any], representation: dict[str, Any], tier: dict[str, Any]) -> dict[str, Any]:
            runtime.set_pixel_target(processor, tier["pixels"])
            torch.cuda.synchronize()
            preprocessing_started = time.monotonic()
            with Image.open(representation["decoded"]) as source:
                image = source.convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image"},
                        {"type": "text", "text": qwen.prompt(row["question"])}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt",
                               min_pixels=tier["pixels"], max_pixels=tier["pixels"])
            measures = runtime.visual_measurements(inputs, processor, int(image_token_id))
            inputs = inputs.to("cuda")
            torch.cuda.synchronize()
            preprocessing_seconds = time.monotonic() - preprocessing_started
            generation_started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=protocol["max_new_tokens"])
            torch.cuda.synchronize()
            receiver_seconds = time.monotonic() - generation_started
            continuation = generated[:, measures["input_tokens"]:]
            prediction = processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()
            return {"prediction": prediction, **measures, "source_image_hw": [image.height, image.width],
                    "generated_tokens": int(continuation.shape[1]), "receiver_seconds": receiver_seconds,
                    "preprocessing_seconds": preprocessing_seconds, "pixel_target": tier["pixels"],
                    "energy_j": None, "energy_reason": "No validated energy measurement; latency is not energy"}

        warmups = []
        for tier in tiers:
            measured = infer(rows[0], representations[(rows[0]["id"], 4000)], tier)
            warmups.append({"tier": tier["name"], "actual_visual_tokens": measured["actual_visual_tokens"],
                            "image_grid_thw": measured["image_grid_thw"], "receiver_seconds": measured["receiver_seconds"]})
        counts = [warmup["actual_visual_tokens"] for warmup in warmups]
        if counts != sorted(set(counts)):
            raise ValueError("Warmups show ineffective visual tier controls")
        save(output / "inference_warmup_latest.json", {"excluded_from_metrics": True, "examples": warmups})
        tier_lookup = {tier["name"]: tier for tier in tiers}
        for index, row in enumerate(rows):
            for budget, tier_name in runtime.configuration_order(budgets, tiers, seed, index):
                key = row["id"], budget, tier_name
                if key in seen:
                    continue
                representation = representations[(row["id"], budget)]
                codec_payload = Path(representation["packet"]).read_bytes()
                frame = pack_frame(codec_payload, tier_name)
                if unpack_frame(frame) != (tier_name, codec_payload) or len(frame) != representation["image_bytes"]:
                    raise ValueError("Routing frame failed byte/tier roundtrip")
                if sha(Path(representation["decoded"])) != representation["decoded_sha256"]:
                    raise ValueError("Decoded RGB changed during inference")
                measured = infer(row, representation, tier_lookup[tier_name])
                record = {"id": row["id"], "image_id": row["image_id"], "split": row["split"],
                          "question_type": row["question_type"], "budget": budget, "tier": tier_name, **measured,
                          "frame_sha256": hashlib.sha256(frame).hexdigest(),
                          **{key: representation[key] for key in ("image_bytes", "codec_image_bytes",
                             "packet_sha256", "decoded_sha256", "encode_seconds", "decode_seconds")},
                          "receiver_sha256": receiver_hash, "protocol_sha256": protocol_hash}
                records.append(record)
                seen.add(key)
                save(records_path, records)
                save(output / "inference_status.json", {"stage": "SUPERVISION_INFERENCE", "done": len(selected & seen),
                     "total": len(selected), "cached_records": len(records), "elapsed_seconds": time.monotonic() - started})

    selected_records = [record for record in records if (record["id"], record["budget"], record["tier"]) in selected]
    if len(selected_records) != len(selected):
        raise RuntimeError("Supervision grid did not complete")
    runtime.verify_tier_separation(selected_records, tiers)
    for path, expected in fingerprint["sha256"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Frozen source/input changed during inference: {path}")
    result = {"state": "SMOKE_COMPLETE" if args.smoke else "SUPERVISION_COMPLETE", "images": len(rows),
              "records": len(selected_records), "cached_records": len(records), "actual_tiers_verified": True,
              "receiver_sha256": receiver_hash, "protocol_sha256": protocol_hash,
              "records_sha256": sha(records_path), "labels_loaded": False, "sealed_test_opened": False,
              "elapsed_seconds_this_invocation": time.monotonic() - started, "energy_j": None,
              "timing_scope": "CUDA-synchronized generation; preprocessing separately; three warmups excluded"}
    save(output / ("inference_smoke_complete.json" if args.smoke else "supervision_complete.json"), result)
    save(output / "inference_status.json", result)
    return result


def main() -> None:
    import json

    args = arguments()
    try:
        result = run(args)
    except Exception as error:
        runtime = load_module("joint_selector_failure_runtime", args.grid_code / "runtime.py")
        runtime.save_json(args.output / "inference_failure.json", {
            "state": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
