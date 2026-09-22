"""Measure a frozen RGB bitrate × Qwen pixel-target grid, development data only.

No labels are loaded; scoring is a separate step. Legacy cached packets remain
unchanged. Three pixel-area targets are NOT exact visual-token caps.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from runtime import (configuration_order, development_rows, freeze_or_verify,
                     load_stage1, read_json, record_key, save_json, set_pixel_target,
                     sha256, validate_protocol, validate_resume,
                     verify_representations, verify_tier_separation, visual_measurements)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True,
                        help="Historical stage 1 directory containing cached dev RGB reconstructions")
    parser.add_argument("--adapter", type=Path, required=True,
                        help="Frozen standard receiver LoRA directory")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Six images, one per question type; 54 measured calls")
    parser.add_argument("--limit", type=int, help="Additional bounded limit; a positive multiple of six")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    root, adapter, output = args.stage1_root.resolve(), args.adapter.resolve(), args.output.resolve()
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    budgets, tiers = validate_protocol(protocol)
    seed = protocol["seed"]
    manifest_path, representations_path = root / "dev_manifest.json", root / "representations.json"
    completion_path = root / "encoding_complete.json"
    if sha256(representations_path) != read_json(completion_path)["representations_sha256"]:
        raise ValueError("Historical representations manifest hash changed")
    all_rows = development_rows(read_json(manifest_path), seed)
    if len(all_rows) != protocol["expected_questions"]:
        raise ValueError("Development question count differs from the frozen protocol")
    count = 6 if args.smoke else len(all_rows)
    if args.limit is not None:
        if args.limit <= 0 or args.limit % 6:
            raise ValueError("--limit must be a positive multiple of six")
        count = min(count, args.limit)
    rows = all_rows[:count]
    representations = verify_representations(read_json(representations_path), all_rows, budgets)
    adapter_path = adapter / "adapter_model.safetensors"
    receiver_hash, protocol_hash = sha256(adapter_path), sha256(protocol_path)
    expected_receiver = protocol["receiver_sha256"]
    if expected_receiver != receiver_hash:
        raise ValueError("Receiver is not the protocol's frozen adapter")
    expected_manifest = protocol["dev_manifest_sha256"]
    if expected_manifest != sha256(manifest_path):
        raise ValueError("Development manifest differs from protocol")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "status.json", {"stage": "VERIFYING_RUNTIME", "selected_images": count})

    stage1 = load_stage1(root / "run_stage1.py")
    _, qwen = stage1.context()
    # Imports above load old helper implementations without invoking stage1.verify(),
    # which would otherwise inspect sealed-test files unnecessarily.
    files = {Path(__file__).resolve(), Path(__file__).with_name("runtime.py").resolve(),
             protocol_path, manifest_path, representations_path, completion_path,
             adapter_path, adapter / "adapter_config.json", root / "run_stage1.py"}
    for module in list(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source:
            path = Path(source).resolve()
            if path.is_file() and path.suffix == ".py" and root.parent in path.parents and not any(
                    part in ("venv", ".venv", "site-packages") for part in path.parts):
                files.add(path)
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "bitsandbytes", "Pillow")}
    fingerprint = {"schema": "rgb-rate-visual-budget-dev-v1", "runtime": versions,
                   "sha256": {str(path): sha256(path) for path in sorted(files)},
                   "receiver_sha256": receiver_hash, "protocol_sha256": protocol_hash}
    frozen_path, records_path = output / "frozen_inputs.json", output / "records.json"
    if records_path.exists() and not frozen_path.exists():
        raise ValueError("Inference cache lacks frozen input provenance; refusing to adopt it")
    freeze_or_verify(frozen_path, fingerprint)
    records = read_json(records_path) if records_path.exists() else []
    allowed = {(r["id"], b, t["name"]) for r in all_rows for b in budgets for t in tiers}
    seen = validate_resume(records, allowed, receiver_hash, protocol_hash)
    selected_keys = {(r["id"], b, t["name"]) for r in rows for b in budgets for t in tiers}
    pending = selected_keys - seen
    started = time.monotonic()

    if pending:
        import torch
        from PIL import Image

        torch.set_num_threads(2)
        random.seed(seed)
        torch.manual_seed(seed)
        save_json(output / "status.json", {"stage": "LOADING_RECEIVER", "pending": len(pending)})
        model, processor = stage1.load_receiver(adapter, False)
        if not torch.cuda.is_available():
            raise RuntimeError("This receiver benchmark requires CUDA; CPU substitution is not allowed")
        # Legacy loader fixes its own seed; restore the frozen grid seed afterward.
        random.seed(seed)
        torch.manual_seed(seed)
        freeze_or_verify(output / "model_runtime.json", {
            "base_model_name_or_path": str(getattr(model.config, "_name_or_path", "")),
            "model_config": model.config.to_dict(),
            "image_processor_class": type(processor.image_processor).__name__,
            "image_processor_initial_config": processor.image_processor.to_dict(),
            "gpu": torch.cuda.get_device_name(), "runtime": versions, "torch_num_threads": torch.get_num_threads(),
        })
        token_id = getattr(model.config, "image_token_id", None)
        if token_id is None:
            token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        image_token_id = int(token_id)
        tier_lookup = {tier["name"]: tier for tier in tiers}

        def infer(row: dict[str, Any], representation: dict[str, Any], tier: dict[str, Any]) -> dict[str, Any]:
            set_pixel_target(processor, tier["pixels"])
            torch.cuda.synchronize()
            preprocessing_started = time.monotonic()
            with Image.open(representation["decoded"]) as source:
                image = source.convert("RGB")
            original_hw = [image.height, image.width]
            messages = [{"role": "user", "content": [{"type": "image"},
                        {"type": "text", "text": qwen.prompt(row["question"])}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt",
                               min_pixels=tier["pixels"], max_pixels=tier["pixels"])
            measures = visual_measurements(inputs, processor, image_token_id)
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
            return {"prediction": prediction, **measures, "source_image_hw": original_hw,
                    "generated_tokens": int(continuation.shape[1]),
                    "receiver_seconds": receiver_seconds, "preprocessing_seconds": preprocessing_seconds,
                    "energy_j": None, "energy_reason": "GPU energy and power sensors unavailable; latency is not energy",
                    "pixel_target": tier["pixels"]}

        warmups = []
        warmup_representation = representations[(rows[0]["id"], 4000)]
        for tier in tiers:
            measurement = infer(rows[0], warmup_representation, tier)
            warmups.append({"tier": tier["name"], "actual_visual_tokens": measurement["actual_visual_tokens"],
                            "image_grid_thw": measurement["image_grid_thw"],
                            "receiver_seconds": measurement["receiver_seconds"]})
        counts = [r["actual_visual_tokens"] for r in warmups]
        if counts != sorted(set(counts)):
            raise ValueError(f"Warmup proves tier controls ineffective: actual token counts {counts}")
        save_json(output / "warmup_latest.json", {"excluded_from_metrics": True, "examples": warmups,
                  "gpu": torch.cuda.get_device_name(), "runtime": versions, "torch_num_threads": torch.get_num_threads()})

        for index, row in enumerate(rows):
            for budget, tier_name in configuration_order(budgets, tiers, seed, index):
                key = row["id"], budget, tier_name
                if key in seen:
                    continue
                representation = representations[(row["id"], budget)]
                # Check just before opening as well as the initial whole-manifest audit.
                if sha256(Path(representation["decoded"])) != representation["decoded_sha256"]:
                    raise ValueError("Decoded RGB changed while inference was running")
                measured = infer(row, representation, tier_lookup[tier_name])
                record = {"id": row["id"], "image_id": row["image_id"], "question_type": row["question_type"],
                          "budget": budget, "tier": tier_name, **measured,
                          "image_bytes": representation["image_bytes"],
                          "packet_sha256": representation["packet_sha256"],
                          "decoded_sha256": representation["decoded_sha256"],
                          "receiver_sha256": receiver_hash, "protocol_sha256": protocol_hash}
                records.append(record)
                seen.add(key)
                save_json(records_path, records)
                save_json(output / "status.json", {"stage": "DEV_INFERENCE", "done": len(seen & selected_keys),
                          "total": len(selected_keys), "elapsed_seconds": time.monotonic() - started})

    selected_records = [record for record in records if record_key(record) in selected_keys]
    if len(selected_records) != len(selected_keys):
        raise RuntimeError("Grid did not complete")
    verify_tier_separation(selected_records, tiers)
    for path, expected in fingerprint["sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Frozen input changed during inference: {path}")
    result = {"state": "SMOKE_COMPLETE" if args.smoke else "DEV_GRID_COMPLETE",
              "images": len(rows), "records": len(selected_records), "cached_records": len(records),
              "receiver_sha256": receiver_hash, "protocol_sha256": protocol_hash,
              "records_sha256": sha256(records_path), "actual_tiers_verified": True,
              "elapsed_seconds_this_invocation": time.monotonic() - started,
              "sealed_test_opened": False, "labels_loaded": False,
              "timing_scope": "CUDA-synchronized model.generate only; preprocessing separately; tier warmups excluded",
              "energy_j": None, "energy_reason": "GPU energy and power sensors unavailable"}
    save_json(output / ("smoke_complete.json" if args.smoke else "complete.json"), result)
    save_json(output / "status.json", result)
    return result


def main() -> None:
    args = arguments()
    try:
        result = run(args)
    except Exception as error:
        save_json(args.output / "failure.json", {"state": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
