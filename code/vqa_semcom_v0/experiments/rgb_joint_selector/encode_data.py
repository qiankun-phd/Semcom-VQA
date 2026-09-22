"""Encode the frozen fresh selector dataset with the audited ordinary codec.

Raw legacy packets remain unchanged. The selector's one-byte route identifier is
counted separately for every arm; it is not silently added inside codec packets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
from pathlib import Path
import sys
import time
from typing import Any

try:
    from .prepare_data import TASKS, SPLITS, SEED, check_hashes, read, save, sha, verify_data
except ImportError:
    from prepare_data import TASKS, SPLITS, SEED, check_hashes, read, save, sha, verify_data

BUDGETS = (2000, 4000, 8000)
ROUTE_BYTES = 1


def smoke_rows(rows: list[dict]) -> list[dict]:
    return [min((row for row in rows if row["question_type"] == task),
                key=lambda row: hashlib.sha256(f"{SEED}|{task}|{row['id']}".encode()).hexdigest())
            for task in sorted(TASKS)]


def load_stage1(stage1: Path) -> Any:
    path = stage1 / "run_stage1.py"
    spec = importlib.util.spec_from_file_location("joint_selector_audited_stage1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load audited codec context: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dependency_snapshot(stage1: Path, protocol: Path, output: Path) -> dict:
    outputs = stage1.parent.resolve()
    paths = {Path(__file__).resolve(), Path(__file__).with_name("prepare_data.py").resolve(),
             protocol.resolve(), output / "frozen_data.json", stage1 / "run_stage1.py"}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.is_file() and path.suffix == ".py" and outputs in path.parents and not any(
                name in path.parts for name in ("venv", ".venv", "site-packages")):
            paths.add(path)
    weight = outputs / "receiver_reconstruction_demo_20260917/weights/bmshj2018-hyperprior-3-6d87be32.pth.tar"
    if not weight.is_file():
        raise FileNotFoundError(f"Required audited codec checkpoint absent: {weight}")
    paths.add(weight)
    return {"sha256": {str(p): sha(p) for p in sorted(paths)},
            "runtime": {name: importlib.metadata.version(name) for name in ("torch", "compressai", "Pillow")},
            "codec": "bmshj2018-hyperprior-q3-source-coordinate-quant-v2-uniform-level8",
            "side": 320, "gain_search_iterations": 12, "raw_caps_bytes": list(BUDGETS),
            "route_bytes": ROUTE_BYTES, "threads": 4}


def publish_bytes(path: Path, payload: bytes) -> None:
    """Recover an orphan only when a deterministic replay exactly matches it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing unregistered/registered artifact differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def validate_record(record: dict) -> None:
    if not record.get("roundtrip") or record["budget"] not in BUDGETS:
        raise ValueError("Unregistered codec representation")
    if not 0 < record["codec_image_bytes"] <= record["budget"]:
        raise ValueError("Raw codec budget exceeded")
    if record["image_bytes"] != record["codec_image_bytes"] + ROUTE_BYTES:
        raise ValueError("One-byte route overhead missing or counted twice")
    packet, decoded = Path(record["packet"]), Path(record["decoded"])
    if packet.stat().st_size != record["codec_image_bytes"]:
        raise ValueError("Raw packet size disagrees with metadata")
    if sha(packet) != record["packet_sha256"] or sha(decoded) != record["decoded_sha256"]:
        raise ValueError("Codec artifact hash mismatch")
    if min(record["encode_seconds"], record["decode_seconds"]) < 0:
        raise ValueError("Negative runtime")


def recover_records(output: Path, rows: list[dict]) -> list[dict]:
    expected = {(row["id"], budget) for row in rows for budget in BUDGETS}
    mapped: dict[tuple[str, int], dict] = {}
    existing = read(output / "representations.json") if (output / "representations.json").exists() else []
    for record in existing:
        key = record["id"], record["budget"]
        if key in mapped:
            raise ValueError("Duplicate codec representation")
        mapped[key] = record
    for path in sorted((output / "encoding_records").glob("*.json")):
        record = read(path)
        key = record["id"], record["budget"]
        if key in mapped and mapped[key] != record:
            raise ValueError("Aggregate and per-record journals disagree")
        mapped[key] = record
    if not set(mapped) <= expected:
        raise ValueError("Encoding record outside frozen data/configuration grid")
    for record in mapped.values():
        validate_record(record)
    return [mapped[(row["id"], budget)] for row in rows for budget in BUDGETS
            if (row["id"], budget) in mapped]


def encode(stage1: Path, protocol: Path, output: Path, smoke_only: bool) -> None:
    verify_data(output, protocol)
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    old = load_stage1(stage1)
    region, _ = old.context()  # Never old.verify(): it reads sealed-test truth dependencies.
    codec = region.alignment.parent.train.load_codec("cpu")
    codec.eval()
    snapshot = dependency_snapshot(stage1, protocol, output)
    snapshot_path = output / "codec_frozen_inputs.json"
    if snapshot_path.exists():
        if read(snapshot_path) != snapshot:
            raise ValueError("Codec/protocol/runtime dependency changed after encoding freeze")
        check_hashes(read(snapshot_path)["sha256"])
    else:
        save(snapshot_path, snapshot)
    all_rows = sum((read(output / f"{s}_manifest.json") for s in SPLITS), [])
    target_rows = smoke_rows(read(output / "train_manifest.json")) if smoke_only else all_rows
    records = recover_records(output, all_rows)
    seen = {(r["id"], r["budget"]): r for r in records}
    completion = output / "encoding_complete.json"
    if completion.exists():
        done = read(completion)
        if sha(output / "representations.json") != done["representations_sha256"] or len(records) != 1800:
            raise ValueError("Completed encoding manifest changed")
        return
    started = time.monotonic()
    newly_encoded = 0
    for index, row in enumerate(target_rows, 1):
        if all((row["id"], budget) in seen for budget in BUDGETS):
            continue
        preprocess_start = time.perf_counter()
        with Image.open(row["file"]) as original:
            source_hw = [original.height, original.width]
            tensor, hw = region.alignment.parent.train.image_tensor(original.convert("RGB"), 320, device="cpu")
        with torch.inference_mode():
            latent = codec.g_a(tensor)
        preprocess_seconds = time.perf_counter() - preprocess_start
        padded_hw = tuple(tensor.shape[-2:])
        for budget in BUDGETS:
            if (row["id"], budget) in seen:
                continue
            indices = torch.full((5, 5), 8, dtype=torch.int64)
            encode_start = time.perf_counter()
            with torch.inference_mode():
                selected = region.alignment.old.select_global_gain(
                    lambda gain: region.alignment.encode_latent(
                        codec, latent, hw, padded_hw, indices, gain,
                        region.alignment.parent.wire, aligned=True), budget, iterations=12)
            encode_search_seconds = time.perf_counter() - encode_start
            payload = selected.payload
            if payload is None or not 0 < len(payload) <= budget:
                raise ValueError(f"Could not encode registered budget {row['id']}/{budget}")
            decode_start = time.perf_counter()
            with torch.inference_mode():
                rgb, header = region.alignment.decode(codec, payload, region.alignment.parent.wire)
                image = region.alignment.parent.canonical_reconstruction_image(rgb, header["image_hw"])
            decode_seconds = time.perf_counter() - decode_start
            pixels_sha256 = region.alignment.parent.reconstruction_sha256(image)
            with torch.inference_mode():
                repeat_rgb, repeat_header = region.alignment.decode(codec, payload, region.alignment.parent.wire)
                repeat_image = region.alignment.parent.canonical_reconstruction_image(repeat_rgb, repeat_header["image_hw"])
            if pixels_sha256 != region.alignment.parent.reconstruction_sha256(repeat_image):
                raise ValueError("Packet-only roundtrip was not deterministic")
            folder = output / "decoded" / row["id"]
            packet, png = folder / f"{budget}.bin", folder / f"{budget}.png"
            orphan_replayed = packet.exists() or png.exists()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            publish_bytes(packet, payload)
            publish_bytes(png, buffer.getvalue())
            record = {"id": row["id"], "budget": budget, "arm": "ordinary_uniform",
                      "codec_image_bytes": len(payload), "route_bytes": ROUTE_BYTES,
                      "image_bytes": len(payload) + ROUTE_BYTES, "packet": str(packet),
                      "packet_sha256": sha(packet), "decoded": str(png), "decoded_sha256": sha(png),
                      "pixels_sha256": pixels_sha256, "source_hw": source_hw, "image_hw": list(hw),
                      "padded_hw": list(padded_hw), "codec_side": 320, "roundtrip": True,
                      "global_gain": selected.gain, "encode_seconds": preprocess_seconds + encode_search_seconds,
                      "preprocess_and_g_a_seconds": preprocess_seconds, "entropy_gain_search_seconds": encode_search_seconds,
                      "decode_seconds": decode_seconds, "orphan_deterministic_replay": orphan_replayed,
                      "timing_scope": "encode=source JPEG load+RGB+resize/pad+g_a+12-step gain search; decode=packet parsing+entropy decoding+g_s+canonical RGB; excludes artifact writes and verification replay",
                      "shared_g_a_cost_charged_per_candidate": True, "device": "cpu", "cpu_threads": 4}
            validate_record(record)
            save(output / "encoding_records" / f"{row['id']}-{budget}.json", record)
            seen[(row["id"], budget)] = record
            records = [seen[(r["id"], b)] for r in all_rows for b in BUDGETS if (r["id"], b) in seen]
            save(output / "representations.json", records)
            newly_encoded += 1
        save(output / "encoding_status.json", {"state": "SMOKE_ENCODING" if smoke_only else "ENCODING",
             "images_done": index, "images_target": len(target_rows), "representations": len(records),
             "elapsed_seconds": time.monotonic() - started})
    elapsed = time.monotonic() - started
    summary = {"state": "SMOKE_COMPLETE" if smoke_only else "COMPLETE", "representations": len(records),
               "newly_encoded": newly_encoded, "elapsed_seconds": elapsed,
               "representations_sha256": sha(output / "representations.json"),
               "estimated_remaining_seconds": (1800 - len(records)) * elapsed / newly_encoded if newly_encoded else None,
               "raw_caps_bytes": list(BUDGETS), "route_bytes_per_arm": ROUTE_BYTES,
               "test_opened": False, "all_packet_only_roundtrip": True}
    if not smoke_only:
        if len(records) != 1800:
            raise ValueError("Incomplete full codec grid")
        save(completion, summary)
    else:
        summary["smoke_ids"] = [r["id"] for r in target_rows]
        save(output / "encoding_smoke_complete.json", summary)
    save(output / "encoding_status.json", summary)
    print({key: summary[key] for key in ("state", "representations", "elapsed_seconds", "estimated_remaining_seconds")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke6", action="store_true")
    args = parser.parse_args()
    encode(args.stage1_root.resolve(), args.protocol.resolve(), args.output.resolve(), args.smoke6)


if __name__ == "__main__":
    main()
