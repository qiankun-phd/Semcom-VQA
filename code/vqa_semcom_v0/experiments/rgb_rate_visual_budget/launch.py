"""Bounded smoke -> paired development grid -> analysis; never unseals test."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)
    protocol_hash = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    lock = args.output / "supervisor.lock"
    try:
        with lock.open("x") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        parser.error("supervisor.lock exists; inspect running process before explicit recovery")
    started = time.monotonic()
    status_path = args.output / "supervisor.json"
    common = ["--stage1-root", str(args.stage1_root.resolve()), "--adapter",
              str(args.adapter.resolve()), "--protocol", str(args.protocol.resolve()),
              "--output", str(args.output.resolve())]
    commands = [
        ("SMOKE", [sys.executable, str(HERE / "run_grid.py"), *common, "--smoke"]),
        ("GRID", [sys.executable, str(HERE / "run_grid.py"), *common]),
        ("ANALYSIS", [sys.executable, str(HERE / "analyze_grid.py"),
                      "--records", str(args.output / "records.json"),
                      "--truth", str(args.stage1_root / "dev_truth.sealed.json"),
                      "--protocol", str(args.protocol), "--output", str(args.output / "analysis")]),
    ]
    metadata = {"pid": os.getpid(), "protocol_sha256": protocol_hash,
                "started_unix": time.time(), "test_unsealed": False,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(HERE.glob("*.py"))}}
    try:
        for stage, command in commands:
            remaining = args.timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Total supervisor runtime exhausted")
            save(status_path, {**metadata, "stage": stage, "state": "RUNNING"})
            # Keep raw model/loading/per-image diagnostics out of the conversation.
            with (logs / f"{stage.lower()}.log").open("a") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=remaining)
        decision_path = args.output / "analysis" / "decision.json"
        decision = json.loads(decision_path.read_text())
        save(status_path, {**metadata, "state": "DEVELOPMENT_COMPLETE",
                           "elapsed_seconds": time.monotonic() - started,
                           "decision": decision, "stage3": "NOT_STARTED_REQUIRES_FROZEN_POLICY"})
        print(json.dumps({"state": "DEVELOPMENT_COMPLETE", "test_unsealed": False}))
        return 0
    except Exception as exc:
        save(status_path, {**metadata, "state": "FAILED", "error": str(exc),
                           "elapsed_seconds": time.monotonic() - started})
        print(json.dumps({"state": "FAILED", "error_type": type(exc).__name__}))
        return 1
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
