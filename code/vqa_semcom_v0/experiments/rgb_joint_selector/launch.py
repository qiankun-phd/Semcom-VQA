"""Bounded EXP-012 supervisor; never opens a test split or starts a channel sweep.

Prepare data, features, and the six-image codec/inference smoke before launch.
All expensive stages are sequential. Their detailed output stays in local logs.
An advisory lock prevents a second supervisor; immutable hashes guard resumes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

try:
    from .prepare_data import check_hashes, read, save, sha, verify_data
except ImportError:
    from prepare_data import check_hashes, read, save, sha, verify_data


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive(path: Path):
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A supervisor already owns this run") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_step(command: list[str], log: Path, seconds_left: float) -> None:
    if seconds_left <= 0:
        raise TimeoutError("Registered total wall-clock budget exhausted")
    with log.open("ab") as handle:
        handle.write(f"\nSUPERVISOR STAGE START {utc()}\n".encode())
        handle.flush()
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            code = process.wait(timeout=seconds_left)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    if code:
        raise RuntimeError(f"Stage exited {code}; inspect {log.name}")


def preflight(output: Path, protocol: Path) -> None:
    verify_data(output, protocol)
    features = read(output / "features_complete.json")
    if (features.get("state") != "COMPLETE" or features.get("records") != 720 or
            features.get("features_sha256") != sha(output / "features.json") or
            features.get("frozen_sha256") != sha(output / "features_frozen.json")):
        raise ValueError("Complete unchanged 600 new + 120 legacy features are required")
    check_hashes(read(output / "features_frozen.json")["sha256"])
    codec = read(output / "encoding_smoke_complete.json")
    if codec.get("representations") != 18:
        raise ValueError("Six-image, three-rate codec smoke is required")
    inference = read(output / "inference_smoke_complete.json")
    if (inference.get("state") != "SMOKE_COMPLETE" or inference.get("records") != 54 or
            not inference.get("actual_tiers_verified") or inference.get("protocol_sha256") != sha(protocol)):
        raise ValueError("Six-image, nine-action inference smoke is required")


def verify_completion(stage: str, output: Path) -> dict:
    if stage == "encode":
        result = read(output / "encoding_complete.json")
        if result.get("representations") != 1800 or result.get("representations_sha256") != sha(output / "representations.json"):
            raise ValueError("Incomplete or changed 1800-representation output")
    elif stage == "supervision":
        result = read(output / "supervision_complete.json")
        if (result.get("state") != "SUPERVISION_COMPLETE" or result.get("records") != 5400 or
                result.get("records_sha256") != sha(output / "supervision_records.json")):
            raise ValueError("Incomplete or changed 5400-record supervision output")
    else:
        result = read(output / "training_complete.json")
        if result.get("state") != "COMPLETE" or result.get("jobs") != 24 or result.get("full_cost_gate") != "PENDING":
            raise ValueError("Incomplete training or invalid full-cost gate")
        if result.get("test_or_snr_started") is not False:
            raise ValueError("Test/SNR expansion was not authorized by this pipeline")
        if result.get("controller_sha256") != sha(output / "controller_frozen.json"):
            raise ValueError("Frozen controller changed")
        check_hashes(read(output / "controller_frozen.json")["sha256"])
    return result


def launch(output: Path, protocol: Path, stage1: Path, adapter: Path, grid: Path) -> dict:
    code = Path(__file__).resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    with exclusive(output / "supervisor.lock"):
        preflight(output, protocol)
        common = ["--output", str(output), "--protocol", str(protocol)]
        commands = [
            ("encode", [sys.executable, str(code / "encode_data.py"), *common, "--stage1-root", str(stage1)]),
            ("supervision", [sys.executable, str(code / "infer_supervision.py"), *common,
                             "--stage1-root", str(stage1), "--adapter", str(adapter), "--grid-code", str(grid / "code")]),
            ("training", [sys.executable, str(code / "train_selector.py"), *common,
                           "--legacy-stage1", str(stage1), "--legacy-grid", str(grid)]),
        ]
        paths = sorted(path for path in code.glob("*.py") if not path.name.startswith("test_")) + [protocol]
        fingerprint = {"sha256": {str(path): sha(path) for path in paths}, "commands": commands,
                       "interpreter": sys.executable, "wallclock_limit_seconds": read(protocol)["wallclock_limit_seconds"]}
        # JSON arrays do not preserve Python tuples.
        fingerprint["commands"] = [[name, command] for name, command in commands]
        frozen_path = output / "supervisor_frozen.json"
        if frozen_path.exists():
            frozen = read(frozen_path)
            if frozen["fingerprint"] != fingerprint:
                raise ValueError("Supervisor code/configuration changed after launch")
        else:
            now = time.time()
            frozen = {"fingerprint": fingerprint, "started_utc": utc(),
                      "deadline_epoch": now + fingerprint["wallclock_limit_seconds"]}
            save(frozen_path, frozen)
        status_path = output / "supervisor_status.json"
        finished = []
        try:
            for name, command in commands:
                check_hashes(frozen["fingerprint"]["sha256"])
                save(status_path, {"state": "RUNNING", "stage": name, "completed_stages": finished,
                                   "updated_utc": utc(), "deadline_epoch": frozen["deadline_epoch"],
                                   "test_or_snr_started": False, "full_cost_gate": "PENDING"})
                run_step(command, output / f"supervisor_{name}.log", frozen["deadline_epoch"] - time.time())
                verify_completion(name, output)
                finished.append(name)
            result = {"state": "COMPLETE", "completed_stages": finished, "updated_utc": utc(),
                      "test_or_snr_started": False, "full_cost_gate": "PENDING",
                      "training_complete_sha256": sha(output / "training_complete.json")}
            save(status_path, result)
            return result
        except BaseException as error:
            save(status_path, {"state": "FAILED", "stage": name, "completed_stages": finished,
                               "updated_utc": utc(), "error": str(error), "test_or_snr_started": False,
                               "full_cost_gate": "PENDING"})
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "protocol", "stage1-root", "adapter", "legacy-grid"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(launch(args.output.resolve(), args.protocol.resolve(), args.stage1_root.resolve(),
                 args.adapter.resolve(), args.legacy_grid.resolve()))


if __name__ == "__main__":
    main()
