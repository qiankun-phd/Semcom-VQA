#!/usr/bin/env python3
"""Conditional C1000 convergence runner (G1V-frozen, resumable).

Authority:
  - BUBBLES_VQA_RL_1000_3000_EPISODE_CONVERGENCE_PLAN_2026-07-23
  - BUBBLES_VQA_RL_PPO_STABILITY_REPAIR_CONTRACT_2026-07-23
  - BUBBLES_VQA_RL_G1U_IMPLEMENTATION_AND_G1V_GATE_REPORT_2026-07-23

Hard guards:
  - final seeds 80000001-80000020 never accessed
  - no C3000 start / queue
  - no live retune of G1V frozen algorithm / env / LR / entropy / dual / network
  - checkpoint selection uses paired validation constrained objective only
  - refuse start if code dirty after freeze snapshot (when required)
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import socket
import struct
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bubbles_vqa.joint_control.config import JointControlConfig
from bubbles_vqa.joint_control.gym_env import JointControlGymEnv
from bubbles_vqa.rl_interface.scripted import (
    ForecastQuotaMobilityPolicyAdapter,
    NullJointControlPolicy,
    StayOnlyMobilityPolicy,
)
from bubbles_vqa.rl_training.checkpoint import load_checkpoint, save_checkpoint
from bubbles_vqa.rl_training.config import RLTrainConfig, resolve_device, scheduled_entropy_weight
from bubbles_vqa.rl_training.duals import DualState, init_dual_state, update_duals
from bubbles_vqa.rl_training.networks import JointTwoTimescaleActorCritic
from bubbles_vqa.rl_training.obs_normalization import load_obs_norm_stats
from bubbles_vqa.rl_training.ppo import epoch0_logprob_ratio, two_timescale_ppo_update
from bubbles_vqa.rl_training.rollout import collect_episode
from bubbles_vqa.rl_training.trainer import set_global_seeds

# ---------------------------------------------------------------------------
# Frozen constants (G1V)
# ---------------------------------------------------------------------------

CONTRACT = "BUBBLES_VQA_RL_1000_3000_EPISODE_CONVERGENCE_PLAN_2026-07-23"
STABILITY_CONTRACT = "BUBBLES_VQA_RL_PPO_STABILITY_REPAIR_CONTRACT_2026-07-23"

TRAIN_SEEDS = list(range(60000001, 60000021))  # 60000001-60000020 cycle
VAL_SEEDS = list(range(70000001, 70000006))  # 70000001-70000005
FINAL_SEEDS = frozenset(range(80000001, 80000021))  # locked denylist
FORBIDDEN_TRAIN_SEEDS = frozenset(VAL_SEEDS) | FINAL_SEEDS

DEFAULT_G1V_DIR = REPO / "outputs" / "rl" / "g1u_g1v_stability_repair_2026-07-23"
DEFAULT_OUT = REPO / "outputs" / "rl" / "convergence_c1000_2026-07-23"
DEFAULT_QUALITY = REPO / "outputs" / "quality" / "online_quality_table.json"
DEFAULT_SHARED_INIT = (
    DEFAULT_G1V_DIR / "shared_init" / "checkpoint_shared_init_with_obs_norm.pt"
)
DEFAULT_OBS_NORM = DEFAULT_G1V_DIR / "obs_norm_stats.json"

EXPECTED_OBS_NORM_SHA = "a0f8c24fb10d8f4c44c5a0896314486944c8f971f86618bf08ac37974dd929e0"
EXPECTED_QUALITY_SHA_PREFIX = "837d9f7f"
LEARNING_RATE = 3e-5
TARGET_KL_GUARD = 0.15
MAX_EPISODES = 1000
CKPT_INTERVAL = 25
VAL_INTERVAL = 50
MIN_DECISION_EPISODE = 300
N_VAL_WINDOW = 5
HORIZON_S = 900.0
LAMBDA_TOTAL = 0.5
EXPECTED_N_FAST = 3000
EXPECTED_N_SLOW = 60

# Decision labels
C1000_CONVERGED = "C1000_CONVERGED"
C1000_STABLE_STILL_IMPROVING = "C1000_STABLE_STILL_IMPROVING"
C1000_PLATEAU_BELOW_TARGET = "C1000_PLATEAU_BELOW_TARGET"
C1000_UNSTABLE_OR_INVALID = "C1000_UNSTABLE_OR_INVALID"
DECISION_LABELS = (
    C1000_CONVERGED,
    C1000_STABLE_STILL_IMPROVING,
    C1000_PLATEAU_BELOW_TARGET,
    C1000_UNSTABLE_OR_INVALID,
)

STOP_IMMEDIATE = {
    C1000_CONVERGED,
    C1000_PLATEAU_BELOW_TARGET,
    C1000_UNSTABLE_OR_INVALID,
}

# Gate thresholds from plan §4.2
REL_IMPROVE_PLATEAU = 0.01
DEGRADE_MAX = 0.02
MEDIAN_KL_MAX = 0.10
MEDIAN_CLIP_MAX = 0.30
RATIO_LO, RATIO_HI = 0.5, 1.5
KL_GUARD_FRAC_MAX = 0.20
ENTROPY_MIN = 1e-4
LAMBDA_MAX_PIN_STREAK = 3
SEED_CONSISTENCY_MIN = 4  # of 5
ORACLE_GAP_FRACTION = 0.80

LOCK_NAME = "runner.lock"
PID_NAME = "pid_or_unit.json"
STATUS_NAME = "runner_status.json"
DECISION_NAME = "convergence_decision.json"
TRAIN_METRICS = "training_metrics.jsonl"
VAL_METRICS = "validation_metrics.jsonl"
CKPT_INDEX = "checkpoint_index.json"
RUN_MANIFEST = "run_manifest.json"
CONFIG_NAME = "config.json"
MONITOR_MD = "monitoring_summary.md"
RECONCILE_LOG = "reconciliation_log.jsonl"
WRITE_JOURNAL = "write_journal.jsonl"

# Keys ignored when comparing train/val rows for conflict detection.
_ROW_COMPARE_EXCLUDE = frozenset(
    {
        "timestamp_utc",
        "wall_time_s",
        "gpu_peak_memory_mib",
        "journal_sequence",
    }
)

# Required fields for a recoverable validation checkpoint row.
_VAL_REQUIRED_FIELDS = (
    "episode_completed",
    "constrained_objective",
    "quality_utility_mean",
    "deadline_cost_mean",
    "finite_ok",
    "cadence_ok",
    "task_conservation_ok",
    "provenance_ok",
    "no_param_update",
    "dual_unchanged",
    "model_hash_pre",
    "model_hash_post",
    "optimizer_hash_pre",
    "optimizer_hash_post",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    payload = json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid JSONL at {path}:{line_no}: {exc}"
            ) from exc
    return rows


def atomic_rewrite_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically rewrite a JSONL file (no leftover tmp after success)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.rewrite.{os.getpid()}"
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def append_reconciliation_log(out_dir: Path, event: Mapping[str, Any]) -> None:
    payload = dict(event)
    payload.setdefault("timestamp_utc", utc_now())
    append_jsonl(Path(out_dir) / RECONCILE_LOG, payload)


def canonical_row_for_compare(row: Mapping[str, Any]) -> str:
    """Stable content fingerprint excluding volatile timing fields."""
    filtered = {
        k: v
        for k, v in sorted(row.items())
        if k not in _ROW_COMPARE_EXCLUDE
    }
    return json.dumps(filtered, sort_keys=True, default=str)


def is_stale_temp_path(path: Path) -> bool:
    """True for confirmed-invalid atomic-write leftovers / temp sidecars."""
    name = path.name
    if not path.is_file():
        return False
    # library save_checkpoint on *.pt.tmp.<pid> leaves *.pt.tmp.json
    if name.endswith(".pt.tmp.json"):
        return True
    if ".tmp." in name:
        return True
    if ".rewrite." in name:
        return True
    if ".writing." in name:
        return True
    if name.endswith(".tmp"):
        return True
    return False


def cleanup_stale_temp_files(root: Path) -> list[str]:
    """Delete confirmed-invalid temp files under root; return cleaned paths."""
    root = Path(root)
    cleaned: list[str] = []
    if not root.exists():
        return cleaned
    for p in sorted(root.rglob("*")):
        if is_stale_temp_path(p):
            try:
                p.unlink()
                cleaned.append(str(p))
            except OSError:
                pass
    return cleaned


def reconcile_training_metrics(
    metrics_path: Path,
    next_episode_index: int,
    *,
    out_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Make training JSONL match checkpoint authority ``next_episode_index``.

    - Parse and validate JSONL.
    - Deduplicate by ``episode_index`` (identical content kept once; conflict fails closed).
    - Drop / archive all rows with ``episode_index >= next_episode_index``.
    - Keep the complete unique prefix for ``episode_index < next_episode_index``.
    - Atomically rewrite and log reconciliation when anything changes.
    """
    metrics_path = Path(metrics_path)
    out_dir = Path(out_dir) if out_dir is not None else metrics_path.parent
    next_episode_index = int(next_episode_index)
    if next_episode_index < 0:
        raise RuntimeError(f"invalid next_episode_index={next_episode_index}")

    raw = read_jsonl(metrics_path) if metrics_path.exists() else []
    by_ep: dict[int, dict[str, Any]] = {}
    duplicate_eps: list[int] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, dict) or "episode_index" not in row:
            raise RuntimeError(
                f"training metrics row {idx} missing episode_index: {row!r}"
            )
        try:
            ei = int(row["episode_index"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"training metrics row {idx} has non-int episode_index"
            ) from exc
        if ei in by_ep:
            if canonical_row_for_compare(by_ep[ei]) != canonical_row_for_compare(row):
                raise RuntimeError(
                    f"conflicting training_metrics rows for episode_index={ei}"
                )
            duplicate_eps.append(ei)
            continue
        by_ep[ei] = dict(row)

    truncated_eps = sorted(ei for ei in by_ep if ei >= next_episode_index)
    expected = list(range(next_episode_index))
    missing = [ei for ei in expected if ei not in by_ep]
    if missing:
        raise RuntimeError(
            "training_metrics incomplete vs checkpoint authority "
            f"next_episode_index={next_episode_index}: missing episode_index={missing}"
        )
    kept = [by_ep[ei] for ei in expected]

    # Detect whether on-disk content already matches the reconciled prefix.
    changed = bool(duplicate_eps) or bool(truncated_eps) or len(raw) != len(kept)
    if not changed:
        for a, b in zip(raw, kept):
            if int(a["episode_index"]) != int(b["episode_index"]):
                changed = True
                break

    if changed:
        archive_path: str | None = None
        if metrics_path.exists():
            archive = out_dir / f"training_metrics.prereconcile.{os.getpid()}.jsonl"
            try:
                archive.write_bytes(metrics_path.read_bytes())
                archive_path = str(archive)
            except OSError:
                archive_path = None
        atomic_rewrite_jsonl(metrics_path, kept)
        append_reconciliation_log(
            out_dir,
            {
                "event": "reconcile_training_metrics",
                "next_episode_index": next_episode_index,
                "raw_rows": len(raw),
                "kept_rows": len(kept),
                "duplicate_episode_indices": duplicate_eps,
                "truncated_episode_indices": truncated_eps,
                "archive": archive_path,
                "metrics_path": str(metrics_path),
            },
        )
    else:
        append_reconciliation_log(
            out_dir,
            {
                "event": "reconcile_training_metrics",
                "next_episode_index": next_episode_index,
                "raw_rows": len(raw),
                "kept_rows": len(kept),
                "duplicate_episode_indices": [],
                "truncated_episode_indices": [],
                "unchanged": True,
                "metrics_path": str(metrics_path),
            },
        )
    return kept


def validation_row_is_acceptable(
    row: Mapping[str, Any],
    *,
    next_episode_index: int,
) -> tuple[bool, str]:
    """Return (ok, reason). Only fully provenance-backed no-update rows accepted."""
    if not isinstance(row, Mapping):
        return False, "not_a_mapping"
    for key in _VAL_REQUIRED_FIELDS:
        if key not in row:
            return False, f"missing_field:{key}"
    try:
        ep = int(row["episode_completed"])
    except (TypeError, ValueError):
        return False, "bad_episode_completed"
    if ep > int(next_episode_index):
        return False, "ahead_of_checkpoint"
    if ep < 0:
        return False, "negative_episode_completed"
    if not bool(row.get("no_param_update", False)):
        return False, "validation_update_not_recorded"
    if not bool(row.get("dual_unchanged", False)):
        return False, "dual_changed"
    if str(row.get("model_hash_pre")) != str(row.get("model_hash_post")):
        return False, "model_hash_mismatch"
    if str(row.get("optimizer_hash_pre")) != str(row.get("optimizer_hash_post")):
        return False, "optimizer_hash_mismatch"
    if not bool(row.get("provenance_ok", False)):
        return False, "provenance_not_ok"
    # Prefer explicit hash when present
    qhash = row.get("quality_table_sha256") or row.get("quality_table_sha")
    if qhash is not None and not str(qhash):
        return False, "empty_quality_hash"
    if not bool(row.get("finite_ok", False)):
        return False, "nonfinite"
    if not bool(row.get("cadence_ok", False)):
        return False, "cadence_fail"
    if not bool(row.get("task_conservation_ok", False)):
        return False, "task_conservation_fail"
    return True, "ok"


def merge_validation_history(
    checkpoint_history: Sequence[Mapping[str, Any]],
    disk_rows: Sequence[Mapping[str, Any]],
    *,
    next_episode_index: int,
    out_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Merge checkpoint validation_history with validation JSONL.

    - Deduplicate by ``episode_completed``.
    - Accept only rows with episode_completed <= next_episode_index that are
      schema/provenance/hash complete and validation-no-update recorded.
    - Disk rows absent from checkpoint are recovered when acceptable.
    - Conflicting content for the same episode_completed fails closed.
    """
    next_episode_index = int(next_episode_index)
    by_ep: dict[int, dict[str, Any]] = {}
    sources: dict[int, str] = {}
    recovered: list[int] = []
    skipped_ahead: list[int] = []
    skipped_invalid: list[dict[str, Any]] = []

    def _try_episode(row: Mapping[str, Any]) -> int | None:
        try:
            return int(row.get("episode_completed"))  # type: ignore[arg-type]
        except Exception:
            return None

    def _ingest(rows: Sequence[Mapping[str, Any]], source: str) -> None:
        for row in rows:
            ok, reason = validation_row_is_acceptable(
                row, next_episode_index=next_episode_index
            )
            if not ok:
                if reason == "ahead_of_checkpoint":
                    ep_ahead = _try_episode(row)
                    if ep_ahead is not None:
                        skipped_ahead.append(ep_ahead)
                    continue
                ep_try = _try_episode(row)
                if ep_try is not None and 0 <= ep_try <= next_episode_index:
                    raise RuntimeError(
                        f"invalid validation row from {source} "
                        f"episode_completed={ep_try}: {reason}"
                    )
                skipped_invalid.append({"source": source, "reason": reason})
                continue
            ep = int(row["episode_completed"])
            payload = dict(row)
            if ep in by_ep:
                if canonical_row_for_compare(by_ep[ep]) != canonical_row_for_compare(
                    payload
                ):
                    raise RuntimeError(
                        "validation history conflict at "
                        f"episode_completed={ep} between {sources[ep]} and {source}"
                    )
                continue
            by_ep[ep] = payload
            sources[ep] = source
            if source == "disk":
                recovered.append(ep)

    _ingest(checkpoint_history, "checkpoint")
    _ingest(disk_rows, "disk")

    merged = [by_ep[k] for k in sorted(by_ep)]
    if out_dir is not None:
        append_reconciliation_log(
            out_dir,
            {
                "event": "merge_validation_history",
                "next_episode_index": next_episode_index,
                "checkpoint_rows": len(list(checkpoint_history)),
                "disk_rows": len(list(disk_rows)),
                "merged_rows": len(merged),
                "recovered_episode_completed": recovered,
                "skipped_ahead": skipped_ahead,
                "skipped_invalid_count": len(skipped_invalid),
            },
        )
    return merged


def git_head(repo: Path = REPO) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
        ).strip()
    except Exception:
        return "unknown"


def git_dirty(repo: Path = REPO) -> bool:
    """True if tracked files differ from HEAD.

    Untracked outputs/ artifacts must not block formal start after a freeze
    snapshot (code tree remains clean while run products accumulate).
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repo),
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return True


def git_diff_check(repo: Path = REPO) -> str:
    try:
        return subprocess.check_output(
            ["git", "diff", "--check"], cwd=str(repo), text=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as exc:
        return (exc.output or "") + f"\nexit={exc.returncode}"
    except Exception as exc:
        return f"git diff --check failed: {exc}"


def assert_not_final_seed(seed: int, context: str = "") -> None:
    si = int(seed)
    if si in FINAL_SEEDS:
        raise RuntimeError(
            f"FINAL SEED DENIED ({context}): seed={si} in locked set "
            f"80000001-80000020; C1000 must never access final seeds"
        )


def assert_allowed_train_seed(seed: int) -> None:
    si = int(seed)
    assert_not_final_seed(si, "train")
    if si in FORBIDDEN_TRAIN_SEEDS:
        raise RuntimeError(f"refusing non-training seed for train path: {si}")
    if si not in TRAIN_SEEDS:
        raise RuntimeError(
            f"train seed {si} not in deterministic cycle {TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}"
        )


def assert_allowed_val_seed(seed: int) -> None:
    si = int(seed)
    assert_not_final_seed(si, "validation")
    if si not in VAL_SEEDS:
        raise RuntimeError(f"validation seed {si} not in {VAL_SEEDS}")


def seed_at_episode(episode_index: int) -> int:
    return int(TRAIN_SEEDS[int(episode_index) % len(TRAIN_SEEDS)])


def median(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    return float(np.median(np.asarray(xs, dtype=np.float64)))


def mean(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    return float(np.mean(np.asarray(xs, dtype=np.float64)))


def is_finite_number(x: Any) -> bool:
    try:
        return bool(math.isfinite(float(x)))
    except Exception:
        return False


def state_dict_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k in sorted(model.state_dict().keys()):
        t = model.state_dict()[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def optimizer_state_sha(optimizer: torch.optim.Optimizer) -> str:
    # Byte-stable dump of optimizer state (CPU floats).
    blob = repr(optimizer.state_dict()).encode("utf-8")
    return sha256_bytes(blob)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"]
    if not torch.is_tensor(torch_state):
        torch_state = torch.tensor(torch_state, dtype=torch.uint8)
    else:
        torch_state = torch_state.detach().cpu().to(dtype=torch.uint8).contiguous()
    torch.set_rng_state(torch_state)
    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda_states = []
        for t in state["cuda"]:
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.uint8)
            else:
                t = t.detach().cpu().to(dtype=torch.uint8).contiguous()
            cuda_states.append(t)
        torch.cuda.set_rng_state_all(cuda_states)


def rng_state_to_cpu_blob(state: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize RNG state into a torch-saveable CPU structure."""
    out: dict[str, Any] = {
        "python": state["python"],
        "numpy": state["numpy"],
        "torch": state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"],
        "cuda": None,
    }
    if state.get("cuda") is not None:
        out["cuda"] = [
            t.cpu() if torch.is_tensor(t) else t for t in state["cuda"]
        ]
    return out


def load_quality_table_meta(table_path: Path) -> dict[str, Any]:
    doc = json.loads(Path(table_path).read_text(encoding="utf-8"))
    prov = dict(doc.get("provenance") or {})
    levels = doc.get("levels") or []
    return {
        "path": str(Path(table_path).resolve()),
        "sha256": sha256_file(table_path),
        "quality_version": prov.get("quality_version"),
        "provenance_hash": prov.get("provenance_hash"),
        "calibration_version": prov.get("calibration_version"),
        "quality_source": prov.get("quality_source"),
        "n_levels": len(levels),
        "level_lcbs": {
            str(int(row["service_level"])): float(row["calibrated_accuracy_lcb"])
            for row in levels
        },
        "schema": doc.get("schema"),
    }


def engine_task_counts(env: JointControlGymEnv) -> dict[str, int]:
    eng = getattr(env, "_engine", None)
    counters = dict(getattr(eng, "counters", {}) or {})
    return {
        "completed": int(counters.get("completed", 0)),
        "expired": int(counters.get("expired", 0)),
        "rejected": int(counters.get("rejected", 0)),
        "deferred": int(counters.get("deferred", 0)),
        "created": int(counters.get("created", 0)),
        "admitted": int(counters.get("admitted", 0)),
        "dropped": int(counters.get("dropped", 0)),
        "abandoned": int(counters.get("abandoned", 0)),
    }


def build_env(
    *,
    horizon_s: float,
    lambda_total_per_s: float,
    quality_table: Path,
) -> JointControlGymEnv:
    jc = JointControlConfig.external_structural(
        operation_duration_s=float(horizon_s),
        lambda_total_per_s=float(lambda_total_per_s),
    )
    jc = replace(jc, online_quality_table_path=str(Path(quality_table).resolve()))
    env = JointControlGymEnv(jc)
    if getattr(env, "_quality_adapter", None) is None:
        raise RuntimeError(f"quality adapter not bound: {quality_table}")
    return env


def gpu_snapshot() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        name, used, total, util = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        return {
            "name": name,
            "memory_used_mib": used,
            "memory_total_mib": total,
            "utilization_pct": util,
            "memory_free_mib": total - used,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Constrained objective + decision math (pure, unit-testable)
# ---------------------------------------------------------------------------


def constrained_objective_from_episode(
    *,
    quality_utility: float,
    deadline_cost_sum: float,
    lambda_deadline: float,
) -> float:
    """Validation constrained objective (Lagrangian form; no train return)."""
    return float(quality_utility) - float(lambda_deadline) * float(deadline_cost_sum)


def aggregate_validation_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_seeds": 0,
            "constrained_objective": float("nan"),
            "quality_utility_mean": float("nan"),
            "deadline_cost_mean": float("nan"),
            "deadline_satisfied": False,
            "per_seed_objective": [],
        }
    objs = [float(r["constrained_objective"]) for r in rows]
    qus = [float(r["quality_utility"]) for r in rows]
    dcs = [float(r["deadline_cost_mean"]) for r in rows]
    # per-seed deadline mean vs limit
    limit = float(rows[0].get("deadline_cost_limit", 0.025))
    satisfied = all(float(r["deadline_cost_mean"]) <= limit + 1e-12 for r in rows)
    return {
        "n_seeds": len(rows),
        "constrained_objective": mean(objs),
        "quality_utility_mean": mean(qus),
        "deadline_cost_mean": mean(dcs),
        "deadline_satisfied": bool(satisfied),
        "per_seed_objective": objs,
        "per_seed_quality_utility": qus,
        "finite_ok": all(bool(r.get("finite_ok")) for r in rows),
        "cadence_ok": all(bool(r.get("cadence_ok")) for r in rows),
        "task_conservation_ok": all(bool(r.get("task_conservation_ok")) for r in rows),
        "provenance_ok": all(bool(r.get("provenance_ok", True)) for r in rows),
    }


def linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    x_mean = x.mean()
    y_mean = y.mean()
    den = float(((x - x_mean) ** 2).sum())
    if den <= 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / den)


def bootstrap_slope_ci(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 424242,
    alpha: float = 0.05,
) -> dict[str, float]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    n = len(x)
    point = linear_slope(x, y)
    if n < 2:
        return {
            "slope": point,
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "includes_zero": True,
        }
    rng = np.random.default_rng(int(seed))
    slopes = np.empty(n_boot, dtype=np.float64)
    idx = np.arange(n)
    for i in range(n_boot):
        sample = rng.choice(idx, size=n, replace=True)
        slopes[i] = linear_slope(x[sample], y[sample])
    lo = float(np.quantile(slopes, alpha / 2.0))
    hi = float(np.quantile(slopes, 1.0 - alpha / 2.0))
    return {
        "slope": float(point),
        "ci_lo": lo,
        "ci_hi": hi,
        "includes_zero": bool(lo <= 0.0 <= hi),
    }


def seed_directional_consistency(
    window: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count validation seeds with non-decreasing objective first→last in window."""
    if not window:
        return {"n_consistent": 0, "n_seeds": 0, "ok": False}
    first = window[0]
    last = window[-1]
    first_ps = list(first.get("per_seed_objective") or [])
    last_ps = list(last.get("per_seed_objective") or [])
    n = min(len(first_ps), len(last_ps))
    consistent = 0
    for i in range(n):
        # directionally consistent: last >= first - small tol, or both finite and same sign of improvement
        if is_finite_number(first_ps[i]) and is_finite_number(last_ps[i]):
            if float(last_ps[i]) + 1e-9 >= float(first_ps[i]):
                consistent += 1
    return {
        "n_consistent": int(consistent),
        "n_seeds": int(n),
        "ok": bool(consistent >= SEED_CONSISTENCY_MIN and n >= 5),
    }


def compute_value_diagnostics(window: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Track value error / explained variance trajectory over validation window."""
    ve = [float(v.get("value_error_mean", float("nan"))) for v in window]
    ev = [float(v.get("explained_variance", float("nan"))) for v in window]
    # "no longer deteriorate": last <= first for value error; last >= first for EV
    ve_ok = True
    ev_ok = True
    if all(is_finite_number(x) for x in ve):
        ve_ok = float(ve[-1]) <= float(ve[0]) + 1e-6 * max(1.0, abs(float(ve[0])))
    if all(is_finite_number(x) for x in ev):
        ev_ok = float(ev[-1]) + 1e-9 >= float(ev[0])
    return {
        "value_error_mean_series": ve,
        "explained_variance_series": ev,
        "value_error_not_deteriorating": bool(ve_ok),
        "explained_variance_not_deteriorating": bool(ev_ok),
        "ok": bool(ve_ok and ev_ok),
    }


def check_safety_window(
    window: Sequence[Mapping[str, Any]],
    train_tail: Sequence[Mapping[str, Any]],
    *,
    deadline_lambda_max: float = 20.0,
) -> dict[str, Any]:
    """Plan §4.2 items 3–7 safety/stability gates on the five-checkpoint window."""
    reasons: list[str] = []

    # 3. finite / cadence / task conservation / provenance
    for i, v in enumerate(window):
        if not v.get("finite_ok", False):
            reasons.append(f"finite_fail@val[{i}]")
        if not v.get("cadence_ok", False):
            reasons.append(f"cadence_fail@val[{i}]")
        if not v.get("task_conservation_ok", False):
            reasons.append(f"task_conservation_fail@val[{i}]")
        if not v.get("provenance_ok", True):
            reasons.append(f"provenance_fail@val[{i}]")

    # 4. deadline constraint or dual stable not pinned at lambda_max
    dual_pinned_streak = 0
    dual_ok = True
    for v in window:
        sat = bool(v.get("deadline_satisfied", False))
        lam = float(v.get("lambda_deadline", 0.0))
        pinned = lam >= float(deadline_lambda_max) - 1e-9
        if sat:
            dual_pinned_streak = 0
        elif pinned:
            dual_pinned_streak += 1
        else:
            dual_pinned_streak = 0
        if dual_pinned_streak >= LAMBDA_MAX_PIN_STREAK and not sat:
            dual_ok = False
    if not dual_ok:
        reasons.append("deadline_dual_pinned_at_lambda_max")
    # if never satisfied and dual not handling — soft fail if mean cost >> limit across window
    if not any(bool(v.get("deadline_satisfied")) for v in window):
        # still allow if dual not pinned (learning dual)
        if dual_pinned_streak >= LAMBDA_MAX_PIN_STREAK:
            reasons.append("deadline_unsatisfied_and_dual_pinned")

    # 5. entropy
    ents = [float(v.get("entropy_mean", float("nan"))) for v in window]
    if not all(is_finite_number(e) and float(e) > ENTROPY_MIN for e in ents):
        reasons.append("entropy_collapse_or_nonfinite")

    # 6. KL / clip / ratio / guard frequency from *training* updates in the window span
    if train_tail:
        kls = [float(r.get("approx_kl_max_over_epochs", float("nan"))) for r in train_tail]
        clips = [float(r.get("clip_fraction_max_over_epochs", float("nan"))) for r in train_tail]
        ratios = [float(r.get("ratio_mean_final_executed_epoch", float("nan"))) for r in train_tail]
        guards = [bool(r.get("kl_guard_stopped_early")) for r in train_tail]
        med_kl = median([k for k in kls if is_finite_number(k)])
        med_clip = median([c for c in clips if is_finite_number(c)])
        med_ratio = median([r for r in ratios if is_finite_number(r)])
        guard_frac = float(sum(1 for g in guards if g) / max(1, len(guards)))
        if not (is_finite_number(med_kl) and med_kl <= MEDIAN_KL_MAX):
            reasons.append(f"median_kl={med_kl}")
        if not (is_finite_number(med_clip) and med_clip <= MEDIAN_CLIP_MAX):
            reasons.append(f"median_clip={med_clip}")
        if not (is_finite_number(med_ratio) and RATIO_LO <= med_ratio <= RATIO_HI):
            reasons.append(f"median_ratio={med_ratio}")
        if guard_frac >= KL_GUARD_FRAC_MAX:
            reasons.append(f"kl_guard_frac={guard_frac}")
        kl_block = {
            "median_kl": med_kl,
            "median_clip": med_clip,
            "median_ratio": med_ratio,
            "kl_guard_fraction": guard_frac,
        }
    else:
        kl_block = {
            "median_kl": float("nan"),
            "median_clip": float("nan"),
            "median_ratio": float("nan"),
            "kl_guard_fraction": float("nan"),
        }

    # 7. value diagnostics
    value_diag = compute_value_diagnostics(window)
    if not value_diag["ok"]:
        reasons.append("value_error_or_ev_deteriorating")

    # 8. seed consistency
    seed_cons = seed_directional_consistency(window)
    if not seed_cons["ok"]:
        reasons.append(
            f"seed_consistency={seed_cons['n_consistent']}/{seed_cons['n_seeds']}"
        )

    return {
        "ok": len(reasons) == 0,
        "reasons": reasons,
        "kl_block": kl_block,
        "value_diag": value_diag,
        "seed_consistency": seed_cons,
        "dual_ok": dual_ok,
    }


def performance_vs_comparators(
    last_obj: float,
    *,
    best_causal_baseline: float | None,
    oracle_upper: float | None,
) -> dict[str, Any]:
    beats = False
    closes_gap = False
    details: dict[str, Any] = {
        "best_causal_baseline": best_causal_baseline,
        "oracle_upper": oracle_upper,
        "last_obj": last_obj,
    }
    if best_causal_baseline is not None and is_finite_number(best_causal_baseline):
        beats = bool(last_obj > float(best_causal_baseline))
        details["beats_best_causal_baseline"] = beats
        if (
            oracle_upper is not None
            and is_finite_number(oracle_upper)
            and float(oracle_upper) > float(best_causal_baseline)
        ):
            gap = float(oracle_upper) - float(best_causal_baseline)
            closed = (float(last_obj) - float(best_causal_baseline)) / gap
            closes_gap = bool(closed >= ORACLE_GAP_FRACTION)
            details["gap_closed_fraction"] = closed
            details["closes_80pct_oracle_gap"] = closes_gap
        else:
            details["gap_closed_fraction"] = None
            details["closes_80pct_oracle_gap"] = False
    else:
        details["beats_best_causal_baseline"] = False
        details["closes_80pct_oracle_gap"] = False
        details["gap_closed_fraction"] = None
    details["meets_performance_target"] = bool(beats or closes_gap)
    return details


def decide_c1000(
    validation_history: Sequence[Mapping[str, Any]],
    training_history: Sequence[Mapping[str, Any]],
    *,
    episode_index: int,  # 0-based last completed episode
    best_causal_baseline: float | None,
    oracle_upper: float | None,
    deadline_lambda_max: float = 20.0,
    force_terminal: bool = False,
) -> dict[str, Any]:
    """Return decision payload. STABLE_STILL_IMPROVING only when episode+1 >= 1000.

    episode_index is the last completed training episode (0-based). After episode
    0 completes, episode_index=0; after 300 episodes, episode_index=299.
    Completed count = episode_index + 1.
    """
    completed = int(episode_index) + 1
    base = {
        "completed_episodes": completed,
        "episode_index": int(episode_index),
        "evaluated_utc": utc_now(),
        "decision": None,
        "stop": False,
        "reasons": [],
        "metrics": {},
    }

    # Immediate invalidity from latest training row
    if training_history:
        last_tr = training_history[-1]
        if not last_tr.get("finite_ok", True):
            base.update(
                {
                    "decision": C1000_UNSTABLE_OR_INVALID,
                    "stop": True,
                    "reasons": ["training_nonfinite"],
                }
            )
            return base
        ent = float(last_tr.get("entropy", float("nan")))
        if is_finite_number(ent) and ent <= ENTROPY_MIN:
            base.update(
                {
                    "decision": C1000_UNSTABLE_OR_INVALID,
                    "stop": True,
                    "reasons": ["entropy_collapse"],
                }
            )
            return base
        if not last_tr.get("cadence_ok", True) or not last_tr.get(
            "task_conservation_ok", True
        ):
            base.update(
                {
                    "decision": C1000_UNSTABLE_OR_INVALID,
                    "stop": True,
                    "reasons": ["cadence_or_task_conservation_fail"],
                }
            )
            return base

    if completed < MIN_DECISION_EPISODE and not force_terminal:
        base["decision"] = None
        base["reasons"] = [f"before_min_decision_episode_{MIN_DECISION_EPISODE}"]
        return base

    if len(validation_history) < N_VAL_WINDOW:
        if completed >= MAX_EPISODES or force_terminal:
            base.update(
                {
                    "decision": C1000_UNSTABLE_OR_INVALID,
                    "stop": True,
                    "reasons": ["insufficient_validation_history_at_terminal"],
                }
            )
            return base
        base["decision"] = None
        base["reasons"] = ["insufficient_validation_window"]
        return base

    window = list(validation_history[-N_VAL_WINDOW:])
    objs = [float(v["constrained_objective"]) for v in window]
    xs = [float(v.get("episode_completed", i + 1)) for i, v in enumerate(window)]
    # Prefer episode markers if present
    if all("episode_completed" in v for v in window):
        xs = [float(v["episode_completed"]) for v in window]

    first_j, last_j = float(objs[0]), float(objs[-1])
    denom = max(abs(first_j), 1e-8)
    rel_improve = (last_j - first_j) / denom

    slope_info = bootstrap_slope_ci(xs, objs, seed=70000000 + completed)
    # best previous (including current window)
    all_objs = [float(v["constrained_objective"]) for v in validation_history]
    best_j = max(all_objs) if all_objs else last_j
    degrade = (best_j - last_j) / max(abs(best_j), 1e-8)

    # training rows spanning the window episodes
    ep_lo = int(window[0].get("episode_completed", 0))
    ep_hi = int(window[-1].get("episode_completed", completed))
    train_tail = [
        r
        for r in training_history
        if ep_lo <= int(r.get("episode_index", -1)) + 1 <= ep_hi
    ]

    safety = check_safety_window(
        window, train_tail, deadline_lambda_max=deadline_lambda_max
    )
    perf = performance_vs_comparators(
        last_j,
        best_causal_baseline=best_causal_baseline,
        oracle_upper=oracle_upper,
    )

    plateau = (abs(rel_improve) < REL_IMPROVE_PLATEAU) and bool(
        slope_info["includes_zero"]
    )
    no_material_degrade = degrade <= DEGRADE_MAX
    # materially positive slope: point estimate > 0 and CI entirely above 0
    still_improving = (
        is_finite_number(slope_info["slope"])
        and float(slope_info["slope"]) > 0.0
        and float(slope_info["ci_lo"]) > 0.0
    )

    metrics = {
        "window_episodes": xs,
        "window_objectives": objs,
        "relative_improvement": rel_improve,
        "slope": slope_info,
        "best_objective": best_j,
        "last_objective": last_j,
        "degradation_from_best": degrade,
        "plateau": plateau,
        "no_material_degradation": no_material_degrade,
        "still_improving": still_improving,
        "safety": safety,
        "performance": perf,
    }
    base["metrics"] = metrics

    if not safety["ok"]:
        # destructive / invalid
        severe = any(
            any(k in r for k in ("finite", "cadence", "task", "entropy", "kl", "ratio"))
            for r in safety["reasons"]
        )
        if severe or any(
            "finite" in r or "entropy" in r or "cadence" in r or "task" in r
            for r in safety["reasons"]
        ):
            base.update(
                {
                    "decision": C1000_UNSTABLE_OR_INVALID,
                    "stop": True,
                    "reasons": list(safety["reasons"]),
                }
            )
            return base
        # softer safety fail at terminal → still unstable
        if completed >= MAX_EPISODES or force_terminal:
            base.update(
                {
                    "decision": C1000_UNSTABLE_OR_INVALID,
                    "stop": True,
                    "reasons": list(safety["reasons"]),
                }
            )
            return base
        # mid-run soft fail: continue unless clearly invalid
        base["decision"] = None
        base["reasons"] = ["safety_soft_fail_continue"] + list(safety["reasons"])
        return base

    # Convergence rule (plan §4.2): plateau + no degrade + safety + constraints
    converged_rule = bool(
        plateau
        and no_material_degrade
        and safety["ok"]
        and bool(window[-1].get("deadline_satisfied", False) or safety.get("dual_ok"))
    )

    if converged_rule and perf.get("meets_performance_target"):
        base.update(
            {
                "decision": C1000_CONVERGED,
                "stop": True,
                "reasons": ["convergence_rule_and_performance_target"],
            }
        )
        return base

    if converged_rule and not perf.get("meets_performance_target"):
        base.update(
            {
                "decision": C1000_PLATEAU_BELOW_TARGET,
                "stop": True,
                "reasons": ["plateau_below_baseline_or_oracle_gap"],
            }
        )
        return base

    # Still improving — only emit at episode 1000
    if still_improving and safety["ok"] and no_material_degrade:
        if completed >= MAX_EPISODES or force_terminal:
            base.update(
                {
                    "decision": C1000_STABLE_STILL_IMPROVING,
                    "stop": True,
                    "reasons": ["stable_still_improving_at_c1000_cap"],
                }
            )
            return base
        base["decision"] = None
        base["reasons"] = ["still_improving_continue_to_1000"]
        return base

    # Terminal without clear plateau / improvement
    if completed >= MAX_EPISODES or force_terminal:
        if safety["ok"] and plateau:
            base.update(
                {
                    "decision": C1000_PLATEAU_BELOW_TARGET,
                    "stop": True,
                    "reasons": ["terminal_plateau_or_no_target"],
                }
            )
        elif safety["ok"] and still_improving:
            base.update(
                {
                    "decision": C1000_STABLE_STILL_IMPROVING,
                    "stop": True,
                    "reasons": ["terminal_still_improving"],
                }
            )
        elif safety["ok"]:
            base.update(
                {
                    "decision": C1000_PLATEAU_BELOW_TARGET,
                    "stop": True,
                    "reasons": ["terminal_no_clear_improvement"],
                }
            )
        else:
            base.update(
                {
                    "decision": C1000_UNSTABLE_OR_INVALID,
                    "stop": True,
                    "reasons": list(safety["reasons"]) or ["terminal_unstable"],
                }
            )
        return base

    base["decision"] = None
    base["reasons"] = ["no_stop_condition"]
    return base


def select_best_checkpoint(
    validation_history: Sequence[Mapping[str, Any]],
    checkpoint_index: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Select highest validation constrained objective among constraint-satisfying ckpts."""
    eligible_eps = {
        int(v["episode_completed"])
        for v in validation_history
        if bool(v.get("deadline_satisfied")) and bool(v.get("finite_ok", True))
    }
    if not eligible_eps:
        # fall back to best objective regardless of constraint (recorded as such)
        if not validation_history:
            return None
        best_v = max(
            validation_history, key=lambda v: float(v.get("constrained_objective", -1e300))
        )
        match = None
        for c in checkpoint_index:
            if int(c.get("episode_completed", -1)) == int(best_v["episode_completed"]):
                match = c
                break
        return {
            "selection_rule": "best_objective_unconstrained_fallback",
            "validation": best_v,
            "checkpoint": match,
        }

    best_v = None
    for v in validation_history:
        if int(v["episode_completed"]) not in eligible_eps:
            continue
        if best_v is None or float(v["constrained_objective"]) > float(
            best_v["constrained_objective"]
        ):
            best_v = v
    if best_v is None:
        return None
    match = None
    for c in checkpoint_index:
        if int(c.get("episode_completed", -1)) == int(best_v["episode_completed"]):
            match = c
            break
    return {
        "selection_rule": "max_validation_constrained_objective_among_feasible",
        "validation": best_v,
        "checkpoint": match,
    }


# ---------------------------------------------------------------------------
# Single-instance lock
# ---------------------------------------------------------------------------


class SingleInstanceLock:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._fh.seek(0)
            existing = self._fh.read().strip()
            raise RuntimeError(
                f"another C1000 runner holds lock {self.path}: {existing}"
            ) from exc
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(
            json.dumps(
                {"pid": os.getpid(), "started_utc": utc_now(), "host": socket.gethostname()}
            )
            + "\n"
        )
        self._fh.flush()

    def release(self) -> None:
        if self._fh is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None


# ---------------------------------------------------------------------------
# Full-state checkpoint (model + optimizer + dual + RNG + indices)
# ---------------------------------------------------------------------------


def save_full_checkpoint(
    path: Path,
    *,
    model: JointTwoTimescaleActorCritic,
    optimizer: torch.optim.Optimizer,
    dual: DualState,
    cfg: RLTrainConfig,
    episode_index: int,
    seed_cycle_position: int,
    rng_state: Mapping[str, Any],
    provenance: dict[str, Any],
    validation_history: list[dict[str, Any]],
    decision: dict[str, Any] | None,
    last_train_row: Mapping[str, Any] | None = None,
    journal_sequence: int = 0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a unique temp name that cannot collide with the final sidecar
    # (library save_checkpoint writes path.with_suffix(".json")).
    tmp = path.parent / f".{path.name}.writing.{os.getpid()}"
    lib_sidecar = tmp.with_suffix(".json")

    try:
        # Use library save for model schema, then enrich with full resume payload.
        save_checkpoint(
            tmp,
            model,
            cfg,
            provenance={
                **provenance,
                "c1000_full_resume": True,
                "episode_index": int(episode_index),
                "seed_cycle_position": int(seed_cycle_position),
            },
            dual_state=dual.to_trace(),
            optimizer_steps=(int(episode_index) + 1) * int(cfg.update_epochs),
        )
        # Drop the library sidecar for the temp path immediately — never leave
        # checkpoint_latest.pt.tmp.json or similar leftovers.
        if lib_sidecar.exists() and lib_sidecar.resolve() != path.with_suffix(".json").resolve():
            try:
                lib_sidecar.unlink()
            except OSError:
                pass

        payload = torch.load(tmp, map_location="cpu", weights_only=False)
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["rng_state"] = rng_state_to_cpu_blob(rng_state)
        payload["resume"] = {
            "episode_index": int(episode_index),
            "next_episode_index": int(episode_index) + 1,
            "seed_cycle_position": int(seed_cycle_position),
            "entropy_weight": float(scheduled_entropy_weight(cfg, episode_index)),
            "obs_norm_sha256": getattr(model, "obs_norm_sha256", None),
            "validation_history": list(validation_history),
            "last_train_row": dict(last_train_row) if last_train_row is not None else None,
            "journal_sequence": int(journal_sequence),
            "decision": decision,
            "c1000": True,
            "c3000_forbidden": True,
        }
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        if lib_sidecar.exists() and (
            not path.with_suffix(".json").exists()
            or lib_sidecar.resolve() != path.with_suffix(".json").resolve()
        ):
            try:
                lib_sidecar.unlink()
            except OSError:
                pass

    # Final sidecar summary only (never a .tmp.json).
    side = {
        "path": str(path),
        "episode_index": int(episode_index),
        "next_episode_index": int(episode_index) + 1,
        "seed_cycle_position": int(seed_cycle_position),
        "obs_norm_sha256": getattr(model, "obs_norm_sha256", None),
        "dual_state": dual.to_trace(),
        "journal_sequence": int(journal_sequence),
        "validation_history_len": len(validation_history),
        "timestamp_utc": utc_now(),
    }
    atomic_write_json(path.with_suffix(".json"), side)
    # Sweep any residual temp sidecars next to this checkpoint.
    for leftover in path.parent.glob(f"{path.name}*"):
        if is_stale_temp_path(leftover):
            try:
                leftover.unlink()
            except OSError:
                pass
    return path


def load_full_checkpoint(
    path: Path,
    *,
    device: str,
    model: JointTwoTimescaleActorCritic,
    optimizer: torch.optim.Optimizer,
    dual: DualState,
) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    ds = payload.get("dual_state") or {}
    dual.deadline = float(ds.get("lambda_deadline", dual.deadline))
    dual.battery = float(ds.get("lambda_battery", dual.battery))
    dual.conflict = float(ds.get("lambda_conflict", 0.0))
    dual.coverage = float(ds.get("lambda_coverage", 0.0))
    if payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])
    resume = dict(payload.get("resume") or {})
    on = payload.get("obs_normalization") or {}
    model.obs_norm_enabled = bool(on.get("enabled", getattr(model, "obs_norm_enabled", False)))
    model.obs_norm_sha256 = on.get("sha256", getattr(model, "obs_norm_sha256", None))
    model.obs_norm_meta = dict(on.get("meta") or getattr(model, "obs_norm_meta", {}) or {})
    return resume


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------


def run_train_episode(
    *,
    env: JointControlGymEnv,
    model: JointTwoTimescaleActorCritic,
    optimizer: torch.optim.Optimizer,
    dual: DualState,
    cfg: RLTrainConfig,
    seed: int,
    episode_index: int,
    device: str,
    quality_meta: Mapping[str, Any],
) -> dict[str, Any]:
    assert_allowed_train_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    set_global_seeds(int(seed))
    probe = env.reset(seed=int(seed))
    obs0 = probe[0] if isinstance(probe, tuple) else probe
    from bubbles_vqa.rl_training.obs_normalization import assert_no_forbidden_keys

    assert_no_forbidden_keys(obs0, context=f"train_seed={seed}")
    rollout = collect_episode(env, model, seed=int(seed), cfg=cfg, dual=dual)

    with torch.no_grad():
        lp, old = epoch0_logprob_ratio(model, rollout)
        ratio0 = torch.exp(lp - old)
        epoch0_ratio_mean = float(ratio0.mean().item())
        finite_lp = bool(
            torch.isfinite(lp).all().item() and torch.isfinite(old).all().item()
        )

    stats = two_timescale_ppo_update(
        model,
        optimizer,
        rollout,
        cfg,
        episode=episode_index,
        target_kl_guard=TARGET_KL_GUARD,
        record_group_diagnostics=True,
    )
    update_duals(dual, rollout, cfg, episode=episode_index)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated())
    else:
        peak = 0

    task_counts = engine_task_counts(env)
    quality_utility = float(np.sum(rollout["utility_rewards"]))
    shaped_return = float(np.sum(rollout["rewards"]))
    deadline_cost_sum = float(np.sum(rollout["deadline_costs"]))
    deadline_cost_mean = float(np.mean(rollout["deadline_costs"])) if rollout["deadline_costs"] else 0.0
    epochs = list(stats.get("epoch_stats") or [])
    kls = [float(e["approx_kl"]) for e in epochs]
    clips = [float(e["clip_fraction"]) for e in epochs]
    executed = [e for e in epochs if e.get("optimizer_step_applied", True)]
    if executed:
        ratio_final = float(executed[-1]["ratio_mean"])
    else:
        ratio_final = float("nan")

    # value explained variance on this rollout (post-update values re-eval not required;
    # use epoch returns vs pre-update values already in buffer)
    values = np.asarray(rollout["values"], dtype=np.float64)
    returns = np.asarray(rollout.get("returns") or [], dtype=np.float64)
    if returns.size == 0 and "advantages" in rollout:
        returns = values + np.asarray(rollout["advantages"], dtype=np.float64)
    if returns.size == values.size and values.size > 0:
        var_r = float(np.var(returns))
        if var_r > 1e-12:
            explained_variance = float(1.0 - np.var(returns - values) / var_r)
        else:
            explained_variance = 0.0
        value_error_mean = float(np.mean((returns - values) ** 2))
    else:
        explained_variance = float("nan")
        value_error_mean = float(stats.get("value_loss", float("nan")))

    cadence_ok = (
        int(rollout["meta"]["n_fast"]) == EXPECTED_N_FAST
        and int(rollout["meta"]["n_slow"]) == EXPECTED_N_SLOW
    )
    # task conservation: created >= completed + expired-ish counters finite
    tc = task_counts
    task_conservation_ok = cadence_ok and all(
        isinstance(tc.get(k), int) and tc.get(k) >= 0
        for k in ("completed", "created", "expired", "rejected")
    )
    finite_ok = bool(
        finite_lp
        and all(is_finite_number(x) for x in kls + clips)
        and is_finite_number(ratio_final)
        and is_finite_number(quality_utility)
        and is_finite_number(shaped_return)
    )
    obj = constrained_objective_from_episode(
        quality_utility=quality_utility,
        deadline_cost_sum=deadline_cost_sum,
        lambda_deadline=float(dual.deadline),
    )
    row = {
        "episode_index": int(episode_index),
        "episode_completed": int(episode_index) + 1,
        "seed": int(seed),
        "phase": "train",
        "device": device,
        "wall_time_s": float(time.perf_counter() - t0),
        "gpu_peak_memory_mib": float(peak) / (1024.0 * 1024.0),
        "n_fast": int(rollout["meta"]["n_fast"]),
        "n_slow": int(rollout["meta"]["n_slow"]),
        "return": shaped_return,
        "quality_utility": quality_utility,
        "constrained_objective": obj,
        "completed": int(task_counts["completed"]),
        "task_counts": task_counts,
        "deadline_cost_sum": deadline_cost_sum,
        "deadline_cost_mean": deadline_cost_mean,
        "deadline_cost_limit": float(cfg.deadline_cost_limit),
        "deadline_satisfied": bool(deadline_cost_mean <= float(cfg.deadline_cost_limit) + 1e-12),
        "dual": dual.to_trace(),
        "lambda_deadline": float(dual.deadline),
        "learning_rate": float(cfg.learning_rate),
        "entropy_weight": float(scheduled_entropy_weight(cfg, episode_index)),
        "epoch0_ratio_mean": epoch0_ratio_mean,
        "epoch0_logprob_finite": finite_lp,
        "update_epoch_stats": epochs,
        "n_update_epochs_executed": int(stats.get("n_update_epochs_executed") or 0),
        "kl_guard_stopped_early": bool(stats.get("kl_guard_stopped_early")),
        "kl_guard_stop_reason": stats.get("kl_guard_stop_reason"),
        "approx_kl_max_over_epochs": float(max(kls)) if kls else float("nan"),
        "clip_fraction_max_over_epochs": float(max(clips)) if clips else float("nan"),
        "ratio_mean_final_executed_epoch": ratio_final,
        "approx_kl": float(stats.get("approx_kl", float("nan"))),
        "clip_fraction": float(stats.get("clip_fraction", float("nan"))),
        "ratio_mean": float(stats.get("ratio_mean", float("nan"))),
        "entropy": float(stats.get("entropy", float("nan"))),
        "value_mean": float(stats.get("value_mean", float("nan"))),
        "return_mean": float(stats.get("return_mean", float("nan"))),
        "normalized_advantage_mean": float(
            stats.get("normalized_advantage_mean", float("nan"))
        ),
        "value_error_mean": value_error_mean,
        "explained_variance": explained_variance,
        "quality_utility_equals_completed": bool(
            abs(quality_utility - float(task_counts["completed"])) < 1e-9
        ),
        "finite_ok": finite_ok,
        "cadence_ok": cadence_ok,
        "task_conservation_ok": task_conservation_ok,
        "provenance_ok": bool(quality_meta.get("sha256")),
        "obs_norm_sha256": getattr(model, "obs_norm_sha256", None),
        "quality_table_sha256": quality_meta.get("sha256"),
        "result_grade": False,
        "c3000": False,
        "timestamp_utc": utc_now(),
    }
    return row


@torch.no_grad()
def run_validation_episode(
    *,
    env: JointControlGymEnv,
    model: JointTwoTimescaleActorCritic,
    dual: DualState,
    cfg: RLTrainConfig,
    seed: int,
    episode_completed: int,
    quality_meta: Mapping[str, Any],
) -> dict[str, Any]:
    assert_allowed_val_seed(seed)
    t0 = time.perf_counter()
    was_training = bool(model.training)
    model.eval()
    set_global_seeds(int(seed))
    # dual frozen: pass a copy so collect_episode can read λ for shaping but we never write back
    dual_frozen = DualState(
        deadline=float(dual.deadline),
        battery=float(dual.battery),
        conflict=float(dual.conflict),
        coverage=float(dual.coverage),
    )
    rollout = collect_episode(env, model, seed=int(seed), cfg=cfg, dual=dual_frozen)
    if was_training:
        model.train()

    task_counts = engine_task_counts(env)
    quality_utility = float(np.sum(rollout["utility_rewards"]))
    shaped_return = float(np.sum(rollout["rewards"]))
    deadline_cost_sum = float(np.sum(rollout["deadline_costs"]))
    deadline_cost_mean = (
        float(np.mean(rollout["deadline_costs"])) if rollout["deadline_costs"] else 0.0
    )
    values = np.asarray(rollout["values"], dtype=np.float64)
    returns = np.asarray(rollout.get("returns") or [], dtype=np.float64)
    if returns.size == 0 and rollout.get("advantages") is not None:
        returns = values + np.asarray(rollout["advantages"], dtype=np.float64)
    if returns.size == values.size and values.size > 0:
        var_r = float(np.var(returns))
        explained_variance = (
            float(1.0 - np.var(returns - values) / var_r) if var_r > 1e-12 else 0.0
        )
        value_error_mean = float(np.mean((returns - values) ** 2))
    else:
        explained_variance = float("nan")
        value_error_mean = float("nan")

    entropies = []
    # approximate mean entropy from stored old log-probs not available; use 0 placeholder
    # Validation row will also carry training entropy from last train update when aggregated.
    cadence_ok = (
        int(rollout["meta"]["n_fast"]) == EXPECTED_N_FAST
        and int(rollout["meta"]["n_slow"]) == EXPECTED_N_SLOW
    )
    tc = task_counts
    task_conservation_ok = cadence_ok and all(
        isinstance(tc.get(k), int) and tc.get(k) >= 0
        for k in ("completed", "created", "expired", "rejected")
    )
    finite_ok = bool(
        is_finite_number(quality_utility)
        and is_finite_number(shaped_return)
        and is_finite_number(deadline_cost_sum)
    )
    obj = constrained_objective_from_episode(
        quality_utility=quality_utility,
        deadline_cost_sum=deadline_cost_sum,
        lambda_deadline=float(dual.deadline),
    )
    return {
        "seed": int(seed),
        "episode_completed": int(episode_completed),
        "phase": "validation",
        "wall_time_s": float(time.perf_counter() - t0),
        "n_fast": int(rollout["meta"]["n_fast"]),
        "n_slow": int(rollout["meta"]["n_slow"]),
        "quality_utility": quality_utility,
        "return": shaped_return,
        "constrained_objective": obj,
        "completed": int(task_counts["completed"]),
        "task_counts": task_counts,
        "deadline_cost_sum": deadline_cost_sum,
        "deadline_cost_mean": deadline_cost_mean,
        "deadline_cost_limit": float(cfg.deadline_cost_limit),
        "deadline_satisfied": bool(
            deadline_cost_mean <= float(cfg.deadline_cost_limit) + 1e-12
        ),
        "lambda_deadline": float(dual.deadline),
        "value_error_mean": value_error_mean,
        "explained_variance": explained_variance,
        "finite_ok": finite_ok,
        "cadence_ok": cadence_ok,
        "task_conservation_ok": task_conservation_ok,
        "provenance_ok": bool(quality_meta.get("sha256")),
        "obs_norm_sha256": getattr(model, "obs_norm_sha256", None),
        "quality_table_sha256": quality_meta.get("sha256"),
        "result_grade": False,
        "timestamp_utc": utc_now(),
    }


def run_paired_validation(
    *,
    env: JointControlGymEnv,
    model: JointTwoTimescaleActorCritic,
    optimizer: torch.optim.Optimizer,
    dual: DualState,
    cfg: RLTrainConfig,
    episode_completed: int,
    quality_meta: Mapping[str, Any],
    last_train_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """No-update validation on VAL_SEEDS; asserts model/optimizer hash unchanged."""
    pre_model = state_dict_sha(model)
    pre_opt = optimizer_state_sha(optimizer)
    pre_dual = dual.to_trace()
    rows = []
    for seed in VAL_SEEDS:
        rows.append(
            run_validation_episode(
                env=env,
                model=model,
                dual=dual,
                cfg=cfg,
                seed=seed,
                episode_completed=episode_completed,
                quality_meta=quality_meta,
            )
        )
    post_model = state_dict_sha(model)
    post_opt = optimizer_state_sha(optimizer)
    post_dual = dual.to_trace()
    if pre_model != post_model or pre_opt != post_opt or pre_dual != post_dual:
        raise RuntimeError(
            "VALIDATION MUTATED STATE: model/optimizer/dual hash changed during validation"
        )
    agg = aggregate_validation_metrics(rows)
    # attach entropy from last train for safety gates
    entropy_mean = float(last_train_row["entropy"]) if last_train_row else float("nan")
    out = {
        "episode_completed": int(episode_completed),
        "seeds": list(VAL_SEEDS),
        "per_seed": rows,
        **agg,
        "entropy_mean": entropy_mean,
        "lambda_deadline": float(dual.deadline),
        "model_hash_pre": pre_model,
        "model_hash_post": post_model,
        "optimizer_hash_pre": pre_opt,
        "optimizer_hash_post": post_opt,
        "dual_unchanged": pre_dual == post_dual,
        "no_param_update": True,
        "timestamp_utc": utc_now(),
        "result_grade": False,
    }
    return out


# ---------------------------------------------------------------------------
# Baselines / upper bounds (public env interface only)
# ---------------------------------------------------------------------------


def evaluate_scripted_on_validation(
    *,
    env: JointControlGymEnv,
    policy: Any,
    seeds: Sequence[int],
    policy_id: str,
    cfg: RLTrainConfig,
) -> dict[str, Any]:
    """Run a scripted policy on validation seeds via public Gym API only."""
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        assert_allowed_val_seed(int(seed))
        set_global_seeds(int(seed))
        reset_out = env.reset(seed=int(seed))
        if isinstance(reset_out, tuple):
            obs, _info = reset_out
        else:
            obs = reset_out
        policy.reset(
            seed=int(seed),
            action_space=env.action_space,
            observation_space=env.observation_space,
        )
        utilities: list[float] = []
        deadlines: list[float] = []
        n_fast = 0
        n_slow = 0
        while True:
            due = bool(getattr(env, "pre_step_mobility_decision_due", False))
            if due:
                n_slow += 1
            decision = policy.act(
                obs, deterministic=True, mobility_decision_due=due
            )
            step_out = env.step(decision.action)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, info = step_out
            channels = dict(info.get("reward_channels") or {})
            util = float(channels.get("utility", reward))
            utilities.append(util)
            deadlines.append(float(channels.get("deadline", 0.0) or 0.0))
            n_fast += 1
            obs = next_obs
            if done:
                break
        task_counts = engine_task_counts(env)
        qu = float(sum(utilities))
        dc_mean = float(np.mean(deadlines)) if deadlines else 0.0
        # baselines use λ=0 for objective reporting (primal utility); also record
        # constrained form at λ=0 identical to utility
        per_seed.append(
            {
                "seed": int(seed),
                "quality_utility": qu,
                "constrained_objective": qu,
                "deadline_cost_mean": dc_mean,
                "deadline_satisfied": bool(
                    dc_mean <= float(cfg.deadline_cost_limit) + 1e-12
                ),
                "completed": int(task_counts["completed"]),
                "task_counts": task_counts,
                "n_fast": n_fast,
                "n_slow": n_slow,
                "cadence_ok": n_fast == EXPECTED_N_FAST and n_slow == EXPECTED_N_SLOW,
                "finite_ok": is_finite_number(qu),
            }
        )
    objs = [float(r["constrained_objective"]) for r in per_seed]
    return {
        "policy_id": policy_id,
        "status": "OK",
        "seeds": list(map(int, seeds)),
        "per_seed": per_seed,
        "constrained_objective_mean": mean(objs),
        "quality_utility_mean": mean([float(r["quality_utility"]) for r in per_seed]),
        "causal": True,
        "result_grade": False,
        "timestamp_utc": utc_now(),
    }


def run_comparator_inventory_and_baselines(
    *,
    env: JointControlGymEnv,
    cfg: RLTrainConfig,
    out_dir: Path,
) -> dict[str, Any]:
    inventory: list[dict[str, Any]] = []
    baseline_doc: dict[str, Any] = {
        "schema": "bubbles_vqa.c1000.baseline_validation.v1",
        "seeds": list(VAL_SEEDS),
        "created_utc": utc_now(),
        "comparators": {},
    }
    upper_doc: dict[str, Any] = {
        "schema": "bubbles_vqa.c1000.upper_bound_validation.v1",
        "seeds": list(VAL_SEEDS),
        "created_utc": utc_now(),
        "comparators": {},
    }

    # B1 / null / stay-only
    for name, factory, causal in (
        ("B1_null", NullJointControlPolicy, True),
        ("B1_stay_only", StayOnlyMobilityPolicy, True),
    ):
        try:
            pol = factory()
            res = evaluate_scripted_on_validation(
                env=env, policy=pol, seeds=VAL_SEEDS, policy_id=name, cfg=cfg
            )
            baseline_doc["comparators"][name] = res
            inventory.append(
                {
                    "id": name,
                    "status": "OK",
                    "causal": causal,
                    "role": "baseline",
                    "policy_id": getattr(pol, "policy_id", name),
                }
            )
        except Exception as exc:
            baseline_doc["comparators"][name] = {
                "policy_id": name,
                "status": "UNAVAILABLE",
                "reason": str(exc),
            }
            inventory.append(
                {
                    "id": name,
                    "status": "UNAVAILABLE",
                    "causal": causal,
                    "role": "baseline",
                    "reason": str(exc),
                }
            )

    # Greedy / primal-dual: no audited public adapter in this worktree
    for name, reason in (
        (
            "greedy_executable",
            "No audited public greedy joint-control policy adapter is wired "
            "for external_structural via the public Gym interface.",
        ),
        (
            "primal_dual_baseline",
            "No audited executable primal-dual baseline adapter exists in "
            "bubbles_vqa.rl_interface for this environment; dual is training-only.",
        ),
    ):
        baseline_doc["comparators"][name] = {
            "policy_id": name,
            "status": "UNAVAILABLE",
            "reason": reason,
        }
        inventory.append(
            {
                "id": name,
                "status": "UNAVAILABLE",
                "causal": True,
                "role": "baseline",
                "reason": reason,
            }
        )

    # B3 forecast/quota — adapter marked NOT_IMPLEMENTED
    b3 = ForecastQuotaMobilityPolicyAdapter
    reason_b3 = (
        "ForecastQuotaMobilityPolicyAdapter.NOT_IMPLEMENTED=True; requires private "
        "engine state (contract §7). Not substituted."
    )
    baseline_doc["comparators"]["B3_forecast_quota"] = {
        "policy_id": getattr(b3, "policy_id", "forecast_quota_b3_v1"),
        "status": "UNAVAILABLE",
        "reason": reason_b3,
        "implementation_status": getattr(b3, "IMPLEMENTATION_STATUS", "NOT_IMPLEMENTED"),
    }
    inventory.append(
        {
            "id": "B3_forecast_quota",
            "status": "UNAVAILABLE",
            "causal": True,
            "role": "baseline",
            "reason": reason_b3,
        }
    )

    # B4 oracle — non-causal upper bound; no public adapter
    reason_b4 = (
        "No public-oracle joint-control adapter is implemented for validation traces. "
        "Marked UNAVAILABLE rather than fabricating a substitute. Non-causal upper bound."
    )
    upper_doc["comparators"]["B4_oracle"] = {
        "policy_id": "B4_oracle",
        "status": "UNAVAILABLE",
        "reason": reason_b4,
        "causal": False,
        "role": "non_causal_upper_bound",
    }
    inventory.append(
        {
            "id": "B4_oracle",
            "status": "UNAVAILABLE",
            "causal": False,
            "role": "non_causal_upper_bound",
            "reason": reason_b4,
        }
    )

    # best causal among available
    causal_scores = []
    for k, v in baseline_doc["comparators"].items():
        if v.get("status") == "OK" and is_finite_number(v.get("constrained_objective_mean")):
            causal_scores.append((k, float(v["constrained_objective_mean"])))
    best_causal = None
    best_causal_id = None
    if causal_scores:
        best_causal_id, best_causal = max(causal_scores, key=lambda t: t[1])
    baseline_doc["best_causal_baseline_id"] = best_causal_id
    baseline_doc["best_causal_baseline_objective"] = best_causal
    upper_doc["oracle_upper_objective"] = None
    upper_doc["oracle_status"] = "UNAVAILABLE"

    inv_doc = {
        "schema": "bubbles_vqa.c1000.comparator_inventory.v1",
        "created_utc": utc_now(),
        "inventory": inventory,
        "best_causal_baseline_id": best_causal_id,
        "best_causal_baseline_objective": best_causal,
        "oracle_upper_objective": None,
        "note": "Unavailable comparators must not be substituted with invented data.",
    }

    atomic_write_json(out_dir / "baseline_validation.json", baseline_doc)
    atomic_write_json(out_dir / "upper_bound_validation.json", upper_doc)
    atomic_write_json(out_dir / "comparator_inventory.json", inv_doc)
    return {
        "inventory": inv_doc,
        "baseline": baseline_doc,
        "upper": upper_doc,
        "best_causal_baseline": best_causal,
        "oracle_upper": None,
    }


# ---------------------------------------------------------------------------
# Monitoring / status
# ---------------------------------------------------------------------------


def write_monitoring_summary(
    path: Path,
    *,
    status: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any] | None,
    gpu: Mapping[str, Any],
    eta_s: float | None,
) -> None:
    lines = [
        "# C1000 Monitoring Summary",
        "",
        f"- updated_utc: {utc_now()}",
        f"- status: {status.get('status')}",
        f"- episodes_completed: {status.get('episodes_completed')}",
        f"- decision: {(decision or {}).get('decision')}",
        f"- gpu: {json.dumps(gpu, sort_keys=True)}",
        f"- eta_s: {eta_s}",
        "",
        "## Latest train metrics",
    ]
    if train_rows:
        r = train_rows[-1]
        lines.extend(
            [
                f"- episode_index: {r.get('episode_index')}",
                f"- seed: {r.get('seed')}",
                f"- quality_utility: {r.get('quality_utility')}",
                f"- constrained_objective: {r.get('constrained_objective')}",
                f"- approx_kl_max: {r.get('approx_kl_max_over_epochs')}",
                f"- clip_max: {r.get('clip_fraction_max_over_epochs')}",
                f"- ratio_final: {r.get('ratio_mean_final_executed_epoch')}",
                f"- entropy: {r.get('entropy')}",
                f"- lambda_deadline: {r.get('lambda_deadline')}",
                f"- finite_ok: {r.get('finite_ok')} cadence_ok: {r.get('cadence_ok')}",
            ]
        )
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Latest validation")
    if val_rows:
        v = val_rows[-1]
        lines.extend(
            [
                f"- episode_completed: {v.get('episode_completed')}",
                f"- constrained_objective: {v.get('constrained_objective')}",
                f"- quality_utility_mean: {v.get('quality_utility_mean')}",
                f"- deadline_satisfied: {v.get('deadline_satisfied')}",
            ]
        )
    else:
        lines.append("- (none)")
    lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_claude_failure_package(out_dir: Path, decision: Mapping[str, Any]) -> None:
    """Read-only attribution package + Claude Opus instruction (no algorithm edits)."""
    pkg = out_dir / "claude_opus_failure_package"
    pkg.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        pkg / "decision_snapshot.json",
        dict(decision),
    )
    instr = f"""# Claude Opus read-only failure attribution (C1000)

## Model requirement
Use Claude model **opus** only.

## Hard prohibitions
- Do **not** modify the failed run artifacts or code in-place.
- Do **not** access final seeds 80000001-80000020.
- Do **not** start C3000.
- Do **not** retune LR / entropy / dual / network / reward on this failed run.
- Grok must not change the algorithm; only Opus may recommend a supervisor-visible amendment.

## Decision
```json
{json.dumps(decision, indent=2, sort_keys=True, default=str)}
```

## Required classification (one or more)
- OPTIMIZATION_FAILURE
- EXPLORATION_OR_ENTROPY_FAILURE
- VALUE_OR_CREDIT_FAILURE
- DUAL_OR_CONSTRAINT_FAILURE
- REWARD_OR_OBJECTIVE_MISALIGNMENT
- ACTION_OR_ENVIRONMENT_INTERFACE_FAILURE
- SCENE_HAS_NO_LEARNABLE_ADVANTAGE
- BASELINE_OR_UPPER_BOUND_INVALID
- INSUFFICIENT_BUDGET_STILL_IMPROVING

## Inputs
Read-only under:
- `{out_dir}`
- G1V freeze: `{DEFAULT_G1V_DIR}`
- Contracts dated 2026-07-23

## Output
Write a structured attribution report under this package directory. Recommend an
amendment only if evidence supports it; do not edit the failed run.
"""
    (pkg / "CLAUDE_OPUS_INSTRUCTIONS.md").write_text(instr, encoding="utf-8")
    atomic_write_json(
        pkg / "package_manifest.json",
        {
            "created_utc": utc_now(),
            "model_required": "opus",
            "read_only": True,
            "final_seeds_access": False,
            "c3000_started": False,
            "decision": decision.get("decision"),
        },
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


class C1000Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.lock = SingleInstanceLock(self.out_dir / LOCK_NAME)
        self._stop_requested = False
        self._interrupted = False
        self.device = resolve_device(args.device)
        self.cfg = RLTrainConfig(
            device=self.device,
            update_epochs=4,
            train_episodes=MAX_EPISODES,
            learning_rate=LEARNING_RATE,
            result_grade=False,
        )
        self.cfg.device = self.device
        self.quality_meta = load_quality_table_meta(Path(args.online_quality_table))
        self.env: JointControlGymEnv | None = None
        self.model: JointTwoTimescaleActorCritic | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.dual: DualState | None = None
        self.train_rows: list[dict[str, Any]] = []
        self.val_rows: list[dict[str, Any]] = []
        self.checkpoint_index: list[dict[str, Any]] = []
        self.validation_history: list[dict[str, Any]] = []
        self.decision: dict[str, Any] | None = None
        self.best_causal_baseline: float | None = None
        self.oracle_upper: float | None = None
        self.start_episode = 0
        self.episode_wall_times: list[float] = []
        self.freeze_commit = str(args.freeze_commit or "") or None
        self.provenance: dict[str, Any] = {}
        self.journal_sequence: int = 0
        self.last_train_row: dict[str, Any] | None = None

    def _next_journal_sequence(self) -> int:
        self.journal_sequence = int(self.journal_sequence) + 1
        return self.journal_sequence

    def _append_journal(self, kind: str, payload: Mapping[str, Any]) -> int:
        seq = self._next_journal_sequence()
        entry = {
            "journal_sequence": seq,
            "kind": kind,
            "timestamp_utc": utc_now(),
            **dict(payload),
        }
        append_jsonl(self.out_dir / WRITE_JOURNAL, entry)
        return seq

    def _install_signals(self) -> None:
        def _handler(signum, _frame):
            self._stop_requested = True
            self._interrupted = True
            print(f"[C1000] signal {signum} → INTERRUPTED_RESUMABLE save", flush=True)

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    def _write_status(self, **kwargs: Any) -> None:
        path = self.out_dir / STATUS_NAME
        cur = {}
        if path.exists():
            try:
                cur = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                cur = {}
        cur.update(kwargs)
        cur["updated_utc"] = utc_now()
        cur["pid"] = os.getpid()
        cur["result_grade"] = False
        cur["c3000_started"] = False
        atomic_write_json(path, cur)

    def _refuse_if_decided(self) -> None:
        dec_path = self.out_dir / DECISION_NAME
        if dec_path.exists():
            doc = json.loads(dec_path.read_text(encoding="utf-8"))
            d = doc.get("decision")
            if d in DECISION_LABELS:
                raise SystemExit(
                    f"refusing to continue: already decided {d} in {dec_path}"
                )

    def _refuse_if_dirty_after_freeze(self) -> None:
        if not self.args.require_clean_tree:
            return
        if git_dirty(REPO):
            raise SystemExit(
                "refusing to start formal C1000: git worktree dirty after freeze snapshot"
            )
        if self.freeze_commit:
            head = git_head(REPO)
            if head != self.freeze_commit:
                raise SystemExit(
                    f"refusing to start: HEAD {head} != freeze_commit {self.freeze_commit}"
                )

    def _refuse_c3000_flags(self) -> None:
        # Hard guard: this binary never starts C3000
        if getattr(self.args, "start_c3000", False):
            raise SystemExit("C3000 start is forbidden in this runner")

    def setup(self) -> None:
        self._refuse_c3000_flags()
        self._refuse_if_decided()
        self._refuse_if_dirty_after_freeze()
        self.lock.acquire()
        atexit.register(self.lock.release)
        self._install_signals()

        # Drop confirmed-invalid atomic-write leftovers before any resume I/O.
        cleaned = cleanup_stale_temp_files(self.out_dir)
        if cleaned:
            append_reconciliation_log(
                self.out_dir,
                {
                    "event": "cleanup_stale_temp_files",
                    "cleaned": cleaned,
                    "count": len(cleaned),
                },
            )
            print(f"[C1000] cleaned {len(cleaned)} stale temp file(s)", flush=True)

        # Quality / obs-norm freeze checks
        qsha = str(self.quality_meta.get("sha256") or "")
        if not qsha.startswith(EXPECTED_QUALITY_SHA_PREFIX):
            raise RuntimeError(
                f"quality table SHA mismatch: got {qsha}, expected prefix {EXPECTED_QUALITY_SHA_PREFIX}"
            )
        stats_path = Path(self.args.obs_norm_stats)
        stats = load_obs_norm_stats(stats_path, verify_sha=True)
        if stats.sha256 != EXPECTED_OBS_NORM_SHA:
            raise RuntimeError(
                f"obs_norm SHA mismatch: got {stats.sha256}, expected {EXPECTED_OBS_NORM_SHA}"
            )

        self.env = build_env(
            horizon_s=float(self.args.horizon_s),
            lambda_total_per_s=float(self.args.lambda_total_per_s),
            quality_table=Path(self.args.online_quality_table),
        )
        if int(self.env.horizon_slots) != EXPECTED_N_FAST:
            if float(self.args.horizon_s) == HORIZON_S and not self.args.allow_short_horizon:
                raise RuntimeError(
                    f"expected horizon_slots={EXPECTED_N_FAST}, got {self.env.horizon_slots}"
                )

        self.provenance = {
            "contract": CONTRACT,
            "stability_contract": STABILITY_CONTRACT,
            "git_commit": git_head(REPO),
            "freeze_commit": self.freeze_commit,
            "dirty_at_start": git_dirty(REPO),
            "device": self.device,
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "learning_rate": LEARNING_RATE,
            "target_kl_guard": TARGET_KL_GUARD,
            "train_seeds": TRAIN_SEEDS,
            "val_seeds": VAL_SEEDS,
            "final_seeds_access": False,
            "c3000_started": False,
            "c3000_authorized": False,
            "quality_table": self.quality_meta,
            "obs_norm_sha256": stats.sha256,
            "obs_norm_path": str(stats_path.resolve()),
            "shared_init": str(Path(self.args.load_init_checkpoint).resolve()),
            "shared_init_sha256": sha256_file(self.args.load_init_checkpoint),
            "created_utc": utc_now(),
            "result_grade": False,
            "max_episodes": MAX_EPISODES,
            "checkpoint_interval": CKPT_INTERVAL,
            "validation_interval": VAL_INTERVAL,
            "min_decision_episode": MIN_DECISION_EPISODE,
            "dry_run": bool(self.args.dry_run),
        }

        # Load G1V shared init with obs norm
        model, _cfg, _payload = load_checkpoint(
            Path(self.args.load_init_checkpoint), device=self.device
        )
        model.to(self.device)
        model.train()
        if not model.obs_norm_enabled or model.obs_norm_sha256 != EXPECTED_OBS_NORM_SHA:
            # re-install frozen stats if needed
            model.set_obs_normalization(stats)
        if model.obs_norm_sha256 != EXPECTED_OBS_NORM_SHA:
            raise RuntimeError(
                f"model obs_norm_sha256={model.obs_norm_sha256} != {EXPECTED_OBS_NORM_SHA}"
            )
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(self.cfg.learning_rate)
        )
        self.dual = init_dual_state(self.cfg)

        # Resume if requested / latest exists. Checkpoint next_episode_index is
        # the sole authority for training progress; JSONL is reconciled to it.
        resume_path = Path(self.args.resume) if self.args.resume else None
        if resume_path is None and (self.out_dir / "checkpoints" / "checkpoint_latest.pt").exists():
            if self.args.auto_resume:
                resume_path = self.out_dir / "checkpoints" / "checkpoint_latest.pt"
        ckpt_validation_history: list[dict[str, Any]] = []
        if resume_path is not None and resume_path.exists():
            resume = load_full_checkpoint(
                resume_path,
                device=self.device,
                model=self.model,
                optimizer=self.optimizer,
                dual=self.dual,
            )
            self.start_episode = int(resume.get("next_episode_index", 0))
            ckpt_validation_history = list(resume.get("validation_history") or [])
            self.journal_sequence = int(resume.get("journal_sequence") or 0)
            ltr = resume.get("last_train_row")
            self.last_train_row = dict(ltr) if isinstance(ltr, Mapping) else None
            print(
                f"[C1000] resumed from {resume_path} next_episode={self.start_episode}",
                flush=True,
            )
        else:
            self.start_episode = 0
            self.journal_sequence = 0
            self.last_train_row = None

        # Training JSONL: checkpoint next_episode_index is authoritative.
        self.train_rows = reconcile_training_metrics(
            self.out_dir / TRAIN_METRICS,
            self.start_episode,
            out_dir=self.out_dir,
        )
        if self.train_rows:
            self.last_train_row = dict(self.train_rows[-1])

        # Validation: merge checkpoint history with disk JSONL (recover legal
        # disk rows the last checkpoint may not yet contain).
        disk_val_rows = read_jsonl(self.out_dir / VAL_METRICS)
        self.validation_history = merge_validation_history(
            ckpt_validation_history,
            disk_val_rows,
            next_episode_index=self.start_episode,
            out_dir=self.out_dir,
        )
        self.val_rows = list(self.validation_history)
        # Persist recovered validation history so disk matches logical state.
        if (
            canonical_row_for_compare({"rows": self.validation_history})
            != canonical_row_for_compare({"rows": disk_val_rows})
            or len(disk_val_rows) != len(self.validation_history)
        ):
            atomic_rewrite_jsonl(self.out_dir / VAL_METRICS, self.validation_history)

        if (self.out_dir / CKPT_INDEX).exists():
            self.checkpoint_index = json.loads(
                (self.out_dir / CKPT_INDEX).read_text(encoding="utf-8")
            ).get("checkpoints", [])
        if (self.out_dir / DECISION_NAME).exists():
            self.decision = json.loads(
                (self.out_dir / DECISION_NAME).read_text(encoding="utf-8")
            )

        # Config / manifest (do not clobber freeze fields on resume)
        if not (self.out_dir / CONFIG_NAME).exists() or self.start_episode == 0:
            atomic_write_json(
                self.out_dir / CONFIG_NAME,
                {**self.cfg.to_dict(), **self.provenance},
            )
        atomic_write_json(
            self.out_dir / "pid_or_unit.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_utc": utc_now(),
                "unit": os.environ.get("INVOCATION_ID") or None,
                "out_dir": str(self.out_dir),
            },
        )
        self._write_status(
            status="STARTING",
            stage="C1000",
            episodes_completed=self.start_episode,
            provenance=self.provenance,
            journal_sequence=self.journal_sequence,
            validation_history_len=len(self.validation_history),
        )

    def ensure_baselines(self) -> None:
        assert self.env is not None and self.cfg is not None
        base_path = self.out_dir / "baseline_validation.json"
        inv_path = self.out_dir / "comparator_inventory.json"
        if base_path.exists() and inv_path.exists() and not self.args.rerun_baselines:
            inv = json.loads(inv_path.read_text(encoding="utf-8"))
            self.best_causal_baseline = inv.get("best_causal_baseline_objective")
            self.oracle_upper = inv.get("oracle_upper_objective")
            print(
                f"[C1000] loaded baselines best_causal={self.best_causal_baseline}",
                flush=True,
            )
            return
        print("[C1000] running baseline/upper-bound precheck on validation seeds", flush=True)
        res = run_comparator_inventory_and_baselines(
            env=self.env, cfg=self.cfg, out_dir=self.out_dir
        )
        self.best_causal_baseline = res["best_causal_baseline"]
        self.oracle_upper = res["oracle_upper"]

    def _maybe_checkpoint(self, episode_index: int) -> None:
        assert self.model and self.optimizer and self.dual and self.cfg
        completed = episode_index + 1
        if completed % CKPT_INTERVAL != 0 and completed < (
            self.args.max_episodes if self.args.dry_run else MAX_EPISODES
        ):
            # still always keep latest after each episode for resume
            pass
        ckpt_dir = self.out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        rng = capture_rng_state()
        latest = ckpt_dir / "checkpoint_latest.pt"
        last_row = self.last_train_row
        if last_row is None and self.train_rows:
            last_row = self.train_rows[-1]
        jseq = self._append_journal(
            "checkpoint",
            {
                "episode_index": int(episode_index),
                "next_episode_index": int(episode_index) + 1,
                "validation_history_len": len(self.validation_history),
            },
        )
        save_full_checkpoint(
            latest,
            model=self.model,
            optimizer=self.optimizer,
            dual=self.dual,
            cfg=self.cfg,
            episode_index=episode_index,
            seed_cycle_position=(episode_index + 1) % len(TRAIN_SEEDS),
            rng_state=rng,
            provenance=self.provenance,
            validation_history=self.validation_history,
            decision=self.decision,
            last_train_row=last_row,
            journal_sequence=jseq,
        )
        entry = {
            "episode_index": episode_index,
            "episode_completed": completed,
            "path": str(latest),
            "timestamp_utc": utc_now(),
            "is_interval": completed % CKPT_INTERVAL == 0,
            "journal_sequence": jseq,
        }
        if completed % CKPT_INTERVAL == 0:
            named = ckpt_dir / f"checkpoint_ep{completed:04d}.pt"
            save_full_checkpoint(
                named,
                model=self.model,
                optimizer=self.optimizer,
                dual=self.dual,
                cfg=self.cfg,
                episode_index=episode_index,
                seed_cycle_position=(episode_index + 1) % len(TRAIN_SEEDS),
                rng_state=rng,
                provenance=self.provenance,
                validation_history=self.validation_history,
                decision=self.decision,
                last_train_row=last_row,
                journal_sequence=jseq,
            )
            entry["interval_path"] = str(named)
            entry["path"] = str(named)
            self.checkpoint_index.append(entry)
            atomic_write_json(
                self.out_dir / CKPT_INDEX,
                {"checkpoints": self.checkpoint_index, "updated_utc": utc_now()},
            )
        else:
            # update latest pointer only
            atomic_write_json(
                self.out_dir / CKPT_INDEX,
                {
                    "checkpoints": self.checkpoint_index,
                    "latest": entry,
                    "updated_utc": utc_now(),
                },
            )

        # monitoring every 25
        if completed % CKPT_INTERVAL == 0:
            eta = None
            if self.episode_wall_times:
                rem = (self.args.max_episodes if self.args.dry_run else MAX_EPISODES) - completed
                eta = float(np.mean(self.episode_wall_times[-10:])) * rem
            write_monitoring_summary(
                self.out_dir / MONITOR_MD,
                status={
                    "status": "RUNNING",
                    "episodes_completed": completed,
                },
                train_rows=self.train_rows,
                val_rows=self.validation_history,
                decision=self.decision,
                gpu=gpu_snapshot(),
                eta_s=eta,
            )

    def _run_validation_if_due(self, episode_index: int, train_row: dict[str, Any]) -> None:
        assert self.env and self.model and self.optimizer and self.dual and self.cfg
        completed = episode_index + 1
        if completed % VAL_INTERVAL != 0:
            return
        print(f"[C1000] validation at episode_completed={completed}", flush=True)
        val = run_paired_validation(
            env=self.env,
            model=self.model,
            optimizer=self.optimizer,
            dual=self.dual,
            cfg=self.cfg,
            episode_completed=completed,
            quality_meta=self.quality_meta,
            last_train_row=train_row,
        )
        jseq = self._append_journal(
            "validation_metrics",
            {"episode_completed": int(completed)},
        )
        val = dict(val)
        val["journal_sequence"] = jseq
        self.validation_history.append(val)
        append_jsonl(self.out_dir / VAL_METRICS, val)
        self.val_rows = self.validation_history
        # Re-save latest so validation history is not lost if we crash before
        # the next train episode checkpoint.
        self._maybe_checkpoint(episode_index)

    def _evaluate_decision(self, episode_index: int) -> dict[str, Any]:
        completed = episode_index + 1
        # Formal terminal only at the 1000-episode cap. Dry-runs must never
        # emit UNSTABLE solely because the short budget lacks a 5-checkpoint window.
        force_terminal = (not self.args.dry_run) and completed >= MAX_EPISODES
        decision = decide_c1000(
            self.validation_history,
            self.train_rows,
            episode_index=episode_index,
            best_causal_baseline=self.best_causal_baseline,
            oracle_upper=self.oracle_upper,
            deadline_lambda_max=float(self.cfg.deadline_lambda_max),
            force_terminal=force_terminal,
        )
        # Never emit STABLE_STILL_IMPROVING before 1000 formal episodes
        if (
            decision.get("decision") == C1000_STABLE_STILL_IMPROVING
            and completed < MAX_EPISODES
        ):
            decision = dict(decision)
            decision["decision"] = None
            decision["stop"] = False
            decision["reasons"] = list(decision.get("reasons") or []) + [
                "stable_still_improving_suppressed_before_1000"
            ]
        # Dry-run: do not freeze formal decision artifacts
        if self.args.dry_run and decision.get("decision") in DECISION_LABELS:
            decision = dict(decision)
            decision["dry_run_suppressed_decision"] = decision.get("decision")
            decision["decision"] = None
            decision["stop"] = False
            decision["reasons"] = list(decision.get("reasons") or []) + [
                "dry_run_no_formal_decision"
            ]
        # Attach best checkpoint selection when deciding
        if decision.get("decision") in DECISION_LABELS:
            decision["selected_checkpoint"] = select_best_checkpoint(
                self.validation_history, self.checkpoint_index
            )
            decision["c3000_started"] = False
            decision["c3000_authorized"] = (
                decision.get("decision") == C1000_STABLE_STILL_IMPROVING
            )
            decision["final_seeds_access"] = False
            atomic_write_json(self.out_dir / DECISION_NAME, decision)
            self.decision = decision
            if decision.get("decision") == C1000_UNSTABLE_OR_INVALID:
                write_claude_failure_package(self.out_dir, decision)
        return decision

    def run(self) -> int:
        assert self.env and self.model and self.optimizer and self.dual
        self.ensure_baselines()
        max_ep = int(self.args.max_episodes) if self.args.dry_run else MAX_EPISODES
        if self.args.dry_run:
            print(f"[C1000] DRY RUN max_episodes={max_ep} out={self.out_dir}", flush=True)

        self._write_status(status="RUNNING", episodes_completed=self.start_episode)
        atomic_write_json(
            self.out_dir / RUN_MANIFEST,
            {
                "status": "RUNNING",
                "started_utc": utc_now(),
                "start_episode": self.start_episode,
                "max_episodes": max_ep,
                "dry_run": bool(self.args.dry_run),
                "provenance": self.provenance,
                "c3000_started": False,
                "final_seeds_access": False,
                "result_grade": False,
            },
        )

        for episode_index in range(self.start_episode, max_ep):
            if self._stop_requested:
                break
            seed = seed_at_episode(episode_index)
            print(
                f"[C1000] train ep={episode_index} seed={seed} "
                f"lr={self.cfg.learning_rate} ent_w={scheduled_entropy_weight(self.cfg, episode_index):.4g}",
                flush=True,
            )
            row = run_train_episode(
                env=self.env,
                model=self.model,
                optimizer=self.optimizer,
                dual=self.dual,
                cfg=self.cfg,
                seed=seed,
                episode_index=episode_index,
                device=self.device,
                quality_meta=self.quality_meta,
            )
            jseq = self._append_journal(
                "training_metrics",
                {"episode_index": int(episode_index), "seed": int(seed)},
            )
            row = dict(row)
            row["journal_sequence"] = jseq
            self.train_rows.append(row)
            self.last_train_row = row
            append_jsonl(self.out_dir / TRAIN_METRICS, row)
            self.episode_wall_times.append(float(row["wall_time_s"]))

            # always save resumable latest (includes last_train_row + val history)
            self._maybe_checkpoint(episode_index)

            # Cooperative interrupt after durable train checkpoint.
            if self._stop_requested:
                break

            # validation
            try:
                self._run_validation_if_due(episode_index, row)
            except Exception as exc:
                print(f"[C1000] validation error: {exc}", flush=True)
                decision = {
                    "decision": C1000_UNSTABLE_OR_INVALID,
                    "stop": True,
                    "reasons": [f"validation_exception:{exc}"],
                    "completed_episodes": episode_index + 1,
                    "timestamp_utc": utc_now(),
                    "c3000_started": False,
                    "final_seeds_access": False,
                }
                atomic_write_json(self.out_dir / DECISION_NAME, decision)
                write_claude_failure_package(self.out_dir, decision)
                self._write_status(status=C1000_UNSTABLE_OR_INVALID, decision=decision)
                return 2

            decision = self._evaluate_decision(episode_index)
            eta = None
            if self.episode_wall_times:
                rem = max_ep - (episode_index + 1)
                eta = float(np.mean(self.episode_wall_times[-10:])) * rem
            # Never paint RUNNING over an in-flight cooperative interrupt.
            if self._stop_requested or self._interrupted:
                break
            self._write_status(
                status="RUNNING" if not decision.get("stop") else decision.get("decision"),
                episodes_completed=episode_index + 1,
                last_seed=seed,
                last_quality_utility=row.get("quality_utility"),
                last_kl=row.get("approx_kl_max_over_epochs"),
                eta_s=eta,
                gpu=gpu_snapshot(),
                decision=decision.get("decision"),
                journal_sequence=self.journal_sequence,
            )
            print(
                f"[C1000] ep={episode_index} done q_util={row['quality_utility']:.4f} "
                f"kl_max={row['approx_kl_max_over_epochs']:.6g} "
                f"clip={row['clip_fraction_max_over_epochs']:.4g} "
                f"ratio={row['ratio_mean_final_executed_epoch']:.4g} "
                f"ent={row['entropy']:.4g} guard={row['kl_guard_stopped_early']} "
                f"decision={decision.get('decision')}",
                flush=True,
            )

            if decision.get("stop") and decision.get("decision") in STOP_IMMEDIATE:
                break
            if decision.get("stop") and decision.get("decision") == C1000_STABLE_STILL_IMPROVING:
                break
            if not row.get("finite_ok", True):
                decision = {
                    "decision": C1000_UNSTABLE_OR_INVALID,
                    "stop": True,
                    "reasons": ["nonfinite_train_metrics"],
                    "completed_episodes": episode_index + 1,
                    "c3000_started": False,
                    "final_seeds_access": False,
                    "timestamp_utc": utc_now(),
                }
                atomic_write_json(self.out_dir / DECISION_NAME, decision)
                write_claude_failure_package(self.out_dir, decision)
                break

        # Interrupted? Durable save first, then status must leave RUNNING.
        if self._interrupted or self._stop_requested:
            if self.train_rows:
                self._maybe_checkpoint(int(self.train_rows[-1]["episode_index"]))
            completed = len(self.train_rows)
            self._write_status(
                status="INTERRUPTED_RESUMABLE",
                episodes_completed=completed,
                ended_utc=utc_now(),
                journal_sequence=self.journal_sequence,
                validation_history_len=len(self.validation_history),
            )
            atomic_write_json(
                self.out_dir / RUN_MANIFEST,
                {
                    "status": "INTERRUPTED_RESUMABLE",
                    "ended_utc": utc_now(),
                    "episodes_completed": completed,
                    "start_episode_on_resume": completed,
                    "c3000_started": False,
                    "final_seeds_access": False,
                    "result_grade": False,
                    "journal_sequence": self.journal_sequence,
                },
            )
            print("[C1000] INTERRUPTED_RESUMABLE", flush=True)
            return 130

        final_decision = self.decision
        if final_decision is None and self.train_rows:
            final_decision = self._evaluate_decision(
                int(self.train_rows[-1]["episode_index"])
            )

        status = (final_decision or {}).get("decision") or "COMPLETED_NO_DECISION"
        self._write_status(
            status=status,
            episodes_completed=len(self.train_rows),
            decision=(final_decision or {}).get("decision"),
            ended_utc=utc_now(),
        )
        atomic_write_json(
            self.out_dir / RUN_MANIFEST,
            {
                "status": status,
                "ended_utc": utc_now(),
                "episodes_completed": len(self.train_rows),
                "decision": final_decision,
                "c3000_started": False,
                "c3000_authorized": bool(
                    (final_decision or {}).get("decision") == C1000_STABLE_STILL_IMPROVING
                ),
                "final_seeds_access": False,
                "result_grade": False,
                "best_checkpoint": (final_decision or {}).get("selected_checkpoint"),
            },
        )
        write_monitoring_summary(
            self.out_dir / MONITOR_MD,
            status={"status": status, "episodes_completed": len(self.train_rows)},
            train_rows=self.train_rows,
            val_rows=self.validation_history,
            decision=final_decision,
            gpu=gpu_snapshot(),
            eta_s=0.0,
        )
        print(json.dumps({"status": status, "decision": final_decision}, indent=2, default=str), flush=True)
        if (final_decision or {}).get("decision") == C1000_UNSTABLE_OR_INVALID:
            return 2
        return 0

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda", "auto"))
    p.add_argument("--horizon-s", type=float, default=HORIZON_S)
    p.add_argument("--lambda-total-per-s", type=float, default=LAMBDA_TOTAL)
    p.add_argument("--online-quality-table", type=Path, default=DEFAULT_QUALITY)
    p.add_argument("--load-init-checkpoint", type=Path, default=DEFAULT_SHARED_INIT)
    p.add_argument("--obs-norm-stats", type=Path, default=DEFAULT_OBS_NORM)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--auto-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Independent dry-run path")
    p.add_argument(
        "--max-episodes",
        type=int,
        default=MAX_EPISODES,
        help="Only honored with --dry-run; formal run is capped at 1000",
    )
    p.add_argument("--require-clean-tree", action="store_true")
    p.add_argument("--freeze-commit", type=str, default="")
    p.add_argument("--rerun-baselines", action="store_true")
    p.add_argument(
        "--allow-short-horizon",
        action="store_true",
        help="Test-only: allow non-900s horizon (never for formal C1000)",
    )
    p.add_argument(
        "--baselines-only",
        action="store_true",
        help="Only freeze comparator inventory / baselines and exit",
    )
    p.add_argument(
        "--start-c3000",
        action="store_true",
        help=argparse.SUPPRESS,  # trap; always rejected
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if abs(float(args.horizon_s) - HORIZON_S) > 1e-9 and not args.allow_short_horizon:
        if not args.dry_run:
            raise SystemExit(
                f"formal C1000 freezes horizon_s={HORIZON_S}; got {args.horizon_s}"
            )
    if not args.dry_run and int(args.max_episodes) != MAX_EPISODES:
        # ignore silent extension
        args.max_episodes = MAX_EPISODES
    if args.dry_run and int(args.max_episodes) > MAX_EPISODES:
        raise SystemExit("dry-run max-episodes cannot exceed 1000")

    # Absolute denylist scan of argv
    for a in (argv or sys.argv[1:]):
        if a.isdigit() and int(a) in FINAL_SEEDS:
            raise SystemExit(f"refusing final seed in argv: {a}")

    runner = C1000Runner(args)
    try:
        runner.setup()
        if args.baselines_only:
            runner.ensure_baselines()
            runner._write_status(status="BASELINES_FROZEN", ended_utc=utc_now())
            return 0
        return runner.run()
    except SystemExit:
        raise
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr)
        try:
            runner._write_status(
                status="FAILED",
                error=str(exc),
                ended_utc=utc_now(),
            )
            decision = {
                "decision": C1000_UNSTABLE_OR_INVALID,
                "stop": True,
                "reasons": [f"exception:{exc}"],
                "timestamp_utc": utc_now(),
                "c3000_started": False,
                "final_seeds_access": False,
            }
            atomic_write_json(runner.out_dir / DECISION_NAME, decision)
            write_claude_failure_package(runner.out_dir, decision)
        except Exception:
            pass
        return 1
    finally:
        runner.close()
        try:
            runner.lock.release()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
