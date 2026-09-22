from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from vqa_semcom.config import ensure_parent

LUTKey = tuple[str, int, str, str, str, str]


@dataclass(frozen=True)
class SemanticServiceEntry:
    accuracy: float
    payload_bytes: float
    sample_count: int = 0
    std_or_ci: float = 0.0
    missing: bool = False
    link_quality_bin: str = ""
    link_quality_source: str = ""
    snr_db: float | None = None


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    episodes: int
    tasks: int
    task_success_rate: float
    average_accuracy: float
    average_payload_mb: float
    average_delay: float
    average_energy: float
    quality_violation_rate: float
    deadline_violation_rate: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_payload_bytes(row: dict[str, str], fallback_payload_mb: float) -> float:
    for field in ("payload_bytes", "avg_payload_bytes", "payload_b", "bytes"):
        if row.get(field, "") != "":
            return float(row[field])
    for field in ("payload_mb", "avg_payload_mb"):
        if row.get(field, "") != "":
            return float(row[field]) * 1_000_000.0
    return fallback_payload_mb * 1_000_000.0


def _nonempty(row: dict[str, str], *fields: str) -> str:
    for field in fields:
        value = row.get(field, "")
        if value != "":
            return value
    return ""


def _snr_to_link_quality(snr_db: float, bad_snr_db: float = 5.0, good_snr_db: float = 15.0) -> str:
    if snr_db < bad_snr_db:
        return "bad"
    if snr_db < good_snr_db:
        return "medium"
    return "good"


def _normalize_link_quality(value: str) -> str:
    aliases = {
        "low": "bad",
        "poor": "bad",
        "weak": "bad",
        "mid": "medium",
        "med": "medium",
        "moderate": "medium",
        "high": "good",
        "strong": "good",
        "excellent": "good",
    }
    normalized = value.strip().lower()
    return aliases.get(normalized, normalized)


def _row_link_quality(row: dict[str, str]) -> tuple[str, str, float | None]:
    snr_bin = _nonempty(row, "snr_bin", "link_quality_bin", "link_quality")
    snr_text = _nonempty(row, "snr_db", "snr", "mean_snr_db", "avg_snr_db")
    snr_db = float(snr_text) if snr_text != "" else None
    if snr_bin:
        return _normalize_link_quality(snr_bin), "snr_bin", snr_db
    if snr_db is not None:
        return _snr_to_link_quality(snr_db), "snr_db", snr_db
    channel_bin = _nonempty(row, "channel_bin")
    if channel_bin:
        return _normalize_link_quality(channel_bin), "channel_bin", None
    raise KeyError("LUT row needs one of snr_bin, snr_db, or channel_bin")


def _row_accuracy(row: dict[str, str]) -> float:
    return float(_nonempty(row, "expected_accuracy", "accuracy", "acc", "mean_accuracy"))


def _row_std_or_ci(row: dict[str, str]) -> float:
    value = _nonempty(row, "std_or_ci", "std_or_confidence_interval", "confidence_interval", "ci95")
    return float(value) if value != "" else 0.0


def load_semantic_lut(path: Path, payload_mb_by_level: dict[str, float] | None = None) -> dict[LUTKey, SemanticServiceEntry]:
    payload_mb_by_level = payload_mb_by_level or {}
    table: dict[LUTKey, SemanticServiceEntry] = {}
    for row in read_csv(path):
        link_quality, link_source, snr_db = _row_link_quality(row)
        key = (
            row["question_type"],
            int(row["service_level"]),
            link_quality,
            row["view_quality_bin"],
            row["freshness_bin"],
            row["risk_level"],
        )
        level = str(key[1])
        fallback_mb = float(payload_mb_by_level.get(level, 0.0))
        table[key] = SemanticServiceEntry(
            accuracy=_row_accuracy(row),
            payload_bytes=_row_payload_bytes(row, fallback_mb),
            sample_count=int(float(row.get("sample_count", 0) or 0)),
            std_or_ci=_row_std_or_ci(row),
            link_quality_bin=link_quality,
            link_quality_source=link_source,
            snr_db=snr_db,
        )
    return table


def load_lut(path: Path) -> dict[LUTKey, float]:
    return {key: entry.accuracy for key, entry in load_semantic_lut(path).items()}


def _choice(rng: random.Random, values: list[str]) -> str:
    return values[rng.randrange(len(values))]


def _lookup_entry(
    lut: dict[LUTKey, float | SemanticServiceEntry],
    task: dict[str, str],
    level: int,
    channel: str,
    freshness: str,
    cfg: dict[str, Any],
) -> SemanticServiceEntry:
    key = (
        task["question_type"],
        level,
        channel,
        task["view_quality_bin"],
        freshness,
        task["risk_level"],
    )
    if key in lut:
        value = lut[key]
        if isinstance(value, SemanticServiceEntry):
            return value
        payload_mb = float(cfg.get("resource_env", {}).get("payload_mb_by_level", {}).get(str(level), 0.0))
        return SemanticServiceEntry(float(value), payload_mb * 1_000_000.0)
    # Sparse fallback keeps the simulation running but remains conservative.
    payload_mb = float(cfg.get("resource_env", {}).get("payload_mb_by_level", {}).get(str(level), 0.0))
    return SemanticServiceEntry(0.0, payload_mb * 1_000_000.0, missing=True)


def _lookup(
    lut: dict[LUTKey, float | SemanticServiceEntry],
    task: dict[str, str],
    level: int,
    channel: str,
    freshness: str,
    cfg: dict[str, Any],
) -> float:
    return _lookup_entry(lut, task, level, channel, freshness, cfg).accuracy


def _choose_level(policy: str, task: dict[str, str], channel: str, freshness: str, lut: dict[LUTKey, float | SemanticServiceEntry], cfg: dict[str, Any]) -> int:
    levels = [int(level) for level in cfg.get("bins", {}).get("service_levels", [0, 1, 2, 3])]
    if policy == "always_cache":
        return 0
    if policy == "always_light":
        return 1
    if policy in {"always_image", "always_full_image"}:
        return 2
    if policy == "always_roi":
        return 3 if 3 in levels else levels[-1]
    if policy == "greedy_min_sufficient_evidence":
        epsilon = float(task["epsilon_k"])
        for level in levels:
            if _lookup(lut, task, level, channel, freshness, cfg) >= epsilon:
                return level
        return levels[-1]
    raise ValueError(f"unknown policy: {policy}")


def run_simulation(tasks: list[dict[str, str]], lut: dict[LUTKey, float | SemanticServiceEntry], cfg: dict[str, Any], episodes: int) -> list[PolicyResult]:
    rng = random.Random(cfg["simulation"]["seed"])
    policies = cfg["simulation"]["policies"]
    channel_bins = cfg["bins"]["channel"]
    freshness_bins = cfg["bins"]["freshness"]
    tasks_per_episode = int(cfg["simulation"]["tasks_per_episode"])
    delay_by_level = {int(k): float(v) for k, v in cfg["simulation"]["delay_by_level"].items()}
    energy_by_level = {int(k): float(v) for k, v in cfg["simulation"]["energy_by_level"].items()}
    channel_delay = cfg["simulation"]["channel_delay_multiplier"]
    aggregates: dict[str, list[dict[str, float]]] = defaultdict(list)
    if not tasks:
        raise ValueError("simulation needs at least one task")
    for _episode in range(episodes):
        episode_tasks = [tasks[rng.randrange(len(tasks))] for _ in range(tasks_per_episode)]
        for policy in policies:
            for task in episode_tasks:
                channel = _choice(rng, channel_bins)
                freshness = _choice(rng, freshness_bins)
                level = _choose_level(policy, task, channel, freshness, lut, cfg)
                entry = _lookup_entry(lut, task, level, channel, freshness, cfg)
                acc = entry.accuracy
                payload_mb = entry.payload_bytes / 1_000_000.0
                delay = delay_by_level[level] * float(channel_delay[channel])
                energy = energy_by_level[level]
                epsilon = float(task["epsilon_k"])
                tau = float(task["tau_k"])
                quality_ok = acc >= epsilon
                deadline_ok = delay <= tau
                success = quality_ok and deadline_ok
                aggregates[policy].append(
                    {
                        "success": float(success),
                        "accuracy": acc,
                        "payload_mb": payload_mb,
                        "delay": delay,
                        "energy": energy,
                        "quality_violation": float(not quality_ok),
                        "deadline_violation": float(not deadline_ok),
                    }
                )
    results: list[PolicyResult] = []
    for policy in policies:
        rows = aggregates[policy]
        denom = max(1, len(rows))
        results.append(
            PolicyResult(
                policy=policy,
                episodes=episodes,
                tasks=len(rows),
                task_success_rate=round(sum(r["success"] for r in rows) / denom, 6),
                average_accuracy=round(sum(r["accuracy"] for r in rows) / denom, 6),
                average_payload_mb=round(sum(r["payload_mb"] for r in rows) / denom, 6),
                average_delay=round(sum(r["delay"] for r in rows) / denom, 6),
                average_energy=round(sum(r["energy"] for r in rows) / denom, 6),
                quality_violation_rate=round(sum(r["quality_violation"] for r in rows) / denom, 6),
                deadline_violation_rate=round(sum(r["deadline_violation"] for r in rows) / denom, 6),
            )
        )
    return results


def write_results(results: list[PolicyResult], csv_path: Path, md_path: Path) -> None:
    ensure_parent(csv_path)
    ensure_parent(md_path)
    fieldnames = list(PolicyResult.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    lines = ["# V0 Resource Simulation Summary", ""]
    lines.append("| policy | success | accuracy | payload MB | delay | energy | quality violation | deadline violation |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.policy} | {r.task_success_rate:.3f} | {r.average_accuracy:.3f} | "
            f"{r.average_payload_mb:.3f} | {r.average_delay:.3f} | {r.average_energy:.3f} | "
            f"{r.quality_violation_rate:.3f} | {r.deadline_violation_rate:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
