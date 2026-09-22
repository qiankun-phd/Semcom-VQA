from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vqa_semcom.rl.diengine_hybrid_env import HybridVQADingEnv, hybrid_env_spec


CONSTRAINT_KEYS = ("quality", "deadline", "resource", "conflict", "battery", "gpu_memory")


@dataclass
class TCHPPOConfig:
    """Lagrangian control parameters for constrained hybrid PPO."""

    lambda_lr: float = 0.05
    lambda_max: float = 10.0
    initial_lambda: float = 0.0
    targets: dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in CONSTRAINT_KEYS})


class TCHDualController:
    """Online dual variable controller used by the TCH-PPO reward wrapper."""

    def __init__(self, config: TCHPPOConfig | None = None) -> None:
        self.config = config or TCHPPOConfig()
        self.lambdas = {key: float(self.config.initial_lambda) for key in CONSTRAINT_KEYS}

    def penalty(self, costs: dict[str, float]) -> float:
        total = 0.0
        for key in CONSTRAINT_KEYS:
            violation = max(0.0, float(costs.get(key, 0.0)) - self.target(key))
            total += self.lambdas[key] * violation
        return float(total)

    def update(self, costs: dict[str, float]) -> dict[str, float]:
        for key in CONSTRAINT_KEYS:
            delta = float(costs.get(key, 0.0)) - self.target(key)
            value = self.lambdas[key] + float(self.config.lambda_lr) * delta
            self.lambdas[key] = min(max(value, 0.0), float(self.config.lambda_max))
        return dict(self.lambdas)

    def target(self, key: str) -> float:
        return float(self.config.targets.get(key, 0.0))


def tch_costs_from_info(info: dict[str, Any]) -> dict[str, float]:
    """Map environment diagnostics to normalized constraint costs."""

    resource_violation = max(_float(info, "resource_violation_rate"), _float(info, "raw_resource_violation"))
    return {
        "quality": _float(info, "quality_violation_rate"),
        "deadline": _float(info, "deadline_violation_rate"),
        "resource": resource_violation,
        "conflict": _float(info, "airspace_conflict_rate"),
        "battery": max(0.0, 1.0 - _float(info, "battery_ok_rate", default=1.0)),
        "gpu_memory": max(0.0, 1.0 - _float(info, "gpu_memory_ok_rate", default=1.0)),
    }


class TCHPPOEnv(HybridVQADingEnv):
    """Centralized constrained hybrid-action environment for TCH-PPO.

    The base VQA-SemCom environment remains the single source of task rewards.
    This wrapper only adds a Lagrangian penalty and exposes the original reward,
    constraint costs, and dual variables through ``info``/CSV traces.
    """

    def __init__(
        self,
        config_path: str | Path = "configs/v0.yaml",
        scenario: str | None = "literature_demo",
        seed: int | None = None,
        reward_scale: float = 1.0,
        tch_config: TCHPPOConfig | None = None,
        trace_path: str | Path | None = None,
        append_trace: bool = False,
    ) -> None:
        super().__init__(config_path=config_path, scenario=scenario, seed=seed, reward_scale=reward_scale)
        self.dual = TCHDualController(tch_config)
        self.trace_path = Path(trace_path) if trace_path else None
        self.append_trace = bool(append_trace)
        self._trace_rows: list[dict[str, Any]] = []
        self._global_step = 0
        self._episode_step = 0
        self._trace_header_written = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._episode_step = 0
        return super().reset(seed=seed, options=options)

    def step(self, action: Any):
        step_out = super().step(action)
        if len(step_out) == 5:
            obs, raw_reward, terminated, truncated, info = step_out
            shaped_reward, shaped_info = self._shape_reward(float(raw_reward), info)
            return obs, shaped_reward, terminated, truncated, shaped_info
        obs, raw_reward, done, info = step_out
        shaped_reward, shaped_info = self._shape_reward(float(raw_reward), info)
        return obs, shaped_reward, done, shaped_info

    def lambda_trace(self) -> list[dict[str, Any]]:
        return list(self._trace_rows)

    def write_lambda_trace(self, path: str | Path) -> None:
        rows = self.lambda_trace()
        if not rows:
            return
        trace_path = Path(path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _shape_reward(self, raw_reward: float, info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        costs = tch_costs_from_info(info)
        penalty = self.dual.penalty(costs)
        tch_reward = float(raw_reward - penalty)
        lambdas = self.dual.update(costs)
        self._global_step += 1
        self._episode_step += 1
        shaped_info = {
            **info,
            "raw_reward": float(raw_reward),
            "tch_reward": tch_reward,
            "tch_penalty": float(penalty),
            "constraint_cost": float(sum(max(0.0, costs[key] - self.dual.target(key)) for key in CONSTRAINT_KEYS)),
        }
        for key in CONSTRAINT_KEYS:
            shaped_info[f"cost_{key}"] = float(costs[key])
            shaped_info[f"lambda_{key}"] = float(lambdas[key])
            shaped_info[f"target_{key}"] = self.dual.target(key)
        row = self._trace_row(shaped_info)
        self._trace_rows.append(row)
        if self.trace_path and self.append_trace:
            self._append_trace_row(row)
        return tch_reward, shaped_info

    def _trace_row(self, info: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {
            "global_step": self._global_step,
            "episode_step": self._episode_step,
            "episode": info.get("episode", self._episode),
            "slot": info.get("slot", 0),
            "scenario": self.scenario or "",
            "raw_reward": info.get("raw_reward", 0.0),
            "tch_reward": info.get("tch_reward", 0.0),
            "tch_penalty": info.get("tch_penalty", 0.0),
            "unique_task_success_rate": info.get("unique_task_success_rate", 0.0),
            "average_semantic_utility": info.get("average_semantic_utility", 0.0),
            "average_delay": info.get("average_delay", 0.0),
            "average_energy": info.get("average_energy", 0.0),
            "average_payload_mb": info.get("average_payload_mb", 0.0),
            "quality_violation_rate": info.get("quality_violation_rate", 0.0),
            "deadline_violation_rate": info.get("deadline_violation_rate", 0.0),
            "resource_violation_rate": info.get("resource_violation_rate", 0.0),
            "airspace_conflict_rate": info.get("airspace_conflict_rate", 0.0),
            "gpu_memory_ok_rate": info.get("gpu_memory_ok_rate", 0.0),
            "battery_ok_rate": info.get("battery_ok_rate", 0.0),
            "service_level_0_rate": info.get("service_level_0_rate", 0.0),
            "service_level_1_rate": info.get("service_level_1_rate", 0.0),
            "service_level_2_rate": info.get("service_level_2_rate", 0.0),
            "service_level_3_rate": info.get("service_level_3_rate", 0.0),
        }
        for key in CONSTRAINT_KEYS:
            row[f"cost_{key}"] = info.get(f"cost_{key}", 0.0)
            row[f"lambda_{key}"] = info.get(f"lambda_{key}", 0.0)
        return row

    def _append_trace_row(self, row: dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.trace_path.exists() and self.trace_path.stat().st_size > 0
        with self.trace_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists and not self._trace_header_written:
                writer.writeheader()
                self._trace_header_written = True
            writer.writerow(row)


def make_tch_vqa_env(
    config_path: str | Path = "configs/v0.yaml",
    scenario: str | None = "literature_demo",
    seed: int | None = None,
    tch_config: TCHPPOConfig | None = None,
    trace_path: str | Path | None = None,
    append_trace: bool = False,
) -> TCHPPOEnv:
    return TCHPPOEnv(
        config_path=config_path,
        scenario=scenario,
        seed=seed,
        tch_config=tch_config,
        trace_path=trace_path,
        append_trace=append_trace,
    )


def tch_env_spec(config_path: str | Path = "configs/v0.yaml", scenario: str | None = "literature_demo"):
    return hybrid_env_spec(config_path=config_path, scenario=scenario)


def _float(info: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = info.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
