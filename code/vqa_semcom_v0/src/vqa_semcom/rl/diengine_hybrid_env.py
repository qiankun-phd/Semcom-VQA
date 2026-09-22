from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vqa_semcom.rl.diengine_env import _load_env, _require_gym, gym, spaces
from vqa_semcom.sim.vqa_resource_env import AllocationDecision, VQAResourceEnv, summarize_outcomes


SENSING_DECISIONS = ["reuse_cache", "observe", "revisit"]


@dataclass(frozen=True)
class HybridVQAEnvSpec:
    obs_shape: int
    max_tasks: int
    num_uavs: int
    service_levels: tuple[int, ...]
    action_args_shape: int
    action_type_shape: tuple[int, ...]


def hybrid_env_spec(config_path: str | Path = "configs/v0.yaml", scenario: str | None = "literature_demo") -> HybridVQAEnvSpec:
    env = _load_env(config_path, scenario=scenario, seed=0)
    max_tasks = int(env.cfg["observation"]["max_tasks"])
    num_uavs = int(env.cfg["num_uavs"])
    service_levels = tuple(env.service_levels())
    action_type_shape: list[int] = []
    for _ in range(max_tasks):
        action_type_shape.extend([num_uavs, len(SENSING_DECISIONS), len(service_levels)])
    return HybridVQAEnvSpec(
        obs_shape=len(env.observation_vector()),
        max_tasks=max_tasks,
        num_uavs=num_uavs,
        service_levels=service_levels,
        action_args_shape=max_tasks * 4,
        action_type_shape=tuple(action_type_shape),
    )


class HybridVQADingEnv(gym.Env if gym is not None else object):
    """Gym-compatible hybrid-action adapter for the VQA resource MDP."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | Path = "configs/v0.yaml",
        scenario: str | None = "literature_demo",
        seed: int | None = None,
        reward_scale: float = 1.0,
    ) -> None:
        _require_gym()
        self.config_path = Path(config_path)
        self.scenario = scenario
        self.reward_scale = float(reward_scale)
        self._seed = 0 if seed is None else int(seed)
        self._episode = -1
        self._env = _load_env(self.config_path, self.scenario, self._seed)
        self.spec_info = hybrid_env_spec(self.config_path, self.scenario)
        self._last_resource_diagnostics: dict[str, float] = {}
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=0.0, high=1.0, shape=(self.spec_info.obs_shape,), dtype=np.float32),
                "observation": spaces.Box(low=0.0, high=1.0, shape=(self.spec_info.obs_shape,), dtype=np.float32),
                "action_mask": self._mask_space(),
            }
        )
        self.action_space = spaces.Dict(
            {
                "assigned_uav": spaces.MultiDiscrete(np.full(self.spec_info.max_tasks, self.spec_info.num_uavs, dtype=np.int64)),
                "sensing_decision": spaces.MultiDiscrete(np.full(self.spec_info.max_tasks, len(SENSING_DECISIONS), dtype=np.int64)),
                "service_level": spaces.MultiDiscrete(np.full(self.spec_info.max_tasks, len(self.spec_info.service_levels), dtype=np.int64)),
                "bandwidth_share": spaces.Box(0.0, 1.0, shape=(self.spec_info.max_tasks,), dtype=np.float32),
                "power_scalar": spaces.Box(0.0, 1.0, shape=(self.spec_info.max_tasks,), dtype=np.float32),
                "cpu_share": spaces.Box(0.0, 1.0, shape=(self.spec_info.max_tasks,), dtype=np.float32),
                "gpu_share": spaces.Box(0.0, 1.0, shape=(self.spec_info.max_tasks,), dtype=np.float32),
            }
        )
        self.reward_range = (-float("inf"), float("inf"))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._seed = int(seed)
        options = options or {}
        self._episode += 1
        scenario = options.get("scenario", self.scenario)
        self._env.rng.seed(self._seed + self._episode)
        self._env.reset(self._episode, scenario=scenario, mdp_mode=True)
        obs = self._observation()
        if gym.__name__ == "gymnasium":
            return obs, self._info([])
        return obs

    def step(self, action: Any):
        decisions = self.action_to_decisions(action)
        _state, outcomes, done, info = self._env.step(decisions, policy="diengine_hybrid")
        reward = self.reward_scale * sum(o.reward for o in outcomes)
        obs = self._observation()
        info = {**info, **self._info(outcomes)}
        if gym.__name__ == "gymnasium":
            return obs, float(reward), bool(done), False, info
        return obs, float(reward), bool(done), info

    def seed(self, seed: int | None = None) -> list[int]:
        self._seed = 0 if seed is None else int(seed)
        self._env.rng.seed(self._seed)
        return [self._seed]

    def close(self) -> None:
        return None

    def action_to_decisions(self, action: Any) -> list[AllocationDecision]:
        parsed = self._parse_action(action)
        masks = self.per_task_action_mask()
        active_ids = {task.task_id for task in self._env.active_tasks()}
        tasks = self._env.tasks[: self.spec_info.max_tasks]
        active_indices = [idx for idx, task in enumerate(tasks) if task.task_id in active_ids]
        raw_bw = float(sum(float(parsed["bandwidth_share"][idx]) for idx in active_indices))
        raw_cpu = float(sum(float(parsed["cpu_share"][idx]) for idx in active_indices))
        raw_gpu = float(sum(float(parsed["gpu_share"][idx]) for idx in active_indices))
        bw = self._normalize_resource(parsed["bandwidth_share"], active_ids, tasks)
        cpu = self._normalize_resource(parsed["cpu_share"], active_ids, tasks)
        gpu = self._normalize_resource(parsed["gpu_share"], active_ids, tasks)
        self._last_resource_diagnostics = {
            "raw_bandwidth_sum": raw_bw,
            "raw_cpu_sum": raw_cpu,
            "raw_gpu_sum": raw_gpu,
            "normalized_bandwidth_sum": float(sum(float(bw[idx]) for idx in active_indices)),
            "normalized_cpu_sum": float(sum(float(cpu[idx]) for idx in active_indices)),
            "normalized_gpu_sum": float(sum(float(gpu[idx]) for idx in active_indices)),
            "raw_resource_violation": float(raw_bw > 1.0 or raw_cpu > 1.0 or raw_gpu > 1.0),
        }
        power_min = min(float(x) for x in self._env.cfg["power_levels_w"])
        power_max = max(float(x) for x in self._env.cfg["power_levels_w"])
        decisions: list[AllocationDecision] = []
        for idx, task in enumerate(tasks):
            if task.task_id not in active_ids:
                continue
            uav = self._masked_index(int(parsed["assigned_uav"][idx]), masks["assigned_uav_mask"][idx])
            sensing_idx = self._masked_index(int(parsed["sensing_decision"][idx]), masks["sensing_mask"][idx])
            service_idx = self._masked_index(int(parsed["service_level"][idx]), masks["service_level_mask"][idx])
            decisions.append(
                AllocationDecision(
                    task_id=task.task_id,
                    assigned_uav=uav,
                    sensing_decision=SENSING_DECISIONS[sensing_idx],
                    service_level=self.spec_info.service_levels[service_idx],
                    bandwidth_share=float(bw[idx]),
                    power_w=power_min + float(parsed["power_scalar"][idx]) * (power_max - power_min),
                    cpu_share=float(cpu[idx]),
                    gpu_share=float(gpu[idx]),
                )
            )
        return decisions

    def per_task_action_mask(self) -> dict[str, np.ndarray]:
        base = self._env.action_mask()
        active_ids = {task.task_id for task in self._env.active_tasks()}
        tasks = self._env.tasks[: self.spec_info.max_tasks]
        assigned = np.ones((self.spec_info.max_tasks, self.spec_info.num_uavs), dtype=np.float32)
        sensing = np.ones((self.spec_info.max_tasks, len(SENSING_DECISIONS)), dtype=np.float32)
        service = np.ones((self.spec_info.max_tasks, len(self.spec_info.service_levels)), dtype=np.float32)
        active = np.zeros((self.spec_info.max_tasks,), dtype=np.float32)
        battery_ok = base.get("battery_ok_by_uav", {})
        uav_ok = np.asarray([1.0 if battery_ok.get(uid, True) else 0.0 for uid in range(self.spec_info.num_uavs)], dtype=np.float32)
        if not np.any(uav_ok):
            best = int(np.argmax([uav.battery for uav in self._env.uavs[: self.spec_info.num_uavs]]))
            uav_ok[best] = 1.0
        level_allowed = base.get("service_level_allowed", {})
        service_ok = np.asarray(
            [1.0 if level == 0 or level_allowed.get(level, True) else 0.0 for level in self.spec_info.service_levels],
            dtype=np.float32,
        )
        if not np.any(service_ok):
            service_ok[0] = 1.0
        for idx, task in enumerate(tasks):
            assigned[idx] = uav_ok
            service[idx] = service_ok
            if task.task_id in active_ids:
                active[idx] = 1.0
        return {
            "assigned_uav_mask": assigned,
            "sensing_mask": sensing,
            "service_level_mask": service,
            "active_task_mask": active,
        }

    def _observation(self) -> dict[str, Any]:
        state = np.asarray(self._env.observation_vector(), dtype=np.float32)
        return {
            "state": state,
            "observation": state.copy(),
            "action_mask": self.per_task_action_mask(),
        }

    def _mask_space(self):
        return spaces.Dict(
            {
                "assigned_uav_mask": spaces.Box(0.0, 1.0, shape=(self.spec_info.max_tasks, self.spec_info.num_uavs), dtype=np.float32),
                "sensing_mask": spaces.Box(0.0, 1.0, shape=(self.spec_info.max_tasks, len(SENSING_DECISIONS)), dtype=np.float32),
                "service_level_mask": spaces.Box(0.0, 1.0, shape=(self.spec_info.max_tasks, len(self.spec_info.service_levels)), dtype=np.float32),
                "active_task_mask": spaces.Box(0.0, 1.0, shape=(self.spec_info.max_tasks,), dtype=np.float32),
            }
        )

    def _parse_action(self, action: Any) -> dict[str, np.ndarray]:
        if isinstance(action, dict) and ("action_type" in action or "action_args" in action):
            return self._parse_diengine_action(action)
        if not isinstance(action, dict):
            return self._parse_diengine_action({"action_type": [], "action_args": action})
        return {
            "assigned_uav": self._int_array(action.get("assigned_uav", 0), self.spec_info.max_tasks),
            "sensing_decision": self._int_array(action.get("sensing_decision", 0), self.spec_info.max_tasks),
            "service_level": self._int_array(action.get("service_level", 0), self.spec_info.max_tasks),
            "bandwidth_share": self._unit_array(action.get("bandwidth_share", 0.0), self.spec_info.max_tasks),
            "power_scalar": self._unit_array(action.get("power_scalar", action.get("power", 0.0)), self.spec_info.max_tasks),
            "cpu_share": self._unit_array(action.get("cpu_share", 0.0), self.spec_info.max_tasks),
            "gpu_share": self._unit_array(action.get("gpu_share", 0.0), self.spec_info.max_tasks),
        }

    def _parse_diengine_action(self, action: dict[str, Any]) -> dict[str, np.ndarray]:
        action_type = self._int_array(action.get("action_type", []), self.spec_info.max_tasks * 3)
        action_args = self._unit_array(action.get("action_args", []), self.spec_info.max_tasks * 4)
        assigned = np.zeros(self.spec_info.max_tasks, dtype=np.int64)
        sensing = np.zeros(self.spec_info.max_tasks, dtype=np.int64)
        service = np.zeros(self.spec_info.max_tasks, dtype=np.int64)
        for idx in range(self.spec_info.max_tasks):
            assigned[idx] = action_type[idx * 3]
            sensing[idx] = action_type[idx * 3 + 1]
            service[idx] = action_type[idx * 3 + 2]
        args = action_args.reshape(self.spec_info.max_tasks, 4)
        return {
            "assigned_uav": assigned,
            "sensing_decision": sensing,
            "service_level": service,
            "bandwidth_share": args[:, 0],
            "power_scalar": args[:, 1],
            "cpu_share": args[:, 2],
            "gpu_share": args[:, 3],
        }

    @staticmethod
    def _masked_index(value: int, mask: np.ndarray) -> int:
        valid = np.flatnonzero(mask > 0.0)
        if valid.size == 0:
            return 0
        idx = int(abs(value)) % len(mask)
        if mask[idx] > 0.0:
            return idx
        return int(valid[0])

    @staticmethod
    def _unit_array(value: Any, size: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            arr = np.zeros(size, dtype=np.float32)
        if arr.size < size:
            arr = np.pad(arr, (0, size - arr.size))
        arr = arr[:size]
        if np.any(arr < 0.0):
            arr = (np.clip(arr, -1.0, 1.0) + 1.0) / 2.0
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _int_array(value: Any, size: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            arr = np.zeros(size, dtype=np.int64)
        if arr.size < size:
            arr = np.pad(arr, (0, size - arr.size))
        return arr[:size].astype(np.int64)

    @staticmethod
    def _normalize_resource(values: np.ndarray, active_ids: set[str], tasks: list[Any]) -> np.ndarray:
        out = np.clip(values.astype(np.float32), 0.0, 1.0)
        active_indices = [idx for idx, task in enumerate(tasks) if task.task_id in active_ids]
        total = float(sum(out[idx] for idx in active_indices))
        if total > 1.0:
            for idx in active_indices:
                out[idx] /= total
        return out

    def _info(self, outcomes: list[Any]) -> dict[str, Any]:
        summary = summarize_outcomes(outcomes) if outcomes else {}
        return {
            "episode": self._env.episode,
            "slot": self._env.slot,
            "active_tasks": len(self._env.active_tasks()),
            "completed_tasks": sum(int(t.completed) for t in self._env.tasks),
            **self._last_resource_diagnostics,
            **summary,
        }


def make_hybrid_vqa_env(
    config_path: str | Path = "configs/v0.yaml",
    scenario: str | None = "literature_demo",
    seed: int | None = None,
) -> HybridVQADingEnv:
    return HybridVQADingEnv(config_path=config_path, scenario=scenario, seed=seed)
