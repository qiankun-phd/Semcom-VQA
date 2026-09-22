from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Prefer gymnasium, but keep compatibility with older DI-engine forks.
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    try:
        import gym
        from gym import spaces
    except ModuleNotFoundError as exc:  # pragma: no cover
        gym = None
        spaces = None
        _GYM_IMPORT_ERROR = exc
    else:
        _GYM_IMPORT_ERROR = None
else:
    _GYM_IMPORT_ERROR = None

import numpy as np

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.sim.vqa_resource_env import (
    VQAResourceEnv,
    load_resource_inputs,
    summarize_outcomes,
)


@dataclass(frozen=True)
class VQASemComEnvSpec:
    obs_shape: int
    action_shape: int
    max_tasks: int
    action_width: int = 7


def _require_gym() -> None:
    if gym is None or spaces is None:
        raise ModuleNotFoundError(
            "VQASemComDingEnv requires gymnasium or gym. Install one of them in the DI-engine environment."
        ) from _GYM_IMPORT_ERROR


def _load_env(config_path: str | Path, scenario: str | None, seed: int | None) -> VQAResourceEnv:
    cfg = load_config(config_path)
    tasks_path = resolve_path(cfg["paths"]["tasks_csv"])
    preferred_lut = cfg["paths"].get("vlm_lut_csv", cfg["paths"]["lut_csv"])
    lut_path = resolve_path(preferred_lut)
    if not lut_path.exists() and preferred_lut != cfg["paths"]["lut_csv"]:
        lut_path = resolve_path(cfg["paths"]["lut_csv"])
    if not tasks_path.exists() or not lut_path.exists():
        raise RuntimeError("Tasks/LUT not found. Run scripts/build_v0_lut.py first.")
    tasks, lut = load_resource_inputs(tasks_path, lut_path, cfg)
    env = VQAResourceEnv(tasks, lut, cfg, seed=seed)
    env.reset(episode=0, scenario=scenario, mdp_mode=True)
    return env


def env_spec(config_path: str | Path = "configs/v0.yaml", scenario: str | None = "literature_demo") -> VQASemComEnvSpec:
    env = _load_env(config_path, scenario=scenario, seed=0)
    max_tasks = int(env.cfg["observation"]["max_tasks"])
    return VQASemComEnvSpec(
        obs_shape=len(env.observation_vector()),
        action_shape=max_tasks * 7,
        max_tasks=max_tasks,
    )


class VQASemComDingEnv(gym.Env if gym is not None else object):
    """Gym-compatible low-level controller for DI-engine PPO/DDPG-style runs.

    The action is a flat continuous vector. Every active task consumes seven
    entries: UAV id, sensing action, service level, bandwidth share, power
    scalar, CPU share, and GPU share. The underlying V0 environment clips and
    converts the vector into ``AllocationDecision`` objects.
    """

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
        self._episode = -1
        self._seed = 0 if seed is None else int(seed)
        self._env = _load_env(self.config_path, self.scenario, self._seed)
        self.spec_info = env_spec(self.config_path, self.scenario)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.spec_info.obs_shape,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.spec_info.action_shape,),
            dtype=np.float32,
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
        obs = np.asarray(self._env.observation_vector(), dtype=np.float32)
        if gym.__name__ == "gymnasium":
            return obs, self._info([])
        return obs

    def step(self, action: Any):
        vector = self._normalize_action(action)
        decisions = self._env.action_from_vector(vector.tolist())
        _state, outcomes, done, info = self._env.step(decisions, policy="diengine_ppo")
        reward = self.reward_scale * sum(o.reward for o in outcomes)
        obs = np.asarray(self._env.observation_vector(), dtype=np.float32)
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

    def _normalize_action(self, action: Any) -> np.ndarray:
        if isinstance(action, dict):
            action = action.get("action_args", action.get("action", []))
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < self.spec_info.action_shape:
            arr = np.pad(arr, (0, self.spec_info.action_shape - arr.size))
        arr = arr[: self.spec_info.action_shape]
        # DI-engine tanh actors often emit [-1, 1]; VQAResourceEnv expects
        # continuous resource entries in [0, 1], while discrete entries are
        # rounded modulo their domain.
        scaled = (np.clip(arr, -1.0, 1.0) + 1.0) / 2.0
        out = np.zeros_like(scaled)
        for start in range(0, scaled.size, self.spec_info.action_width):
            out[start] = round(scaled[start] * max(1, int(self._env.cfg["num_uavs"]) - 1))
            out[start + 1] = round(scaled[start + 1] * 2.0)
            out[start + 2] = round(scaled[start + 2] * (len(self._env.service_levels()) - 1))
            out[start + 3 : start + 7] = scaled[start + 3 : start + 7]
        return out

    def _info(self, outcomes: list[Any]) -> dict[str, Any]:
        summary = summarize_outcomes(outcomes) if outcomes else {}
        return {
            "episode": self._env.episode,
            "slot": self._env.slot,
            "active_tasks": len(self._env.active_tasks()),
            "completed_tasks": sum(int(t.completed) for t in self._env.tasks),
            **summary,
        }


def make_vqa_semcom_env(
    config_path: str | Path = "configs/v0.yaml",
    scenario: str | None = "literature_demo",
    seed: int | None = None,
) -> VQASemComDingEnv:
    return VQASemComDingEnv(config_path=config_path, scenario=scenario, seed=seed)
