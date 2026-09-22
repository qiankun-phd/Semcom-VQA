from __future__ import annotations

from typing import Any


class OldGymAPIAdapter:
    """Expose a Gymnasium env through the old Gym API expected by this DI-engine fork."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reward_range = getattr(env, "reward_range", (-float("inf"), float("inf")))
        self.metadata = getattr(env, "metadata", {})

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0]
        return out

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return obs, reward, bool(terminated or truncated), info
        return out

    def seed(self, seed: int | None = None):
        if hasattr(self.action_space, "seed"):
            self.action_space.seed(seed)
        self.env.reset(seed=seed)
        return [0 if seed is None else int(seed)]

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str):
        return getattr(self.env, name)


def ensure_ding_gym_env_registered() -> None:
    """Register DI-engine's lightweight Gym wrapper when a fork omits it."""

    from ding.utils import ENV_REGISTRY

    if "gym" in ENV_REGISTRY:
        return
    try:
        import gym
        import numpy as np
        from ding.envs import DingEnvWrapper
    except (ImportError, AttributeError) as exc:  # pragma: no cover - depends on DI-engine fork
        raise RuntimeError("DI-engine does not expose a registerable gym_env wrapper.") from exc

    class LenientDingEnvWrapper(DingEnvWrapper):
        def _judge_action_type(self, action):
            if isinstance(action, dict):
                return {key: self._judge_action_type(value) for key, value in action.items()}
            if isinstance(action, (list, tuple)):
                return np.asarray(action)
            return super()._judge_action_type(action)

    class GymEnvAdapter:
        @staticmethod
        def create_collector_env_cfg(cfg: dict):
            return LenientDingEnvWrapper.create_collector_env_cfg(cfg)

        @staticmethod
        def create_evaluator_env_cfg(cfg: dict):
            return LenientDingEnvWrapper.create_evaluator_env_cfg(cfg)

        def __new__(cls, cfg: dict, seed_api: bool = True, **kwargs):
            caller = "collector" if bool(cfg.get("is_train", True)) else "evaluator"
            try:
                raw_env = gym.make(cfg.env_id, disable_env_checker=True, **kwargs)
            except TypeError:
                raw_env = gym.make(cfg.env_id, **kwargs)
            return LenientDingEnvWrapper(OldGymAPIAdapter(raw_env), cfg=cfg, seed_api=seed_api, caller=caller)

    ENV_REGISTRY.register("gym", GymEnvAdapter)
