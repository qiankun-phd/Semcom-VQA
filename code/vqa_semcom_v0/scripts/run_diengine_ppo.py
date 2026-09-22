#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.rl.diengine_env import VQASemComDingEnv, env_spec


def _scenario_list(scenario: str) -> list[str]:
    return [item.strip() for item in scenario.split(",") if item.strip()]


def smoke(config: str, scenario: str, episodes: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    for scenario_name in _scenario_list(scenario):
        env = VQASemComDingEnv(config_path=config, scenario=scenario_name, seed=seed)
        returns: list[float] = []
        for episode in range(episodes):
            reset_out = env.reset(seed=seed + episode)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            ep_return = 0.0
            while not done:
                action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
                step_out = env.step(action)
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, info = step_out
                ep_return += float(reward)
            returns.append(ep_return)
            print(
                f"scenario={scenario_name} episode={episode} return={ep_return:.3f} "
                f"unique_success={info.get('unique_task_success_rate', 0.0):.3f} "
                f"obs_dim={len(obs)}"
            )
        print(f"scenario={scenario_name} mean_return={sum(returns) / max(1, len(returns)):.3f}")
    return 0


def train(config: str, scenario: str, seed: int, max_train_iter: int) -> int:
    try:
        import gym
        from ding.entry import serial_pipeline_onpolicy
        from easydict import EasyDict
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "DI-engine training dependencies are missing. Use --smoke locally, "
            f"or run this script inside the DI-engine environment. Missing: {exc.name or exc}."
        ) from exc

    from vqa_semcom.rl.ding_compat import ensure_ding_gym_env_registered

    ensure_ding_gym_env_registered()
    train_scenario = _scenario_list(scenario)[0]
    spec = env_spec(config, train_scenario)

    def _make_env(**kwargs):
        return VQASemComDingEnv(config_path=config, scenario=train_scenario, seed=seed)

    try:
        gym.envs.registration.register(
            id="VQASemCom-v0",
            entry_point=_make_env,
            max_episode_steps=64,
        )
    except Exception as exc:
        if "Cannot re-register" not in str(exc) and "already registered" not in str(exc):
            raise
    main_config = EasyDict(
        dict(
            exp_name=f"vqa_semcom_ppo_{train_scenario}_seed{seed}",
            env=dict(
                collector_env_num=4,
                evaluator_env_num=1,
                n_evaluator_episode=2,
                stop_value=1e9,
                env_id="VQASemCom-v0",
            ),
            policy=dict(
                cuda=False,
                model=dict(
                    obs_shape=spec.obs_shape,
                    action_shape=spec.action_shape,
                    action_space="continuous",
                    encoder_hidden_size_list=[256, 128, 64],
                    sigma_type="fixed",
                    fixed_sigma_value=0.35,
                    bound_type="tanh",
                ),
                learn=dict(
                    epoch_per_collect=4,
                    batch_size=128,
                    learning_rate=3e-4,
                    value_weight=1.0,
                    entropy_weight=0.01,
                    clip_ratio=0.2,
                    adv_norm=True,
                    value_norm=True,
                ),
                collect=dict(
                    n_sample=512,
                    unroll_len=1,
                    discount_factor=0.99,
                    gae_lambda=0.95,
                ),
                eval=dict(evaluator=dict(eval_freq=20)),
                on_policy=True,
            ),
        )
    )
    create_config = EasyDict(
        dict(
            env=dict(type="gym", import_names=[]),
            env_manager=dict(type="base"),
            policy=dict(type="ppo"),
        )
    )
    serial_pipeline_onpolicy([main_config, create_config], seed=seed, max_train_iter=max_train_iter)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "v0.yaml"))
    parser.add_argument("--scenario", default="literature_demo,cache_freshness")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-train-iter", type=int, default=200)
    parser.add_argument("--smoke", action="store_true", help="Run random-action Gym smoke test instead of DI-engine training.")
    args = parser.parse_args()
    if args.smoke:
        return smoke(args.config, args.scenario, args.episodes, args.seed)
    return train(args.config, args.scenario, args.seed, args.max_train_iter)


if __name__ == "__main__":
    raise SystemExit(main())
