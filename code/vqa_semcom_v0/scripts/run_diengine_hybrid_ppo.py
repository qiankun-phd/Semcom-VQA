#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.rl.diengine_hybrid_env import HybridVQADingEnv, hybrid_env_spec


def _scenario_list(scenario: str) -> list[str]:
    return [item.strip() for item in scenario.split(",") if item.strip()]


def _step_env(env: HybridVQADingEnv, action: dict[str, Any]):
    step_out = env.step(action)
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = step_out
    return obs, float(reward), bool(done), info


def _random_diengine_action(env: HybridVQADingEnv, rng: np.random.Generator) -> dict[str, np.ndarray]:
    spec = env.spec_info
    action_type = []
    for _ in range(spec.max_tasks):
        action_type.extend(
            [
                int(rng.integers(0, spec.num_uavs)),
                int(rng.integers(0, 3)),
                int(rng.integers(0, len(spec.service_levels))),
            ]
        )
    return {
        "action_type": np.asarray(action_type, dtype=np.int64),
        "action_args": rng.uniform(-1.0, 1.0, size=spec.action_args_shape).astype(np.float32),
    }


def smoke(config: str, scenario: str, episodes: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    for scenario_name in _scenario_list(scenario):
        env = HybridVQADingEnv(config_path=config, scenario=scenario_name, seed=seed)
        returns: list[float] = []
        for episode in range(episodes):
            reset_out = env.reset(seed=seed + episode)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            ep_return = 0.0
            use_structured = episode % 2 == 0
            while not done:
                action = env.action_space.sample() if use_structured else _random_diengine_action(env, rng)
                obs, reward, done, info = _step_env(env, action)
                ep_return += reward
            returns.append(ep_return)
            mask = obs["action_mask"]
            print(
                f"scenario={scenario_name} episode={episode} return={ep_return:.3f} "
                f"unique_success={info.get('unique_task_success_rate', 0.0):.3f} "
                f"state_dim={len(obs['state'])} active_mask_sum={float(mask['active_task_mask'].sum()):.1f}"
            )
        print(f"scenario={scenario_name} mean_return={sum(returns) / max(1, len(returns)):.3f}")
    return 0


def train(config: str, scenario: str, seed: int, max_train_iter: int, model_name: str) -> int:
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
    spec = hybrid_env_spec(config, train_scenario)

    def _make_env(**kwargs):
        return HybridVQADingEnv(config_path=config, scenario=train_scenario, seed=seed)

    try:
        gym.envs.registration.register(
            id="HybridVQASemCom-v0",
            entry_point=_make_env,
            max_episode_steps=64,
        )
    except Exception as exc:
        if "Cannot re-register" not in str(exc) and "already registered" not in str(exc):
            raise

    main_config = EasyDict(
        dict(
            exp_name=f"hybrid_vqa_semcom_ppo_{train_scenario}_seed{seed}",
            env=dict(
                collector_env_num=4,
                evaluator_env_num=1,
                n_evaluator_episode=2,
                stop_value=1e9,
                env_id="HybridVQASemCom-v0",
            ),
            policy=dict(
                cuda=False,
                model=dict(
                    obs_shape=spec.obs_shape,
                    action_shape=dict(
                        action_type_shape=list(spec.action_type_shape),
                        action_args_shape=spec.action_args_shape,
                    ),
                    action_space="hybrid",
                    encoder_hidden_size_list=[256, 128, 64],
                    sigma_type="fixed",
                    fixed_sigma_value=0.25,
                    bound_type="tanh",
                ),
                learn=dict(
                    epoch_per_collect=4,
                    batch_size=128,
                    learning_rate=3e-4,
                    value_weight=1.0,
                    entropy_weight=0.02,
                    clip_ratio=0.2,
                    adv_norm=True,
                    value_norm=True,
                    ppo_param_init=True,
                    grad_clip_type="clip_norm",
                    grad_clip_value=0.5,
                    lr_scheduler=None,
                ),
                collect=dict(
                    n_sample=512,
                    unroll_len=1,
                    discount_factor=0.99,
                    gae_lambda=0.95,
                ),
                eval=dict(evaluator=dict(eval_freq=20)),
                other=dict(),
                on_policy=True,
                action_space="hybrid",
                priority=False,
                priority_IS_weight=False,
                recompute_adv=True,
                transition_with_policy_data=True,
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
    model = None
    if model_name == "hybrid_vqa":
        from vqa_semcom.rl.hybrid_vqa_model import HybridVQAModel

        model = HybridVQAModel(
            obs_shape=spec.obs_shape,
            max_tasks=spec.max_tasks,
            num_uavs=spec.num_uavs,
            num_service_levels=len(spec.service_levels),
            hidden_size=128,
        )
    serial_pipeline_onpolicy([main_config, create_config], seed=seed, model=model, max_train_iter=max_train_iter)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "v0.yaml"))
    parser.add_argument("--scenario", default="literature_demo,cache_freshness")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-train-iter", type=int, default=200)
    parser.add_argument("--model", choices=["vac", "hybrid_vqa"], default="vac")
    parser.add_argument("--smoke", action="store_true", help="Run random-action hybrid smoke test instead of DI-engine training.")
    args = parser.parse_args()
    if args.smoke:
        return smoke(args.config, args.scenario, args.episodes, args.seed)
    return train(args.config, args.scenario, args.seed, args.max_train_iter, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
