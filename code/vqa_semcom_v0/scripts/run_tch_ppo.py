#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.rl.tch_ppo import CONSTRAINT_KEYS, TCHPPOConfig, TCHPPOEnv, tch_env_spec


METRIC_KEYS = (
    "unique_task_success_rate",
    "average_semantic_utility",
    "average_delay",
    "average_energy",
    "average_payload_mb",
    "quality_violation_rate",
    "deadline_violation_rate",
    "resource_violation_rate",
    "airspace_conflict_rate",
    "gpu_memory_ok_rate",
    "battery_ok_rate",
    "service_level_0_rate",
    "service_level_1_rate",
    "service_level_2_rate",
    "service_level_3_rate",
)


def _scenario_list(scenario: str) -> list[str]:
    return [item.strip() for item in scenario.split(",") if item.strip()]


def _step_env(env: TCHPPOEnv, action: dict[str, Any]):
    step_out = env.step(action)
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = step_out
    return obs, float(reward), bool(done), info


def _random_diengine_action(env: TCHPPOEnv, rng: np.random.Generator) -> dict[str, np.ndarray]:
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


def _tch_config(args: argparse.Namespace) -> TCHPPOConfig:
    targets = {key: 0.0 for key in CONSTRAINT_KEYS}
    return TCHPPOConfig(
        lambda_lr=args.lambda_lr,
        lambda_max=args.lambda_max,
        initial_lambda=args.initial_lambda,
        targets=targets,
    )


def smoke(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "tch_ppo_results.csv"
    summary_path = output_dir / "tch_ppo_summary.md"
    lambda_path = output_dir / "tch_ppo_lambda_trace.csv"
    rng = np.random.default_rng(args.seed)
    episode_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for scenario_name in _scenario_list(args.scenario):
        env = TCHPPOEnv(
            config_path=args.config,
            scenario=scenario_name,
            seed=args.seed,
            tch_config=_tch_config(args),
            reward_scale=args.reward_scale,
        )
        for episode in range(args.episodes):
            reset_out = env.reset(seed=args.seed + episode)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            ep_return = 0.0
            ep_raw_return = 0.0
            step_infos: list[dict[str, Any]] = []
            use_structured = episode % 2 == 0
            while not done:
                action = env.action_space.sample() if use_structured else _random_diengine_action(env, rng)
                obs, reward, done, info = _step_env(env, action)
                ep_return += reward
                ep_raw_return += float(info.get("raw_reward", reward))
                step_infos.append(info)
            row = _episode_summary(scenario_name, args.seed, episode, ep_return, ep_raw_return, step_infos)
            episode_rows.append(row)
            mask = obs["action_mask"]
            print(
                f"scenario={scenario_name} episode={episode} tch_return={ep_return:.3f} "
                f"raw_return={ep_raw_return:.3f} success={row['unique_task_success_rate']:.3f} "
                f"lambda_resource={row['lambda_resource']:.3f} active_mask_sum={float(mask['active_task_mask'].sum()):.1f}"
            )
        trace_rows.extend(env.lambda_trace())
    _write_csv(results_path, episode_rows)
    _write_csv(lambda_path, trace_rows)
    _write_summary(summary_path, episode_rows, trace_rows, title="TCH-PPO Smoke Summary")
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {lambda_path}")
    return 0


def train(args: argparse.Namespace) -> int:
    try:
        import gym
        from ding.entry import serial_pipeline_onpolicy
        from easydict import EasyDict
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "DI-engine training dependencies are missing. Run --smoke locally, "
            f"or use the remote RA_DI conda environment. Missing: {exc.name or exc}."
        ) from exc

    from vqa_semcom.rl.ding_compat import ensure_ding_gym_env_registered

    ensure_ding_gym_env_registered()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_scenario = _scenario_list(args.scenario)[0]
    spec = tch_env_spec(args.config, train_scenario)
    lambda_path = output_dir / "tch_ppo_lambda_trace.csv"
    results_path = output_dir / "tch_ppo_results.csv"
    summary_path = output_dir / "tch_ppo_summary.md"
    if lambda_path.exists():
        lambda_path.unlink()
    tch_config = _tch_config(args)

    def _make_env(**kwargs):
        return TCHPPOEnv(
            config_path=args.config,
            scenario=train_scenario,
            seed=args.seed,
            tch_config=tch_config,
            reward_scale=args.reward_scale,
            trace_path=lambda_path,
            append_trace=True,
        )

    env_id = "TCHVQASemCom-v0"
    try:
        gym.envs.registration.register(
            id=env_id,
            entry_point=_make_env,
            max_episode_steps=64,
        )
    except Exception as exc:
        if "Cannot re-register" not in str(exc) and "already registered" not in str(exc):
            raise

    main_config = EasyDict(
        dict(
            exp_name=str(output_dir / f"tch_ppo_{train_scenario}_seed{args.seed}"),
            env=dict(
                collector_env_num=args.collector_env_num,
                evaluator_env_num=1,
                n_evaluator_episode=2,
                stop_value=1e9,
                env_id=env_id,
            ),
            policy=dict(
                cuda=bool(args.cuda),
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
                    learning_rate=args.learning_rate,
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
                    n_sample=args.n_sample,
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
    if args.model == "hybrid_vqa":
        from vqa_semcom.rl.hybrid_vqa_model import HybridVQAModel

        model = HybridVQAModel(
            obs_shape=spec.obs_shape,
            max_tasks=spec.max_tasks,
            num_uavs=spec.num_uavs,
            num_service_levels=len(spec.service_levels),
            hidden_size=args.hidden_size,
        )
    serial_pipeline_onpolicy([main_config, create_config], seed=args.seed, model=model, max_train_iter=args.max_train_iter)
    trace_rows = _read_csv(lambda_path)
    episode_rows = _summarize_trace_rows(trace_rows, train_scenario, args.seed)
    _write_csv(results_path, episode_rows)
    _write_summary(summary_path, episode_rows, trace_rows, title="TCH-PPO Training Summary")
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {lambda_path}")
    return 0


def _episode_summary(
    scenario: str,
    seed: int,
    episode: int,
    ep_return: float,
    ep_raw_return: float,
    infos: list[dict[str, Any]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "algorithm": "tch_ppo",
        "scenario": scenario,
        "seed": seed,
        "episode": episode,
        "steps": len(infos),
        "tch_return": ep_return,
        "raw_return": ep_raw_return,
        "tch_penalty": sum(float(info.get("tch_penalty", 0.0)) for info in infos),
    }
    for key in METRIC_KEYS:
        row[key] = _mean(infos, key)
    for key in CONSTRAINT_KEYS:
        row[f"cost_{key}"] = _mean(infos, f"cost_{key}")
        row[f"lambda_{key}"] = float(infos[-1].get(f"lambda_{key}", 0.0)) if infos else 0.0
    return row


def _summarize_trace_rows(rows: list[dict[str, Any]], scenario: str, seed: int) -> list[dict[str, Any]]:
    if not rows:
        return [
            {
                "algorithm": "tch_ppo",
                "scenario": scenario,
                "seed": seed,
                "episode": 0,
                "steps": 0,
                "tch_return": 0.0,
                "raw_return": 0.0,
                "tch_penalty": 0.0,
            }
        ]
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("episode", "0"))
        by_episode.setdefault(key, []).append(row)
    summaries = []
    for episode, episode_rows in sorted(by_episode.items(), key=lambda item: int(float(item[0]))):
        infos = [_coerce_row(row) for row in episode_rows]
        summaries.append(
            _episode_summary(
                scenario=scenario,
                seed=seed,
                episode=int(float(episode)),
                ep_return=sum(float(row.get("tch_reward", 0.0)) for row in infos),
                ep_raw_return=sum(float(row.get("raw_reward", 0.0)) for row in infos),
                infos=infos,
            )
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_summary(path: Path, episode_rows: list[dict[str, Any]], trace_rows: list[dict[str, Any]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    avg_success = _mean(episode_rows, "unique_task_success_rate")
    avg_utility = _mean(episode_rows, "average_semantic_utility")
    avg_delay = _mean(episode_rows, "average_delay")
    avg_energy = _mean(episode_rows, "average_energy")
    avg_payload = _mean(episode_rows, "average_payload_mb")
    avg_penalty = _mean(episode_rows, "tch_penalty")
    final_lambdas = {key: 0.0 for key in CONSTRAINT_KEYS}
    if trace_rows:
        last = _coerce_row(trace_rows[-1])
        final_lambdas = {key: float(last.get(f"lambda_{key}", 0.0)) for key in CONSTRAINT_KEYS}
    lines = [
        f"# {title}",
        "",
        f"- episodes: {len(episode_rows)}",
        f"- trace rows: {len(trace_rows)}",
        f"- unique task success: {avg_success:.4f}",
        f"- semantic utility: {avg_utility:.4f}",
        f"- delay: {avg_delay:.4f}",
        f"- energy: {avg_energy:.4f}",
        f"- payload MB: {avg_payload:.4f}",
        f"- total penalty/episode: {avg_penalty:.4f}",
        "",
        "| lambda | value |",
        "|---|---:|",
    ]
    for key in CONSTRAINT_KEYS:
        lines.append(f"| {key} | {final_lambdas[key]:.6f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(_to_float(row.get(key, 0.0)) for row in rows) / max(1, len(rows))


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _to_float(value) if _looks_numeric(value) else value for key, value in row.items()}


def _looks_numeric(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str) or not value:
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "v0.yaml"))
    parser.add_argument("--scenario", default="literature_demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-train-iter", type=int, default=200)
    parser.add_argument("--smoke", action="store_true", help="Run random-action TCH smoke test instead of DI-engine training.")
    parser.add_argument("--model", choices=["hybrid_vqa", "vac"], default="hybrid_vqa")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "rl"))
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--lambda-lr", type=float, default=0.05)
    parser.add_argument("--lambda-max", type=float, default=10.0)
    parser.add_argument("--initial-lambda", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-sample", type=int, default=512)
    parser.add_argument("--collector-env-num", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        return smoke(args)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
