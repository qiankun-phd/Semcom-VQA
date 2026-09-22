#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.sim.allocators import ALLOCATORS
from vqa_semcom.sim.vqa_resource_env import VQAResourceEnv, load_resource_inputs, summarize_outcomes, write_resource_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--scenario",
        default=None,
        help="Resource scenario: literature_demo, clear_area_nominal, cache_freshness, critical_preemption, area4d_conflict, bad_channel_stress, or sampled.",
    )
    parser.add_argument("--report", action="store_true", help="Generate the companion report and SVG figures.")
    parser.add_argument("--mode", choices=["batch", "mdp"], default="batch", help="batch keeps the v0 one-shot demo; mdp rolls out multiple slots.")
    parser.add_argument(
        "--policy",
        default="all",
        choices=["all", *sorted(ALLOCATORS.keys())],
        help="Resource allocation policy to evaluate.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    tasks_path = resolve_path(cfg["paths"]["tasks_csv"])
    preferred_lut = cfg["paths"].get("vlm_lut_csv", cfg["paths"]["lut_csv"])
    lut_path = resolve_path(preferred_lut)
    if not lut_path.exists() and preferred_lut != cfg["paths"]["lut_csv"]:
        lut_path = resolve_path(cfg["paths"]["lut_csv"])
    if not tasks_path.exists() or not lut_path.exists():
        raise RuntimeError("Tasks/LUT not found. Run scripts/build_v0_lut.py first.")

    tasks, lut = load_resource_inputs(tasks_path, lut_path, cfg)
    policies = sorted(ALLOCATORS.keys()) if args.policy == "all" else [args.policy]
    outcomes = []
    for policy_name in policies:
        allocator = ALLOCATORS[policy_name]
        env = VQAResourceEnv(tasks, lut, cfg)
        for episode in range(args.episodes):
            env.reset(episode, scenario=args.scenario, mdp_mode=args.mode == "mdp")
            done = False
            while not done:
                if args.mode == "mdp":
                    original_tasks = env.tasks
                    env.tasks = env.active_tasks()
                    decisions = allocator.allocate(env) if env.tasks else []
                    env.tasks = original_tasks
                else:
                    decisions = allocator.allocate(env)
                _state, episode_outcomes, done, _info = env.step(decisions, policy=policy_name)
                outcomes.extend(episode_outcomes)
                if args.mode == "batch":
                    break

    write_resource_outputs(
        outcomes,
        resolve_path(cfg["paths"].get("resource_results_csv", "outputs/resource_alloc/v0_resource_results.csv")),
        resolve_path(cfg["paths"].get("resource_summary_md", "outputs/resource_alloc/v0_resource_summary.md")),
    )
    for policy_name in policies:
        rows = [o for o in outcomes if o.policy == policy_name]
        summary = summarize_outcomes(rows)
        print(
            f"{policy_name}: attempt_success={summary['attempt_success_rate']:.3f} "
            f"unique_success={summary['unique_task_success_rate']:.3f} "
            f"delay={summary['average_delay']:.3f} energy={summary['average_energy']:.3f}"
        )
    if args.report:
        from report_resource_alloc import main as report_main

        old_argv = sys.argv
        sys.argv = ["report_resource_alloc.py", "--config", args.config]
        try:
            report_main()
        finally:
            sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
