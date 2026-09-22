#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.sim.resource_env import load_lut, read_csv, run_simulation, write_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tasks_path = resolve_path(cfg["paths"]["tasks_csv"])
    lut_path = resolve_path(cfg["paths"]["lut_csv"])
    if not tasks_path.exists() or not lut_path.exists():
        raise RuntimeError("Tasks/LUT not found. Run scripts/build_v0_lut.py first.")
    tasks = read_csv(tasks_path)
    lut = load_lut(lut_path)
    results = run_simulation(tasks, lut, cfg, episodes=args.episodes)
    write_results(results, resolve_path(cfg["paths"]["sim_results_csv"]), resolve_path(cfg["paths"]["sim_summary_md"]))
    for r in results:
        print(
            f"{r.policy}: success={r.task_success_rate:.3f} "
            f"acc={r.average_accuracy:.3f} delay={r.average_delay:.3f} energy={r.average_energy:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
