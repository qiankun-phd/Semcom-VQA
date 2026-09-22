# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`vqa_semcom_v0` is a clean-room V0 implementation of task-conditioned UAV VQA semantic-communication quality modeling and resource-allocation simulation. It deliberately does **not** import prior HPPO-VQA/UAV-MEC code. The pipeline:

```
VisDrone annotations -> VQA-style tasks -> semantic quality LUT -> resource simulation -> UAV-VQA resource allocation -> RL (DI-engine PPO / hybrid PPO / TCH-PPO)
```

This directory is a subproject of the larger `HPPO-VQA` git repo (paper sources live in `../paper`, plans in `../plan`).

## Commands

There is no packaging/build step. Scripts insert `src/` into `sys.path` themselves; run everything from this directory.

```bash
# Tests (main pipeline — stdlib only, fast)
python -m unittest discover -s tests
# Single test
python -m unittest tests.test_v0_pipeline.V0PipelineTest.test_task_generator_creates_required_question_types

# Pipeline stages (in order)
python scripts/build_v0_lut.py --config configs/v0.yaml --limit-images 100   # add --demo for built-in fixture (no VisDrone needed)
python scripts/run_v0_sim.py --config configs/v0.yaml --episodes 10
python scripts/run_resource_alloc.py --config configs/v0.yaml --episodes 5 --scenario literature_demo --report

# RL adapters — always verify with --smoke first (works without DI-engine/torch training deps)
python scripts/run_diengine_ppo.py --config configs/v0.yaml --scenario literature_demo,cache_freshness --smoke
python scripts/run_diengine_hybrid_ppo.py --config configs/v0.yaml --scenario literature_demo --smoke
python scripts/run_tch_ppo.py --config configs/v0.yaml --scenario literature_demo --smoke --episodes 2
# Real training: drop --smoke, add --max-train-iter N; hybrid PPO takes --model vac | hybrid_vqa
```

All outputs go under `outputs/` (`tasks/`, `lut/`, `sim/`, `resource_alloc/`, `rl/`); paths are defined in the config's `paths` block.

## Key facts that are easy to get wrong

- **`configs/v0.yaml` is JSON**, not YAML. `vqa_semcom.config.load_config` uses `json.loads` to avoid a PyYAML dependency. Keep edits JSON-valid (dict keys for levels are strings: `"0"`–`"3"`).
- **Core code is stdlib-only** (no OpenCV, numpy/pandas optional). Only the RL training paths need torch/DI-engine.
- **Service levels are fixed**: 0=cache answer, 1=lightweight evidence (tags/boxes/tokens), 2=full image, 3=ROI/crop. `critical` is a **risk level**, not a service level.
- **Quality and deadline constraints are intentionally separate**: quality satisfied iff `A_k >= epsilon_k`; deadline satisfied iff `T_k <= tau_k`. Don't merge them into one penalty.
- The LUT is keyed as `LUT[question_type, service_level, channel_bin, view_quality_bin, freshness_bin, risk_level]`.
- **Heavy training runs remotely**, not on this Mac. The remote env (see README) is conda `RA_DI` / `uav_semcom` on lab servers (`lab-s1`, `lab-s2`), reached via ssh/scp; results are pulled back with rsync/scp.
- `scripts/joint_control_rl_conditional_c1000.py` imports `bubbles_vqa.*`, a package that exists **only on the remote machine** — it will not run locally. It enforces frozen experiment contracts (train seeds 60000001–60000020, val 70000001–70000005, final seeds 80000001–80000020 are a locked denylist; no live retune of the G1V-frozen config). Never weaken these guards. Its tests (`tests/rl_training/test_conditional_c1000_runner.py`) use pytest and need numpy/torch.

## Architecture

`src/vqa_semcom/` layers, bottom-up:

- `data/visdrone.py` — VisDrone DET annotation parser (category map, skips `ignored`), plus `demo_objects()` fixture used by `--demo` and tests.
- `tasks/generate_tasks.py` — turns per-image objects into VQA tasks across five question types (presence, counting, risk, attribute, relation) with a view-quality proxy (`_view_quality`: scale/occlusion/truncation/density).
- `quality/lut_builder.py` — deterministic proxy evaluator (`estimate_accuracy`) that composes base accuracy × service gain × channel/view/freshness/risk factors (all coefficients in config `evaluator` block, incl. `empirical_calibration`) into the semantic-quality LUT.
- `sim/resource_env.py` — simple per-task policy simulation over the LUT (the `run_v0_sim.py` baselines: always_cache/light/image/roi, greedy_min_sufficient_evidence).
- `sim/vqa_resource_env.py` — the real multi-UAV environment `VQAResourceEnv`: A2G channel model (LoS/NLoS, fading, interference), semantic cache with freshness, InterUSS-style `Area4D` (2D footprint + altitude band + time window) conflicts, battery/CPU/GPU/GPU-memory accounting, per-decision feasibility diagnostics. Scenarios (`literature_demo`, `cache_freshness`, `critical_preemption`, `area4d_conflict`, `bad_channel_stress`, `clear_area_nominal`) are constructed here.
- `sim/allocators.py` — heuristic allocation policies behind a `ResourceAllocator` Protocol, registered in `ALLOCATORS` (cache_first, min_sufficient_evidence, deadline_aware_greedy, goal_oriented_priority, joint_greedy_resource). Add new baselines here.
- `rl/` — adapters wrapping `VQAResourceEnv` for DI-engine:
  - `diengine_env.py` — flat Gym env; action = 7 continuous values per task (`uav_id, sensing_mode, service_level, bandwidth_share, power_scalar, cpu_share, gpu_share`).
  - `diengine_hybrid_env.py` — hybrid action space; discrete heads per task are `assigned_uav, sensing_decision, service_level`, continuous args are `bandwidth_share, power_scalar, cpu_share, gpu_share` (keep this ordering).
  - `hybrid_vqa_model.py` — HPPO-ready actor-critic skeleton (`--model hybrid_vqa`).
  - `tch_ppo.py` — Lagrangian penalty layer over six costs (`quality, deadline, resource, conflict, battery, gpu_memory`), writes lambda traces.
  - `ding_compat.py` — Gymnasium→old-Gym API shim for the DI-engine fork in use.

`tests/test_v0_pipeline.py` (unittest, 39 tests) covers the whole local pipeline and is the fastest regression check after any change to the modules above.

Baseline design rationale and the planned baseline roadmap are in `docs/resource_allocation_baselines.md`.
