# UAV-VQA Semantic Communication V0

This is a clean V0 implementation for task-conditioned UAV VQA semantic
communication quality modeling and resource simulation.

It does not import or reuse previous HPPO-VQA/UAV-MEC code. The V0 pipeline is:

```text
VisDrone annotations -> VQA-style tasks -> semantic quality LUT -> resource simulation
-> UAV-VQA resource allocation
```

## Environment

Use the existing remote environment:

```bash
source /home/qiankun/miniconda3/etc/profile.d/conda.sh
conda activate RA_DI
export PYTHONPATH=/home/qiankun/HPPO-VQA/hppo-uav:/home/qiankun/HPPO-VQA/vqa_semcom_v0/src:$PYTHONPATH
```

The implementation avoids OpenCV and uses only the Python standard library plus
NumPy/Pandas where available.

## Download VisDrone DET valset

```bash
python scripts/download_visdrone_det.py --config configs/v0.yaml --split val
```

The script downloads the official VisDrone2019-DET valset Google Drive file and
extracts it to:

```text
data/raw/visdrone/DET/val/
```

If Google Drive blocks command-line download, manually download the valset from:

```text
https://github.com/VisDrone/VisDrone-Dataset
```

and place/extract it so that this file layout exists:

```text
data/raw/visdrone/DET/val/annotations/*.txt
data/raw/visdrone/DET/val/images/*
```

## Build V0 LUT

```bash
python scripts/build_v0_lut.py --config configs/v0.yaml --limit-images 100
```

Outputs:

```text
outputs/tasks/v0_tasks.csv
outputs/lut/v0_semantic_quality_lut.csv
outputs/lut/v0_semantic_quality_summary.json
outputs/lut/v0_semantic_quality_summary.md
```

If VisDrone is not available, add `--demo` to run a tiny built-in fixture:

```bash
python scripts/build_v0_lut.py --config configs/v0.yaml --demo
```

## Run Resource Simulation

```bash
python scripts/run_v0_sim.py --config configs/v0.yaml --episodes 10
```

Outputs:

```text
outputs/sim/v0_results.csv
outputs/sim/v0_summary.md
```

Baselines:

- `always_cache`
- `always_light`
- `always_image`
- `greedy_min_sufficient_evidence`

## Run UAV-VQA Resource Allocation

```bash
python scripts/run_resource_alloc.py --config configs/v0.yaml --episodes 5 --scenario literature_demo --report
```

Outputs:

```text
outputs/resource_alloc/v0_resource_results.csv
outputs/resource_alloc/v0_resource_summary.md
outputs/resource_alloc/v0_resource_report.md
outputs/resource_alloc/figures/*.svg
```

Allocation policies:

- `cache_first`
- `min_sufficient_evidence`
- `deadline_aware_greedy`
- `goal_oriented_priority`
- `joint_greedy_resource`

## DI-engine PPO Adapters

The flat adapter exposes the V0 MDP as a Gym-compatible continuous-vector
environment for DI-engine PPO/SAC smoke runs:

```bash
python scripts/run_diengine_ppo.py --config configs/v0.yaml --scenario literature_demo,cache_freshness --smoke
```

Run without `--smoke` inside a DI-engine environment to start PPO training:

```bash
python scripts/run_diengine_ppo.py --config configs/v0.yaml --scenario literature_demo --max-train-iter 200
```

The adapter action vector uses 7 values per task:

```text
uav_id, sensing_mode, service_level, bandwidth_share, power_scalar, cpu_share, gpu_share
```

The formal hybrid adapter exposes structured task actions and also accepts
DI-engine hybrid PPO actions:

```bash
python scripts/run_diengine_hybrid_ppo.py --config configs/v0.yaml --scenario literature_demo,cache_freshness --smoke
```

Run hybrid PPO with the DI-engine `VAC(action_space="hybrid")` baseline:

```bash
python scripts/run_diengine_hybrid_ppo.py --config configs/v0.yaml --scenario literature_demo --max-train-iter 200 --model vac
```

Or use the HPPO-ready `HybridVQAModel` actor-critic skeleton:

```bash
python scripts/run_diengine_hybrid_ppo.py --config configs/v0.yaml --scenario literature_demo --max-train-iter 200 --model hybrid_vqa
```

TCH-PPO uses the same centralized hybrid scheduler, with a Lagrangian penalty
over quality, deadline, resource, conflict, battery, and GPU-memory costs:

```bash
python scripts/run_tch_ppo.py --config configs/v0.yaml --scenario literature_demo --smoke --episodes 2
python scripts/run_tch_ppo.py --config configs/v0.yaml --scenario literature_demo --max-train-iter 20 --seed 0
```

TCH-PPO writes:

```text
outputs/rl/tch_ppo_results.csv
outputs/rl/tch_ppo_summary.md
outputs/rl/tch_ppo_lambda_trace.csv
```

Hybrid discrete heads are ordered per task:

```text
assigned_uav, sensing_decision, service_level
```

Hybrid continuous arguments are ordered per task:

```text
bandwidth_share, power_scalar, cpu_share, gpu_share
```

Baseline planning notes are in:

```text
docs/resource_allocation_baselines.md
```

Scenarios:

- `literature_demo`: calibrated mix of cache-feasible, light-feasible,
  image-required, and infeasible tasks.
- `clear_area_nominal`: no 4D airspace conflict.
- `cache_freshness`: fresh/stale/expired semantic cache cases.
- `critical_preemption`: urgent high-priority VQA task competing with normal tasks.
- `area4d_conflict`: InterUSS-style overlapping operational areas.
- `bad_channel_stress`: adverse wireless channel stress test.

This model adds InterUSS-inspired 4D task areas without depending on InterUSS:

```text
Area4D = 2D footprint + altitude band + time window
```

The allocator jointly decides UAV assignment, sensing/cache action, evidence
level, bandwidth share, power, and CPU share. The environment reports separate
quality, deadline, resource, and airspace-conflict violations.

Each allocation outcome also includes feasibility diagnostics:

```text
min_quality_level, min_delay_level, feasible_service_levels, feasibility_class
```

## Run Tests

```bash
python -m unittest discover -s tests
```

## Modeling Notes

The lookup table estimates expected VQA accuracy:

```text
A_k = LUT[question_type, service_level, channel_bin, view_quality_bin, freshness_bin, risk_level]
```

Quality and deadline constraints are intentionally separate:

```text
quality satisfied if A_k >= epsilon_k
deadline satisfied if T_k <= tau_k
```

Service levels are fixed:

- `0`: cache answer
- `1`: lightweight evidence, e.g. tags / boxes / semantic tokens
- `2`: full image evidence
- `3`: ROI/crop evidence

`critical` is a risk level, not a service level.
