# UAV-VQA Resource Allocation Baselines

This note translates the June 2026 meeting framework into an experiment plan for
`vqa_semcom_v0`.

## Problem Scope

The low-level controller serves VQA tasks every slot. Its state includes
question type, deadline, accuracy requirement, priority, cache freshness,
channel/view quality, UAV positions, battery, and edge CPU/GPU load. Its action
chooses UAV assignment, sensing mode, semantic service level, bandwidth, power,
CPU, and GPU share.

The service levels follow the meeting slides:

- `0`: cached answer
- `1`: lightweight tags/boxes/tokens
- `2`: crop or full-image evidence

The reward should report both task utility and constraint violations:

```text
priority * success - delay penalty - energy penalty
- quality violation - deadline violation - airspace conflict
```

## Literature-Informed Baseline Set

Use the following baseline families in the first paper-grade comparison:

| Family | Baseline | Why include it | V0 status |
|---|---|---|---|
| Non-learning lower bound | random feasible action | sanity floor for RL | add later |
| Non-learning lower bound | equal bandwidth/CPU/GPU | common resource allocation baseline | add later |
| Fixed evidence | always cache / always light / always image | isolates semantic evidence level | available in `run_v0_sim.py` |
| Task-oriented heuristic | cache-first | common semantic-cache baseline | available |
| Task-oriented heuristic | minimum sufficient evidence | matches task-oriented SemCom objective | available |
| Task-oriented heuristic | deadline-aware greedy | EDF/priority style resource scheduling | available |
| Optimization oracle | joint greedy / exhaustive over discretized actions | practical upper reference for small action grids | available as `joint_greedy_resource` |
| Online learning | contextual bandit over service level/resource bins | lightweight alternative to DRL | planned |
| DRL | PPO continuous-vector controller | first DI-engine adapter target | added |
| Hybrid DRL | PDQN/PADDPG/HPPO | hybrid-action comparison for the final algorithm | planned after PPO smoke |
| Multi-agent DRL | MAPPO/HAPPO | needed when high-level UAV mobility/matching is fully multi-UAV | later stage |

## Source Anchors

The baseline choices follow common patterns in task-oriented semantic
communication and wireless resource allocation:

- Task-oriented bandwidth/power allocation papers typically compare optimized
  allocation against equal/static allocation and heuristic resource allocation.
- Semantic communication papers often include semantic-aware allocation versus
  conventional bit-throughput or non-semantic allocation.
- DRL resource-allocation papers usually compare PPO/DDPG-style learning against
  greedy, random, fixed allocation, and optimization-based solvers where the
  action grid is small.

Concrete papers used as anchors:

- Chuanhong Liu et al., "Bandwidth and Power Allocation for Task-Oriented
  Semantic Communication", 2022.
- Zhiyu Shao et al., "Semantic-Aware Resource Allocation Based on Deep
  Reinforcement Learning for 5G-V2X HetNets", 2024.
- Xinyi Lin et al., "RL-Driven Semantic Compression Model Selection and
  Resource Allocation in Semantic Communication Systems", 2025.

## Immediate Experiment Order

1. Rebuild demo LUT and run all existing heuristic baselines.
2. Run DI-engine adapter smoke test with random actions.
3. Train PPO on `literature_demo` for a short interface run.
4. Compare PPO against `cache_first`, `min_sufficient_evidence`,
   `deadline_aware_greedy`, `goal_oriented_priority`, and
   `joint_greedy_resource`.
5. Add equal-share and random feasible baselines before reporting first tables.
6. Add PDQN/PADDPG or HPPO only after PPO shows stable environment interaction.
