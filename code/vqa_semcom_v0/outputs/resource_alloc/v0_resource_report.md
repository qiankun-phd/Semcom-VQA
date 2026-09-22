# UAV-VQA Goal-oriented Resource Allocation Report

Batch mode is one-shot allocation. MDP mode is multi-slot rollout; use `unique task success` and `completion` when comparing MDP policies.

## Policy Comparison

| policy | attempt success | unique task success | completion | attempts/task | semantic utility | delay | energy | rate | SNR | SINR | quality sat. | deadline sat. | feasible | conflict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cache_first | 0.150 | 0.150 | 0.150 | 1.000 | 0.572 | 4.278 | 2.072 | 9.743 | 32.803 | 19.444 | 0.562 | 0.338 | 0.675 | 0.000 |
| deadline_aware_greedy | 0.150 | 0.150 | 0.150 | 1.000 | 0.572 | 4.257 | 2.044 | 9.767 | 32.807 | 19.517 | 0.562 | 0.362 | 0.675 | 0.000 |
| goal_oriented_priority | 0.312 | 0.312 | 0.312 | 1.000 | 0.422 | 1.358 | 0.313 | 10.482 | 64.208 | 33.117 | 0.362 | 0.900 | 0.675 | 0.000 |
| joint_greedy_resource | 0.275 | 0.275 | 0.275 | 1.000 | 0.672 | 2.827 | 0.574 | 10.925 | 27.237 | 21.875 | 0.675 | 0.512 | 0.675 | 0.000 |
| min_sufficient_evidence | 0.150 | 0.150 | 0.150 | 1.000 | 0.572 | 4.278 | 2.072 | 9.743 | 32.803 | 19.444 | 0.562 | 0.338 | 0.675 | 0.000 |

## Channel Consistency

| policy | effective bad | effective medium | effective good | impairment dB | dist m | elev deg | LoS prob. | path loss | interference | fading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cache_first | 0.125 | 0.562 | 0.312 | 0.000 | 260.562 | 18.650 | 0.319 | 104.060 | -93.863 | 0.038 |
| deadline_aware_greedy | 0.125 | 0.562 | 0.312 | 0.000 | 260.562 | 18.650 | 0.319 | 104.060 | -93.924 | 0.038 |
| goal_oriented_priority | 0.125 | 0.388 | 0.487 | 0.000 | 260.562 | 18.650 | 0.319 | 104.060 | -114.539 | 0.038 |
| joint_greedy_resource | 0.125 | 0.450 | 0.425 | 0.000 | 260.562 | 18.650 | 0.319 | 104.060 | -105.890 | 0.038 |
| min_sufficient_evidence | 0.125 | 0.562 | 0.312 | 0.000 | 260.562 | 18.650 | 0.319 | 104.060 | -93.863 | 0.038 |

## Delay Breakdown

| policy | travel | sensing | upload | queue | model load | inference |
|---|---:|---:|---:|---:|---:|---:|
| cache_first | 0.631 | 0.656 | 0.960 | 0.282 | 0.374 | 1.374 |
| deadline_aware_greedy | 0.631 | 0.656 | 0.950 | 0.282 | 0.374 | 1.364 |
| goal_oriented_priority | 0.187 | 0.213 | 0.103 | 0.282 | 0.400 | 0.173 |
| joint_greedy_resource | 0.480 | 0.400 | 0.365 | 0.282 | 0.374 | 0.926 |
| min_sufficient_evidence | 0.631 | 0.656 | 0.960 | 0.282 | 0.374 | 1.374 |

## Semantic Cache, MEC, and UAV State

| policy | cache hit prob. | semantic cache hit | acc. gain | payload MB | semantic eff. | LUT missing | utility/latency | GPU mem ok | battery ok | battery remaining | UAV util. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cache_first | 0.515 | 0.125 | 0.236 | 0.961 | 0.257 | 0.000 | 0.210 | 1.000 | 1.000 | 99.931 | 0.500 |
| deadline_aware_greedy | 0.515 | 0.125 | 0.236 | 0.961 | 0.257 | 0.000 | 0.210 | 1.000 | 1.000 | 99.932 | 0.500 |
| goal_oriented_priority | 0.515 | 0.425 | 0.109 | 0.355 | 0.095 | 0.000 | 0.230 | 1.000 | 1.000 | 99.981 | 0.500 |
| joint_greedy_resource | 0.515 | 0.125 | 0.187 | 0.433 | 0.325 | 0.000 | 0.259 | 1.000 | 1.000 | 99.967 | 0.500 |
| min_sufficient_evidence | 0.515 | 0.125 | 0.236 | 0.961 | 0.257 | 0.000 | 0.210 | 1.000 | 1.000 | 99.931 | 0.500 |

## Violation Breakdown

| policy | quality violation | deadline violation | resource violation | airspace conflict |
|---|---:|---:|---:|---:|
| cache_first | 0.438 | 0.662 | 0.000 | 0.000 |
| deadline_aware_greedy | 0.438 | 0.637 | 0.000 | 0.000 |
| goal_oriented_priority | 0.637 | 0.100 | 0.000 | 0.000 |
| joint_greedy_resource | 0.325 | 0.488 | 0.000 | 0.000 |
| min_sufficient_evidence | 0.438 | 0.662 | 0.000 | 0.000 |

## Service Level Distribution

| policy | cache s=0 | tags/tokens s=1 | full image s=2 | ROI/crop s=3 |
|---|---:|---:|---:|---:|
| cache_first | 0.125 | 0.125 | 0.500 | 0.250 |
| deadline_aware_greedy | 0.125 | 0.125 | 0.500 | 0.250 |
| goal_oriented_priority | 0.750 | 0.000 | 0.200 | 0.050 |
| joint_greedy_resource | 0.375 | 0.125 | 0.125 | 0.375 |
| min_sufficient_evidence | 0.125 | 0.125 | 0.500 | 0.250 |

## Representative Task Trace

| policy | task | attempt | sensing | s | acc | gain | payload MB | cache p | delay | upload | infer | channel | SNR | SINR | LoS | path loss | intf | GPU | mem ok | batt ok | LUT miss | feasible | success | completed |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| cache_first | ep0_cache_feasible_0 | 1 | reuse_cache | 0 | 0.667 | 0.000 | 0.010 | 0.96 | 0.806 | 0.000 | 0.030 | good | 38.22 | 38.07 | 0.60 | 97.8 | -120.0 | 0.12 | 1 | 1 | 0 | feasible | 1 | 1 |
| cache_first | ep0_light_feasible_1 | 1 | observe | 1 | 0.516 | 0.081 | 0.180 | 0.68 | 2.504 | 0.197 | 0.720 | medium | 31.40 | 14.55 | 0.21 | 104.6 | -88.5 | 0.12 | 1 | 1 | 0 | feasible | 0 | 0 |
| cache_first | ep0_image_required_2 | 1 | observe | 2 | 0.691 | 0.433 | 1.600 | 0.28 | 5.844 | 1.708 | 1.774 | medium | 31.73 | 14.90 | 0.28 | 106.5 | -88.5 | 0.12 | 1 | 1 | 0 | feasible | 0 | 0 |
| cache_first | ep0_infeasible_bad_channel_3 | 1 | observe | 3 | 0.285 | 0.062 | 0.550 | 0.28 | 3.858 | 0.348 | 1.574 | bad | 39.81 | 25.34 | 0.49 | 98.4 | -90.9 | 0.12 | 1 | 1 | 0 | quality_limited | 0 | 0 |
| cache_first | ep0_critical_preempt_4 | 1 | observe | 2 | 0.691 | 0.433 | 1.600 | 0.28 | 6.307 | 1.911 | 1.774 | medium | 30.16 | 13.24 | 0.16 | 108.1 | -88.4 | 0.12 | 1 | 1 | 0 | deadline_limited | 0 | 0 |
| cache_first | ep0_normal_competing_5 | 1 | observe | 2 | 0.620 | 0.185 | 1.600 | 0.68 | 5.565 | 1.434 | 1.774 | medium | 34.38 | 17.85 | 0.29 | 101.7 | -88.8 | 0.12 | 1 | 1 | 0 | feasible | 0 | 0 |
| cache_first | ep0_attribute_damage_6 | 1 | observe | 2 | 0.698 | 0.287 | 1.600 | 0.68 | 6.200 | 2.026 | 1.774 | medium | 29.40 | 12.44 | 0.26 | 106.6 | -88.4 | 0.12 | 1 | 1 | 0 | feasible | 0 | 0 |
| cache_first | ep0_relation_person_vehicle_7 | 1 | observe | 3 | 0.526 | 0.278 | 0.550 | 0.28 | 4.167 | 0.470 | 1.574 | medium | 35.12 | 18.71 | 0.27 | 100.9 | -88.9 | 0.12 | 1 | 1 | 0 | quality_limited | 0 | 0 |
| cache_first | ep1_cache_feasible_0 | 1 | reuse_cache | 0 | 0.667 | 0.000 | 0.010 | 0.96 | 0.684 | 0.000 | 0.030 | good | 34.86 | 34.72 | 0.60 | 101.2 | -120.0 | 0.12 | 1 | 1 | 0 | feasible | 1 | 1 |
| cache_first | ep1_light_feasible_1 | 1 | observe | 1 | 0.516 | 0.081 | 0.180 | 0.68 | 2.407 | 0.221 | 0.720 | medium | 27.96 | 12.85 | 0.21 | 108.1 | -90.3 | 0.12 | 1 | 1 | 0 | feasible | 0 | 0 |
| cache_first | ep1_image_required_2 | 1 | observe | 2 | 0.778 | 0.521 | 1.600 | 0.28 | 5.349 | 1.335 | 1.774 | good | 33.69 | 19.20 | 0.28 | 104.6 | -90.9 | 0.12 | 1 | 1 | 0 | feasible | 0 | 0 |
| cache_first | ep1_infeasible_bad_channel_3 | 1 | observe | 3 | 0.285 | 0.062 | 0.550 | 0.28 | 3.751 | 0.364 | 1.574 | bad | 37.36 | 24.27 | 0.49 | 100.9 | -92.4 | 0.12 | 1 | 1 | 0 | quality_limited | 0 | 0 |
