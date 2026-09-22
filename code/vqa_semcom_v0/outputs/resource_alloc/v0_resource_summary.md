# V0 UAV-VQA Resource Allocation Summary

Batch mode is one-shot allocation. MDP mode is multi-slot rollout, so attempt-level and unique-task-level success are reported separately.

| policy | attempt success | unique task success | completion | attempts/task | semantic utility | accuracy | delay | energy | quality sat. | deadline sat. | feasible | conflict | cache | s0 | s1 | s2 full | s3 ROI | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cache_first | 0.086 | 0.375 | 0.375 | 4.375 | 0.414 | 0.580 | 3.966 | 1.750 | 0.400 | 0.286 | 0.500 | 0.000 | 0.086 | 0.086 | 0.029 | 0.371 | 0.514 | -3.227 |
| deadline_aware_greedy | 0.086 | 0.375 | 0.375 | 4.375 | 0.414 | 0.579 | 3.941 | 1.714 | 0.400 | 0.286 | 0.500 | 0.000 | 0.086 | 0.086 | 0.029 | 0.357 | 0.529 | -3.214 |
| goal_oriented_priority | 0.185 | 0.625 | 0.625 | 3.375 | 0.402 | 0.441 | 1.692 | 0.362 | 0.370 | 0.648 | 0.444 | 0.000 | 0.685 | 0.685 | 0.037 | 0.185 | 0.093 | -1.230 |
| joint_greedy_resource | 0.133 | 0.500 | 0.500 | 3.750 | 0.863 | 0.516 | 2.355 | 0.480 | 0.583 | 0.367 | 0.567 | 0.000 | 0.483 | 0.483 | 0.033 | 0.200 | 0.283 | -1.908 |
| min_sufficient_evidence | 0.086 | 0.375 | 0.375 | 4.375 | 0.414 | 0.580 | 3.966 | 1.750 | 0.400 | 0.286 | 0.500 | 0.000 | 0.086 | 0.086 | 0.029 | 0.371 | 0.514 | -3.227 |

## Feasibility Breakdown

| policy | feasible | deadline-limited | quality-limited | quality violation | deadline violation | resource violation |
|---|---:|---:|---:|---:|---:|---:|
| cache_first | 0.500 | 0.043 | 0.457 | 0.600 | 0.714 | 0.000 |
| deadline_aware_greedy | 0.500 | 0.043 | 0.457 | 0.600 | 0.714 | 0.000 |
| goal_oriented_priority | 0.444 | 0.093 | 0.463 | 0.630 | 0.352 | 0.000 |
| joint_greedy_resource | 0.567 | 0.033 | 0.400 | 0.417 | 0.633 | 0.000 |
| min_sufficient_evidence | 0.500 | 0.043 | 0.457 | 0.600 | 0.714 | 0.000 |

## Communication-Compute Breakdown

| policy | rate Mbps | SNR dB | SINR dB | dist m | elev deg | LoS prob. | path loss | interference | fading | travel delay | upload delay | inference delay | model load delay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cache_first | 11.773 | 33.823 | 17.483 | 237.562 | 20.971 | 0.392 | 102.066 | -89.006 | 0.422 | 0.563 | 0.746 | 1.110 | 0.390 |
| deadline_aware_greedy | 11.834 | 33.827 | 17.552 | 237.562 | 20.971 | 0.392 | 102.066 | -89.062 | 0.422 | 0.563 | 0.733 | 1.100 | 0.390 |
| goal_oriented_priority | 18.190 | 37.536 | 26.788 | 254.698 | 18.954 | 0.328 | 103.226 | -115.854 | 0.607 | 0.228 | 0.131 | 0.264 | 0.381 |
| joint_greedy_resource | 16.072 | 27.347 | 23.813 | 240.948 | 20.491 | 0.383 | 102.244 | -108.654 | 0.483 | 0.361 | 0.247 | 0.634 | 0.338 |
| min_sufficient_evidence | 11.773 | 33.823 | 17.483 | 237.562 | 20.971 | 0.392 | 102.066 | -89.006 | 0.422 | 0.563 | 0.746 | 1.110 | 0.390 |

## Semantic Cache, MEC, and UAV Breakdown

| policy | cache hit prob. | semantic cache hit | acc. gain | payload MB | semantic eff. | LUT missing | utility/latency | GPU mem ok | battery ok | battery remaining | UAV util. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cache_first | 0.858 | 0.086 | 0.120 | 0.883 | 0.122 | 0.000 | 0.132 | 1.000 | 1.000 | 99.474 | 0.513 |
| deadline_aware_greedy | 0.858 | 0.086 | 0.119 | 0.868 | 0.122 | 0.000 | 0.132 | 1.000 | 1.000 | 99.481 | 0.513 |
| goal_oriented_priority | 0.530 | 0.222 | 0.081 | 0.361 | 0.097 | 0.000 | 0.183 | 0.815 | 1.000 | 99.861 | 0.526 |
| joint_greedy_resource | 0.615 | 0.083 | 0.125 | 0.487 | 0.172 | 0.000 | 0.288 | 0.850 | 1.000 | 99.786 | 0.578 |
| min_sufficient_evidence | 0.858 | 0.086 | 0.120 | 0.883 | 0.122 | 0.000 | 0.132 | 1.000 | 1.000 | 99.474 | 0.513 |
