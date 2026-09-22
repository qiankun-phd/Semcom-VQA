# V0.5 LUT Distribution Report

This report treats the LUT as a task-conditioned semantic accuracy model, not as an image-quality score.

- task rows: `20`
- LUT rows: `540`
- accuracy range: `0.182` to `0.954`
- accuracy mean: `0.495`
- nonzero CI cells: `432/540`

## Task Distribution: question_type

| value | count | ratio |
|---|---:|---:|
| attribute | 5 | 0.250 |
| counting | 5 | 0.250 |
| presence | 5 | 0.250 |
| relation | 1 | 0.050 |
| risk | 4 | 0.200 |

## Task Distribution: view_quality_bin

| value | count | ratio |
|---|---:|---:|
| good | 12 | 0.600 |
| medium | 4 | 0.200 |
| poor | 4 | 0.200 |

## Task Distribution: risk_level

| value | count | ratio |
|---|---:|---:|
| critical | 4 | 0.200 |
| normal | 16 | 0.800 |

## Task Distribution: target_class

| value | count | ratio |
|---|---:|---:|
| bus | 3 | 0.150 |
| car | 3 | 0.150 |
| motor | 3 | 0.150 |
| pedestrian | 3 | 0.150 |
| person_vehicle | 1 | 0.050 |
| scene | 4 | 0.200 |
| van | 3 | 0.150 |

## LUT Coverage: service_level

| value | count | ratio |
|---|---:|---:|
| 0 | 135 | 0.250 |
| 1 | 135 | 0.250 |
| 2 | 135 | 0.250 |
| 3 | 135 | 0.250 |

## LUT Coverage: channel_bin

| value | count | ratio |
|---|---:|---:|
| bad | 180 | 0.333 |
| good | 180 | 0.333 |
| medium | 180 | 0.333 |

## LUT Coverage: freshness_bin

| value | count | ratio |
|---|---:|---:|
| expired | 180 | 0.333 |
| fresh | 180 | 0.333 |
| stale | 180 | 0.333 |

## Simulation Results

| policy | success | accuracy | delay | energy | quality violation | deadline violation |
|---|---:|---:|---:|---:|---:|---:|
| always_cache | 0.075 | 0.454 | 0.678 | 0.200 | 0.925 | 0.000 |
| always_light | 0.225 | 0.550 | 2.017 | 1.000 | 0.775 | 0.000 |
| always_image | 0.400 | 0.663 | 3.984 | 2.500 | 0.600 | 0.350 |
| greedy_min_sufficient_evidence | 0.463 | 0.629 | 3.326 | 1.933 | 0.537 | 0.350 |
