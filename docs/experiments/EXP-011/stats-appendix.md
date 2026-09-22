# Statistical appendix

## Design and estimands

120 questions, 120 image clusters, nine paired configurations, one inference run. Accuracy differences are candidate minus reference. Resources are actual image bytes, actual visual tokens, and measured generate latency. No independent training-seed variation is available.

No t-test, normality test, or confirmatory p-value is appropriate for the present reused-development/model-selection screen. Reported 95% percentile ranges use the existing paired image-cluster bootstrap, 2,000 resamples, seed 20260922. Selection is held fixed: the intervals omit policy/configuration-selection uncertainty. They are exploratory descriptive resampling intervals, unadjusted for multiple contrasts, and are not simultaneous or generalization guarantees.

## Paired answer gains and losses at the primary utility weight

| Comparison | Gained | Lost | Net correct | Accuracy difference pp | Descriptive 95% range pp | Utility difference | Descriptive 95% utility range |
|---|---:|---:|---:|---:|---:|---:|---:|
| best_fixed_vs_primary | 5 | 5 | 0 | 0.000 | [-5.000, 5.000] | 0.024476 | [-0.025638, 0.074583] |
| independent_fixed_vs_primary | 5 | 5 | 0 | 0.000 | [-5.000, 5.000] | 0.024476 | [-0.025638, 0.074583] |
| joint_oracle_vs_best_fixed | 7 | 0 | 7 | 5.833 | [1.667, 10.000] | 0.084162 | [0.043710, 0.125537] |
| joint_oracle_vs_rate_only_oracle | 3 | 0 | 3 | 2.500 | [0.000, 5.833] | 0.039271 | [0.015293, 0.071365] |
| joint_oracle_vs_compute_only_oracle | 2 | 0 | 2 | 1.667 | [0.000, 4.167] | 0.029313 | [0.012274, 0.054120] |

## Fixed configuration descriptives

| Byte cap | Tier | Correct / N | Accuracy | Mean image B | Mean symbols | Mean tokens | Median generate s | Latency Q1–Q3 s | Energy J |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | low | 88/120 | 73.33% | 1996.50 | 21420.00 | 43.61 | 0.236785 | 0.233520–0.244715 | unavailable |
| 2000 | medium | 88/120 | 73.33% | 1996.50 | 21420.00 | 110.17 | 0.244846 | 0.242631–0.248289 | unavailable |
| 2000 | high | 87/120 | 72.50% | 1996.50 | 21420.00 | 215.75 | 0.295588 | 0.281095–0.309123 | unavailable |
| 4000 | low | 89/120 | 74.17% | 3995.43 | 42840.00 | 43.61 | 0.238012 | 0.233179–0.242070 | unavailable |
| 4000 | medium | 93/120 | 77.50% | 3995.43 | 42840.00 | 110.17 | 0.245083 | 0.242689–0.248440 | unavailable |
| 4000 | high | 93/120 | 77.50% | 3995.43 | 42840.00 | 215.75 | 0.297832 | 0.280548–0.308334 | unavailable |
| 8000 | low | 89/120 | 74.17% | 7993.07 | 85170.00 | 43.61 | 0.236551 | 0.232752–0.242104 | unavailable |
| 8000 | medium | 92/120 | 76.67% | 7993.07 | 85170.00 | 110.17 | 0.244979 | 0.242836–0.249353 | unavailable |
| 8000 | high | 92/120 | 76.67% | 7993.07 | 85170.00 | 215.75 | 0.289372 | 0.280566–0.307791 | unavailable |

## Sensitivity across the frozen λ grid

| λ | Best fixed | Joint oracle correct | Strongest rate-only correct | Strongest compute-only correct | Joint mean utility |
|---:|---|---:|---:|---:|---:|
| 0.0 | 4000_medium | 100/120 | 97/120 | 98/120 | 0.833333 |
| 0.01 | 4000_medium | 100/120 | 97/120 | 98/120 | 0.828400 |
| 0.025 | 4000_medium | 100/120 | 97/120 | 98/120 | 0.821000 |
| 0.05 | 4000_medium | 100/120 | 97/120 | 98/120 | 0.808666 |
| 0.1 | 2000_low | 100/120 | 94/120 | 98/120 | 0.783999 |
| 0.2 | 2000_low | 100/120 | 94/120 | 94/120 | 0.734665 |
| 0.4 | 2000_low | 100/120 | 94/120 | 94/120 | 0.635997 |

Utility is correctness − λ × (actual bytes / 8000 + actual visual tokens / same-image high-tier tokens). Accuracy/cost λ tradeoffs are descriptive; no λ is selected from sealed-test outcomes.

## Missingness and provenance

Missing energy is never converted to zero. The aggregate mean energy is unavailable if any required observation is missing. No per-question answers, IDs, images, or prompts are included in this bundle.

- 2000_low: energy missing 120/120.
- 2000_medium: energy missing 120/120.
- 2000_high: energy missing 120/120.
- 4000_low: energy missing 120/120.
- 4000_medium: energy missing 120/120.
- 4000_high: energy missing 120/120.
- 8000_low: energy missing 120/120.
- 8000_medium: energy missing 120/120.
- 8000_high: energy missing 120/120.

Input SHA-256 hashes (filenames only; no private host paths):

- `summary.json`: `fba680f9f7620374f5105fa2aa0c613565c2abe8fe5697b9c19566fc6be2a4e7`
- `decision.json`: `077f34081bb9a03dd2fbc2676db54c317785a17aa20965d28509271d06c2efbe`
- `scored.json`: `a67d44012c7a0a3ed4057caaaf06fdf65d514db2c279594f2e1e8f43c5739cd1`
- `report.md`: `c893836cc1ac96de7123b91ac0144339af74a6e60451793a332678fbf4c9cb9a`
