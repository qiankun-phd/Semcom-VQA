# LSS-HSR-L Risk Guard Ablation

Overall: **risk-guard-ablation-ready-real-data-pending**
Input questions: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_real_radar_questions.csv`
Samples: `1455`

| Variant | Cost | Guard count | Floor | Critical floor | Drone miss | False confirm |
|---|---:|---:|---:|---:|---:|---:|
| `cheap_no_guard` | 5119 | 458 | 0.921 | 1.0 | 0.2007 | 0.0 |
| `scene_only_guard` | 5119 | 458 | 0.921 | 1.0 | 0.2007 | 0.0 |
| `target_scene_guard` | 5809 | 573 | 1.0 | 1.0 | 0.0 | 0.0 |
| `learnrisk_guard` | 10273 | 1317 | 1.0 | 1.0 | 0.0 | 0.8435 |

## Selection

- recommended variant: `target_scene_guard`
- selection rule: lowest visual-symbol cost among variants with zero rotary-UAV miss and full critical floor
- safe variant count: `2`
- diagnostic: LearnRisk guard is safe but not cost-selected; it adds 4464.0 symbols and 0.8435 false-confirm rate over target_scene_guard.

## Notes

- LearnRisk guard is safe but high-cost; real calibration should tune thresholds.
