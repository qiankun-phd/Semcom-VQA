# LSS-HSR-L Risk Guard Ablation

Overall: **risk-guard-ablation-ready-real-data-pending**
Input questions: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_demo_questions.csv`
Samples: `4`

| Variant | Cost | Guard count | Floor | Critical floor | Drone miss | False confirm |
|---|---:|---:|---:|---:|---:|---:|
| `cheap_no_guard` | 12 | 1 | 1.0 | 1.0 | 0.0 | 0.0 |
| `scene_only_guard` | 12 | 1 | 1.0 | 1.0 | 0.0 | 0.0 |
| `target_scene_guard` | 12 | 1 | 1.0 | 1.0 | 0.0 | 0.0 |
| `learnrisk_guard` | 18 | 2 | 1.0 | 1.0 | 0.0 | 0.3333 |

## Notes

- Demo guard ablation is safe; real-data ablation remains pending.
