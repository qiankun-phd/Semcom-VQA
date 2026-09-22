# LSS-HSR-L Real-Data Gate

Overall: **real-data-ready-for-derived-labels**
Audit JSON: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_real_dataset_audit.json`

## Checks

| Check | Status | Detail |
|---|---|---|
| `doi` | `pass` | 10.57760/sciencedb.radars.00063 |
| `cstr` | `pass` | 31253.11.sciencedb.radars.00063 |
| `audit_status` | `pass` | patrol-warning-ready-for-label-design |
| `readme_or_usage` | `pass` | pass |
| `license_or_usage_terms` | `pass` | pass |
| `target_group_coverage` | `pass` | pass |
| `doppler_waterfall_or_image` | `pass` | pass |
| `trajectory_data` | `pass` | pass |
| `radar_to_question_feasibility` | `pass` | pass |
| `nine_class_coverage` | `pass` | pass; manual label-file inspection is required if this is review |

## Next Commands If Ready

- `python3 hppo-uav/run/lss_hsr_l_manifest_builder.py --dataset-path /path/to/数据集及使用说明.zip`
- `python3 hppo-uav/run/lss_hsr_l_question_builder.py --input-csv paper/data/lss_hsr_l_sample_manifest.csv`
- `python3 hppo-uav/run/lss_hsr_l_feature_adapter.py --input-csv paper/data/lss_hsr_l_radar_questions.csv`
- `python3 hppo-uav/run/lss_hsr_l_comm_policy_sim.py --input-csv paper/data/lss_hsr_l_radar_questions.csv`

## Notes

- The archive audit supports derived radar-question labels; proceed to manifest and question generation.
- Commercial or engineering deployment still requires reading the exact license terms.
