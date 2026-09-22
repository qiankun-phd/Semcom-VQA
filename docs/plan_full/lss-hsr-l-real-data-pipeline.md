# LSS-HSR-L Real-Data Pipeline

Dataset path: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/数据集及使用说明`
Overall: **real-data-pipeline-complete**
Audit status: `patrol-warning-ready-for-label-design`
Real-data gate: `real-data-ready-for-derived-labels`

## Outputs

- audit_json: `paper/data/lss_hsr_l_real_dataset_audit.json`
- audit_md: `plan/lss-hsr-l-real-dataset-audit.md`
- gate_json: `paper/data/lss_hsr_l_real_real_data_gate.json`
- gate_md: `plan/lss-hsr-l-real-real-data-gate.md`
- manifest_csv: `paper/data/lss_hsr_l_real_sample_manifest.csv`
- manifest_json: `paper/data/lss_hsr_l_real_sample_manifest_summary.json`
- manifest_md: `plan/lss-hsr-l-real-sample-manifest.md`
- mat_features_csv: `paper/data/lss_hsr_l_real_mat_statistics.csv`
- mat_calibrated_manifest_csv: `paper/data/lss_hsr_l_real_mat_calibrated_manifest.csv`
- mat_calibration_json: `paper/data/lss_hsr_l_real_mat_stat_calibration.json`
- mat_calibration_md: `plan/lss-hsr-l-real-mat-stat-calibration.md`
- windowed_features_csv: `paper/data/lss_hsr_l_real_windowed_statistics.csv`
- windowed_features_json: `paper/data/lss_hsr_l_real_windowed_features.json`
- windowed_features_md: `plan/lss-hsr-l-real-windowed-features.md`
- calibrated_manifest_csv: `paper/data/lss_hsr_l_real_calibrated_manifest.csv`
- calibration_json: `paper/data/lss_hsr_l_real_semantic_calibration.json`
- calibration_md: `plan/lss-hsr-l-real-semantic-calibration.md`
- questions_csv: `paper/data/lss_hsr_l_real_radar_questions.csv`
- questions_json: `paper/data/lss_hsr_l_real_radar_questions_summary.json`
- questions_md: `plan/lss-hsr-l-real-radar-question-labels.md`
- features_csv: `paper/data/lss_hsr_l_real_semantic_control_features.csv`
- features_json: `paper/data/lss_hsr_l_real_semantic_control_features.json`
- features_md: `plan/lss-hsr-l-real-semantic-control-features.md`
- sim_csv: `paper/data/lss_hsr_l_real_comm_policy_sim.csv`
- sim_json: `paper/data/lss_hsr_l_real_comm_policy_sim.json`
- sim_md: `plan/lss-hsr-l-real-comm-policy-sim.md`
- visual_escalation_csv: `paper/data/lss_hsr_l_real_visual_escalation_sim.csv`
- visual_escalation_json: `paper/data/lss_hsr_l_real_visual_escalation_sim.json`
- visual_escalation_md: `plan/lss-hsr-l-real-visual-escalation-sim.md`

## Stage Summaries

- manifest rows: `1455`
- unknown target rows: `0`
- calibration rows: `1455`
- all real calibrated: `None`
- windowed feature rows: `1455`
- window count: `6`
- question rows: `5820`
- visual confirm yes: `970`
- feature rows: `5820`
- risk-gated rows: `5268`

## Communication Policy Probe

| Policy | Cost | Avg cost | Floor rate | Drone miss | False confirm |
|---|---:|---:|---:|---:|---:|
| `no_radar_uniform` | 8730 | 6.0 | 1.0 | 0.0 | 1.0 |
| `radar_triggered_greedy` | 7221 | 4.9629 | 1.0 | 0.0 | 0.4501 |
| `typed_price_qrs` | 8649 | 5.9443 | 1.0 | 0.0 | 0.4501 |
| `learnrisk_qrs` | 10273 | 7.0605 | 1.0 | 0.0 | 0.8435 |

## UAV/VQA Escalation Probe

- tier counts: `{'radar_record_only': 501, 'visual_evidence_capture': 210, 'close_vqa_disambiguation': 744}`
- UAV dispatch rate: `0.6557`
- expected VQA error rate: `0.1929`
- expected rotary-UAV miss rate: `0.0191`

## Notes

- This pipeline derives MAT statistics, adds coarse temporal-window features, and then trains an in-repo softmax radar calibration model for confidence, margin, and trajectory-risk fields.
- The UAV/VQA stage is a simulated noisy visual escalation sensor because LSS-HSR-L has no paired radar-image data.
- Policy simulation is an interface probe; final learned-controller claims still require remote multi-seed communication/control experiments.
