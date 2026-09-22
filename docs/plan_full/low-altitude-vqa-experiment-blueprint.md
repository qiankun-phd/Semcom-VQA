# Low-Altitude VQA Experiment Blueprint

Overall: **experiment-blueprint-ready-real-data-pending**

## Experiment Status

| Experiment | Status | Claim | Metric | Pass condition |
|---|---|---|---|---|
| `E1_scene_coverage` | `pass` | LSS-HSR-L supports all low-altitude patrol-warning semantic scenes. | scene_count, target_group coverage, question_type coverage | 5 scene templates and all 4 target groups are covered; real files later confirm class/scene metadata. |
| `E2_radar_triggered_policy_probe` | `pass` | Radar-triggered policies reduce visual-symbol cost while keeping rotary-UAV miss rate at zero. | visual_symbol_cost, drone_miss_rate, warning_floor_rate, false_visual_confirm_rate | at least one radar-triggered policy has zero rotary-UAV miss and lower cost than no_radar_uniform; false confirmations are reported. |
| `E3_calibration_ablation` | `demo-pass-real-data-pending` | Real radar confidence/margin/trajectory calibration changes VQA confirmation decisions beyond priors. | prior-derived rows, real-calibrated rows, confirm decision flips, typed_price_seed drift | real-calibrated manifest exists; prior-vs-real decision differences are quantified. |
| `E4_risk_guard_ablation` | `demo-pass-real-data-pending` | Risk guard prevents cheap actions on airport, ambiguous drone-bird, and high-trajectory-risk scenes. | critical_floor_rate, drone_miss_rate, false_visual_confirm_rate | risk-guarded policy has zero rotary-UAV miss and non-worse critical_floor_rate than unguarded typed price. |
| `E5_remote_multiseed_stress` | `evidence-pending` | The role-split semantic architecture is robust under channel stress and larger UE/query pressure. | question_floor_rate, cost, fallback rate, price saturation, n_seeds | remote multi-seed F2-F5 artifacts pass the existing TCCN evidence contract. |

## Demo Probe Summary

- scene status: `pass`
- scene count: `5`
- feature rows: `16`
- policy status: `pass`
- recommended policy: `radar_triggered_greedy`
- recommended policy cost: `7221`
- LearnRisk cost: `10273`
- Uniform cost: `8730`
- LearnRisk drone miss rate: `0.0`
- risk guard status: `pass`
- recommended risk guard: `target_scene_guard`
- recommended risk guard cost: `5809`
- recommended risk guard false confirm rate: `0.0`
- LearnRisk guard critical floor: `1.0`
- calibration ablation status: `pass`
- calibration confirm flips: `397`

## Command Plan

- `python3 hppo-uav/run/lss_hsr_l_real_data_pipeline.py --dataset-path /path/to/数据集及使用说明.zip`
- `python3 hppo-uav/run/vqa_semantic_scene_builder.py --feature-csv paper/data/lss_hsr_l_real_semantic_control_features.csv`
- `python3 hppo-uav/run/lss_hsr_l_comm_policy_sim.py --input-csv paper/data/lss_hsr_l_real_radar_questions.csv`
- `python3 hppo-uav/run/qrs_tccn_submission_gate.py`

## Claim Boundary

Demo policy probes support interface shape only. Real claims require the downloaded LSS-HSR-L archive, license-safe derived labels, calibrated radar confidence/risk, and remote multi-seed stress evidence.
