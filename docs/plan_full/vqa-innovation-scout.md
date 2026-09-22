# VQA Innovation Scout

Overall: **innovation-map-ready**
Files scanned: `145`

## Innovation Themes

| Theme | Status | Evidence files | Gap |
|---|---|---:|---|
| `question_conditioned_semantic_state` | `evidence-found` | 15 | Needs real-data per-question metrics after LSS-HSR-L audit and feature calibration. |
| `role_split_uav_ue_control` | `evidence-found` | 19 | Final learned slow UAV controller is still future work; current executable slice uses role-split QRS. |
| `typed_semantic_price_bridge` | `evidence-found` | 20 | Real LSS-HSR-L calibration is needed for target-specific price weights. |
| `learned_fast_layer_and_risk` | `evidence-found` | 20 | No real LSS-HSR-L learned-fast-layer dominance claim until multi-seed evidence exists. |
| `radar_triggered_patrol_warning` | `evidence-found` | 20 | Requires downloaded archive, README/license audit, and real classifier/trajectory confidence. |
| `semantic_scene_reconstruction` | `evidence-found` | 11 | Scene priors still need real LSS-HSR-L trajectory/confidence calibration. |
| `real_data_guardrails` | `evidence-found` | 20 | Actual ScienceDB zip is not yet present locally or on the verification server. |

## Theme Details

### question_conditioned_semantic_state

Claim: VQA resource allocation is question-conditioned rather than average-similarity-only.

Gap: Needs real-data per-question metrics after LSS-HSR-L audit and feature calibration.

Evidence files:
- `hppo-uav/dizoo/gym_hybrid/baselines/qrs_bandit.py`
- `hppo-uav/dizoo/gym_hybrid/baselines/qrs_neural_bandit.py`
- `hppo-uav/dizoo/gym_hybrid/baselines/qrs_semantic.py`
- `hppo-uav/dizoo/gym_hybrid/envs/gym-hybrid/gym_hybrid/environments.py`
- `hppo-uav/dizoo/gym_hybrid/envs/gym_hybrid_env.py`
- `hppo-uav/run/dump_qrs_rollout.py`
- `hppo-uav/run/eval_qrs_semantic.py`
- `hppo-uav/run/qrs_component_ablation.py`
- `hppo-uav/run/qrs_question_type_ablation.py`
- `hppo-uav/run/train_qrs_bandit.py`

### role_split_uav_ue_control

Claim: UAV and UE/query decisions are decomposed by timescale and information scope.

Gap: Final learned slow UAV controller is still future work; current executable slice uses role-split QRS.

Evidence files:
- `hppo-uav/dizoo/gym_hybrid/baselines/qrs_bandit.py`
- `hppo-uav/dizoo/gym_hybrid/baselines/qrs_neural_bandit.py`
- `hppo-uav/dizoo/gym_hybrid/baselines/qrs_semantic.py`
- `hppo-uav/run/dump_qrs_rollout.py`
- `hppo-uav/run/eval_qrs_semantic.py`
- `hppo-uav/run/lss_hsr_l_narrative_gate.py`
- `hppo-uav/run/qrs_component_ablation.py`
- `hppo-uav/run/qrs_tccn_innovation_linter.py`
- `hppo-uav/run/train_qrs_neural_bandit.py`
- `hppo-uav/run/vqa_algorithm_architecture_gate.py`

### typed_semantic_price_bridge

Claim: The cross-layer message is target/question-aware semantic pressure, not raw channel state.

Gap: Real LSS-HSR-L calibration is needed for target-specific price weights.

Evidence files:
- `hppo-uav/dizoo/gym_hybrid/baselines/qrs_bandit.py`
- `hppo-uav/dizoo/gym_hybrid/baselines/qrs_neural_bandit.py`
- `hppo-uav/dizoo/gym_hybrid/baselines/qrs_semantic.py`
- `hppo-uav/run/dump_qrs_rollout.py`
- `hppo-uav/run/eval_qrs_semantic.py`
- `hppo-uav/run/low_altitude_vqa_experiment_blueprint.py`
- `hppo-uav/run/lss_hsr_l_calibration_ablation.py`
- `hppo-uav/run/lss_hsr_l_demo_pipeline.py`
- `hppo-uav/run/lss_hsr_l_feature_adapter.py`
- `hppo-uav/run/lss_hsr_l_narrative_gate.py`

### learned_fast_layer_and_risk

Claim: The fast UE/query layer can be learned while retaining deterministic QRS/risk fallback.

Gap: No real LSS-HSR-L learned-fast-layer dominance claim until multi-seed evidence exists.

Evidence files:
- `hppo-uav/dizoo/gym_hybrid/baselines/qrs_neural_bandit.py`
- `hppo-uav/run/REMOTE_QRS_GATE.md`
- `hppo-uav/run/qrs_large_ue_sweep.py`
- `hppo-uav/run/qrs_learned_risk_scaling.py`
- `hppo-uav/run/qrs_learned_risk_sweep.py`
- `hppo-uav/run/qrs_moe_learnrisk_ablation.py`
- `hppo-uav/run/qrs_remote_launch_plan.py`
- `hppo-uav/run/qrs_remote_sync_manifest.py`
- `hppo-uav/run/qrs_robustness_sweep.py`
- `hppo-uav/run/qrs_tccn_gate.py`

### radar_triggered_patrol_warning

Claim: Radar event semantics decide which visual confirmation question is worth asking.

Gap: Requires downloaded archive, README/license audit, and real classifier/trajectory confidence.

Evidence files:
- `hppo-uav/run/low_altitude_vqa_experiment_blueprint.py`
- `hppo-uav/run/low_altitude_vqa_readiness_gate.py`
- `hppo-uav/run/low_altitude_vqa_tccn_gap_audit.py`
- `hppo-uav/run/lss_hsr_l_calibration_ablation.py`
- `hppo-uav/run/lss_hsr_l_comm_policy_sim.py`
- `hppo-uav/run/lss_hsr_l_data_locator.py`
- `hppo-uav/run/lss_hsr_l_dataset_audit.py`
- `hppo-uav/run/lss_hsr_l_demo_pipeline.py`
- `hppo-uav/run/lss_hsr_l_feature_adapter.py`
- `hppo-uav/run/lss_hsr_l_manifest_builder.py`

### semantic_scene_reconstruction

Claim: Low-altitude VQA is reconstructed as explicit semantic scenes that bind target, place, question, and algorithm role.

Gap: Scene priors still need real LSS-HSR-L trajectory/confidence calibration.

Evidence files:
- `hppo-uav/run/low_altitude_vqa_experiment_blueprint.py`
- `hppo-uav/run/low_altitude_vqa_readiness_gate.py`
- `hppo-uav/run/low_altitude_vqa_tccn_gap_audit.py`
- `hppo-uav/run/vqa_innovation_scout.py`
- `hppo-uav/run/vqa_semantic_scene_builder.py`
- `plan/low-altitude-vqa-readiness-gate.md`
- `plan/low-altitude-vqa-tccn-gap-audit.md`
- `plan/low-altitude-vqa-top5-narrative-dashboard.md`
- `plan/vqa-innovation-scout.md`
- `plan/vqa-semantic-innovation-matrix.md`

### real_data_guardrails

Claim: The workflow blocks real-data claims until archive structure and usage terms are checked.

Gap: Actual ScienceDB zip is not yet present locally or on the verification server.

Evidence files:
- `hppo-uav/run/low_altitude_vqa_experiment_blueprint.py`
- `hppo-uav/run/low_altitude_vqa_readiness_gate.py`
- `hppo-uav/run/low_altitude_vqa_tccn_gap_audit.py`
- `hppo-uav/run/lss_hsr_l_data_locator.py`
- `hppo-uav/run/lss_hsr_l_dataset_audit.py`
- `hppo-uav/run/lss_hsr_l_manifest_builder.py`
- `hppo-uav/run/lss_hsr_l_narrative_gate.py`
- `hppo-uav/run/lss_hsr_l_real_data_gate.py`
- `hppo-uav/run/lss_hsr_l_real_data_pipeline.py`
- `hppo-uav/run/lss_hsr_l_remote_plan.py`

## Research Questions

- How much visual semantic bandwidth can radar-triggered confirmation save under zero rotary-UAV miss constraints?
- Can target-aware typed price lambda[m,q,c] reduce false visual confirmations on birds/cars without missing drones?
- Does a fast UE/query learner reduce fallback rate while preserving warning-floor safety?
- When does the slow UAV controller need constrained learning beyond deterministic role-split QRS?
- Which LSS-HSR-L trajectory features best predict visual-confirmation priority?
