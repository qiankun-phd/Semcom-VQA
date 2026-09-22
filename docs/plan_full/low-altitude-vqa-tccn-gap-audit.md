# Low-Altitude VQA TCCN Gap Audit

Overall: **tccn-narrative-innovation-ready-evidence-pending**

## Verdict

The narrative innovation is now locally coherent: radar-triggered semantic scenes, question-conditioned VQA, role-split UAV/UE control, and typed semantic-price/risk guards. The local extracted LSS-HSR-L directory now passes the dataset audit for derived labels under non-commercial ScienceDB CC BY-NC 4.0 metadata, and a trained in-repo radar calibrator now replaces class-only priors in the semantic bridge with temporal/windowed features. It is not TCCN-evidence-ready until remote multi-seed communication/control evidence and stronger sequence radar modeling pass.

## Dimension Audit

| Dimension | Status | Evidence | Gap |
|---|---|---|---|
| `D1_scenario_originality` | `local-ready` | `paper/data/vqa_semantic_scene_library.json` | Scene priors now have trained radar calibration; next step is learned temporal scene modeling. |
| `D2_dataset_fit_and_license` | `real-data-ready` | `paper/data/lss_hsr_l_scidb_metadata_gate.json` | Local extracted data and ScienceDB CC BY-NC 4.0 metadata support non-commercial derived-label research; commercial/deployment use still needs explicit legal review. |
| `D3_radar_to_vqa_task_bridge` | `local-ready` | `paper/data/lss_hsr_l_real_semantic_calibration.json` | Windowed softmax calibration is now present; next step is stronger sequence radar modeling and system stress evidence. |
| `D4_uav_ue_algorithm_architecture` | `local-ready` | `paper/data/vqa_algorithm_claim_scope.json` | Claim is scoped to architecture and executable fast layer; final constrained slow UAV learner remains future work. |
| `D5_system_innovation_map` | `local-ready` | `paper/data/vqa_innovation_scout.json` | Convert calibrated scene priors into trained ablations and remote stress evidence. |
| `D6_tccn_experimental_evidence` | `evidence-pending` | `paper/data/low_altitude_vqa_experiment_blueprint.json` | Trained calibration and risk-guard ablations now run on 1455 samples; E5 remote multi-seed evidence is still pending. |
| `D7_claim_boundary` | `local-ready` | `paper/data/lss_hsr_l_remote_plan.json` | Keep commercial/deployment and paired camera-VQA accuracy claims blocked; current evidence supports radar-triggered semantic confirmation only. |

## Priority Fix Queue

| Priority | Item | Command | Blocked by |
|---|---|---|---|
| P1 | Replace the current windowed-softmax calibrator with a true sequence/window model if reviewer-grade radar recognition is claimed. | `python3 hppo-uav/run/lss_hsr_l_windowed_features.py --input-csv paper/data/lss_hsr_l_real_sample_manifest.csv --feature-csv paper/data/lss_hsr_l_real_mat_statistics.csv --dataset-path 数据集及使用说明` | `windowed-softmax-ready` |
| P2 | Run real-data calibration and risk-guard ablations instead of demo-only ablations. | `python3 hppo-uav/run/lss_hsr_l_calibration_ablation.py --input-csv paper/data/lss_hsr_l_real_calibrated_manifest.csv` | `real_pipeline=real-data-pipeline-complete` |
| P3 | Keep algorithm claim scoped: architecture + executable fast layer, not final constrained UAV slow learning. | `python3 hppo-uav/run/vqa_algorithm_claim_scope.py --strict` | `claim-scope-ready` |
| P4 | Execute the low-altitude VQA experiment blueprint and remote multi-seed evidence contract. | `python3 hppo-uav/run/low_altitude_vqa_experiment_blueprint.py` | `experiment-blueprint-ready-real-data-pending` |
| P5 | Keep claim boundaries explicit: radar data triggers VQA confirmation, not paired camera-VQA accuracy. | `python3 hppo-uav/run/low_altitude_vqa_tccn_gap_audit.py` | `none` |

## Source Status

- readiness: `ready-for-real-data-derived-experiments`
- semantic_scene: `semantic-scene-library-ready-with-feature-probe`
- innovation_scout: `innovation-map-ready`
- algorithm_architecture: `algorithm-architecture-ready`
- scidb_metadata: `scidb-metadata-ready-archive-pending`
- data_locator: `found-local-extracted`
- real_data_gate: `real-data-ready-for-derived-labels`
- real_data_pipeline: `real-data-pipeline-complete`
- remote_plan_local_source_present: `False`
- windowed_features: `windowed-features-ready`
- semantic_calibration_rows: `1455`
- algorithm_claim_scope: `claim-scope-ready`
- experiment_blueprint: `experiment-blueprint-ready-real-data-pending`
- risk_guard_ablation: `risk-guard-ablation-ready-real-data-pending`
- calibration_ablation: `calibration-ablation-ready-real-data-pending`
- qrs_submission: `local-smoke-only`
