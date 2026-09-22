# TCCN Readiness Audit

Overall: **needs-remote-sync**

## Gates

| Gate | Status | Evidence strength | Remaining need |
|---|---|---|---|
| F1 Final method identity | local-smoke | manuscript-audited | Remove any remaining MA-HPPO-as-main performance claim. |
| F2 Greedy-vs-QRS cost/fidelity gate | local-smoke | local-smoke-only | Run server multi-seed, longer-episode final table. |
| F3 Stress safety / jamming-equivalent robustness | local-smoke | local-smoke-only | Run server multi-seed robustness sweep with Mixed-MoE enabled. |
| F4 Component and MoE ablations | local-smoke | local-smoke-only | Expand ablations to multi-seed and per-question metrics. |
| F5 Scaling and large-UE pressure | local-smoke | local-smoke-only | Run server larger M,N / large-UE sweep and final baselines. |

## Metric Evidence

### F2 Greedy-vs-QRS cost/fidelity gate
- Source: `paper/data/qrs_tccn_gate_local_mini.csv`
- Source strength: `local-smoke`
- Rows: `8`
- methods: `['Greedy', 'LearnRisk', 'QRS', 'TypedPrice']`
- scenarios: `['nominal', 'stress']`
- similarity_min: `0.0898`
- similarity_max: `0.8405`
- q_floor_min: `0.0250`
- q_floor_max: `0.9667`
- Missing final evidence:
  - `paper/data/qrs_tccn_gate_remote_mseed.csv`
  - `paper/tables/qrs_tccn_gate_remote_mseed.tex`

### F3 Stress safety / jamming-equivalent robustness
- Source: `paper/data/qrs_robustness_sweep_tiny.csv`
- Source strength: `local-smoke`
- Rows: `4`
- methods: `['Greedy', 'TypedPrice']`
- loss_db: `[0.0, 10.0]`
- similarity_min: `0.1957`
- similarity_max: `0.9002`
- q_floor_min: `0.0500`
- q_floor_max: `1.0000`
- Missing final evidence:
  - `paper/data/qrs_robustness_sweep_remote_mseed.csv`
  - `paper/tables/qrs_robustness_sweep_remote_mseed.tex`

### F4 Component and MoE ablations
- Source: `paper/data/qrs_moe_learnrisk_mini.csv`
- Source strength: `local-smoke`
- Rows: `6`
- variants: `['Mixed-1H', 'Mixed-MoE', 'Single']`
- scenarios: `['nominal', 'stress']`
- similarity_min: `0.0898`
- similarity_max: `0.8405`
- q_floor_min: `0.0250`
- q_floor_max: `0.9500`
- Missing final evidence:
  - `paper/data/qrs_component_ablation_remote_mseed.csv`
  - `paper/tables/qrs_component_ablation_remote_mseed.tex`
  - `paper/data/qrs_moe_learnrisk_remote_mseed.csv`
  - `paper/tables/qrs_moe_learnrisk_remote_mseed.tex`

### F5 Scaling and large-UE pressure
- Source: `paper/data/qrs_large_ue_sweep_tiny.csv`
- Source strength: `local-smoke`
- Rows: `4`
- methods: `['Greedy', 'TypedPrice']`
- N_values: `[4.0, 6.0]`
- similarity_min: `0.6697`
- similarity_max: `0.9002`
- q_floor_min: `0.8000`
- q_floor_max: `1.0000`
- Missing final evidence:
  - `paper/data/qrs_large_ue_sweep_remote_mseed.csv`
  - `paper/tables/qrs_large_ue_sweep_remote_mseed.tex`


## Local Build

- PDF exists: `True`
- LaTeX status: `pass`
- Fatal log patterns: `[]`

## Remote

- Host: `lab-s2`
- Status: `needs-sync`
- Missing files: `37`
  - `hppo-uav/dizoo/gym_hybrid/baselines/qrs_semantic.py`
  - `hppo-uav/dizoo/gym_hybrid/baselines/qrs_bandit.py`
  - `hppo-uav/dizoo/gym_hybrid/baselines/qrs_neural_bandit.py`
  - `hppo-uav/run/eval_qrs_semantic.py`
  - `hppo-uav/run/train_qrs_bandit.py`
  - `hppo-uav/run/train_qrs_neural_bandit.py`
  - `hppo-uav/run/qrs_tccn_gate.py`
  - `hppo-uav/run/qrs_learned_risk_sweep.py`
  - `hppo-uav/run/qrs_learned_risk_scaling.py`
  - `hppo-uav/run/qrs_moe_learnrisk_ablation.py`
  - `hppo-uav/run/qrs_robustness_sweep.py`
  - `hppo-uav/run/qrs_large_ue_sweep.py`
  - `hppo-uav/run/qrs_tccn_readiness_audit.py`
  - `hppo-uav/run/qrs_tccn_claim_linter.py`
  - `hppo-uav/run/qrs_tccn_table_audit.py`
  - `hppo-uav/run/qrs_tccn_evidence_contract.py`
  - `hppo-uav/run/qrs_tccn_submission_gate.py`
  - `hppo-uav/run/qrs_remote_job_status.py`
  - `hppo-uav/run/qrs_remote_preflight.py`
  - `hppo-uav/run/qrs_remote_sync_plan.py`
  - `hppo-uav/run/qrs_remote_smoke_plan.py`
  - `hppo-uav/run/qrs_remote_launch_plan.py`
  - `hppo-uav/run/qrs_remote_collect_plan.py`
  - `hppo-uav/run/qrs_remote_sync_manifest.py`
  - `hppo-uav/run/qrs_component_ablation.py`
  - `hppo-uav/run/qrs_question_type_ablation.py`
  - `hppo-uav/run/dump_qrs_rollout.py`
  - `hppo-uav/tests/test_qrs_evidence_contract.py`
  - `hppo-uav/tests/test_qrs_tccn_claim_linter.py`
  - `hppo-uav/tests/test_qrs_tccn_table_audit.py`
  - `hppo-uav/tests/test_qrs_tccn_submission_gate.py`
  - `hppo-uav/tests/test_qrs_remote_job_status.py`
  - `hppo-uav/tests/test_qrs_remote_sync_plan.py`
  - `hppo-uav/tests/test_qrs_remote_smoke_plan.py`
  - `hppo-uav/tests/test_qrs_remote_launch_plan.py`
  - `hppo-uav/tests/test_qrs_remote_collect_plan.py`
  - `paper/data/summarize_qrs_moe_learnrisk.py`
