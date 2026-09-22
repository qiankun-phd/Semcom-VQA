# LearnRisk-QRS Remote Sync Plan

Mode: `dry-run`
Remote target: `lab-s2:/home/qiankun/HPPO-VQA/`
File count: `44`
Total bytes: `377131`
All local files present: `True`

## Rsync

```bash
rsync -azR \
  hppo-uav/dizoo/gym_hybrid/baselines/greedy.py \
  hppo-uav/dizoo/gym_hybrid/baselines/qrs_semantic.py \
  hppo-uav/dizoo/gym_hybrid/baselines/qrs_bandit.py \
  hppo-uav/dizoo/gym_hybrid/baselines/qrs_neural_bandit.py \
  hppo-uav/dizoo/gym_hybrid/envs/gym-hybrid/gym_hybrid/environments.py \
  hppo-uav/dizoo/gym_hybrid/envs/gym_hybrid_env.py \
  hppo-uav/run/eval_greedy.py \
  hppo-uav/run/eval_qrs_semantic.py \
  hppo-uav/run/train_qrs_bandit.py \
  hppo-uav/run/train_qrs_neural_bandit.py \
  hppo-uav/run/qrs_tccn_gate.py \
  hppo-uav/run/qrs_learned_risk_sweep.py \
  hppo-uav/run/qrs_learned_risk_scaling.py \
  hppo-uav/run/qrs_moe_learnrisk_ablation.py \
  hppo-uav/run/qrs_robustness_sweep.py \
  hppo-uav/run/qrs_large_ue_sweep.py \
  hppo-uav/run/qrs_tccn_readiness_audit.py \
  hppo-uav/run/qrs_tccn_claim_linter.py \
  hppo-uav/run/qrs_tccn_innovation_linter.py \
  hppo-uav/run/qrs_tccn_table_audit.py \
  hppo-uav/run/qrs_tccn_evidence_contract.py \
  hppo-uav/run/qrs_tccn_submission_gate.py \
  hppo-uav/run/qrs_remote_job_status.py \
  hppo-uav/run/qrs_remote_preflight.py \
  hppo-uav/run/qrs_remote_sync_plan.py \
  hppo-uav/run/qrs_remote_smoke_plan.py \
  hppo-uav/run/qrs_remote_launch_plan.py \
  hppo-uav/run/qrs_remote_collect_plan.py \
  hppo-uav/run/qrs_remote_sync_manifest.py \
  hppo-uav/run/qrs_component_ablation.py \
  hppo-uav/run/qrs_question_type_ablation.py \
  hppo-uav/run/dump_qrs_rollout.py \
  hppo-uav/tests/test_qrs_evidence_contract.py \
  hppo-uav/tests/test_qrs_tccn_claim_linter.py \
  hppo-uav/tests/test_qrs_tccn_innovation_linter.py \
  hppo-uav/tests/test_qrs_tccn_table_audit.py \
  hppo-uav/tests/test_qrs_tccn_submission_gate.py \
  hppo-uav/tests/test_qrs_remote_job_status.py \
  hppo-uav/tests/test_qrs_remote_sync_plan.py \
  hppo-uav/tests/test_qrs_remote_smoke_plan.py \
  hppo-uav/tests/test_qrs_remote_launch_plan.py \
  hppo-uav/tests/test_qrs_remote_collect_plan.py \
  hppo-uav/tests/test_uavnet_env.py \
  paper/data/summarize_qrs_moe_learnrisk.py \
  paper/data/qrs_remote_sync_manifest.json \
  lab-s2:/home/qiankun/HPPO-VQA/
```

## Remote Verify

```bash
ssh lab-s2 'set -e
cd /home/qiankun/HPPO-VQA
/home/qiankun/.conda/envs/DI-engine/bin/python hppo-uav/run/qrs_remote_sync_manifest.py --verify-json paper/data/qrs_remote_sync_manifest.json'
```
