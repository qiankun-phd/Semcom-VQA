# TCCN Top-5 Completion Dashboard

Date: 2026-05-30

This dashboard tracks the five highest-priority fixes for moving the
paper from a local prototype toward a TCCN-ready LearnRisk-QRS
submission.

## Current Verdict

Status: **architecture ready, submission evidence not ready**.

The manuscript now has the right center of gravity: LearnRisk-QRS is the
target method, while MA-HPPO/CHPPO are diagnostic baselines. The missing
piece is not another local smoke table; it is remote multi-seed evidence
for F2-F5 plus final claim-wording validation.

Latest read-only server audit on `lab-s2`:

- Remote root `/home/qiankun/HPPO-VQA` exists.
- Remote readiness is `needs-sync`: only 5 tracked files are present and
  37 QRS validation files are missing.
- Remote final job status is `missing`: all 5 multi-seed logs and all
  10 CSV/TEX outputs are absent.
- Therefore the next executable step is guarded sync, not launch or
  collection.

## Top-5 Gates

| Gate | What must be true | Current evidence | Next action |
|---|---|---|---|
| F1 method identity | Whole paper names LearnRisk-QRS as the journal method and keeps MA-HPPO/CHPPO diagnostic | Local manuscript and claim linter pass | Rerun after remote tables replace smoke tables |
| F2 cost/fidelity | Greedy, QRS, TypedPrice, and LearnRisk have nominal/stress multi-seed CSV/TEX | Local mini only | Run `qrs_tccn_gate_remote_mseed` on server |
| F3 stress robustness | 0/5/10/15 dB sweep reports cost, mean similarity, floor rate, and tail metric | Tiny local sweep only | Run `qrs_robustness_sweep_remote_mseed` with Mixed-MoE |
| F4 ablations | no-question, no-UAV-role, no-price, Single, Mixed-1H, and Mixed-MoE are compared multi-seed | Local smoke only | Run component and MoE remote ablations |
| F5 large-UE pressure | N=4/6/8/10 fixed-M sweep includes Greedy, TypedPrice, and Mixed-MoE | Tiny N=4/6 local only | Run `qrs_large_ue_sweep_remote_mseed` |

## Claim Boundaries

Allowed now:

- The paper proposes a VQA-native role-split semantic-control
  architecture.
- Typed semantic price, VQA-risk gating, and nominal/stress experts are
  implemented as executable components.
- Local smoke results are only preflight evidence.

Not allowed until remote evidence passes:

- LearnRisk-QRS dominates Greedy.
- MA-HPPO is the final TCCN method.
- Local smoke proves robustness, scalability, or stress safety.
- The learned neural fast layer is better because it is neural.

## Validation Commands

```bash
python3 hppo-uav/run/qrs_tccn_claim_linter.py
python3 hppo-uav/run/qrs_tccn_table_audit.py
python3 hppo-uav/run/qrs_tccn_evidence_contract.py
python3 hppo-uav/run/qrs_tccn_submission_gate.py
python3 hppo-uav/run/qrs_remote_preflight.py
python3 hppo-uav/run/qrs_remote_smoke_plan.py
python3 hppo-uav/run/qrs_remote_collect_plan.py
```

After explicit upload authorization only:

```bash
python3 hppo-uav/run/qrs_remote_sync_plan.py --execute --confirm SYNC_REMOTE_QRS_FILES
python3 hppo-uav/run/qrs_remote_smoke_plan.py --execute --confirm RUN_REMOTE_QRS_SMOKE
python3 hppo-uav/run/qrs_remote_launch_plan.py --execute --confirm START_REMOTE_QRS_JOBS
python3 hppo-uav/run/qrs_remote_collect_plan.py --execute --confirm COLLECT_REMOTE_QRS_RESULTS
```

## Evidence Authorities

- Claim wording: `hppo-uav/run/qrs_tccn_claim_linter.py`
- Table replacement: `hppo-uav/run/qrs_tccn_table_audit.py`
- Evidence threshold: `hppo-uav/run/qrs_tccn_evidence_contract.py`
- Submission gate: `hppo-uav/run/qrs_tccn_submission_gate.py`
- Remote readiness: `hppo-uav/run/qrs_remote_preflight.py`
- Remote smoke plan: `hppo-uav/run/qrs_remote_smoke_plan.py`
- Remote result collection: `hppo-uav/run/qrs_remote_collect_plan.py`
- Remote runbook: `plan/qrs-remote-experiment-runbook.md`
- Final table insertion: `plan/tccn-final-results-insertion-plan.md`
- Reviewer defense: `plan/tccn-reviewer-novelty-defense.md`
