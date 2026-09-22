# TCCN Final Results Insertion Plan

Date: 2026-05-30

Purpose: make the five highest-priority fixes actionable after remote
validation. This file maps each missing TCCN evidence gate to the exact
artifact, paper location, acceptable claim wording, and fallback wording
if the remote result is weaker than expected.

For reviewer-facing novelty defense, use
`plan/tccn-reviewer-novelty-defense.md` alongside this insertion plan.

## Current Status

The manuscript is now framed around LearnRisk-QRS, but F2-F5 remain
local-smoke only. The paper may describe the architecture and diagnostic
motivation, but it must not present local smoke values as final TCCN
evidence. The executable authority is
`hppo-uav/run/qrs_tccn_evidence_contract.py`; the operational authority
is `plan/qrs-remote-experiment-runbook.md`. The claim-wording guard is
`hppo-uav/run/qrs_tccn_claim_linter.py`; it must pass before any final
table/caption replacement is treated as submission-ready.

The latest read-only server audit found `/home/qiankun/HPPO-VQA`, but
the server still needs sync: 37 QRS validation files are missing and no
remote multi-seed CSV/TEX/log artifacts are present. Do not launch F2-F5
jobs until sync and remote smoke verification pass.

## Gate-to-Paper Map

| Gate | Remote artifact | Paper insertion point | Claim if positive | Wording if weak |
|---|---|---|---|---|
| F1 method identity | whole-paper audit after table insertion | Abstract, Introduction, Methodology, Conclusion | LearnRisk-QRS is the target method; MA-HPPO/CHPPO are diagnostic baselines | State that the paper contributes the VQA semantic-control formulation and validated components, not a final learned UAV controller |
| F2 Greedy-vs-QRS | `paper/data/qrs_tccn_gate_remote_mseed.csv`, `paper/tables/qrs_tccn_gate_remote_mseed.tex` | Section VI, replace the current smoke discussion in `sec:exp:future` | TypedPrice/LearnRisk improves cost or floor safety versus Greedy/QRS under nominal and stress settings | Frame as cost/fidelity trade-off or negative control; do not claim dominance |
| F3 stress robustness | `paper/data/qrs_robustness_sweep_remote_mseed.csv`, `paper/tables/qrs_robustness_sweep_remote_mseed.tex` | New robustness paragraph/table after F2 | Mixed-MoE gives smoother degradation or better floor-rate under 10/15 dB loss | State the unrecoverable stress boundary and use it to justify the risk gate |
| F4 component/MoE ablation | `paper/data/qrs_component_ablation_remote_mseed.csv`, `paper/tables/qrs_component_ablation_remote_mseed.tex`, `paper/data/qrs_moe_learnrisk_remote_mseed.csv`, `paper/tables/qrs_moe_learnrisk_remote_mseed.tex` | Replace smoke component/MoE tables in `sec:exp:future` | Question conditioning, UAV role split, typed price, and MoE each have measurable necessity or a clear negative-control role | Keep only supported components in main claim; move weak modules to limitations |
| F5 large-UE pressure | `paper/data/qrs_large_ue_sweep_remote_mseed.csv`, `paper/tables/qrs_large_ue_sweep_remote_mseed.tex` | Section VI scaling subsection | Role-split/typed-price architecture remains competitive as N grows | Restrict claim to executable scalability and identify the learned UAV slow layer as future work |

## Minimum Table Replacement Rules

- Replace every table caption that says "smoke", "mini", or
  "diagnostic" before submission, unless the table is explicitly kept as
  a negative-control diagnostic.
- Every final table must report at least cost, mean similarity,
  question-floor rate, and one tail metric such as min or low-percentile
  similarity.
- A LaTeX table without a matching nonempty CSV does not count.
- A remote CSV without multiple seeds does not count.
- A learned method is not a claim unless fallback rate, nominal
  similarity, or stress floor-rate improves over deterministic QRS.

## Contribution Wording After Successful F2-F5

Use this contribution structure only if the evidence contract promotes
the gates to `claim-ready` or `tradeoff-ready`:

1. We formulate VQA semantic communication as question-conditioned
   evidence control, where semantic value depends on the question type,
   local DeepSC-VQA LUT gain, deadline slack, and serving-UAV pressure.
2. We design LearnRisk-QRS, a role-split UAV/UE semantic-control
   architecture with slow UAV decisions, fast UE/query adaptation, and a
   typed UAV-by-question semantic-price matrix.
3. We add a learned VQA-risk and nominal/stress MoE fast layer that keeps
   nominal VQA gains while preventing stress-regime semantic collapse.
4. We validate the design with Greedy-vs-QRS, stress robustness,
   component/MoE ablation, and large-UE pressure gates.

## Conservative Wording If Results Are Mixed

If one or more gates remain `tradeoff-ready` or `negative-control`, use
this narrower framing:

1. The paper contributes a VQA-native semantic-control decomposition and
   executable evidence gates for multi-UAV VQA offloading.
2. The deterministic role-split and typed-price components provide
   measurable cost/fidelity or safety trade-offs, while learned fast
   layers require risk calibration to avoid stress collapse.
3. The results identify when question-conditioned semantic prices help,
   when they saturate, and why a monolithic hybrid PPO is not the right
   final architecture.

## Final Audit Sequence

1. Run local preflight: `python3 hppo-uav/run/qrs_remote_preflight.py`.
2. After explicit upload authorization, sync the manifest with
   `python3 hppo-uav/run/qrs_remote_sync_plan.py --execute --confirm SYNC_REMOTE_QRS_FILES`.
3. Run remote smoke verification with
   `python3 hppo-uav/run/qrs_remote_smoke_plan.py --execute --confirm RUN_REMOTE_QRS_SMOKE`.
4. Run F2-F5 remote jobs with
   `python3 hppo-uav/run/qrs_remote_launch_plan.py --execute --confirm START_REMOTE_QRS_JOBS`.
5. Collect finished remote outputs with
   `python3 hppo-uav/run/qrs_remote_collect_plan.py --execute --confirm COLLECT_REMOTE_QRS_RESULTS`.
6. Run `python3 hppo-uav/run/qrs_tccn_evidence_contract.py`.
7. Replace smoke tables and captions according to the gate verdicts.
8. Run `python3 hppo-uav/run/qrs_tccn_readiness_audit.py`.
9. Run `python3 hppo-uav/run/qrs_tccn_claim_linter.py`.
10. Run `python3 hppo-uav/run/qrs_tccn_table_audit.py --strict`.
11. Run `python3 hppo-uav/run/qrs_tccn_submission_gate.py --strict`.
12. Compile `paper/main.tex` and do one whole-paper search for forbidden
   final-claim wording around MA-HPPO/CHPPO.
