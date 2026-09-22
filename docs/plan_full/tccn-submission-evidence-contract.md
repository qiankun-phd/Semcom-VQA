# TCCN Submission Evidence Contract

Date: 2026-05-30

Purpose: define what counts as submission-grade evidence for the current
LearnRisk-QRS rewrite. This contract prevents local smoke runs, empty
remote artifacts, or isolated positive numbers from being treated as a
TCCN-ready result.

## Status Labels

| Label | Meaning | Submission use |
|---|---|---|
| `interface-only` | Code path runs, but the metric does not support a paper claim | Mention only as implementation progress |
| `local-smoke` | Small local run validates plumbing and direction | Not submission evidence |
| `remote-mseed` | Server run with multiple seeds and final CSV/TEX outputs | Candidate submission evidence |
| `claim-ready` | Remote-mseed evidence supports the stated claim and passes this contract | Can appear as a main paper result |
| `tradeoff-ready` | Remote-mseed evidence supports a narrower cost/fidelity or safety trade-off | Can appear with careful wording |
| `negative-control` | Remote-mseed evidence shows a component is ineffective or unsafe | Useful as ablation/limitation evidence |

## Gate Requirements

| Gate | Required artifact | Minimum evidence | Claim decision |
|---|---|---|---|
| F1 Method identity | Whole-paper audit plus final method wording | No remaining claim that MA-HPPO is the final performance method | Main method is LearnRisk-QRS; MA-HPPO/CHPPO are diagnostics |
| F2 Greedy-vs-QRS | `qrs_tccn_gate_remote_mseed.csv` and `.tex` | Greedy, QRS, TypedPrice, LearnRisk over nominal and stress scenarios with nonzero CSV rows | `claim-ready` only if LearnRisk or TypedPrice improves cost or floor safety without hiding a large fidelity loss; otherwise `tradeoff-ready` |
| F3 Stress robustness | `qrs_robustness_sweep_remote_mseed.csv` and `.tex` | 0/5/10/15 dB sweep including Greedy, TypedPrice, and Mixed-MoE | Claim robustness only if stress degradation is smoother or floor-rate is better than Greedy/TypedPrice at comparable cost |
| F4 Component/MoE ablation | component and MoE remote CSV/TEX files | Full stable QRS, no-question, no-UAV-role, no-price, Single, Mixed-1H, Mixed-MoE variants | Each architectural component must either improve a metric or be framed as a negative control |
| F5 Scaling/large-UE | `qrs_large_ue_sweep_remote_mseed.csv` and `.tex` | Fixed-M UE pressure with N=4/6/8/10 and nonzero rows | Scaling claim only if the role-split/price architecture remains competitive as N grows |

## Metric Interpretation Rules

- Mean similarity alone is insufficient. Every main table must report
  question-floor rate or low-percentile/min similarity.
- A method that improves average similarity but collapses under 10 dB
  stress is not claim-ready; it is a negative-control fast-layer result.
- A method that lowers cost while preserving question-floor rate within a
  small tolerance can support a trade-off claim even if mean similarity
  is slightly lower.
- A learned policy is not better merely because it is neural. It must
  reduce fallback rate, improve nominal utility, or improve stress floor
  safety relative to deterministic QRS.
- Empty CSVs, single-seed smoke runs, and generated LaTeX tables without
  matching CSV rows do not count as evidence.

## Innovation-to-Evidence Mapping

| Innovation | Must be supported by |
|---|---|
| Question-conditioned semantic state | F4 no-question ablation and per-question metrics |
| UAV role-split visual-symbol control | F4 no-UAV-role ablation plus F5 large-UE pressure |
| Typed UAV-by-question semantic price | F2 TypedPrice rows and F3 stress price/floor behavior |
| Fast UE/query contextual policy | F2 LearnRisk rows and F4 Single/Mixed/MoE ablation |
| Learned VQA-risk safety | F2/F3 fallback and floor-rate under stress |
| Nominal/stress MoE | F4 Single vs Mixed-1H vs Mixed-MoE table |

## Current Evidence State

As of this contract, the current authoritative readiness status is:

- F1: `manuscript-audited`, but requires one more whole-paper audit after
  final server results are inserted.
- F2-F5: `local-smoke-only`; not submission evidence.
- Remote validation server: `needs-remote-sync`.

The next promotion step is to sync the manifest in
`plan/qrs-remote-sync-manifest.md`, run
`plan/qrs-remote-experiment-runbook.md`, and verify with
`hppo-uav/run/qrs_remote_job_status.py` plus
`hppo-uav/run/qrs_tccn_readiness_audit.py`.

The executable version of this contract is
`hppo-uav/run/qrs_tccn_evidence_contract.py`. It writes
`paper/data/tccn_evidence_contract.json` and
`plan/tccn-evidence-contract-check.md`.
Before remote syncing or job launch, run
`hppo-uav/run/qrs_remote_preflight.py` to verify that the local scripts,
manifest, readiness audit, evidence contract, and job-status reports are
internally consistent.
After remote outputs exist, use
`plan/tccn-final-results-insertion-plan.md` to replace smoke tables,
choose conservative claim wording, and run the final whole-paper audit.
