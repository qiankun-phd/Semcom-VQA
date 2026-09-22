# TCCN Table Replacement Audit

Status: **local-smoke-tables-present**
Table count: `15`
Findings: `9`
Replacement required before submission: `6`

## Final Table Presence

| Gate | Required table input | Referenced | File exists |
|---|---|---:|---:|
| F2 | `tables/qrs_tccn_gate_remote_mseed.tex` | `False` | `False` |
| F3 | `tables/qrs_robustness_sweep_remote_mseed.tex` | `False` | `False` |
| F4-component | `tables/qrs_component_ablation_remote_mseed.tex` | `False` | `False` |
| F4-moe | `tables/qrs_moe_learnrisk_remote_mseed.tex` | `False` | `False` |
| F5 | `tables/qrs_large_ue_sweep_remote_mseed.tex` | `False` | `False` |

## Smoke/Mini/Tiny Findings

- `replace-before-submission` at `paper/sections/06_experiments.tex:483` input=`tables/qrs_neural_bandit_safe_smoke.tex`
  - Smoke/mini/tiny wording appears without an explicit diagnostic or negative-control scope.
  - Caption: Neural UE/query fast-layer smoke test at $\xi_{\mathrm{base}}=0.85$, channel-loss offset $10$\,dB, and fixed floor penalty disabled. Unsafe behavior cloning collapses; the QRS safety gate prevents catastrophic actions; RiskSoft minimizes explicit floor-risk probability mass, while LearnRisk uses a separate learned risk head to trigger fallbacks.
- `diagnostic-smoke-allowed` at `paper/sections/06_experiments.tex:496` input=`tables/qrs_neural_bandit_nominal_smoke.tex`
  - Smoke wording is explicitly diagnostic/interface scoped.
  - Caption: Nominal-channel neural UE/query smoke diagnostic at $\xi_{\mathrm{base}}=0.50$ and no channel-loss offset. The unsafe learned fast layer gives a higher nominal similarity but is unsafe under stress; guarded variants keep the QRS-safe operating point while exposing the remaining safety--performance trade-off. RiskSoft denotes utility-regret plus floor-risk distillation with a soft QRS utility gate; LearnRisk uses a separate risk head and keeps the unsafe nominal gain in this smoke run.
- `replace-before-submission` at `paper/sections/06_experiments.tex:511` input=`tables/qrs_learned_risk_threshold_sweep_smoke.tex`
  - Smoke/mini/tiny wording appears without an explicit diagnostic or negative-control scope.
  - Caption: Learned-risk threshold sweep for the neural UE/query fast layer. The sweep keeps the nominal learned gain while preventing high-floor 10\,dB stress collapse for all tested thresholds, but it is still smoke-scale evidence.
- `replace-before-submission` at `paper/sections/06_experiments.tex:522` input=`tables/qrs_moe_learnrisk_mini.tex`
  - Smoke/mini/tiny wording appears without an explicit diagnostic or negative-control scope.
  - Caption: Regime-aware LearnRisk mini ablation. Single trains/evaluates one action head per scenario; Mixed-1H trains one shared action head on nominal+stress rollouts; Mixed-MoE trains separate nominal and stress UE/query experts with a shared risk head. Mixed-MoE keeps most of the nominal gain while retaining mixed-training stress safety.
- `diagnostic-smoke-allowed` at `paper/sections/06_experiments.tex:534` input=`tables/qrs_typed_price_smoke.tex`
  - Smoke wording is explicitly diagnostic/interface scoped.
  - Caption: Typed semantic-price smoke diagnostic at $\xi_{\mathrm{base}}=0.85$, channel-loss offset $10$\,dB, and fixed floor penalty disabled. The typed-price variant keeps the scalar stable-price similarity gain while reducing average semantic-price pressure.
- `diagnostic-smoke-allowed` at `paper/sections/06_experiments.tex:546` input=`tables/qrs_component_ablation_smoke.tex`
  - Smoke wording is explicitly diagnostic/interface scoped.
  - Caption: Executable QRS component ablation smoke test at $\xi_{\mathrm{base}}=0.85$, channel-loss offset $10$\,dB, and fixed floor penalty disabled. This table verifies ablation interfaces for the full study; it is not yet submission-scale evidence.
- `replace-before-submission` at `paper/sections/06_experiments.tex:575` input=`tables/qrs_learned_risk_scaling_smoke.tex`
  - Smoke/mini/tiny wording appears without an explicit diagnostic or negative-control scope.
  - Caption: LearnRisk QRS scaling smoke values used by Fig.~\ref{fig:scale}.
- `replace-before-submission` at `paper/sections/06_experiments.tex:622` input=`tables/qrs_rollout_nominal_cdf_smoke.tex`
  - Smoke/mini/tiny wording appears without an explicit diagnostic or negative-control scope.
  - Caption: Nominal typed-QRS per-task similarity quantiles from the rollout dump used to replace Fig.~\ref{fig:cdf}. The dump contains 240 seed--episode--step--UE records and should be expanded before submission.
- `replace-before-submission` at `paper/sections/06_experiments.tex:644` input=``
  - Smoke/mini/tiny wording appears without an explicit diagnostic or negative-control scope.
  - Caption: Role-split ablation map for the final TCCN experiment suite. The first executable smoke values are reported in Table~\ref{tab:qrs-component-ablation} and Table~\ref{tab:qrs-moe-learnrisk}; the final submission table should expand each row to multi-seed nominal, channel-stress, and per-question-type results.
