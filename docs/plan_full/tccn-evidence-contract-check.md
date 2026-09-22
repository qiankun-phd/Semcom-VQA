# TCCN Evidence Contract Check

Overall: **local-smoke-only**

| Gate | Source | Rows | Verdict | Missing outputs | Missing required |
|---|---|---:|---|---|---|
| F2 Greedy-vs-QRS cost/fidelity | local-smoke | 8 | local-smoke | `paper/data/qrs_tccn_gate_remote_mseed.csv`<br>`paper/tables/qrs_tccn_gate_remote_mseed.tex` | - |
| F3 Stress/jamming robustness | local-smoke | 4 | local-smoke | `paper/data/qrs_robustness_sweep_remote_mseed.csv`<br>`paper/tables/qrs_robustness_sweep_remote_mseed.tex` | `Mixed-MoE`, `5dB`, `15dB` |
| F4 Component and MoE ablations | local-smoke | 10 | local-smoke | `paper/data/qrs_component_ablation_remote_mseed.csv`<br>`paper/tables/qrs_component_ablation_remote_mseed.tex`<br>`paper/data/qrs_moe_learnrisk_remote_mseed.csv`<br>`paper/tables/qrs_moe_learnrisk_remote_mseed.tex` | - |
| F5 Scaling and large-UE pressure | local-smoke | 4 | local-smoke | `paper/data/qrs_large_ue_sweep_remote_mseed.csv`<br>`paper/tables/qrs_large_ue_sweep_remote_mseed.tex` | `Mixed-MoE`, `N=8`, `N=10` |

## Notes

### F2 Greedy-vs-QRS cost/fidelity
- Source: `paper/data/qrs_tccn_gate_local_mini.csv`
- Remote multi-seed CSV/TEX is missing; local rows are not submission evidence.
- Next action: Run the F2 remote gate in plan/qrs-remote-experiment-runbook.md until qrs_tccn_gate_remote_mseed.csv and .tex contain Greedy, QRS, TypedPrice, and LearnRisk rows for nominal and stress.

### F3 Stress/jamming robustness
- Source: `paper/data/qrs_robustness_sweep_tiny.csv`
- Remote 0/5/10/15 dB multi-seed robustness evidence is missing.
- Next action: Run the F3 robustness sweep over 0/5/10/15 dB with Greedy, TypedPrice, and Mixed-MoE, then regenerate the remote CSV/TEX.

### F4 Component and MoE ablations
- Source: `component+moe`
- Remote component and MoE ablation CSV/TEX files are both required.
- Next action: Run both remote F4 ablations: component variants full_stable/no_question/no_uav_role/no_price and MoE variants Single/Mixed-1H/Mixed-MoE.

### F5 Scaling and large-UE pressure
- Source: `paper/data/qrs_large_ue_sweep_tiny.csv`
- Remote fixed-M N=4/6/8/10 large-UE evidence is missing.
- Next action: Run the fixed-M large-UE sweep for N=4/6/8/10 with Greedy, TypedPrice, and Mixed-MoE, then regenerate the remote CSV/TEX.
