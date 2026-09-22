# Low-Altitude VQA Top-5 Narrative Dashboard

Date: 2026-05-30

Purpose: track the five highest-priority items for turning the current VQA
semantic communication project into a low-altitude patrol-warning system
supported by LSS-HSR-L. This is a narrative and system-architecture dashboard,
not a manuscript-editing checklist.

## Current Verdict

Status: **narrative interface ready, real-data validation pending**.

Single readiness command:

```bash
python3 hppo-uav/run/low_altitude_vqa_readiness_gate.py
```

Current result:

```text
overall_status: local-design-ready-real-data-pending
N1-N4: local-ready
N5: real-data-pending
```

Real-data location check: local workspace and `lab-s2`
were searched on 2026-05-30. No LSS-HSR-L archive was found. Details are
recorded in `plan/lss-hsr-l-real-data-readiness.md`. Download placement and
related ScienceDB radar-dataset context are recorded in
`plan/lss-hsr-l-download-and-related-datasets.md`.

ScienceDB OpenAPI metadata was also checked. It now confirms DOI/title,
unrestricted/free access, `CC BY-NC 4.0`, size `209569478` bytes, all four
target groups, Doppler waterfall, trajectory data, and Shenzhen/Changsha/
Chongqing urban/airport/suburban scenes. This supports non-commercial research
framing, but the downloaded zip README / usage instructions still control
derived-label and redistribution claims.

The remote import path is now scripted:

```bash
python3 hppo-uav/run/lss_hsr_l_remote_plan.py
```

Current dry-run status: `local_source_present: false`. Once the zip is placed
locally, the same script can sync to `lab-s2` and run the
guarded remote audit pipeline with
`--execute --confirm SYNC_REMOTE_LSS_HSR_L_DATA`.

The local repository now has a coherent chain:

```text
LSS-HSR-L metadata
  -> dataset audit
  -> radar sample manifest
  -> semantic calibration
  -> radar-triggered warning questions
  -> low-altitude semantic scene library
  -> semantic-control feature adapter
  -> UAV/UE algorithm split
  -> communication policy probe
```

The current gate command reports:

```bash
python3 hppo-uav/run/lss_hsr_l_narrative_gate.py --strict
```

Result:

```text
overall_status: narrative-interface-ready-real-data-pending
learnrisk_cost: 18
uniform_cost: 24
learnrisk_drone_miss: 0.0
feature_questions: 16
risk_gate_rows: 8
```

These are synthetic pipeline numbers. They prove interface shape, not real
dataset validity.

The algorithm-composition gate reports:

```bash
python3 hppo-uav/run/vqa_algorithm_architecture_gate.py --strict
```

Result:

```text
overall_status: algorithm-architecture-ready
```

The algorithm claim-scope gate reports:

```bash
python3 hppo-uav/run/vqa_algorithm_claim_scope.py --strict
```

Result:

```text
overall_status: claim-scope-ready
main_claim_scope: architecture_and_executable_fast_layer_not_final_slow_uav_learning
```

Allowed algorithm claim: role-specific UAV/UE semantic-control architecture,
typed semantic price, risk guard, and executable fast UE/query selectors.
Disallowed algorithm claim: final constrained learned UAV slow controller is
implemented.

The semantic-scene library reports:

```bash
python3 hppo-uav/run/vqa_semantic_scene_builder.py \
  --feature-csv paper/data/lss_hsr_l_demo_semantic_control_features.csv
```

Result:

```text
overall_status: semantic-scene-library-ready-with-feature-probe
scene_count: 5
target_groups: biological, fixed_rotating, rotary_uav, vehicle
question_types: drone_or_bird, intrusion_risk, target_identity, visual_confirm
feature_rows: 16
risk_gate_rows: 8
```

The local innovation scout reports:

```bash
python3 hppo-uav/run/vqa_innovation_scout.py --strict
```

Result:

```text
overall_status: innovation-map-ready
searched local files: 102
themes found: question-conditioned state, role split, typed semantic price,
learned fast layer/risk, radar-triggered patrol warning, real-data guardrails
```

The TCCN narrative/evidence gap audit reports:

```bash
python3 hppo-uav/run/low_altitude_vqa_tccn_gap_audit.py
```

Result:

```text
overall_status: tccn-narrative-innovation-ready-evidence-pending
D1 scenario originality: local-ready
D2 dataset fit/license: metadata-ready-real-data-pending
D3 radar-to-VQA bridge: local-ready via semantic calibration stage
D4 UAV/UE algorithm architecture: local-ready via claim-scope gate
D5 system innovation map: local-ready
D6 TCCN experimental evidence: evidence-pending
D7 claim boundary: local-ready
```

The low-altitude VQA experiment blueprint reports:

```bash
python3 hppo-uav/run/low_altitude_vqa_experiment_blueprint.py
```

Result:

```text
overall_status: experiment-blueprint-ready-real-data-pending
E1 scene coverage: pass
E2 radar-triggered policy probe: pass
E3 calibration ablation: demo-pass-real-data-pending
E4 risk guard ablation: demo-pass-real-data-pending
E5 remote multiseed stress: evidence-pending
```

The calibration ablation reports:

```bash
python3 hppo-uav/run/lss_hsr_l_calibration_ablation.py \
  --input-csv paper/data/lss_hsr_l_demo_manifest.csv
```

Result:

```text
overall_status: calibration-ablation-ready-real-data-pending
confirm_flip_rows: 1
avg_typed_price_abs_delta: 0.6997
```

The risk guard ablation reports:

```bash
python3 hppo-uav/run/lss_hsr_l_risk_guard_ablation.py \
  --input-csv paper/data/lss_hsr_l_demo_questions.csv
```

Result:

```text
overall_status: risk-guard-ablation-ready-real-data-pending
variants: cheap_no_guard, scene_only_guard, target_scene_guard, learnrisk_guard
```

## Top-5 Items

| Item | What has to be true | Current evidence | Status | Next action |
|---|---|---|---|---|
| N1 dataset usefulness | LSS-HSR-L is framed as a radar semantic event source, not a direct camera-VQA dataset | `lss_hsr_l_scidb_metadata_gate.py` confirms page-level DOI/license/access/modality/classes; system assessment records limits | metadata-ready locally | Download zip and run audit on README/license |
| N2 radar-to-question bridge | Radar target class, trajectory, confidence, and scene map into patrol-warning questions | `lss_hsr_l_manifest_builder.py` + `lss_hsr_l_semantic_calibration.py` + `lss_hsr_l_question_builder.py` + protocol doc | done locally | Replace prior-derived fields with real classifier/trajectory outputs |
| N3 semantic-control interface | Question rows become semantic scenes and control features for typed price, risk gate, and visual symbol floor | `vqa_semantic_scene_builder.py`; `lss_hsr_l_feature_adapter.py`; demo outputs 16 feature rows and 8 risk-gated rows | done locally | Calibrate scene priors and feature weights on real LSS-HSR-L confusion/risk statistics |
| N4 UAV/UE algorithm architecture | UAV and UE/query use different algorithms suitable for their timescales without overclaiming final slow-UAV learning | `vqa_algorithm_architecture_gate.py`; `vqa_algorithm_claim_scope.py`; `plan/uav-ue-algorithm-composition-decision.md` | done locally | Keep current claim scoped; implement constrained slow-UAV learner only after calibrated fast layer is stable |
| N5 evidence boundary | The system does not overclaim VQA accuracy or commercial readiness before license/data inspection | narrative gate checks claim boundary; dataset audit, locator, and remote plan tools exist | done locally | Place zip, run `lss_hsr_l_remote_plan.py`, then audit downloaded archive |

## TCCN Gap Queue

| Priority | What to fix | Current blocker |
|---|---|---|
| P1 | Download/place LSS-HSR-L zip and run archive README/license audit | `local_source_present=false`; page metadata already confirms `CC BY-NC 4.0` |
| P2 | Calibrate radar confidence, drone-bird margin, scene risk, and trajectory risk from real files | `missing-audit`; calibration stage exists but awaits real manifest |
| P3 | Keep algorithm claim scoped to architecture + executable fast layer, not final constrained UAV slow learning | `claim-scope-ready` |
| P4 | Execute low-altitude VQA experiment blueprint and remote multi-seed evidence contract | `experiment-blueprint-ready-real-data-pending`; `local-smoke-only` |
| P5 | Keep claim boundaries explicit: radar triggers VQA confirmation, not paired camera-VQA accuracy | no blocker; continue enforcing |

## Innovation Scout Themes

| Theme | Current claim | Remaining gap |
|---|---|---|
| question-conditioned semantic state | VQA allocation is driven by question type, semantic margin, LUT gain, and deadline slack | Needs real-data per-question metrics after LSS-HSR-L feature calibration |
| role-split UAV/UE control | UAV slow control and UE/query fast control are separated by timescale and information scope | Final learned slow UAV controller is not implemented yet |
| typed semantic price bridge | `lambda[m,q,c]` carries target/question-aware semantic pressure across layers | Real LSS-HSR-L target-specific price calibration is still pending |
| learned fast layer and risk | Contextual/MLP fast layer can be guarded by deterministic QRS or learned risk | No real-data multi-seed learned dominance claim yet |
| radar-triggered patrol warning | Radar event semantics decide whether visual confirmation is worth the symbols | Requires downloaded archive plus classifier/trajectory confidence |
| real-data guardrails | Audit/gate/pipeline prevent accidental real-data overclaiming | Actual ScienceDB zip is not yet present locally or on the server |

## Innovation Claim That Is Now Defensible

The strongest current claim is:

> The project can be reframed as radar-triggered, question-conditioned VQA
> semantic communication for low-altitude patrol warning. LSS-HSR-L supplies
> radar event semantics; the UAV visual link is invoked only when target
> ambiguity, trajectory risk, or scene risk justifies visual confirmation.

This is more original than a generic VQA transmission story because the
communication system is no longer a passive image/question sender. It becomes
an event-triggered semantic control system.

## Algorithm Composition

Recommended split:

| Role | Decision timescale | Suitable algorithm family | Reason |
|---|---|---|---|
| UAV controller | slow | constrained PPO / HPPO / CMDP wrapper | Route, confirmation dispatch, transmit power, and shared visual-symbol floors are coupled and safety-constrained |
| UE/query selector | fast | contextual bandit / PDQN / risk-aware neural bandit | Current question and semantic budget are local, discrete or hybrid, and should adapt quickly |
| Cross-layer interface | medium | target-aware typed price `lambda[m,q,c]` | Links UAV resource pressure, question type, and radar target superclass |
| Safety fallback | event-triggered | risk head or deterministic veto | Prevents cheap semantic actions on ambiguous drone/bird or high-risk trajectory events |

## Real-Data Gate

The remaining blocker for real claims is not narrative design. It is
inspection of the downloaded ScienceDB archive:

```bash
python3 hppo-uav/run/lss_hsr_l_dataset_audit.py \
  --dataset-path /path/to/数据集及使用说明.zip
```

Or run the guarded one-command pipeline:

```bash
python3 hppo-uav/run/lss_hsr_l_real_data_pipeline.py \
  --dataset-path /path/to/数据集及使用说明.zip
```

Then run the promotion gate:

```bash
python3 hppo-uav/run/lss_hsr_l_real_data_gate.py --strict
```

Proceed only if:

- README / usage instructions are present.
- license permits academic derived-label experiments.
- the 9-way class mapping is explicit.
- Doppler waterfall and trajectory files are identifiable.
- trajectory samples can be joined to target labels.
- no restriction forbids the radar-to-question derived task.

The enhanced audit reports these gates explicitly as `gate_checks`.
The promotion gate turns those checks into a single status:
`real-data-ready-for-derived-labels` or `real-data-needs-review`.

## Do Not Claim Yet

- LSS-HSR-L proves final visual VQA answer accuracy.
- The dataset includes paired UAV camera VQA samples.
- The method is ready for commercial low-altitude deployment.
- Synthetic demo metrics are real experimental evidence.
- Remote multi-seed TCCN evidence is complete.
