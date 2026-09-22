# LSS-HSR-L Narrative Gate

Overall: **narrative-interface-ready-real-data-pending**

## Checks

| Check | Status | Evidence |
|---|---|---|
| `dataset_anchor` | `pass` | `plan/lss-hsr-l-vqa-semantic-system-assessment.md` |
| `radar_triggered_question_chain` | `pass` | `plan/lss-hsr-l-radar-question-protocol.md` |
| `semantic_control_interface` | `pass` | `hppo-uav/run/lss_hsr_l_feature_adapter.py` |
| `uav_ue_algorithm_split` | `pass` | `plan/lss-hsr-l-radar-vqa-system-architecture.md` |
| `claim_boundary` | `pass` | `plan/lss-hsr-l-radar-vqa-system-architecture.md` |
| `demo_pipeline` | `pass` | `paper/data/lss_hsr_l_demo_pipeline.json` |

## Demo Metrics

- learnrisk_cost: `18`
- uniform_cost: `24`
- learnrisk_drone_miss: `0.0`
- feature_questions: `16`
- risk_gate_rows: `8`

## Notes

- This gate proves the local narrative/interface chain, not real-data experimental validity.
- Real LSS-HSR-L claims remain pending until the downloaded archive, README, and license are audited.
