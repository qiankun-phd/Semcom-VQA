# LSS-HSR-L Semantic-Control Features

Input questions: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_demo_questions.csv`
Output CSV: `paper/data/lss_hsr_l_demo_semantic_control_features.csv`
Samples: `4`
Question rows: `16`
Risk-gated rows: `8`
Average typed price seed: `3.9582`
Max typed price seed: `10.2789`

## UAV Controller Hints

- `patrol_hold`: `8`
- `reserve_confirmation_budget`: `4`
- `risk_constrained_confirmation`: `2`
- `warning_route_adjustment`: `2`

## UE Selector Hints

- `confirmatory_vqa`: `4`
- `identity_query`: `4`
- `uav_bird_disambiguation`: `4`
- `warning_answer_floor`: `4`

## Interface Meaning

- `typed_price_seed`: initial value for target-aware `lambda[m,q,c]` pricing.
- `risk_threshold`: fallback threshold; lower values trigger safer confirmation sooner.
- `visual_symbol_floor`: minimum visual semantic budget for confirmation-oriented control.
- `uav_controller_hint`: slow UAV-side control role.
- `ue_selector_hint`: fast question/answer selection role.
