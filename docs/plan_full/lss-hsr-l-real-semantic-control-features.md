# LSS-HSR-L Semantic-Control Features

Input questions: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_real_radar_questions.csv`
Output CSV: `paper/data/lss_hsr_l_real_semantic_control_features.csv`
Samples: `1455`
Question rows: `5820`
Risk-gated rows: `5268`
Average typed price seed: `3.12`
Max typed price seed: `6.6087`

## UAV Controller Hints

- `patrol_hold`: `1593`
- `reserve_confirmation_budget`: `1940`
- `risk_constrained_confirmation`: `970`
- `warning_route_adjustment`: `1317`

## UE Selector Hints

- `confirmatory_vqa`: `1455`
- `identity_query`: `1455`
- `uav_bird_disambiguation`: `1455`
- `warning_answer_floor`: `1455`

## Interface Meaning

- `typed_price_seed`: initial value for target-aware `lambda[m,q,c]` pricing.
- `risk_threshold`: fallback threshold; lower values trigger safer confirmation sooner.
- `visual_symbol_floor`: minimum visual semantic budget for confirmation-oriented control.
- `uav_controller_hint`: slow UAV-side control role.
- `ue_selector_hint`: fast question/answer selection role.
