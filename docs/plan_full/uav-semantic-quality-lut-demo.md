# UAV Semantic Communication Quality LUT

Overall: **uav-semantic-quality-lut-ready**

## Model

`A_k = LUT[l_k, s_k, channel_bin, view_quality_bin, freshness_bin, risk_level]`

- Quality constraint: `A_k >= epsilon_k`
- Deadline constraint: `T_k <= tau_k`
- `s_k=0`: cache answer
- `s_k=1`: lightweight evidence, e.g., tags / boxes / semantic tokens
- `s_k=2`: high-fidelity evidence, e.g., crop / full image
- `critical` is represented as `risk_level`, not as a service level.

## Dataset-Derived View Proxy

`phi_mk = f(apparent_object_scale, occlusion, effective_resolution, viewpoint_degradation)`

- apparent object scale: `bbox_area / image_area`
- effective resolution: bbox crop pixel size
- occlusion: VisDrone occlusion tag or annotation proxy
- viewpoint degradation: synthetic rotation / perspective / crop proxy

## Outputs

- task CSV: `paper/data/uav_semantic_quality_tasks_demo.csv`
- LUT CSV: `paper/data/uav_semantic_quality_lut_demo.csv`
- LUT rows: `405`
- task rows: `20`
- service levels: `[0, 1, 2]`
- quality-floor LUT rows: `30`

## Coverage

- `attribute` tasks: `4`
- `counting` tasks: `4`
- `presence` tasks: `4`
- `relation` tasks: `4`
- `risk` tasks: `4`

## Claim Boundary

v0 uses a rule-based evaluator for mechanism validation; final VQA/VLM claims require replacing the evaluator with model-measured accuracy.
