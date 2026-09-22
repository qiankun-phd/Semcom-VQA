# VQA Semantic Scene Library

Overall: **semantic-scene-library-ready-with-feature-probe**

## Innovation Claim

The system reconstructs low-altitude VQA as a radar-triggered semantic-scene controller: radar events choose the question, UAV slow control reserves safe confirmation capacity, UE/query fast selection spends symbols only where the current semantic scene requires them.

## Checks

| Check | Status | Detail |
|---|---|---|
| `scene_library_has_5w1h` | `pass` | Each scene states what/why/who/where/when/how. |
| `lss_target_group_coverage` | `pass` | biological,fixed_rotating,rotary_uav,vehicle |
| `question_type_coverage` | `pass` | drone_or_bird,intrusion_risk,target_identity,visual_confirm |
| `uav_ue_algorithm_split` | `pass` | {"slow controller": true, "fast selector": true, "lambda[m,q,c]": true, "risk": true} |
| `feature_interface_probe` | `pass` | feature_rows=16 |

## Scene Library

| Scene | Where | Target groups | Question types | Bridge |
|---|---|---|---|---|
| `airport_rotary_intrusion` | airport / restricted low-altitude corridor | rotary_uav | drone_or_bird, intrusion_risk, visual_confirm | lambda[m,q,c] increases for rotary_uav + intrusion_risk / visual_confirm |
| `urban_drone_bird_ambiguity` | urban canyon or city park edge | rotary_uav, biological | target_identity, drone_or_bird, visual_confirm | typed price separates biological false-confirm risk from rotary-UAV miss risk |
| `suburban_vehicle_decoy` | suburban road near low-altitude patrol boundary | vehicle | target_identity, intrusion_risk, visual_confirm | lambda[m,q,c] remains low for vehicle + visual_confirm unless trajectory risk spikes |
| `fixed_rotating_false_alarm` | industrial or suburban fixed installation | fixed_rotating | target_identity, intrusion_risk | typed price discounts fixed_rotating unless scene risk is protected |
| `migration_flock_airspace_pressure` | airport suburb or migration corridor | biological | target_identity, drone_or_bird, visual_confirm | lambda[m,q,c] encodes event burst pressure and target superclass |

## Algorithm Roles

### `airport_rotary_intrusion`

- UAV: constrained slow controller reserves confirmation route and visual-symbol floor
- UE/query: risk-aware contextual selector prioritizes intrusion and confirmation answers
- Guard: zero rotary-UAV miss floor before cost minimization
- Evidence needed: real LSS-HSR-L class label, trajectory risk, and airport scene metadata

### `urban_drone_bird_ambiguity`

- UAV: slow controller keeps patrol posture unless risk gate escalates
- UE/query: fast selector spends symbols only on disambiguation-critical question rows
- Guard: fallback to confirmation when margin is below the safety threshold
- Evidence needed: drone-vs-bird confusion statistics and margin calibration

### `suburban_vehicle_decoy`

- UAV: slow controller holds route and does not reserve high visual-symbol floor
- UE/query: fast selector answers cheap identity/risk questions before confirmation
- Guard: only high trajectory risk can override the cheap non-air target policy
- Evidence needed: trajectory features separating vehicles from aerial targets

### `fixed_rotating_false_alarm`

- UAV: slow controller avoids route deviation unless trajectory risk becomes nonzero
- UE/query: fast selector favors identity query with low visual-symbol floor
- Guard: stationary trajectory veto prevents rotor-signature overreaction
- Evidence needed: trajectory stationarity and label agreement from the dataset instructions

### `migration_flock_airspace_pressure`

- UAV: slow controller budgets patrol coverage under event bursts
- UE/query: fast selector batches low-risk biological confirmations
- Guard: biological batching cannot reduce rotary-UAV warning floor
- Evidence needed: temporal grouping or trajectory-window statistics from real files

## Remaining Gap

Real LSS-HSR-L files are still required to calibrate confidence, trajectory risk, event bursts, and license-safe derived labels.
