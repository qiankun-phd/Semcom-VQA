# UAV/UE Algorithm Composition Decision

Date: 2026-05-30

Purpose: make the algorithmic novelty explicit for the low-altitude VQA
semantic communication system. The system should not be framed as borrowing one
generic RL algorithm. It should be framed as a composed architecture where each
role uses the algorithm family that matches its decision timescale and
information scope.

## Decision

Use a split controller:

```text
Radar event semantics
  -> warning question and semantic-control features
  -> slow UAV controller
  -> target-aware typed semantic price
  -> fast UE/query selector
  -> learned or deterministic risk fallback
```

The core algorithmic innovation is the interface and decomposition:

- UAV decisions are slow, coupled, and safety-constrained.
- UE/query decisions are fast, local, and question-conditioned.
- The bridge is not raw channel state; it is semantic pressure:
  `lambda[m,q,c]`, semantic slack, visual-symbol floor, and risk threshold.

## Role-Specific Algorithm Choices

| Role | Decision variables | Recommended algorithm family | Why this role needs it | Current executable evidence |
|---|---|---|---|---|
| Radar event layer | target class, target group, confidence, trajectory risk | supervised radar classifier or calibrated rule baseline | LSS-HSR-L is a radar recognition dataset; this layer should output semantics, not directly solve VQA | `lss_hsr_l_manifest_builder.py`, `lss_hsr_l_question_builder.py` |
| UAV slow controller | confirmation dispatch, patrol/intercept route, UAV transmit power, shared visual-symbol floor | constrained PPO / HPPO / CMDP or Lyapunov wrapper | one UAV action changes future topology, channel pressure, and many downstream warning questions | architecture contract in `lss-hsr-l-radar-vqa-system-architecture.md`; QRS executable slice uses role-split UAV symbol control |
| UE/query fast selector | question action, answer floor, VQA semantic symbol budget, UE power | contextual bandit for discrete local choices; PDQN-style selector when action has discrete question plus continuous/pseudo-continuous budget | each warning question is local and must adapt faster than UAV mobility; query type changes the value of visual evidence | `qrs_semantic.py`, `qrs_bandit.py`, `qrs_neural_bandit.py`, `lss_hsr_l_feature_adapter.py` |
| Cross-layer bridge | target-aware semantic price and risk features | typed semantic-price update `lambda[m,q,c]` plus risk threshold | keeps global UAV pressure and local query utility coupled without a monolithic policy | `typed_price_seed`, `risk_threshold`, `visual_symbol_floor` |
| Safety fallback | visual confirmation veto/fallback | learned risk head plus deterministic QRS fallback | preserves nominal learned gains while preventing false cheap actions on drone/bird ambiguity or airport risk | LearnRisk-QRS docs and communication policy probe |

## Why Not a Single Borrowed Algorithm

A flat hybrid PPO/PDQN style policy would see a large mixed action vector:

```text
(UAV movement, UAV power, UAV symbols, UE symbols, UE power, question action)
```

That is the wrong abstraction for this system because:

1. UAV actions have slow, shared consequences across many UEs and warning
   questions.
2. UE/query actions are local and change whenever the warning question changes.
3. LSS-HSR-L introduces target class and trajectory context before visual VQA
   starts, so the communication controller must decide whether a visual
   question is worth asking.
4. Safety risk is sparse and asymmetric: missing a rotary-UAV warning near an
   airport is not equivalent to over-confirming a car.

Therefore, the novelty is not "apply algorithm X." The novelty is:

For checker clarity: this is not "apply algorithm X" to UAV offloading; it is
a role-specific semantic-control architecture.

```text
radar-triggered question formation
+ target-aware typed semantic price
+ slow UAV constrained control
+ fast UE/query semantic selection
+ risk-gated visual confirmation
```

## LSS-HSR-L-Specific State Injection

The LSS-HSR-L interface adds these features before the UAV visual link spends
bandwidth:

```text
target_group_id
question_type_id
radar_confidence
drone_bird_margin
trajectory_risk_score
scene_risk_weight
semantic_priority_weight
risk_threshold
visual_symbol_floor
typed_price_seed
```

This converts the dataset from a standalone radar recognition benchmark into a
semantic-control trigger for low-altitude patrol warning.

## Algorithm Development Roadmap

1. Current executable slice

   Use the existing QRS/typed-price/risk-gate pipeline to validate the
   interface. Keep claims limited to system readiness and synthetic or
   smoke-scale evidence.

2. Real-data feature calibration

   After the LSS-HSR-L archive passes `lss_hsr_l_real_data_gate.py`, estimate
   class confidence, drone-bird margin, and trajectory risk from real radar
   data. Use those values to calibrate `typed_price_seed` and
   `risk_threshold`.

3. Fast UE/query learner

   Train a contextual bandit or PDQN-style selector on radar-triggered warning
   questions. Compare against deterministic QRS and typed-price QRS.

4. Slow UAV learner

   Add constrained PPO/HPPO or CMDP learning for confirmation dispatch and
   route/power decisions only after the fast layer and risk fallback have a
   stable interface.

5. Joint evaluation

   Evaluate symbol cost, warning-floor rate, rotary-UAV miss rate, false visual
   confirmation rate, and critical-floor rate under nominal and stress channel
   settings.

## Evidence Boundary

Allowed now:

- The repository has a coherent role-specific algorithm architecture.
- LSS-HSR-L can serve as a radar semantic event source for patrol-warning VQA
  communication if license/README checks pass.
- LSS-HSR-L is not a direct camera-VQA dataset; it triggers and prices visual
  confirmation rather than replacing camera/question-answer evidence.
- The interface can be tested before real data through the synthetic demo and
  narrative/real-data gates.

Not allowed yet:

- A claim that the final learned UAV slow controller is implemented.
- A claim that LSS-HSR-L proves visual VQA answer accuracy.
- A claim that any learned fast layer dominates deterministic QRS on real
  LSS-HSR-L data.
- A commercial low-altitude deployment claim before the exact license is read.
