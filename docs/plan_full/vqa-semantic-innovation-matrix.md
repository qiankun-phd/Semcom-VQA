# VQA Semantic Innovation Matrix

Date: 2026-05-30

Purpose: identify self-owned VQA semantic-system innovations for the
TCCN rewrite, beyond applying a borrowed hybrid-RL template. The target
method remains **LearnRisk-QRS**: slow UAV semantic control plus fast
UE/query semantic adaptation with a typed price interface and a learned
risk gate.

## Core Insight

VQA is not a generic semantic-transmission task. The question changes
which visual evidence is valuable: counting and spatial queries need
finer visual/textual semantic detail than yes-no queries, and the same
wireless symbol budget can be excessive for one question type but
insufficient for another. Therefore, the communication system should not
optimize a single average similarity score. It should expose
question-conditioned semantic state, allocate semantic symbols according
to question utility, and protect the low-percentile question floor under
channel stress.

The LSS-HSR-L radar dataset adds a low-altitude patrol-warning version
of this insight. A radar event can determine which VQA question should
be asked next: whether the target is a drone or bird, whether its
trajectory implies intrusion risk, and whether UAV visual confirmation
is worth the semantic-symbol cost. In this setting, radar semantics do
not replace VQA; they trigger question-conditioned visual confirmation
and set the semantic price of the warning task.

## Innovation Matrix

| Layer | Innovation | Existing artifact | Why it is VQA-specific | Required evidence |
|---|---|---|---|---|
| Semantic state | Question-conditioned state features: question type, semantic margin, LUT local gain, deadline slack, UAV semantic load | `semantic_state_features=True` env hook; `tests/test_uavnet_env.py` | The controller sees what the current question needs, not just SNR/cost | Observation ablation and per-question metrics |
| Slow UAV layer | Role-split visual-symbol control per serving UAV | `QuestionAwareSemanticPolicy`, `uav_role_split` toggle | A UAV visual symbol budget affects all questions associated with that UAV | Component ablation: disabling UAV role split should raise cost or floor loss |
| Interface | Typed UAV-by-question semantic-price matrix | `typed_stable_dual_utility`, typed price table | Counting/spatial/yes-no questions create different QoS pressure for the same UAV | Price drift, Q-floor, and per-type floor satisfaction under stress |
| Fast UE/query layer | Local symbol-power utility using question weights and LUT marginal gain | `qrs_semantic.py`, `eval_qrs_semantic.py` | UE decisions optimize question utility rather than max similarity alone | Greedy-vs-QRS cost/fidelity gate |
| Learned fast layer | Neural contextual bandit distilled from QRS rollouts | `qrs_neural_bandit.py`, `train_qrs_neural_bandit.py` | The UE/query policy learns local VQA context: question type, serving UAV, price, slack, LUT slope | Nominal/stress fast-layer ablation |
| Safety | Separate learned VQA-risk head | `safety_mode=learned_risk` and `regime_learned_risk` | The action head can retain nominal VQA gains while the risk head vetoes floor-violating shortcuts | Learned-risk threshold and fallback analysis |
| Regime adaptation | Nominal/stress MoE fast layer | `--use-moe`, `qrs_moe_learnrisk_ablation.py` | Different channel regimes need different symbol-power experts for the same question distribution | Single vs Mixed-1H vs Mixed-MoE remote table |
| Robustness | Channel-loss/jamming-equivalent semantic degradation axis | `qrs_robustness_sweep.py` | Tests whether typed semantic price and risk gate preserve question floors under degraded semantic links | 0/5/10/15 dB multi-seed sweep |
| Scaling | Fixed-M large-UE pressure gate | `qrs_large_ue_sweep.py` | Tests whether UAV association/price remains useful when more questions compete for the same UAV semantic channel | N=4/6/8/10 multi-seed sweep |
| Semantic scene reconstruction | 5 low-altitude patrol-warning scenes bind 5W1H, target group, question type, UAV role, UE role, bridge signal, and guard | `vqa_semantic_scene_builder.py`; `plan/vqa-semantic-scene-library.md` | The VQA task is no longer an isolated question/image pair; it is an event-driven patrol scene that decides whether visual confirmation is semantically worth transmitting | Feature-probe coverage now passes on synthetic demo; real LSS-HSR-L confidence/trajectory calibration still pending |
| Patrol-warning context | Radar-triggered warning question types from LSS-HSR-L target class, trajectory, and scene metadata | `plan/lss-hsr-l-vqa-semantic-system-assessment.md` | Low-altitude warning questions are determined by target class and motion before visual confirmation | Derived radar-question labels; drone-vs-bird confusion; visual-confirm budget gate |

## Algorithm Architecture Choice

The algorithm should be presented as a composed architecture, not as one
monolithic RL agent:

- **UAV layer:** constrained multi-agent control is appropriate because
  UAV motion, channel reuse, and visual-symbol budgets are slow and
  affect many UEs. The long-term target is constrained PPO/HAPPO-style
  learning with semantic floors as constraints. The current executable
  slice keeps Greedy trajectory/channel behavior and learns/ablates the
  semantic-symbol part.
- **UE/query layer:** contextual bandit or PDQN-style parameterized
  action learning is appropriate because each UE chooses a small local
  action `(K_usr, power)` conditioned on question type, serving UAV,
  semantic price, deadline slack, and LUT slope. This layer can adapt
  faster than the UAV layer.
- **Bridge:** typed semantic price, semantic slack, and LUT marginal gain
  are the only cross-layer messages. This is the communication-system
  novelty: the UAV layer does not need query-level action control, and
  UEs do not need global mobility/interference control.
- **Radar-triggered patrol warning:** LSS-HSR-L can provide a
  low-altitude semantic event source. Radar target class, class
  confidence, trajectory risk, and scene type can be added as state
  features that decide whether the VQA query is an identity,
  drone-vs-bird, intrusion-risk, or visual-confirmation question.
- **Semantic scene library:** the current system now names five patrol
  warning scenes: airport rotary intrusion, urban drone-bird ambiguity,
  suburban vehicle decoy, fixed-rotating false alarm, and migration flock
  airspace pressure. Each scene maps the 5W1H context to UAV slow-control
  behavior, UE/query fast selection, typed price, and a risk guard.
- **Safety:** a separate risk head is preferable to a conservative action
  head because the latter loses nominal VQA gains. MoE experts separate
  nominal and stress regimes; the risk head decides when to fall back to
  deterministic QRS.

## Evidence Contract

The five TCCN gates map to the innovation matrix as follows:

| Gate | Innovation proved | Current status |
|---|---|---|
| F1 Method identity | LearnRisk-QRS is the main contribution, MA-HPPO/CHPPO are diagnostics | manuscript-audited |
| F2 Greedy-vs-QRS | Role-split question-aware symbol/power allocation gives a cost/fidelity trade-off | local-smoke-only |
| F3 Stress/jamming | Typed price + MoE/risk gate preserve semantic floors under channel degradation | local-smoke-only |
| F4 Component/MoE ablation | Each architecture component is necessary or has a measurable trade-off | local-smoke-only |
| F5 Scaling/large-UE | The role split remains meaningful under more UE questions per UAV | local-smoke-only |

Submission-grade evidence requires the remote multi-seed files listed in
`plan/qrs-remote-experiment-runbook.md` and `plan/qrs-remote-job-status.md`.
Whether a remote result is strong enough to support a main claim is
decided by `plan/tccn-submission-evidence-contract.md`.
The concrete UAV/UE algorithm composition is specified in
`plan/qrs-algorithm-architecture-spec.md`.
Reviewer-facing novelty risks and evidence-backed responses are mapped
in `plan/tccn-reviewer-novelty-defense.md`.

## Paper Integration Notes

- Introduction: state that VQA makes semantic resource allocation
  question-conditioned, not merely similarity-maximizing.
- Methodology: keep the role-specific algorithm selection paragraph as
  the architecture thesis.
- Experiments: report per-question floor satisfaction and low-percentile
  similarity alongside mean similarity.
- Discussion: frame Greedy as a necessary LUT-aware sanity check; the
  contribution is the role-split semantic system and its learned risk
  extension, not generic hybrid PPO dominance.
