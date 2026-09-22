# LearnRisk-QRS Algorithm Architecture Specification

Date: 2026-05-30

Purpose: turn the VQA semantic-system insight into a concrete algorithm
architecture that is self-owned and implementable. This spec connects the
paper method, the executable QRS code path, and the F1-F5 evidence gates.

## Design Thesis

VQA semantic communication should not be modeled as one flat hybrid-RL
agent over all UAV and UE actions. The question changes which visual
evidence is valuable, and UAV decisions and UE/query decisions operate on
different physical and semantic timescales. Therefore the algorithm should
be a composed semantic-control system:

- Slow UAV layer: network geometry, association/channel pressure, and
  UAV-side visual-symbol budgets.
- Fast UE/query layer: per-question text-symbol cardinality and UE power.
- Bridge: typed semantic price, semantic slack, deadline slack, and local
  LUT marginal gain.
- Safety: learned VQA-risk gate and nominal/stress MoE fast layer.

This is the architecture that should be presented as the journal
contribution. MA-HPPO and MA-CHPPO remain diagnostics and theoretical
scaffolding, not the final performance claim.

## Self-Owned Innovation Boundary

The architecture should be written as a VQA semantic-system design, not
as "we apply algorithm X to UAV resource allocation." Existing algorithm
families only supply local optimization primitives after the system has
been decomposed:

- constrained multi-agent policy optimization is appropriate for the
  slow UAV layer because one UAV action changes topology, interference,
  and many downstream queries;
- contextual bandit / neural parameterized-action selection is
  appropriate for the fast UE/query layer because each query chooses a
  small local symbol-power action under a question-specific semantic
  floor;
- the typed semantic-price matrix is the new cross-layer message, not a
  generic congestion price, because it is indexed by UAV and question
  type and is updated from VQA floor violations;
- the learned risk head is not a generic classifier; it estimates whether
  a candidate UE/query action will miss the current
  question-conditioned semantic floor.

This boundary keeps the claimed novelty inside the paper's own VQA
communication model: question-conditioned evidence value, role-specific
UAV/UE control, and VQA-risk-calibrated action execution.

## Layer Contracts

| Layer | Decision | Algorithm choice | Why this choice fits | Current executable artifact |
|---|---|---|---|---|
| UAV slow layer | `Delta d_m`, `theta_m`, UAV power, channel reuse, `K_U,m` | Constrained multi-agent PPO/HAPPO-style policy for the final learner; deterministic role-split QRS for the executable slice | One UAV action affects many downstream VQA queries and interference relationships | `QuestionAwareSemanticPolicy`, `uav_role_split` |
| UE/query fast layer | `(K_n, p_n)` | Deterministic QRS utility, tabular contextual bandit, neural contextual bandit, later PDQN-style parameterized action | Each query has a small local action conditioned on question type, serving UAV, price, slack, and LUT slope | `qrs_semantic.py`, `qrs_bandit.py`, `qrs_neural_bandit.py` |
| Cross-layer bridge | `lambda_{m,q}`, semantic slack, LUT local gain, deadline slack | Typed semantic-price matrix plus local summaries | Keeps UAVs from controlling every query directly while giving UEs network-level QoS pressure | `typed_stable_dual_utility` |
| Safety/regime layer | fallback/veto and expert routing | Learned risk head plus nominal/stress MoE experts | Avoids nominal shortcut collapse under channel stress without making all regimes conservative | `safety_mode=learned_risk`, `--use-moe` |

## Algorithmic Components

### 1. Question-Conditioned Semantic State

The controller state must expose VQA-native quantities:

- question type;
- question-conditioned floor and margin;
- local DeepSC-VQA LUT gain around the current SNR/symbol bin;
- deadline slack;
- serving UAV and UAV semantic load;
- typed semantic price.

This turns the problem from generic rate allocation into VQA semantic
control. The evidence gate is an observation / no-question ablation and
per-question metrics.

### 2. Slow UAV Role Split

The UAV layer controls variables whose effect is shared across UEs:

```text
a^U_m = (Delta d_m, theta_m, p_U,m, b_U,m, K_U,m)
```

The final learning primitive should be constrained multi-agent policy
optimization because UAV movement/channel choices affect future topology
and many associated UEs. The current executable slice keeps Greedy
trajectory/channel behavior and tests the semantic-symbol role split
through `K_U,m`. This is honest: it proves the interface before claiming
full slow-layer learning.

### 3. Fast UE/Query Action Selector

The UE/query layer controls:

```text
a^Q_n = (K_n, p_n)
```

The deterministic QRS utility is the safe lower-bound policy:

```text
utility = question_weight * similarity
          + question_gain_weight * LUT_local_gain
          - resource_weight * (delay + energy + UAV-rate cost)
          - typed_price * floor_violation
```

The trainable replacement can be:

- tabular contextual bandit for quick online adaptation;
- neural contextual bandit for compact learned fast-layer inference;
- PDQN-style parameterized-action module if the action space expands
  beyond the current discrete LUT grid.

This is not borrowed wholesale from a single prior algorithm. The
algorithm choice follows the VQA system decomposition: slow global UAV
control plus fast local query adaptation.

### 4. Typed Semantic-Price Matrix

The bridge variable is a UAV-by-question-type price:

```text
lambda[m, q] <- clip(lambda[m, q] + eta * w_q * mean_floor_violation[m, q])
```

The stable variant adds:

- per-step price trust region;
- anti-windup cap on stored/effective price;
- question-specific price weights.

This is the central semantic-communication innovation: the message is
not generic congestion price or SNR. It is VQA QoS pressure indexed by
the question type served by each UAV.

### 5. Learned VQA-Risk and MoE Fast Layer

The neural fast layer has:

- nominal action expert;
- stress action expert;
- optional learned router;
- shared risk head.

The risk head predicts whether a candidate local action will miss the
question-conditioned semantic floor. It can veto a neural shortcut and
fall back to deterministic QRS when the shortcut is unsafe. This
separates nominal performance from stress safety: the action expert is
allowed to keep VQA gains, while the risk head handles safety.

## Evidence Mapping

| Claim | Required evidence |
|---|---|
| Role-split semantic control is better than flat hybrid PPO framing | F1 manuscript identity plus F2 Greedy-vs-QRS gate |
| Typed semantic price is useful and stable | F2 TypedPrice rows, F3 stress rows, price-stability ablation |
| Question type matters | F4 no-question ablation and per-question floor/similarity metrics |
| UAV visual-symbol role split matters | F4 no-UAV-role ablation and F5 large-UE pressure |
| Learned fast layer is safe | F2/F3 LearnRisk rows with fallback and floor-rate metrics |
| MoE is justified | F4 Single vs Mixed-1H vs Mixed-MoE remote table |

## Paper Integration

- Abstract and introduction: describe LearnRisk-QRS as a semantic-control
  architecture, not as a generic hybrid-RL algorithm.
- Methodology: keep the role-specific algorithm-selection paragraph and
  Algorithm 1/2 as the formal contract.
- Experiments: report cost, mean similarity, low-percentile/min
  similarity, question-floor rate, fallback rate, and price saturation.
- Discussion: state that Greedy is a necessary LUT-aware sanity check;
  the contribution is the role-split VQA semantic system and learned
  risk-gated fast layer.
- Final result insertion: use
  `plan/tccn-final-results-insertion-plan.md` after F2-F5 remote outputs
  exist so smoke artifacts are replaced rather than overclaimed.

## Current Implementation Status

- Deterministic role-split QRS: implemented.
- Stable scalar and typed semantic price: implemented.
- Component switches: implemented.
- Tabular fast-layer bandit: implemented but not a claim.
- Neural contextual fast layer: implemented.
- Learned-risk safety gate: implemented.
- Nominal/stress MoE: implemented.
- Final constrained learned UAV slow layer: not yet implemented; current
  executable slice reuses Greedy trajectory/channel behavior.
- Submission-grade evidence: pending remote multi-seed F2-F5.
