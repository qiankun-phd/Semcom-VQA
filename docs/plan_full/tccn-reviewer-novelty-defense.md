# TCCN Reviewer Novelty Defense Map

Date: 2026-05-30

Purpose: turn the VQA semantic-system idea into a reviewer-facing
defense. This file is deliberately stricter than the manuscript prose:
it lists likely TCCN objections, the self-owned LearnRisk-QRS response,
and the evidence gate that must support each response.

## Main Defense

The paper should not claim novelty as "hybrid PPO for UAV resource
allocation." That space has close prior work. The defensible claim is:

> VQA semantic communication needs a question-conditioned semantic-control
> system. The question changes the value of visual evidence, so UAV and
> UE decisions should be split by timescale and connected by a typed
> UAV-by-question semantic price plus a VQA-risk gate.

This is the line that separates the work from generic UAV-MEC, generic
semantic communication, and generic hybrid-action MARL.

## Objection-to-Evidence Map

| Reviewer objection | Response | Required evidence |
|---|---|---|
| "This is just MAPPO/HPPO with a semantic reward." | The final method is not monolithic MAPPO/HPPO. MA-HPPO/CHPPO are diagnostics that expose why flat hybrid control is weak; LearnRisk-QRS decomposes UAV and UE/query decisions and adds a typed semantic-price interface. | F1 method identity audit; Section IV \qmethod; F2 Greedy-vs-QRS remote table |
| "Hybrid-action MAPPO already exists for UAV/MEC." | Existing hybrid-action MAPPO controls association, offloading, channel, or power. It does not model question type, DeepSC-VQA LUT marginal gain, question floors, typed UAV-by-question price, or learned VQA floor-risk. | Related-work differentiation; F4 no-question/no-price/no-UAV-role ablation |
| "Wireless VQA semantic communication already exists." | Encoder-side wireless VQA designs optimize what semantic features to transmit. This paper takes a fixed DeepSC-VQA LUT and optimizes the multi-UAV resource-control system around it. | F2 cost/fidelity gate; per-question metrics; CDF/tail similarity |
| "Semantic price is just a Lagrangian multiplier." | The typed price is not one scalar constraint multiplier. It is a UAV-by-question-type pressure matrix updated from question-conditioned floor violations and broadcast to UE/query controllers. | F3 stress sweep; F4 no-price ablation; typed-price diagnostics |
| "A learned neural fast layer is not novel by itself." | Correct. The claim is risk-calibrated execution: a learned action expert may keep nominal gains, while a separate VQA-risk head vetoes actions predicted to miss the semantic floor. | F2 LearnRisk rows; F4 Single vs Mixed-1H vs Mixed-MoE |
| "Why separate UAV and UE algorithms?" | UAV actions affect topology, association, interference, and many queries over a slow timescale. UE/query actions are local symbol-power decisions conditioned on a question. The algorithm choices follow this system decomposition. | F4 no-UAV-role ablation; F5 fixed-M N=4/6/8/10 pressure sweep |
| "Mean similarity hides sacrificed users/questions." | The evidence contract requires question-floor rate and tail/min similarity, not mean alone. | Evidence contract; F2-F5 table schemas; CDF figure |
| "Stress/jamming robustness is post-hoc." | The paper treats channel-loss sweeps as deployment diagnostics. The risk/MoE layer is designed specifically because unsafe nominal shortcuts collapse under stress. | F3 0/5/10/15 dB sweep; F4 MoE ablation |

## Contribution Wording To Prefer

Use:

- "question-conditioned VQA semantic control"
- "role-split UAV/UE semantic architecture"
- "typed UAV-by-question semantic price"
- "learned VQA floor-risk gate"
- "nominal/stress UE-query experts"
- "cost/fidelity and floor-safety trade-off"

Avoid overclaiming:

- "first hybrid PPO for UAV networks"
- "MA-HPPO dominates Greedy"
- "neural fast layer is better because it is neural"
- "semantic price alone solves stress robustness"
- "local-smoke results prove TCCN claims"

## Evidence Dependency

| Claim tier | Allowed before remote F2-F5 | Allowed after remote F2-F5 |
|---|---|---|
| System novelty | Yes: architecture and rationale | Yes, with final tables |
| Algorithmic dominance | No | Only if evidence contract says `claim-ready` |
| Cost/fidelity trade-off | Local smoke only, clearly labeled | If F2/F5 are `tradeoff-ready` or better |
| Stress safety | No final claim | If F3 and F4 MoE are `claim-ready` or `tradeoff-ready` |
| Large-UE scalability | No final claim | If F5 has N=4/6/8/10 remote rows |

## Paper Locations

- Section I: put the VQA question-conditioned evidence-value thesis in
  the contribution list.
- Section II: differentiate against hybrid-action MAPPO by missing VQA
  semantic variables, not by saying hybrid MAPPO does not exist.
- Section IV: make \qmethod the target method and keep \method/\chmethod
  as diagnostics.
- Section VI: replace smoke tables with remote F2-F5 tables before
  submission.
- Section VII/VIII: state limits honestly if any gate is only
  `tradeoff-ready` or `negative-control`.
