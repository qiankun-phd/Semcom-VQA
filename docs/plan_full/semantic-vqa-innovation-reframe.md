# VQA Semantic-System Reframe for TCCN

Date: 2026-05-29

## Architecture thesis for the TCCN rewrite

The paper should now be written around **LearnRisk-QRS** rather than
around MA-HPPO. The core semantic-communication innovation is a
role-split VQA control system:

- **Slow UAV layer:** handles mobility, association/channel reuse, and
  visual-symbol budgets with constrained multi-agent control because one
  UAV decision affects many downstream questions.
- **Fast UE/query layer:** handles text-symbol and UE-power adaptation
  with QRS utility, contextual bandit/neural action selection, and
  PDQN-style parameterized actions because each decision is local to one
  VQA query.
- **Typed semantic-price interface:** sends only UAV-by-question QoS
  pressure, semantic slack, and LUT marginal gain across the layer
  boundary; this is the VQA-native communication primitive.
- **MoE LearnRisk safety layer:** separates a nominal expert, a
  channel-stress expert, and a shared learned-risk head so nominal
  VQA gains are not erased by conservative stress training.

This is the self-owned algorithmic architecture. MA-HPPO/CHPPO remain
diagnostic baselines and theoretical scaffolding; they should not be
presented as the final TCCN performance method unless new evidence
overturns the current Greedy/QRS gate.

## Top-5 completion state

| Fix | Current artifact | Status | Evidence still required |
|---|---|---|---|
| 1. Final method identity | title/abstract/conclusion now use LearnRisk-QRS; §IV has role-specific algorithm selection | local manuscript fix | One more whole-paper audit after server results replace smoke tables |
| 2. Greedy-vs-QRS gate | `run/qrs_tccn_gate.py` and local mini CSV/table | local smoke | Server multi-seed, longer-episode Greedy/TypedPrice/LearnRisk/MoE table |
| 3. Stress/jamming robustness | `run/qrs_robustness_sweep.py` over channel-loss offsets | script-ready + tiny smoke | Server 0/5/10/15 dB sweep with Mixed-MoE enabled and confidence intervals |
| 4. Component/MoE ablation | component toggles, MoE LearnRisk ablation script/table | local smoke | Multi-seed per-question ablation and learned-router as secondary negative control |
| 5. Scaling/large-UE | LearnRisk M,N smoke grid and `run/qrs_large_ue_sweep.py` | script-ready + tiny smoke | Larger fixed-M UE-pressure sweep, preferably N=4/6/8/10 with Greedy/TypedPrice/MoE |

Blocking evidence gap as of 2026-05-30: the validation server
`lab-s2:/home/qiankun/HPPO-VQA` is missing the
LearnRisk-QRS code path. Do not treat the local smoke tables as final
TCCN evidence until the manifest is synced and the remote gates pass.
The concrete server execution sequence is maintained in
`plan/qrs-remote-experiment-runbook.md`.
The VQA-specific system and algorithm innovation map is maintained in
`plan/vqa-semantic-innovation-matrix.md`.
The claim threshold for promoting a result from local smoke to
submission evidence is maintained in
`plan/tccn-submission-evidence-contract.md`, with an executable checker
at `hppo-uav/run/qrs_tccn_evidence_contract.py`.
The self-owned UAV/UE algorithm composition is specified in
`plan/qrs-algorithm-architecture-spec.md`.
The final table/caption replacement path after F2-F5 remote validation
is specified in `plan/tccn-final-results-insertion-plan.md`.

Update 2026-05-30:

- Implemented the first QRS environment hook behind
  `semantic_state_features=True`.
- Default observations remain backward-compatible with the MA-HPPO /
  MA-CHPPO configs.
- When enabled, the env appends and exposes:
  `question_type`, `semantic_margin`, `lut_local_gain`,
  `deadline_slack`, `uav_semantic_load`, and flattened
  `semantic_control_features`.
- Focused test coverage added in `hppo-uav/tests/test_uavnet_env.py`;
  `python3 -m pytest hppo-uav/tests/test_uavnet_env.py -q` passes
  with 11 tests.
- Added an executable UE/query prototype:
  `dizoo/gym_hybrid/baselines/qrs_semantic.py` and
  `run/eval_qrs_semantic.py`.
- Smoke evidence with 2 episodes / 1 seed:
  QRS prototype gets raw cost 6.45, mean similarity 0.600, mean min
  similarity 0.545, semantic-floor rate 0.855; Greedy gets raw cost
  4.02 and mean similarity 0.758. This prototype is therefore a
  runnable interface, not yet an improvement.
- Updated the UE/query prototype to include `mode="utility"`, which
  jointly enumerates `(K_usr, UE power)` and scores candidates by a
  local VQA-aware utility. Then extended the utility mode to the slow
  UAV semantic layer: each UAV chooses its own `K_uav` by aggregating
  the utilities of its associated UEs.
- Clean 100-episode / 3-seed local comparison after fixing
  backward-compatible RNG behavior:
  - Greedy-current: raw cost 4.187, energy 7.375, mean similarity 0.756,
    min similarity 0.647, semantic-floor rate 0.995.
  - QRS-utility: raw cost 3.722, energy 6.444, mean similarity 0.751,
    min similarity 0.654, semantic-floor rate 0.950.
  This is a cost-saving signal, not a full dominance result: QRS reduces
  raw cost by about 11.1% and slightly improves min similarity, at the
  price of lower average similarity and lower floor-rate.
- Added an executable `mode="dual_utility"` hook for semantic-price
  exchange between the slow UAV layer and fast UE/query layer. Each UAV
  updates a non-negative price from question-conditioned semantic-floor
  violations, and UEs include that price in the local symbol/power
  utility. The default M2N4 smoke point currently matches `utility`,
  so this is an architecture hook and ablation target, not yet a new
  performance claim.
- Added `run/qrs_price_sweep.py` and question-conditioned metrics
  (`question_floor_rate`, `semantic_price_mean`, `semantic_price_max`)
  to the QRS evaluator. A small M2N4 sweep over 10 episodes x 2 seeds
  with `floor_penalty=0` shows that price exchange can change decisions
  without increasing cost: at base floor 0.50, dual utility improves
  mean similarity by 0.0015, min similarity by 0.0061, and question-floor
  rate by 0.0063; at base floor 0.85, it improves mean similarity by
  0.0016 and min similarity by 0.0065, with no cost/energy change. This
  is still sensitivity evidence, not a final TCCN-scale result.
- Added a default-off `channel_loss_offset_db` stress knob to emulate
  deployment-time channel degradation / jamming-like loss without
  changing the nominal environment. A 5-episode M2N4 stress sweep over
  offsets 0/5/10/15 dB shows three regimes:
  - 0--10 dB: semantic price gives small similarity or floor-rate gains
    without changing cost at base floor 0.50.
  - 15 dB: the semantic link is unrecoverable and price saturates.
  - 10 dB plus base floor 0.85: price saturates and can explode cost,
    so the final algorithm needs trust-region damping, anti-windup
    price clipping, and an ablation over price stability.
- Implemented `mode="stable_dual_utility"` with two stabilizers:
  per-step price trust region (`price_delta_max`, default 0.05) and an
  anti-windup cap on the stored/effective semantic price
  (`price_effect_max`, default 1.0).
  On the known unstable point (10 dB channel loss, base floor 0.85,
  `floor_penalty=0`), raw `dual_utility` raises mean similarity by
  0.0014 but increases raw cost by 3.71e6. `stable_dual_utility` keeps
  a 0.0011 similarity gain with zero cost/energy increase by capping
  stored/effective price at 1.0. This is the first concrete algorithmic
  architecture improvement beyond a diagnostic heuristic.
- Extended the same stress check to N=6. At 10 dB channel loss and base
  floor 0.85, raw `dual_utility` increases mean similarity by 0.0028
  but explodes raw cost by 8.06e6. `stable_dual_utility` keeps a 0.0017
  similarity gain with zero cost/energy increase. The instability grows
  with UE count, which strengthens the case that stable semantic-price
  control is a necessary architecture component rather than a cosmetic
  hyperparameter tweak.
- Added `paper/data/summarize_qrs_price_ablation.py`, which generates
  `paper/data/qrs_stable_price_ablation.csv` and
  `paper/tables/qrs_stable_price_ablation.tex` from the M2N4/M2N6 sweep
  JSON files. The table is now included in Section VI as the first
  executable role-split price-stability ablation.
- Added QRS component-ablation switches for the next TCCN experiment:
  `question_conditioned_floors=False` disables question-type floors,
  `uav_role_split=False` keeps Greedy's UAV symbol selection, and
  `mode="utility"` removes semantic-price exchange. The new
  `run/qrs_component_ablation.py` script writes JSON/CSV/LaTeX outputs.
  In a 5-episode M2N4 stress smoke run at 10 dB loss and base floor
  0.85, disabling UAV role split raises raw cost from 5.30e5 to 4.24e6
  with nearly unchanged similarity, while removing price slightly lowers
  mean similarity. This is interface/smoke evidence, not final
  submission-scale ablation.
- Added a trainable UE/query layer:
  `dizoo/gym_hybrid/baselines/qrs_bandit.py` implements a tabular
  contextual bandit over `(K_usr, UE power)` with contexts built from
  question type, serving UAV, UAV symbol, UAV/UE SINR bins, semantic
  price bin, and deadline bin. The bandit is initialized from the
  deterministic QRS utility as a prior, then updated online from
  per-query similarity / floor-loss rewards. The smoke command
  `run/train_qrs_bandit.py` writes `paper/data/qrs_bandit_smoke.json`
  and `paper/data/qrs_bandit_smoke_model.json`. Current smoke results
  are negative under the high-floor 10 dB stress point: the bandit is an
  executable trainable interface, not a performance claim yet.
- Added a neural UE/query fast-layer interface:
  `dizoo/gym_hybrid/baselines/qrs_neural_bandit.py` implements an MLP
  action selector over QRS context features, and
  `run/train_qrs_neural_bandit.py` offline-distills it from typed-QRS
  teacher rollouts. The first tiny smoke run collects 240 UE/query
  samples and writes `paper/data/qrs_neural_bandit_smoke.json` plus
  `paper/data/qrs_neural_bandit_smoke_model.json`. Evaluation collapses
  under the high-floor 10 dB stress point, so this is interface evidence
  only. Added a default safety gate: execute the MLP proposal only when
  its local QRS utility is no worse than the deterministic QRS action.
  The safe smoke recovers mean similarity from ~1e-6 to 0.1796 but uses
  fallback on all decisions, proving the guard rather than learned
  dominance. Next step: reduce fallback rate with constrained imitation
  or QRS-advantage-weighted distillation before claiming learned
  fast-layer performance.
- Added question-type resolved evaluation to `run/eval_qrs_semantic.py`
  and `run/qrs_question_type_ablation.py`. The smoke diagnostics compare
  full QRS against `question_conditioned_floors=False` for object,
  attribute, counting, spatial, and yes-no questions. After adding
  action-level question utility/resource weights and a positive local
  LUT marginal-gain bonus, the nominal smoke run now shows small
  per-type similarity gains (largest on spatial questions, about
  +1.49e-3) at the cost of higher raw cost (8.54 -> 9.28). The high-floor
  10 dB stress point still shows no recoverable per-type similarity
  difference. This means question type is now coupled to both utility
  and local LUT slope, while type-specific semantic prices or learned
  UE/query policy weights are still needed for stressed channels.
- Added `mode="typed_stable_dual_utility"` in `qrs_semantic.py`.
  This keeps the stable anti-windup price update but changes the stored
  semantic price from one scalar per UAV to a UAV-by-question-type
  matrix. A small high-floor 10 dB smoke run shows the typed variant
  matches the stable scalar price's similarity gain (+7.41e-4 over
  utility) at identical raw cost, while lowering mean stored price
  (0.891 -> 0.319) and mean effective UE price (0.954 -> 0.469). This is
  a useful interface/diagnostic result, not yet a final performance
  claim. The summarizer `paper/data/summarize_qrs_typed_price.py`
  converts the JSON into `paper/data/qrs_typed_price_smoke.csv` and
  `paper/tables/qrs_typed_price_smoke.tex`.
- Added `run/dump_qrs_rollout.py` for real CDF/trajectory artifacts.
  It exports per-task similarity, question floor, semantic slack,
  question type, serving UAV, selected symbols/channels, UE power,
  semantic price, SINR, and UAV/UE positions. A nominal typed-QRS smoke
  dump now provides 240 per-task records with mean similarity 0.822 and
  Q-floor rate 0.917; the stress dump provides the matching long-tail
  failure diagnostic with mean similarity 0.180 and Q-floor rate 0.050.
  These files replace the previous "no rollout dump yet" blocker for
  Fig. 5/Fig. 6, though they are not submission-scale. Added
  `paper/figures/_make_qrs_rollout_figures.py`, which regenerates
  `fig5_trajectory.pdf` and `fig6_similarity_cdf.pdf` directly from
  those rollout CSVs.
- Added normalized QRS-utility-regret distillation to
  `run/train_qrs_neural_bandit.py`. During teacher rollout collection,
  the script now records each candidate UE/query action's local utility
  regret relative to the deterministic QRS optimum and trains the MLP
  with an optional probability-weighted regret loss. On the same
  high-floor 10 dB smoke point, this reduces the safety-gate fallback
  rate from 1.0000 to 0.1083 while preserving the guarded similarity
  level (0.1796). This is a real fast-layer training improvement, but
  still not a final learned-performance claim because similarity and
  floor-rate do not improve over the guarded teacher fallback.
- Extended the neural fast-layer safety interface with two additional
  gates: a tunable QRS utility margin (`--safety-score-margin`) and a
  semantic-floor risk gate (`--safety-mode semantic_floor`). Nominal
  smoke diagnostics reveal the useful tension: the unsafe learned MLP
  reaches mean similarity 0.8420 at Q-floor 0.95, above the guarded
  typed-QRS operating point, but the same unsafe policy collapses under
  the high-floor 10 dB stress point. Soft utility gating reduces nominal
  fallback from 0.3750 to 0.0083 and does not reintroduce stress
  collapse, but it still does not recover the unsafe nominal gain. This
  gives a sharper next algorithm target: learn a VQA-aware risk
  predictor or calibrate the safety gate from rollout-level outcomes,
  rather than relying only on local QRS utility.
- Added an explicit VQA floor-risk distillation loss
  (`--floor-risk-weight`) to the neural UE/query fast-layer trainer.
  Each replay sample now stores a candidate-action mask indicating
  whether the action is predicted to miss the question-conditioned
  semantic floor, and the MLP can be trained to put less probability on
  those actions. Floor-risk distillation alone does not prevent stress
  collapse when the safety gate is disabled, but combined with soft QRS
  utility gating it reduces stress fallback from 0.1083 to 0.0417 while
  preserving the guarded similarity and slightly improves the nominal
  guarded point (0.8189 -> 0.8214 mean similarity). This supports the
  direction of a VQA-aware risk-calibrated fast layer, while still
  showing that final learned performance needs a stronger rollout-level
  risk predictor.
- Tested a stricter semantic-floor improvement gate
  (`--safety-mode semantic_floor_improve`) that only allows fallback when
  the teacher action has no lower predicted similarity than the neural
  proposal. On the current smoke points it matches the plain
  `semantic_floor` gate rather than recovering the unsafe nominal gain:
  nominal mean similarity remains 0.8199 and stress mean similarity
  remains 0.1796. This negative result narrows the problem: nominal
  loss is not simply caused by the safety gate replacing higher-similarity
  neural actions; the learned fast layer needs better rollout-level risk
  calibration or richer features before it can keep the unsafe nominal
  gain safely.
- Added a separate learned risk head to the neural UE/query fast layer
  (`--risk-head-weight`, `--safety-mode learned_risk`). Unlike the
  floor-risk action penalty, this keeps the action logits free to retain
  nominal learned gains while training an auxiliary per-action predictor
  for whether the candidate misses the question-conditioned floor. This
  is the first smoke variant that preserves the unsafe nominal gain
  (mean similarity 0.8430, Q-floor 0.95, fallback 0) and still prevents
  high-floor 10 dB stress collapse (mean similarity 0.1796, Q-floor
  0.05, fallback 0.1000). RiskSoft has lower stress fallback (0.0417)
  but loses the nominal gain. This supports the final architecture
  direction: UE/query fast control should use a separate VQA risk
  calibrator rather than making the action policy itself conservative.
- Added `run/qrs_learned_risk_sweep.py` to calibrate the learned-risk
  threshold. A smoke sweep over `risk_threshold in {0.30,0.50,0.70}`
  gives identical behavior at the current nominal/stress points:
  nominal mean similarity 0.8430 with Q-floor 0.95 and no fallback, and
  stress mean similarity 0.1796 with Q-floor 0.05 and fallback 0.10.
  This suggests a stable risk-ranking plateau in the learned risk head,
  rather than a brittle one-threshold artifact. The table is written to
  `paper/tables/qrs_learned_risk_threshold_sweep_smoke.tex`.
- Ran a slightly larger three-seed learned-risk follow-up at
  `risk_threshold=0.50`. The nominal setting uses 5 collect episodes and
  5 eval episodes per seed, giving mean similarity 0.8435 +/- 0.0006,
  mean min similarity 0.7151 +/- 0.0012, Q-floor 0.95, and fallback 0.
  The high-floor 10 dB stress setting uses 3 collect episodes and 4 eval
  episodes per seed, preserving mean similarity 0.1796 and Q-floor 0.05
  with fallback 0.1639 +/- 0.0660. This is still small, but it moves
  LearnRisk from a single-seed smoke point toward reproducible
  calibration evidence.
- Added `run/qrs_learned_risk_scaling.py`, a real LearnRisk scaling
  smoke grid over `M in {1,2}` and `N in {3,4,6}`. This regenerates
  `paper/figures/fig4_scaling.pdf` from executable role-split runs and
  writes `paper/data/qrs_learned_risk_scaling_smoke.csv` plus
  `paper/tables/qrs_learned_risk_scaling_smoke.tex`. The six-cell smoke
  grid keeps Q-floor rates between 0.9625 and 1.0 with zero fallback in
  nominal conditions. It replaces the previous mock Fig. 4 with
  smoke-real evidence, though it remains far from submission-scale.
- Reframed the paper's front matter around a self-owned semantic
  communication architecture, **LearnRisk-QRS**, instead of presenting
  MA-HPPO / MA-CHPPO as final winners. The new main idea is:
  slow UAV constrained control + fast UE/query semantic adaptation +
  typed UAV-by-question semantic price + learned VQA-risk veto.
  This directly addresses the user requirement to innovate the VQA
  semantic system and algorithm architecture without depending on a
  single borrowed hybrid-RL template.
- Added the learned-risk gate equations to the methodology:
  the action head proposes UE/query symbol-power actions, while a
  separate risk head predicts whether each candidate misses the
  question-conditioned floor. This separation is the core algorithmic
  novelty: the action head can keep nominal VQA gains, and the risk
  head only vetoes stressed-channel unsafe shortcuts.
- Added `run/qrs_tccn_gate.py`, a common local/server entry point for
  the five pre-submission fixes. It runs a compact nominal/stress gate
  over Greedy, QRS, typed semantic price, and LearnRisk, then writes
  `paper/data/qrs_tccn_gate_smoke.csv` and
  `paper/tables/qrs_tccn_gate_smoke.tex`. The default budget is smoke
  scale; server runs should raise `--n-seeds`, `--eval-episodes`,
  `--collect-episodes`, and `--max-steps`.
- Checked the validation server `lab-s2`. The existing
  `/home/qiankun/HPPO-VQA` directory is not a git repository and lacks
  the new LearnRisk-QRS files. System `python3` lacks `numpy`; the
  usable interpreter is `/home/qiankun/.conda/envs/DI-engine/bin/python`.
  Added `hppo-uav/run/REMOTE_QRS_GATE.md` with the exact sync list and
  smoke/submission-scale gate commands. A structure-preserving rsync
  requires explicit user approval before uploading the local code bundle.
- Strengthened `run/qrs_tccn_gate.py` by adding stress Greedy and stress
  plain-QRS rows. A local mini gate
  (`--n-seeds 2 --eval-episodes 3 --collect-episodes 3 --max-steps 30
  --epochs 20`) now writes
  `paper/data/qrs_tccn_gate_local_mini.csv` and
  `paper/tables/qrs_tccn_gate_local_mini.tex`.
  Key signal:
  - nominal Greedy: cost 18.14, similarity 0.8070, Q-floor 0.9667
  - nominal LearnRisk: cost 22.79, similarity 0.8405, Q-floor 0.9500
  - stress Greedy: cost 1.08e7, similarity 0.1207, Q-floor 0.1583
  - stress QRS/TypedPrice: cost 1.85e6, similarity about 0.179,
    Q-floor 0.0500
  - stress LearnRisk: cost 1.85e6, similarity 0.0898, Q-floor 0.0250
  The algorithmic reading is sharp: LearnRisk improves the nominal
  fast-layer semantic point, but a risk head trained per-scenario is not
  yet a cross-scenario VQA safety calibrator. The next method step should
  train/evaluate the risk head on mixed nominal+stress rollouts so it
  learns channel-regime-aware risk rather than a brittle local gate.
- Added mixed-scenario teacher collection to
  `run/train_qrs_neural_bandit.py`:
  `--collect-base-floor-list`, `--collect-floor-penalty-list`, and
  `--collect-channel-loss-offset-db-list`. Also added
  `--learnrisk-mixed-collect` to `run/qrs_tccn_gate.py`. A local
  two-seed mini check with mixed nominal+stress teacher rollouts shows:
  - mixed LearnRisk nominal: cost 22.79, similarity 0.8200,
    Q-floor 0.9167, fallback 0.0833
  - mixed LearnRisk stress: cost 1.85e6, similarity 0.1796,
    Q-floor 0.0500, fallback 0.1958
  This recovers the stress similarity of typed/QRS and fixes the
  single-scenario LearnRisk stress collapse, but loses the nominal
  semantic gain. The next calibrator should use scenario-balanced
  weighting or a two-head regime-aware risk gate so nominal gains and
  stress safety are both retained.
- Added an explicit channel-regime feature to the neural UE/query
  fast-layer context:
  `channel_loss_offset_db / 20` in
  `dizoo/gym_hybrid/baselines/qrs_neural_bandit.py`. Also added
  `regime_learned_risk` mode with nominal and stress risk thresholds
  (`--risk-threshold`, `--risk-threshold-stress`,
  `--risk-regime-loss-db`). The local mini check is negative:
  - regime-threshold mixed nominal remains at similarity 0.8200 and
    Q-floor 0.9167
  - disabling nominal fallback by setting threshold 1.1 collapses
    nominal similarity to 0.399, proving the mixed action head itself
    loses the nominal policy rather than merely being over-vetoed
  - adding the explicit channel-regime feature keeps stress at 0.1796
    but does not recover nominal gain and raises fallback frequency
  The conclusion is now more precise: the final UE/query fast layer
  should be a regime-aware mixture-of-experts or two-head policy
  (nominal-gain expert + stress-safety expert) with a learned router,
  not a single action head with one auxiliary risk head.
- Implemented the first executable version of that idea: `--use-moe`
  in `run/train_qrs_neural_bandit.py` and `--learnrisk-use-moe` in
  `run/qrs_tccn_gate.py`. The policy now has a nominal action expert,
  a stress action expert, a shared VQA risk head, and a simple channel
  regime router keyed by `channel_loss_offset_db`. Local two-seed mini
  evidence:
  - single-scenario LearnRisk: nominal 0.8405 / stress 0.0898
  - mixed single-head LearnRisk: nominal 0.8200 / stress 0.1796
  - mixed MoE LearnRisk: nominal 0.8341 / stress 0.1796
  This is the first positive architecture-level trade-off: the MoE fast
  layer keeps most of the nominal semantic gain while retaining the
  stress safety of mixed training. It should become the final UE/query
  algorithm candidate, with a learnable router replacing the current
  threshold router in the submission-scale version.
- Implemented a first learned-router MoE variant
  (`--use-learned-router`, `--router-loss-weight`). The router is trained
  to predict whether a sample comes from the stress channel regime and
  blends nominal/stress expert logits. The mini result is negative:
  threshold-router MoE gets nominal/stress similarity 0.8341/0.1796,
  while learned-router MoE gets 0.8212/0.1796 and raises nominal fallback
  to 0.5333. For the next submission-scale run, keep the fixed
  channel-regime router as the main LearnRisk-QRS candidate; revisit the
  learned router only after adding more router supervision or a larger
  mixed dataset.
- Added `run/qrs_moe_learnrisk_ablation.py` as the reproducible
  Single-vs-Mixed-vs-MoE LearnRisk entry point. It reruns nominal and
  stress variants, regenerates `paper/data/qrs_moe_learnrisk_mini.csv`,
  and writes `paper/tables/qrs_moe_learnrisk_mini.tex`. This closes the
  "manual command stitching" gap for the current MoE evidence, but not
  the final TCCN evidence gap: the table still needs a server-side
  multi-seed, longer-episode run before it can be used as the main
  submission claim.
- Added `run/qrs_robustness_sweep.py` as the executable replacement for
  the old mock robustness and jamming panels. It sweeps
  `channel_loss_offset_db` as a shared channel-drift / jamming-equivalent
  degradation axis and reports Greedy, TypedPrice-QRS, and fixed-router
  Mixed-MoE LearnRisk rows. This turns the TCCN robustness requirement
  into a single server command; the local/default budget is intentionally
  smoke-scale.
- Added `run/qrs_large_ue_sweep.py` as the executable replacement for
  the unsupported large-UE table. It sweeps `N` at fixed `M` and reports
  Greedy, TypedPrice-QRS, and optional fixed-router Mixed-MoE LearnRisk.
  The script is framed as a large-UE pressure/scaling gate, not a
  zero-shot transfer claim.
- Added `run/qrs_tccn_readiness_audit.py`, a machine-checkable audit of
  the top-5 gates. It marks local smoke evidence separately from
  server-ready evidence so the paper cannot accidentally treat smoke
  results as final TCCN claims.
- Added `run/qrs_remote_sync_manifest.py` to generate a local hash
  manifest and exact `rsync -azR` command for the server validation
  files. This keeps remote synchronization reproducible once upload is
  authorized.

## 0. Top-5 TCCN Fixes Now Being Tracked

| Fix | Current executable artifact | Still needed before submission |
|---|---|---|
| F1. Final method identity | `\qmethod` / LearnRisk-QRS in title, abstract, methodology, conclusion | Remove remaining MA-HPPO-as-main language across the whole paper |
| F2. Greedy-vs-QRS cost/fidelity gate | `run/qrs_tccn_gate.py`, `eval_greedy.py`, `eval_qrs_semantic.py` | Multi-seed, >=100 episode final table |
| F3. Stress safety under channel loss/jamming | `run/qrs_robustness_sweep.py`, LearnRisk nominal/stress sweeps, fixed-router MoE risk gate | Run server multi-seed jamming/channel-loss sweep and UE-count stress |
| F4. Component ablation | QRS toggles, LearnRisk variants, `run/qrs_moe_learnrisk_ablation.py` | Submission-scale ablations with per-question metrics |
| F5. Scaling/generalization | LearnRisk M,N smoke grid, rollout dump, `run/qrs_large_ue_sweep.py` | Server multi-seed larger M,N / large-UE pressure sweep and final baselines |

## 1. Current Evidence

The current local evidence does not support a submission claim that the
monolithic MA-HPPO or current MA-CHPPO checkpoints beat the LUT-grade
Greedy reference. Verified M2N4 evaluation logs show:

| Method | Raw joint cost | Energy | Mean similarity | Conference SR |
|---|---:|---:|---:|---:|
| Greedy LUT | 3.64 | 6.29 | 0.860 | 0 |
| MAPPO-hybrid | 5.85e4 ± 1.18e4 | 1.17e5 ± 2.37e4 | 0.601 ± 0.021 | 0 |
| MA-HPPO | 1.18e5 ± 1.50e4 | 2.36e5 ± 2.99e4 | 0.559 ± 0.003 | 0 |
| PADDPG | 1.22e5 ± 1.40e5 | 2.45e5 ± 2.80e5 | 0.284 ± 0.115 | 0 |
| MA-CHPPO | 1.34e6 ± 2.80e4 | 2.69e6 ± 5.60e4 | 0.298 ± 0.110 | 0 |

Conclusion: the paper should stop claiming that a flat hybrid PPO
solves the system. The strong contribution should be reframed as a
semantic-system redesign motivated by this failure.

## 2. External Landscape

Recent nearby work narrows the novelty space:

- DSC-UAV proposes context-adaptive digital semantic communication for
  UAV networks, with prompt-aware visual encoding and TQC-based
  trajectory/resource optimization for semantic-structural similarity
  and AoI. Source: arXiv 2601.01430 / Context-Aware Information
  Transfer via Digital Semantic Communication in UAV-Based Networks.
- Relay-aware semantic resource allocation for multi-hop UAV swarm
  systems optimizes semantic distortion over relay routes and discusses
  relay-drift / consistency regularizers for reconstructed imagery.
  Source: Physical Communication 2026 pre-proof.
- Latency-aware human-in-the-loop RL for semantic communications uses a
  constrained MDP and primal-dual PPO with latency-aware shaping in a
  semantic-aware Open RAN setting. Source: arXiv 2602.15640.

Implication: "semantic UAV + RL" and "constrained semantic RL" are no
longer enough. The novelty must be task-specific to VQA and
architecture-specific to UAV/UE role separation.

## 3. Proposed New Core Contribution

Working title:

**Question-Conditioned Role-Split Semantic Control for Multi-UAV VQA
Offloading**

Core claim:

Instead of treating VQA semantic offloading as generic image-feature
transmission, model it as a question-conditioned semantic-control
system. The controller should decide which visual/question semantics
must be preserved, which UAV should shape the channel/association
context, and which UE should adapt symbol/power decisions for the
current query.

This avoids borrowing an existing semantic system wholesale. The
DeepSC-VQA LUT becomes only a measurement oracle; the system
architecture around it is ours.

## 4. Semantic Communication System Rebuild

### 4.1 Question-Conditioned Semantic State

Current state: SINR, positions, channel gain, symbol indices, channel
selectors, step index.

New state additions:

- Question type embedding: object / attribute / counting / spatial /
  yes-no.
- Query urgency / deadline slack.
- Per-UE semantic margin: `mean_similarity - xi_th`.
- LUT local slope: marginal gain from increasing `K_uav`, `K_usr`, or
  SNR bin.
- Serving-UAV semantic load: aggregate margin and interference pressure
  over UEs associated with each UAV.

Why this is VQA-specific:

VQA does not need uniform visual reconstruction. Counting and spatial
questions need different semantic robustness than yes-no or attribute
questions. The semantic resource should be allocated by question
utility, not only by image SNR.

### 4.2 Semantic Action Decomposition

Separate the flat hybrid action into two coupled layers:

| Layer | Agent | Decisions | Timescale | Suggested algorithm |
|---|---|---|---|---|
| Network layer | UAV | trajectory, UE association, channel reuse, jamming avoidance | slow | constrained PPO / HAPPO-style role update |
| Task layer | UE/query | `K_usr`, power, optional query priority, semantic retry | fast | contextual bandit / PDQN / lightweight actor |

The UAV layer should minimize network-level semantic cost and enforce
QoS. The UE layer should maximize per-query semantic utility under the
current serving-UAV context.

### 4.3 Semantic Price Exchange

Exchange only three summaries between layers:

1. `lambda_sem`: global or per-UAV QoS shadow price.
2. `delta_xi`: predicted similarity slack for each UE/query.
3. `marginal_gain`: LUT-derived gain of one more symbol or SNR bin.

This is the publishable architecture: a communication-efficient
semantic-control protocol, not just a bigger neural network.

## 5. Algorithm Architecture

Name candidates:

- RS-SemHPPO: Role-Split Semantic Hybrid PPO
- QRS-CHPPO: Question-conditioned Role-Split Constrained HPPO
- DuoSem-RL: Dual-layer Semantic Reinforcement Learning

Recommended final name:

**QRS-CHPPO**

Architecture:

1. UAV policy `pi_U`:
   - Inputs: UAV positions, UE clusters, channel state, jamming state,
     aggregate semantic margins, lambda price.
   - Outputs: movement, association/channel plan, UAV symbol budget.
   - Algorithm: constrained PPO with per-UAV role heads. HAPPO-style
     sequential update is optional if UAV roles become heterogeneous.

2. UE/query policy `pi_Q`:
   - Inputs: serving UAV, question type, LUT local slope, channel/SINR
     bin, deadline slack.
   - Outputs: UE symbol cardinality and power.
   - Algorithm: contextual bandit if only one-step symbol/power
     adaptation is needed; PDQN if symbol choice and continuous power
     remain coupled.

3. Coordinator:
   - Updates lambda from semantic-constraint violation.
   - Broadcasts semantic price and per-UAV load.
   - Does not centrally choose every action at execution time.

Training objective:

`r = - normalized_delay_energy_cost - lambda * max(0, xi_th - xi_mean)`

but compute the constraint at two levels:

- per-query constraint for UE/query fairness;
- per-UAV aggregate constraint for association and interference control.

## 6. New Experiments Needed

Minimum TCCN-ready set:

1. Main comparison:
   QRS-CHPPO vs Greedy, MA-HPPO, MAPPO-hybrid, PADDPG, PDQN.
   Gate: match or beat Greedy cost while maintaining mean similarity
   above 0.5 and nonzero per-query satisfaction.

2. Ablation:
   - no question type;
   - no LUT local slope;
   - no semantic price exchange;
   - monolithic controller instead of role split.

3. Robustness:
   - channel perturbation;
   - UE-count generalization;
   - jamming;
   - question-type distribution shift.

4. Fairness:
   Report 5th percentile similarity, not only mean similarity.

5. Cost decomposition:
   Separate propulsion, UAV transmission, UE transmission, and delay.

## 7. Paper Reframe

Introduction contribution bullets should become:

1. VQA-specific semantic offloading system with question-conditioned
   semantic state and LUT-derived marginal utility.
2. Diagnostic showing why monolithic hybrid PPO fails under multi-UAV
   semantic interference and saturated delay-energy reward.
3. QRS-CHPPO role-split architecture: slow UAV constrained control plus
   fast UE/query semantic adaptation.
4. Conditional trust-region / primal-dual analysis for the role-split
   controller.
5. Robustness suite including question-distribution shift.

What to avoid:

- Do not claim "first UAV semantic RL".
- Do not claim "new constrained PPO" without qualification.
- Do not claim current MA-CHPPO results are positive.
- Do not make Greedy look like a weak baseline; it is currently the
  strongest evidence anchor.

## 8. Immediate Implementation Path

1. Add question-type and LUT-slope fields to env `info` and observation.
   **Status: done as optional `semantic_state_features=True`.**
2. Implement a UE/query contextual policy module first; keep UAV Greedy
   or fixed trajectory to isolate semantic gains.
   **Status: partially done as deterministic QRS utility, component
   toggles, and a trainable tabular contextual bandit. The current
   bandit smoke run is negative, so it should be treated as an
   interface for the next learned UE/query module rather than the final
   algorithm.**
3. Add UAV constrained PPO after the UE module beats Greedy on
   similarity/cost trade-off in fixed topology.
4. Only then rerun full M2N4 and scaling sweeps.

## 9. Minimal UE/Query Controller Prototype

Before training a full QRS-CHPPO stack, implement a deterministic or
contextual-bandit UE/query controller with fixed UAV behavior:

Inputs per UE/query:

- one-hot `question_type`;
- current `semantic_margin`;
- `lut_local_gain[:, 0:3]`;
- `deadline_slack`;
- serving UAV id from `ue_to_uav`;
- current channel/SINR bin.

Actions:

- choose UE symbol index `K_usr`;
- choose UE transmit power;
- optionally request one-step UAV symbol increase from the slow layer.

Reward:

- positive for increasing 5th-percentile similarity above
  `xi_th`;
- negative for UE energy and delay;
- large penalty for negative semantic margin when deadline slack is low.

Acceptance gate for this prototype:

- On fixed `(M,N)=(2,4)` with existing UAV behavior, the UE/query layer
  should improve either raw joint cost at similar mean similarity or
  mean/5th-percentile similarity at similar cost relative to Greedy.
  If it cannot improve either axis, the role-split story needs a
  stronger semantic encoder or question-type-resolved LUT before it can
  become a TCCN submission.

Current status:

- A deterministic baseline has been implemented:
  `QuestionAwareSemanticPolicy` keeps Greedy's UAV layer and chooses the
  smallest UE symbol index predicted to clear a question-conditioned LUT
  floor.
- The threshold-only mode is negative against Greedy. The issue is that
  minimizing UE symbol cardinality alone reduces similarity too much while
  Greedy's max-power / trajectory behavior already keeps raw cost low.
- The utility mode now searches over `(K_usr, UE power)` jointly and
  also lets each UAV choose `K_uav` from the aggregate utility of its
  associated UEs. It optimizes a local utility
  `utility = similarity - beta_energy * UE_energy - beta_delay * delay`,
  with question-type-specific similarity floors. In smoke tests, this
  produces a cost-saving semantic-control angle: lower raw cost and
  slightly higher min similarity, but not higher mean similarity or
  floor-rate.
- The `dual_utility` mode adds a minimal primal-dual interface:
  UAV-level semantic prices are updated from question-conditioned floor
  violations and then broadcast to UE/query controllers. Smoke tests
  show that the default fixed floor penalty masks the extra price term.
  An isolated `floor_penalty=0` sweep confirms the price channel can
  change decisions: dual utility improves mean similarity by about
  0.0015 and min similarity by about 0.006 with no cost/energy change.
  The next experiment should scale this to more UEs, stronger semantic
  floors, and channel/jamming pressure to see whether the effect becomes
  submission-relevant.
- Component switches now expose the TCCN ablation axes directly:
  question-conditioned floors, UAV role-split symbol selection, and
  semantic-price exchange. The first smoke ablation at 10 dB loss shows
  that the UAV role-split symbol layer is cost-critical under stress,
  while question and price effects need larger question-type-resolved
  runs to become final claims.
- A first trainable UE/query contextual bandit now exists. It uses the
  deterministic utility scores as Q-value priors to avoid cold-start
  low-power collapse, then updates from per-query rollout feedback. The
  high-floor stress smoke run learns 97 contexts and 1600 UE updates,
  but eval similarity remains too low for a claim. Next algorithm step:
  replace tabular bandit with a small neural contextual bandit or PDQN
  UE/query learner and add offline replay from QRS rollouts.
- Question-type diagnostics exposed a design gap and drove an update:
  question type now enters floor thresholds, action-level utility /
  resource weights, and a local LUT marginal-gain reward. Nominal smoke
  evidence shows weak but measurable per-type similarity shifts, while
  channel-stress evidence remains negative. Next VQA-specific redesign:
  type-specific semantic prices and learned UE/query policies, e.g.
  counting/spatial questions should price UAV visual symbols more
  aggressively than yes-no questions under channel stress.

Remote reproducibility command:

```bash
ssh lab-s2 \
  'cd ~/HPPO-VQA/hppo-uav && \
   PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid \
   python3 run/eval_qrs_semantic.py --M 2 --N 4 --n-episodes 100 \
     --n-seeds 3 --mode utility --out ~/HPPO-VQA/runs/qrs_utility_M2N4_eval.json'
```

Price-exchange sensitivity smoke command:

```bash
python3 run/qrs_price_sweep.py --M 2 --N 4 --n-episodes 10 \
  --n-seeds 2 --max-steps 80 --floors 0.50,0.85 \
  --modes utility,dual_utility,stable_dual_utility \
  --floor-penalty 0.0 --channel-loss-offsets 0,5,10,15 \
  --out ~/HPPO-VQA/runs/qrs_price_sweep_penalty0_M2N4_smoke.json
```

N=6 scaling smoke command:

```bash
python3 run/qrs_price_sweep.py --M 2 --N 6 --n-episodes 5 \
  --n-seeds 1 --max-steps 80 --floors 0.50,0.85 \
  --modes utility,dual_utility,stable_dual_utility \
  --floor-penalty 0.0 --channel-loss-offsets 0,10 \
  --out ~/HPPO-VQA/runs/qrs_stable_price_channel_stress_M2N6_smoke.json
```

Component-ablation smoke command:

```bash
python3 run/qrs_component_ablation.py --M 2 --N 4 --n-episodes 5 \
  --n-seeds 1 --max-steps 80 --base-floor 0.85 \
  --floor-penalty 0.0 --channel-loss-offset-db 10.0
```

UE/query bandit smoke command:

```bash
python3 run/train_qrs_bandit.py --M 2 --N 4 --train-episodes 8 \
  --eval-episodes 4 --n-seeds 1 --max-steps 50 \
  --base-floor 0.85 --floor-penalty 0.0 --channel-loss-offset-db 10.0
```

Neural UE/query distillation smoke command:

```bash
python3 run/train_qrs_neural_bandit.py --M 2 --N 4 \
  --collect-episodes 2 --eval-episodes 2 --n-seeds 1 \
  --max-steps 30 --epochs 3 --base-floor 0.85 \
  --floor-penalty 0.0 --channel-loss-offset-db 10.0
```

Regret-distilled neural UE/query smoke command:

```bash
python3 run/train_qrs_neural_bandit.py --M 2 --N 4 \
  --collect-episodes 2 --eval-episodes 2 --n-seeds 1 \
  --max-steps 30 --epochs 50 --lr 0.003 \
  --utility-regret-weight 2.0 --base-floor 0.85 \
  --floor-penalty 0.0 --channel-loss-offset-db 10.0 \
  --out paper/data/qrs_neural_bandit_regret_smoke.json \
  --model-out paper/data/qrs_neural_bandit_regret_smoke_model.json
```

Nominal neural UE/query safety diagnostic command:

```bash
python3 run/train_qrs_neural_bandit.py --M 2 --N 4 \
  --collect-episodes 3 --eval-episodes 3 --n-seeds 1 \
  --max-steps 30 --epochs 50 --lr 0.003 \
  --utility-regret-weight 2.0 --base-floor 0.50 \
  --floor-penalty 1.0 --channel-loss-offset-db 0.0 \
  --safety-score-margin 0.05 \
  --out paper/data/qrs_neural_bandit_nominal_soft_safe_smoke.json \
  --model-out paper/data/qrs_neural_bandit_nominal_soft_safe_smoke_model.json
```

Risk-distilled neural UE/query safety diagnostic command:

```bash
python3 run/train_qrs_neural_bandit.py --M 2 --N 4 \
  --collect-episodes 2 --eval-episodes 2 --n-seeds 1 \
  --max-steps 30 --epochs 60 --lr 0.003 \
  --utility-regret-weight 2.0 --floor-risk-weight 3.0 \
  --base-floor 0.85 --floor-penalty 0.0 \
  --channel-loss-offset-db 10.0 --safety-score-margin 0.05 \
  --out paper/data/qrs_neural_bandit_risk_soft_safe_smoke.json \
  --model-out paper/data/qrs_neural_bandit_risk_soft_safe_smoke_model.json
```

Semantic-floor improvement gate diagnostic command:

```bash
python3 run/train_qrs_neural_bandit.py --M 2 --N 4 \
  --collect-episodes 2 --eval-episodes 2 --n-seeds 1 \
  --max-steps 30 --epochs 20 --lr 0.003 \
  --base-floor 0.85 --floor-penalty 0.0 \
  --channel-loss-offset-db 10.0 \
  --safety-mode semantic_floor_improve \
  --out paper/data/qrs_neural_bandit_bc_floor_improve_smoke.json \
  --model-out paper/data/qrs_neural_bandit_bc_floor_improve_smoke_model.json
```

Learned-risk gate diagnostic command:

```bash
python3 run/train_qrs_neural_bandit.py --M 2 --N 4 \
  --collect-episodes 2 --eval-episodes 2 --n-seeds 1 \
  --max-steps 30 --epochs 40 --lr 0.003 \
  --risk-head-weight 3.0 --base-floor 0.85 \
  --floor-penalty 0.0 --channel-loss-offset-db 10.0 \
  --safety-mode learned_risk --risk-threshold 0.50 \
  --out paper/data/qrs_neural_bandit_learned_risk_smoke.json \
  --model-out paper/data/qrs_neural_bandit_learned_risk_smoke_model.json
```

Learned-risk threshold sweep command:

```bash
python3 run/qrs_learned_risk_sweep.py --thresholds 0.30,0.50,0.70
```

Three-seed learned-risk follow-up command:

```bash
python3 run/qrs_learned_risk_sweep.py --thresholds 0.50 \
  --n-seeds 3 --nominal-collect-episodes 5 \
  --nominal-eval-episodes 5 --stress-collect-episodes 3 \
  --stress-eval-episodes 4 --tag mseed \
  --out-csv paper/data/qrs_learned_risk_threshold_sweep_mseed.csv \
  --out-tex paper/tables/qrs_learned_risk_threshold_sweep_mseed.tex
```

LearnRisk scaling smoke command:

```bash
python3 run/qrs_learned_risk_scaling.py --M-values 1,2 \
  --N-values 3,4,6 --collect-episodes 2 --eval-episodes 2 \
  --n-seeds 1 --max-steps 20 --epochs 30
```

Unsafe neural UE/query distillation smoke command:

```bash
python3 run/train_qrs_neural_bandit.py --M 2 --N 4 \
  --collect-episodes 2 --eval-episodes 2 --n-seeds 1 \
  --max-steps 30 --epochs 3 --base-floor 0.85 \
  --floor-penalty 0.0 --channel-loss-offset-db 10.0 \
  --unsafe-neural --out paper/data/qrs_neural_bandit_unsafe_smoke.json \
  --model-out paper/data/qrs_neural_bandit_unsafe_smoke_model.json
```

Question-type ablation smoke command:

```bash
python3 run/qrs_question_type_ablation.py --M 2 --N 4 --n-episodes 5 \
  --n-seeds 1 --max-steps 80 --mode stable_dual_utility \
  --base-floor 0.85 --floor-penalty 0.0 --channel-loss-offset-db 10.0
```

Typed semantic-price smoke command:

```bash
python3 run/qrs_price_sweep.py --M 2 --N 4 --n-episodes 5 \
  --n-seeds 1 --max-steps 80 --floors 0.85 \
  --channel-loss-offsets 10.0 --floor-penalty 0.0 \
  --modes utility,stable_dual_utility,typed_stable_dual_utility \
  --out paper/data/qrs_typed_price_smoke.json
```

Nominal rollout/CDF dump smoke command:

```bash
python3 run/dump_qrs_rollout.py --M 2 --N 4 --n-episodes 2 \
  --n-seeds 1 --max-steps 30 --mode typed_stable_dual_utility \
  --base-floor 0.50 --floor-penalty 1.0 --channel-loss-offset-db 0.0 \
  --out-prefix paper/data/qrs_rollout_nominal_smoke \
  --out-tex paper/tables/qrs_rollout_nominal_cdf_smoke.tex
```

Stress rollout/CDF dump smoke command:

```bash
python3 run/dump_qrs_rollout.py --M 2 --N 4 --n-episodes 2 \
  --n-seeds 1 --max-steps 30 --mode typed_stable_dual_utility \
  --base-floor 0.85 --floor-penalty 0.0 --channel-loss-offset-db 10.0 \
  --out-prefix paper/data/qrs_rollout_smoke \
  --out-tex paper/tables/qrs_rollout_cdf_smoke.tex
```

Local result files:

- `paper/data/greedy_M2N4_current_eval.json`
- `paper/data/qrs_utility_M2N4_eval.json`
- `paper/data/qrs_price_sweep_penalty0_M2N4_smoke.json`
- `paper/data/qrs_stable_price_channel_stress_M2N4_smoke.json`
- `paper/data/qrs_stable_price_channel_stress_M2N6_smoke.json`
- `paper/data/qrs_stable_price_ablation.csv`
- `paper/tables/qrs_stable_price_ablation.tex`
- `paper/data/qrs_component_ablation_smoke.json`
- `paper/data/qrs_component_ablation_smoke.csv`
- `paper/tables/qrs_component_ablation_smoke.tex`
- `paper/data/qrs_bandit_smoke.json`
- `paper/data/qrs_bandit_smoke_model.json`
- `paper/data/qrs_question_type_ablation_smoke.json`
- `paper/data/qrs_question_type_ablation_smoke.csv`
- `paper/tables/qrs_question_type_ablation_smoke.tex`
- `paper/data/qrs_typed_price_smoke.json`
- `paper/data/qrs_typed_price_smoke.csv`
- `paper/tables/qrs_typed_price_smoke.tex`
- `paper/data/qrs_neural_bandit_smoke.json`
- `paper/data/qrs_neural_bandit_smoke_model.json`
- `paper/data/qrs_neural_bandit_unsafe_smoke.json`
- `paper/data/qrs_neural_bandit_unsafe_smoke_model.json`
- `paper/data/qrs_neural_bandit_safe_smoke.json`
- `paper/data/qrs_neural_bandit_safe_smoke_model.json`
- `paper/data/qrs_neural_bandit_regret_smoke.json`
- `paper/data/qrs_neural_bandit_regret_smoke_model.json`
- `paper/data/qrs_neural_bandit_soft_safe_smoke.json`
- `paper/data/qrs_neural_bandit_soft_safe_smoke_model.json`
- `paper/data/qrs_neural_bandit_floor_safe_smoke.json`
- `paper/data/qrs_neural_bandit_floor_safe_smoke_model.json`
- `paper/data/qrs_neural_bandit_nominal_unsafe_smoke.json`
- `paper/data/qrs_neural_bandit_nominal_unsafe_smoke_model.json`
- `paper/data/qrs_neural_bandit_nominal_regret_smoke.json`
- `paper/data/qrs_neural_bandit_nominal_regret_smoke_model.json`
- `paper/data/qrs_neural_bandit_nominal_soft_safe_smoke.json`
- `paper/data/qrs_neural_bandit_nominal_soft_safe_smoke_model.json`
- `paper/data/qrs_neural_bandit_nominal_floor_safe_smoke.json`
- `paper/data/qrs_neural_bandit_nominal_floor_safe_smoke_model.json`
- `paper/data/qrs_neural_bandit_risk_unsafe_smoke.json`
- `paper/data/qrs_neural_bandit_risk_unsafe_smoke_model.json`
- `paper/data/qrs_neural_bandit_risk_soft_safe_smoke.json`
- `paper/data/qrs_neural_bandit_risk_soft_safe_smoke_model.json`
- `paper/data/qrs_neural_bandit_nominal_risk_unsafe_smoke.json`
- `paper/data/qrs_neural_bandit_nominal_risk_unsafe_smoke_model.json`
- `paper/data/qrs_neural_bandit_nominal_risk_soft_safe_smoke.json`
- `paper/data/qrs_neural_bandit_nominal_risk_soft_safe_smoke_model.json`
- `paper/data/qrs_neural_bandit_bc_floor_improve_smoke.json`
- `paper/data/qrs_neural_bandit_bc_floor_improve_smoke_model.json`
- `paper/data/qrs_neural_bandit_nominal_bc_floor_improve_smoke.json`
- `paper/data/qrs_neural_bandit_nominal_bc_floor_improve_smoke_model.json`
- `paper/data/qrs_neural_bandit_learned_risk_smoke.json`
- `paper/data/qrs_neural_bandit_learned_risk_smoke_model.json`
- `paper/data/qrs_neural_bandit_nominal_learned_risk_smoke.json`
- `paper/data/qrs_neural_bandit_nominal_learned_risk_smoke_model.json`
- `paper/data/qrs_learned_risk_threshold_sweep_smoke.csv`
- `paper/tables/qrs_learned_risk_threshold_sweep_smoke.tex`
- `paper/data/qrs_learned_risk_threshold_sweep_mseed.csv`
- `paper/tables/qrs_learned_risk_threshold_sweep_mseed.tex`
- `paper/data/qrs_neural_bandit_nominal_learned_risk_t0p50_mseed.json`
- `paper/data/qrs_neural_bandit_stress_learned_risk_t0p50_mseed.json`
- `paper/data/qrs_learned_risk_scaling_smoke.csv`
- `paper/tables/qrs_learned_risk_scaling_smoke.tex`
- `paper/figures/fig4_scaling.pdf`
- `paper/data/qrs_neural_bandit_safe_smoke.csv`
- `paper/data/qrs_neural_bandit_nominal_smoke.csv`
- `paper/tables/qrs_neural_bandit_safe_smoke.tex`
- `paper/tables/qrs_neural_bandit_nominal_smoke.tex`
- `paper/data/qrs_rollout_nominal_smoke.tasks.csv`
- `paper/data/qrs_rollout_nominal_smoke.trajectory.csv`
- `paper/data/qrs_rollout_nominal_smoke.summary.json`
- `paper/tables/qrs_rollout_nominal_cdf_smoke.tex`
- `paper/data/qrs_rollout_smoke.tasks.csv`
- `paper/data/qrs_rollout_smoke.trajectory.csv`
- `paper/data/qrs_rollout_smoke.summary.json`
- `paper/tables/qrs_rollout_cdf_smoke.tex`
