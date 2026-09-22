# EXP-012 post-run diagnosis (2026-09-22)

Approved scope: explain the completed selector's failures using existing cached
nine-action outcomes, low-cost features, and frozen MLP checkpoints. No codec or
VLM inference, optimizer updates, fresh labels, test300 access, wireless sweep,
paper modification, or new trained-policy claim is authorized by this diagnosis.

This is **post-run exploratory diagnosis**, not a new independent evaluation or
a replacement of the original preregistration. Do not change EXP-012's protocol,
checkpoints, scaler, selected fixed baselines, lambda, or historical outcomes.

## Questions fixed before the additional calculations

1. How many images have identical correctness across all nine actions? Starting
   at the cheapest 2k/low action, how many failures can other actions rescue,
   and how many cases become wrong when resources increase? Distinguish
   rate-only, resolution-only, and jointly necessary increases.
2. How close is the trained policy to the answer-informed oracle? Compare with
   all fixed actions and single-axis oracles using the strongest fixed other
   axis. Oracles use labels and are upper bounds, never deployable performance.
3. Does the frozen model rank correct versus incorrect actions **within the
   same image**, or mostly estimate whether a question is easy for all actions?
   Distinguish within-image AUC from pooled image/action AUC. Repeated actions
   from an image are not independent statistical units.
4. Does the registered resource penalty cause the failure? Evaluate exactly
   lambda=0 versus registered lambda=.05 as a diagnostic intervention on frozen
   probabilities. Do not search or fit a best lambda; no changed deployment.
5. What do training/validation differences, seed variation, block scales,
   question-only and image-shuffle controls establish about features? Numerical
   feature imbalance is a hypothesis, not causal proof without controlled fits.

## Artifacts and statistics

New code lives only in this diagnostics directory. Raw diagnostic files and
identity-level records stay in ignored `paper/outputs/rgb_joint_selector_20260922/diagnostics/`.
No private addresses/paths, labels, or weights are added to public source.

Audit source hashes and frozen model selections before and after CPU forward
export. Recompute correctness with the original normalization and keep all
images in their original split. Actual image bytes include the same 1-byte
tier prefix for every action. Nominal selection and actual evaluation costs
must remain separate.

Use paired image bootstrap intervals (2000 resamples) and three seed-level
descriptions where relevant. Binary paired differences use exact McNemar with
Holm correction within each stated split/family. Do not test dependent action
pairs as independent observations. Validation selected models, legacy dev is
reused, and train is in-sample; label all three explicitly.

Done means: a concise diagnosis report, exact summary tables, statistics
appendix, at least two actual diagnostic figures with interpretation notes,
tests for new computations, and a bounded next-step decision. No automatic
retraining or further task spawning after conclusions are available.
