# EXP-013: bounded selector factorial experiment

Read the pre-training [protocol](protocol.json). This experiment uses only the
EXP-012 cached features and nine-action outcomes. It does not run a codec, VLM,
wireless channel, or sealed test. All new outputs must be outside the frozen
source run. Training runs on the existing experiment server, with two CPU threads.

## Four interventions

|Group|Input image block|Training objective|
|---|---|---|
|original_absolute|Train-standardized, unchanged|Nine-head correctness BCE|
|balanced_absolute|Train-standardized, divided by sqrt(83)|Nine-head correctness BCE|
|original_gain|Train-standardized, unchanged|Eight non-reference signed-gain MSE|
|balanced_gain|Train-standardized, divided by sqrt(83)|Eight non-reference signed-gain MSE|

Question features are the same L2-normalized 256-dimensional hashed text vector.
Scaling is a fixed block multiplier, not per-example normalization; it preserves
relative image magnitudes. All models have the same 339→128→64→9 architecture,
dropout, seeds 7/17/27, optimizer settings, and bounded early stopping.

Action 0 is 2000-byte-cap/low-resolution. Relative gain is
`correct(action) - correct(action 0)`, including -1 when an upgrade hurts. The
model outputs `tanh(logit(action) - logit(action 0))`; reference gain is exactly
zero. Predicted gain is **not a probability**. At deployment, gain is compared
with the fixed extra nominal resource cost. No true correctness enters the
deployment decision. Absolute and gain variants differ in parameterization,
loss, and checkpoint criterion together; they are not a pure loss-free change
of label format.

## Boundaries and reproducibility

- The original-scale absolute group must reproduce EXP-012 frozen joint9 outputs
  to 1e-6 and all decisions. A failed reproduction blocks interpretation.
- All twelve checkpoints must be frozen before legacy-development truth is
  scored. This ordering does not make the previously inspected split independent.
- Report all groups and seeds. Do not use these results to search lambda,
  choose an unregistered replacement, or automatically open test300.
- Code, protocol, inputs, checkpoints, and exported scores are hashed. Private
  manifests and raw per-image outputs remain in ignored output directories.
- Use raw codec length + one visual-tier header byte for every method, including
  fixed controls. Symbols are accounting only: `510*ceil(framed_bytes/48)`.
- Delivery rate, joules, and live end-to-end latency are not measured here.

## Analysis package

The result bundle contains `analysis-report.md`, `stats-appendix.md`,
`figure-catalog.md`, and two actual PDF/PNG figures. The primary mechanism metric
is image-macro AUC over the **same** mixed-outcome images for all four groups.
Rescue/harm and realized resource/utility metrics describe deployment choices.
Bootstrap units are images, not nine dependent actions. Three factorial
contrasts over AUC/utility and the two development splits form one family of
twelve exploratory sign-flip tests, with Holm correction. Repeated development
inspection and fitted-model selection remain limitations.

The system screening conditions remain those of EXP-012, including strong
learned single-axis controls and the pending full-cost gate. A better loss or
ranking alone does not establish a useful communication system.

## Commands and completed result

Run with the frozen source environment, substituting the existing run paths:

```bash
python train_factorial.py --source-run OLD_RUN --output NEW_RUN --protocol PROTOCOL --source-code OLD_RUN/code --baseline-scores OLD_DIAGNOSTICS/scored_inputs.json
python evaluate_factorial.py --source-dir OLD_RUN --output NEW_RUN --protocol PROTOCOL --source-code OLD_RUN/code --source-protocol OLD_RUN/code/protocol.json --legacy-grid OLD_GRID --legacy-truth OLD_DEV_TRUTH
python build_report.py --evaluation NEW_RUN/evaluation.json --output-dir NEW_RUN/analysis-output
```

The optional explicit legacy paths refer only to the already-used development
grid/truth, not the sealed test. They are checked against the old report's
fingerprints; no copies need to be written into the old run. The first evaluation
attempt stopped at a missing legacy cache path; resolving that path did not
modify the protocol, models, or data and did not cause additional training.

All 12 fits and the complete evaluation finished. The original group reproduced
25,920 scores exactly and 2,880 decisions without mismatch. None of the four
groups passes the unchanged preliminary system screen on either development
split. See the [Chinese results summary](RESULTS.md). No test or SNR expansion
was initiated.
