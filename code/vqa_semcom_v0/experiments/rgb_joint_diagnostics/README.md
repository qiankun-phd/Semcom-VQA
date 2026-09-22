# Frozen-cache diagnosis of EXP-012

Read [DIAGNOSTIC_SCOPE.md](DIAGNOSTIC_SCOPE.md) first. This is post-run analysis,
not a new training run or a replacement policy. The original EXP-012 artifacts
and all test300 inputs stay unchanged.

## Modules

- `export_scores.py`: CPU eval/inference-mode forward of twelve existing
  checkpoints (four families, three seeds), all 720 cached feature vectors;
  validates original source hashes and 3840 reported decisions. Writes outside
  the source run. Optional `--first-layer-only` decomposes the frozen first
  linear layer into question/image signals, without fitting or intervention.
- `opportunities.py`: cache-only nine-action correctness, resource response,
  non-monotonicity, rescue/harm and explicitly answer-informed oracle bounds.
- `score_diagnostics.py`: mixed-image macro AUC, image-level intervals, calibration,
  frozen lambda=0/.05 mechanism comparison, and seed stability. Never scans lambda.
- `feature_diagnostics.py`: train-only scaler reproduction, block scale and
  question overlap/hash-collision description. Does not modify feature inputs.
- `build_report.py`: exact tables, paired exploratory tests, two real figures,
  interpretation notes, and an analysis/stats/figure-catalog bundle.

## Inputs and reproduction

The ignored local run folder contains the existing `supervision_records.json`,
train/validation truth and manifests, legacy grid records, legacy development
truth/manifest, original reports, `features.json`, `policy.json` and histories.
Private data and weights are not packaged in this source directory.

Use the frozen training environment for forward export on its owning host:

```bash
python export_scores.py --source FROZEN_RUN --destination NEW_DIAGNOSTIC_ROOT
python export_scores.py --source FROZEN_RUN --destination NEW_DIAGNOSTIC_ROOT --first-layer-only
```

Keep original forward-export code with its recorded hash when adding a later
diagnostic. Do not overwrite existing raw exports. Copy outputs locally under
`paper/outputs/rgb_joint_selector_20260922/diagnostics/`.

From the repository root (replace `RUN`/`DIAGNOSTICS` with local artifact paths):

```bash
python code/vqa_semcom_v0/experiments/rgb_joint_diagnostics/opportunities.py --input-dir RUN
python code/vqa_semcom_v0/experiments/rgb_joint_diagnostics/feature_diagnostics.py --root RUN --output DIAGNOSTICS/feature_diagnostics.json
python code/vqa_semcom_v0/experiments/rgb_joint_diagnostics/score_diagnostics.py --root RUN --scored-inputs DIAGNOSTICS/scored_inputs.json --output DIAGNOSTICS
python code/vqa_semcom_v0/experiments/rgb_joint_diagnostics/build_report.py --root DIAGNOSTICS
python -m unittest discover -s code/vqa_semcom_v0/experiments/rgb_joint_diagnostics -p 'test_*.py'
```

Local analysis uses NumPy and Matplotlib; only forward export needs PyTorch.
No VLM, neural codec, detector, dataset downloads, optimizer steps, model
selection, wireless simulation, or hidden-test access is performed here.

## Result boundary

New validation has nine failures of fixed 2k/low recoverable somewhere in the
grid; legacy development has twelve. The original joint policy rescues/hurts
1/1 and 2/2 respectively. Its within-image ranking AUC is about .495/.558, with
wide intervals spanning .5. Turning off the penalty raises communication but
does not improve new-validation total correctness. These observations identify
scoring/generalization problems, not an experimentally isolated causal fix.

At the close of this diagnosis, the 2×2 feature-scale / incremental-gain-target
probe was a recommendation only. A later, separately authorized EXP-013 has now
completed it; see [its result](../rgb_selector_factorial/RESULTS.md). The original
diagnosis and frozen EXP-012 run remain unchanged. Neither development analysis
can be promoted to a paper's final test result or an energy-saving claim.
