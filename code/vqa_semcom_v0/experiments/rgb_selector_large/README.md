# EXP-014: large-data RGB VQA resource-selector experiment

Read [protocol.json](protocol.json) before running. User authorized expansion on
2026-09-22. This is a scale experiment, not a new codec or VLM training run.

## Data and comparisons

- 4,800 train images: TDIUC train2014, 800 per each of six types.
- 1,200 validation + 2,400 new test images: TDIUC val2014, 200/400 per type.
- One question per image; all 8,400 image identities disjoint; known historical
  experiments, receiver-adapter training and old sealed300 excluded.
- val2014 alone cannot provide the proposed 1,400 unique images per type:
  activity has only 847 after the initial 7,154-image exclusion inventory.
  The existing train2014 source supplies enough fresh training images. Exact
  joint class/image allocation, not separate marginal counts, is required.
- A nested 480-image subset of the new train source supplies a same-source small
  control. Same epochs/batch size imply different optimizer update counts; this
  is a fixed-training-recipe scale comparison, not a pure sample-count effect.

Four recipes: original/balanced image-block scale × absolute BCE/signed gain MSE.
Each large recipe trains joint9 plus six single-axis families, each at seeds
7/17/27: 84 fits. Four nested joint recipes add 12, for at most 96 small CPU fits.
All use the same MLP hidden128/64, maximum50/minimum10 epochs, patience8, batch32,
AdamW1e-3 and weight decay1e-4. Gain uses the family's cheapest action as its
reference and keeps negative harm. Different-family raw gain scores are never
compared directly; choose families by validation realized utility.

The main joint recipe, strongest learned rate/compute controllers and strongest
fixed action are selected on validation and frozen before new test images are
downloaded. All other recipes/seeds remain visible as preregistered secondary
results; none may replace the frozen main method based on test performance.

## Frozen physical/model interface

Ordinary pretrained CompressAI bmshj2018-hyperprior q3, longedge320, uniform
quantization map and 12-step global gain search → real raw codec packet at
2000/4000/8000-byte cap → common one-byte receiver-tier field → RGB decoding →
full-view pixel target50176/100352/200704 → frozen Qwen3-VL-4B NF4 + standard LoRA.
The codec runs on CPU and VLM inference requires CUDA. An online selector chooses
one action; the nine-way grid is offline supervision/evaluation, not nine live
transmissions per question.

All image byte counts include framing. Symbols are accounting only:
`510*ceil(framed_bytes/48)`. Delivery rate, joules and live end-to-end latency
remain unmeasured. No wireless sweep or manuscript change is part of this run.

## Execution and restart boundaries

```bash
python launch.py --output RUN --protocol CODE/protocol.json --source-code EXP012/code --project-root PROJECT --stage1-root STAGE1 --grid-code EXP011/code --adapter STANDARD_ADAPTER --detach
```

The sequential supervisor performs preparation/features, six-image codec and
GPU smoke, full train/validation grid, all selector fits, controller freeze,
new test preparation/features/grid, and one frozen test evaluation. Six smoke
images are reused. Budget is 25,200 codec representations and 75,600 unique VLM
outcomes; bounded warmups are excluded. One absolute24-hour deadline survives
resumption, with an8GiB free-space floor and one supervisor lock. Logs stay in
the output directory. No automatic error retries or policy retuning occur.

Per-image codec/inference journals avoid repeatedly rewriting a growing
54,000-record JSON after every answer. Finished stage receipts are hashed and
skipped on resume; completion artifacts are not rewritten. A partly fitted MLP
run is fail-closed, not automatically granted another96 fits. A started test
evaluation without a complete receipt also stops for explicit diagnosis.

Preparation writes new test annotations into a sealed file to define the dataset;
training and VLM inference never read that file. The existing old test300 truth
and JPEGs remain untouched. New test truth is opened only after verifying all
96 checkpoints, selection, source inputs, and evaluator code. Independent test
artifacts never append to or change frozen train/validation files.

## Status and evidence

Inspect `supervisor_status.json` first. Stage details are in
`data_status_trainval.json`, `features_status.json`, `encoding_status.json`,
`inference_status.json`, and `training_status.json`. Artifacts include
`capacity_report.json`, `selection_frozen.json`, `frozen_data.json`,
`validation_report.json`, `controller_frozen.json`, `test_report.json`, and
`test_evaluation_complete.json`. Raw identities, images and weights stay outside
version control under the ignored experiment output root.

This larger evaluation improves precision; it does not guarantee resolving a
1pp difference. Preserve the old accuracy screen as a proportion1/120 (up to
10/1,200 or20/2,400 net lost answers), not as one answer at every dataset size.
Screening point estimates are not proof of statistical noninferiority. The
three primary utility contrasts against zero share a Holm correction; evidence
against zero is not itself evidence that improvement exceeds0.005.
