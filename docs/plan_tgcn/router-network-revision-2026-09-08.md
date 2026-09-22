# Router network/target development — frozen protocol

User explicitly authorized new network-focused experiments. Existing manuscript
and all verified outputs remain unchanged. CPU only, 1 BLAS thread, nice10,
CUDA disabled, RA_DI environment on lab-s2. No dependency installation,
VLM/detector/codec call, extra imagery, or new input feature is authorized here.

## Fixed data and comparison

Reuse crossreceiver_v2 frozen arrays: three receivers; identical VisDrone
Rician keys, six SNRs, original image partition, shared train-only calibration,
18 sender-box-verified features and existing branch correctness labels.
Train/validation/test decisions are 9150/2646/2808. Main metric is full pooled
answer accuracy, lambda=0. Test performance has already been viewed: this is
subsequent development on an existing benchmark, not a pristine unseen holdout.

Four candidates, fixed before screening:

| ID | Target | Hidden widths | L2 alpha | Parameters |
|---|---|---|---:|---:|
| original_bce | Independent correctness BCE for two branches |32,16 per branch|0.0001|2306|
| wide_bce | Same independent BCE |64,32 per branch|0.001|6658|
| advantage_mse | Regress y_image minus y_detection |32,16|0.001|1153|
| disagreement_bce | Image-uniquely-correct classification on disagreeing branches only |32,16|0.001|1153|

All use ReLU, Adam lr0.001, batch200, max300 epochs, min_delta1e-4 and patience10.
The original candidate exactly retains per-branch validation BCE and best-state
restoration. Regression early-stops on full validation MSE; disagreement model
early-stops on validation disagreement BCE. No threshold search: score>0 chooses
image and exact ties choose detection. The conditional disagreement probability
does not serve as an energy-priced branch probability difference.

## Selection gate

Each receiver: train all four candidates at seeds0,1,2; choose greatest mean
full validation answer accuracy. Exact ties favor fewer parameters, then ID.
Keep every candidate/seed, including failures. Save global
validation_selection.json for all receivers before reading test arrays.
Then train selected configuration at seeds0–9 and evaluate the frozen test set.
No test-driven follow-up search or configuration revision. Thus 36 screening
fits and 30 final fits; a dual-head fit includes both branch predictors.

All stage models, best-head checkpoints, validation loss curves, probabilities
or scores, picks, parameter counts, and wall-clock times are saved. Before formal
launch, run targeted unit tests and a validation-only 2epoch smoke for all four
candidates and all three receivers. Test arrays remain unread in smoke.

## Reporting and inference boundary

Retain original MLP, linear and fixed baselines. Report all final seeds, pooled
mean±sample SD, every question type, image-use rate and all screening validation
results. Inference timing uses the same 2646-row validation batch (3 warmups,
20 repeats), plus first100 validation queries individually. CPU timing excludes
VLM, detector and communications, and is not energy or end-to-end latency.

Only descriptive conclusions initially; any image-cluster inference must state
selection/prior-test-exposure limitations. Do not turn a seed SD into a population
confidence interval or an unsupported superiority claim. No automatic paper edits
or DV reruns follow this task.

## Execution, completion and QA

Five targeted unit tests passed. Validation-only real-data smoke completed all
four candidates on all three receivers at seed0/two epochs without loading test
arrays. Formal tmux `tgcn-router-network-20260908`, PID1250874, started at
UNIX1788874289.201465; log is
`/home/qiankun/phd_research/vqa_semcom/outputs/router_network_revision_20260908.log`.
Runner SHA256: `6db900599be76151dd95e6f0f548edcfbfe956d4f2668abd49e14fb7517508d4`.

Formal status COMPLETE: 36 screening fits plus 30 final fits, 67.019 seconds.
Global configuration selection was saved at UNIX1788874322.4912362 before final
test loading; SHA256 `df46c35dcdb0c9a9df8aeedd602d86e9b947a194f1570968281a13cf36252322`.
No further model/threshold/configuration changes followed test inspection.

Post-run QA reloaded all66 saved model sets and reproduced saved predictions,
decisions and metrics exactly. Nine original-BCE screening predictions matched
the prior verified crossreceiver_v2 validation scores element-for-element.
Every NPZ passed CRC and finite-array checks. All candidate records and best-head
checkpoints remain; no unsuccessful validation candidate was removed. Logs have
no Traceback/Warning/Error.

343 files synchronized to local `paper/outputs/router_network_revision_20260908`.
The sorted relative-path/per-file-SHA256 manifest digest agrees on both hosts:
`6e6a42cf65968eb909bcc2f5135c58e45af1156bc5fb0afb9a97e56200cede1a`.
Two PDF/SVG/600dpi PNG figures were visually inspected: no clipping/overlap;
dot plots disclose zoomed accuracy axes and seed-SD error bars.

## Findings without a universal-superiority claim

Accuracy values are full2808-key test percentages, mean±sample SD over10 seeds.

| Receiver | Validation-selected configuration | New | Original MLP | Linear | New minus original (pp) |
|---|---|---:|---:|---:|---:|
| Qwen2-VL | wide_bce |69.8967±0.1951|69.6261±0.3759|69.5869|+0.2707|
| Qwen2.5-VL | advantage_mse |68.9815±0.1862|68.6218±0.2836|68.3761|+0.3597|
| SmolVLM | advantage_mse |69.4302±0.3173|69.2913±0.5666|68.1268|+0.1389|

The improvement is small, not a substantial jump in overall accuracy. Network
capacity alone did not win every receiver. The single-head advantage target
is a useful compact candidate but was not selected on Qwen2; disagreement-only
classification won none of the receivers. No DV experiment was started.

| Receiver | Original/new parameters | Original/new fit seconds | Original/new single-query microseconds | Original/new batch microseconds per query |
|---|---:|---:|---:|---:|
| Qwen2-VL |2306/6658|1.150/2.023|177.7/171.6|0.361/0.643|
| Qwen2.5-VL |2306/1153|1.309/0.856|185.8/71.1|0.357/0.161|
| SmolVLM |2306/1153|0.891/0.372|168.6/71.4|0.350/0.160|

Original cost reference averages the three original-BCE screening runs; selected
cost averages ten final runs. Same single-thread host and timing code. Fit wall
time includes best-checkpoint I/O; inference timing excludes checkpoints. CPU
Python-call overhead and batch amortization dominate these tiny networks, so a
slightly smaller single-call number for the wider Qwen2 network is not evidence
that widening makes computation cheaper. None of these are hardware energy or
end-to-end VQA latency measurements.

The analysis folder contains `analysis-report.md`, `stats-appendix.md`,
`figure-catalog.md`, per-type comparisons, cost values, QA, and all six exploratory
paired contrasts. Image-cluster bootstrap and sign-flip results are conditional
post-development sensitivity checks; Holm-adjusted p-values across six contrasts
are all above0.05 (minimum0.2304). Prior benchmark/test exposure and scene
dependence are not removed by these calculations. Do not write “significantly
improved” from these results or treat seed SD as a population confidence bound.
