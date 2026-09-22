# Accuracy upgrade: frozen development protocol (2026-09-08)

Only new scripts and outputs are owned by this experiment. No manuscript or old
cache is modified. Existing test results have been examined: this is subsequent
development, not a pristine holdout. No test-driven second search is permitted.

## A: sender-observable features (primary deliverable)

Keep crossreceiver_v2 keys, labels, train-only calibration, and image partitions:
9150 training, 2646 validation, 2808 test decisions. Lambda=0 pooled accuracy.
Use the previous validation-selected network unchanged: Qwen2 wide_bce (64,32,
alpha .001); Qwen2.5 and Smol advantage_mse (32,16, alpha .001). Optimizer,
batch200, lr .001, max300, patience10, min_delta1e-4 and best-state recovery
remain identical. No hyperparameter or routing-threshold search.

Four feature groups are declared before any enhanced validation performance:

- base18: the verified original 18 features, unchanged and unstandardized.
- quality: base18 plus ten sender-box statistics: target confidence mean/max/SD,
  fraction below .4, target normalized box-area mean/max/sum, total predicted
  box count /100, all-box confidence mean, all-box normalized area sum.
- semantics: base18 plus 24 question/box features: ten secondary-class one-hot
  indicators; secondary-present, threshold-present, explicit-negation flags;
  primary/secondary/difference/sum/min/max predicted counts divided by60;
  primary/(primary+secondary+1), both-counts-positive; question threshold /60,
  (predicted primary count - threshold)/60, predicted count >= threshold.
  Threshold-dependent fields are zero when no threshold exists. The currently
  supported exact templates contain no negation, so its flag is zero; unknown
  or negated templates fail explicitly rather than deriving a flag from answers.
- combined: base18 plus both groups, 52 dimensions.

Counts and confidence/area statistics come only from the original canonical
sender detector boxes, never received counts or answers. Areas use actual source
image dimensions. Exact full-match question regex extracts class names and
threshold integers; parsed primary class must equal the frozen key class.
The canonical detector file also contains one `others` box. It contributes only
to all-box quality statistics; it is not a task category. The first smoke stopped
on this schema discovery before any fit, and its failed directory is retained.
GT, correctness, received count, answer polarity, and annotations are prohibited
inputs. Correctness is used only as the frozen supervised training target.
Only appended columns are standardized, using training means/SDs (zero SD->1).

Screen each group with seeds0,1,2 on training/validation. Select per receiver
highest mean full validation answer accuracy, ties fewer input dimensions then
group ID. Save one global validation_selection.json before reading test arrays.
Then fit selected group seeds0-9 and evaluate test once. Original18 screening
predictions must exactly reproduce the previous corresponding network runs.
All candidates, checkpoints, seed metrics, source hashes, costs and failures stay.

## B/C: fixed validation-only pilots, not main pooled comparisons

Use the first eight lexicographically sorted validation images having all five
question types in common_keys. B uses every original predicted/GT class on those
images. Four predeclared YOLO settings: (imgsz640,conf.25), (1280,.25),
(640,.15), (1280,.15), existing VisDrone weights, same category mapping and
remaining Ultralytics settings. No SAHI because dependency is absent. Report
counts, one-to-one IoU.5 precision/recall and unmatched detections (not all
unmatched boxes are duplicates), plus class-aware overlapping-pair counts.
This is detector-only validation diagnosis unless channel/packet replay is
explicitly implemented; changed boxes cannot reuse old received branch labels.

C uses these same images, lexicographically first question per type, and SNR
-5,10,20: maximum120 fixed validation decisions per model. Same received JPEG,
prompt, scoring, processor min200704/max802816 pixels, max_new_tokens24,
deterministic generation. Prefer Qwen2.5-VL-7B vs3B under equal precision. An
isolated environment may add quantization support; do not mutate old envs.
8GB GPU prevents ordinary7B BF16; if both models can use NF4, compare equal NF4
configurations and separately retain cached3B results as contextual reference.
No silent resizing on OOM and no model/prompt selection by accuracy. Model
download is public and no paid API is allowed.

One GPU inference hour total. After a first complete image (15 decisions/model),
estimate throughput and cap evaluation to the largest common image-prefix
(2..8) estimated to fit50min, reserving10min for detector/loading overhead.
This budget choice uses time only, not answers. If even two images or equal
precision cannot be run, retain smoke evidence and report C blocked/incomplete.
GPU must be idle; no other jobs are stopped. No invented energy: old32.31J and
.4275J are not applicable. GPU power may be unavailable on the RTX4060.

## Validation and delivery

CPU routing is nice10 with one BLAS thread and CUDA disabled. Test strict parser,
feature isolation, training-only scaling, dimensional parameter counts; run real
2epoch smoke before formal launch. All outputs are new, exist_ok=False.
Report mean +/- sample SD, all seeds and types, originalMLP/linear/last-network
comparisons; seed SD is not population uncertainty. Any image-cluster inference
is exploratory and must disclose prior test exposure and multiple comparisons.
Deliver analysis report, exact tables, figures, protocol/hash manifests and
separate A/B/C states. No automatic manuscript update or DV expansion.

## Execution and verified A results

Five unit tests and all12 validation-only two-epoch smoke fits passed. The first
smoke stopped before fitting on the known `others` category; it is retained.
Formal A used tmux `tgcn-accuracy-upgrade-20260908`, PID1281959, nice10/BLAS1,
and completed36screen+30final fits in59.2644seconds. Selection hash:
`991e49a8627c5e53f20709f7cb222f1fa4c8ad2ddd324cc50a48c64ebed80861`.
All66 saved models reproduce their saved predictions/metrics exactly; all9base18
screening results reproduce the previous same-network experiment exactly.

| Receiver | Selected input | Parameters | Accuracy mean ± SD (%) | Same-network original18 (%) | Gain (pp) |
|---|---|---:|---:|---:|---:|
| Qwen2-VL | combined52 |11010|72.1011±0.4191|69.8967±0.1951|+2.2044|
| Qwen2.5-VL | semantics42 |1921|73.2870±0.3981|68.9815±0.1862|+4.3056|
| SmolVLM | semantics42 |1921|73.3511±0.4015|69.4302±0.3173|+3.9209|

All values retain the full2808test decisions and10seeds. OriginalMLP and linear
are preserved in `analysis/comparisons.json`. Full train+validation feature
construction took0.7974seconds with no detector/VLM call. The exploratory
image-cluster appendix retains intervals and all9Holm-adjusted comparisons;
prior test exposure and scene dependence prevent pristine confirmatory claims.
Two actual figures (PDF/SVG/600dpiPNG) were inspected for clipping and overlap.

## B and cache diagnosis

Eight-image/four-setting detector pilot completed in2.32seconds after imports,
with model load and warmup separated from per-image prediction timing. Original
640/.25 per-class counts exactly reproduce the canonical cache on all8images.
1280/.25 changes count MAE4.35→2.6625 and IoU.5 recall28.31→45.37%; this is not
official VisDrone AP (ignored-region suppression is not implemented).
Before-link, uncalibrated symbolic answers on40unique fixed questions are
70%,67.5%,52.5%,65% for the four settings in declaration order. Consequently
better detection counts alone did not establish better overall question answering.
No changed detections were mixed with old received-branch outcomes.

Three-receiver training/validation cached answers were rescored with the original
checker: zero mismatches. Qwen2 threshold unknown parses occur158/2058training
and46/606validation rows; no answers or parser rules were rewritten. A processor-
only image audit records original/received dimensions and actual image grids.
The first metadata-only processor probe attempted an optional online config lookup
despite local_files_only and was safely stopped; `diagnostics_v2` uses explicit
cached snapshot paths and offline mode and is complete. Both folders are retained.

## C resources and controlled precision

7B public checkpoint downloaded in317.26seconds through the HF transport mirror;
snapshot commit `cc594898137f460bfe9f0759e9844b3ce807cfb5`. The isolated
`quant_env` inherits existing packages and adds only bitsandbytes0.48.2. Its wheel
SHA256 was checked against official PyPI:
`cd289562cb7308ee2a707e6884fecca9bbbcfc9ec33a86df2a45e0779692c1a3`.
No old environment was upgraded. A slow original wheel transfer was stopped
after validating its exact owned PID; the verified mirror copy was installed.

Tmux `tgcn-accuracy-vlm-20260908`, PID1292499, uses both7B and3B in equal NF4
double quantization/BF16 compute, sameJPEG/prompt/greedy24tokens/processor budget.
The first7Bimage/15questions took18.7631seconds; the time-only budget rule froze
all8images/120decisions per model. GPU occupancy was6290MiB/100%; power wasN/A.
No image resizing was changed to fit memory and no old Joule values were reused.

Implementation references: [official Qwen model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
for model-specific processor/vision input preparation, and [Transformers
bitsandbytes documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
for the NF4/BF16 configuration. Precision is matched between the new pilots;
cached3B BF16 results are only contextual, not a precision-controlled scale test.

## C completion and final handoff

C completed240GPUanswers in252.7794seconds, leaving the GPU idle afterward.
All120paired keys passed identicalJPEG/prompt/grid/question/GT and original
scoring checks. 7B-NF4 accuracy61.6667%,3B-NF4 48.3333% (+13.3333pp);
inference0.92768s vs0.64289s per query. Comparison questions are an adverse
exception:7B54.1667% vs3B62.5%. This8imagepilot is not the full table4test set.
Exact image-cluster sign-flip p=.0625, with only8clusters including related
frames: do not claim statistical/general superiority. Cached3B is47.5% on
these keys and is retained only as context.

Additional decoding provenance: official generation_config uses do_sample=True,
temperature1e-6 and repetition_penalty1.05. The old evaluator passes only
max_new_tokens, so the historical cache is near-greedy low-temperature sampling,
not an explicitly greedy call. Both newNF4models explicitly override
do_sample=False while retaining the same penalty. Thus the new paired control is
valid, but new-vs-cached differences are not attributed solely to quantization.
The stored model_metadata generation_config is the pre-override default; the
protocol/code records the actual explicit call overrides.

Primary delivery: `paper/outputs/accuracy_upgrade_20260908/analysis/analysis-report.md`.
Separate detector/VLMpilot delivery: `pilot_analysis/analysis-report.md`.
Original scripts/results/manuscript are untouched. Checkpoint hashes and source
image dimensions/hashes are stored without copying large images/model weights.
The bundle manifest excludes the isolated environment and wheel download but
includes archived code, logs, models, scores, full candidate history and figures.
Six final targeted unit tests passed, including a label/received-value mutation
test proving those unused fields do not alter the feature vector.
