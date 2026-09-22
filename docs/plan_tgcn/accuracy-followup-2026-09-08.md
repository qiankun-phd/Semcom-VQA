# Accuracy follow-up: frozen protocol

Use new outputs only, preserve all previous caches/models/manuscript. Existing
test exposure is explicit: development evaluation, not pristine confirmation.
No nonlinear architecture or feature search, no DroneVehicle expansion.

## A: matched-input linear controls

Reuse accuracy_upgrade_20260908 phaseA exact train/validation/test matrices and
labels (9150/2646/2808 rows): Qwen2combined52, Qwen2.5/Smolsemantics42, training-
fitted appended-column scaling unchanged. Reuse and verify all30enhanced-network
models/scores; no nonlinear retraining. Preserve old18-dimensional linear.

Each receiver: reproduce original double-logistic optimizer (zero initialization,
400 full-batch GD updates, step.5, L2alpha.001 including bias). Also solve the
same objective by fixed L-BFGS(maxiter5000,gtol1e-8,ftol1e-12) to diagnose and
avoid relying on an underconverged optimizer. The latter is the main converged
BCE linear baseline, not a validation-selected hyperparameter. Keep both.
For Qwen2.5/Smol, also fit a single closed-form ridge model to y_image-y_detection,
objective mean squared error/2 +.001*||w||²/2 including bias. No regularizer search.
Record all training objectives, gradient norms, convergence and validation values.
Freeze models and all validation policies before one final test evaluation.

## B: accuracy-resource tradeoffs

Scores are p_image-p_detection for dualBCE models and raw predicted expected
correctness difference for advantage regressors. No conditional-disagreement
score is used. Report regression predictions outside[-1,1], do not silently clip.

Common metric: image-use fraction, no physical energy assumption. All receivers
use a dimensionless penalty kappa: choose image iff score>kappa. Freeze26points:
0 plus logspace(-4,0,25). This is not the physical energy price lambda and does
not imply an unknown Joule conversion. Fixed branches are separate endpoints.
For every model/seed select the smallest validation image-use among grid points
meeting the same absolute validation accuracy target0.74, ties smaller penalty.
Report infeasible models/seeds explicitly. Also preserve original own-zero-price
minus1pp policy as a clearly labeled relative-constraint secondary analysis;
do not claim dominance from methods with different accuracy targets. Show full
curves and both validation/test outcomes of validation-selected points.

For Qwen2 only, optionally repeat with the original26physical-lambda points
0+logspace(-5,-1,25) and original per-query payload/detector/VLM accounting.
Gate all queried JPEG hashes against the matching formal payload audit, unchanged
codec source hash and original power artifact. If any identity is mismatched,
do not borrow unmatched payload energy or pretend the full curve is calibrated;
report the blocker and retain valid image-use results. Qwen2.5/Smol have no
independent energy measurement and receive no Joule claims. Additional feature
construction/router costs are unmeasured and not silently included. New7B/new
detector answers never enter A/B labels or energy models.

## C: complete fixed validation set, matched3B/7B NF4

Freeze all2646common validation keys,101images: presence552/counting276/
comparison606/co_presence606/threshold606; per-type image coverage must also be
reported, not claimed uniform. Same original receivedJPEG and prompt; NF4double
quantization, BF16compute/nonquantized modules, official processor min200704/
max802816pixels, SDPA, greedy24newtokens, repetition_penalty1.05. This explicit
greedy flag differs from historical near-greedy sampling at temperature1e-6.
No prompt repair, new question selection, test inference, receiver training,
or silent OOM resizing. Model checkpoints already cached; isolated quant_env only.

Reuse prior120answers/model only after exact model/config/key/JPEG/prompt/grid
and scoring verification. Compute only missing2526/model, save each prediction
atomically with reused/new provenance and timing. Full set is frozen before any
new answers: earlier measured.93s/.64s implies roughly70–90min including
processing, within a2h GPU budget. Stop safely and preserve incomplete outputs
if occupied/OOM/budget reached; do not choose a favorable subset afterward.
Runner supports verified resume. GPU must be idle before each model load; do not
kill any other work. NVML instantaneous power.draw is N/A; a read-only follow-up
check confirms nvidia-smi Power Samples remains available. This run does not have
continuous, phase-aligned samples and a new idle baseline, so report time/memory,
not new Joules. Do not claim that the hardware has no power sensor at all.

## Analysis and completion

daily-coding: read source, scoped scripts, unit checks, execution and reproducible
QA. results-analysis: retain every baseline/seed/curve, exact tables, actual
figures, image-cluster exploratory intervals/tests with multiple-comparison
scope and limitations. Prior benchmark development and correlated video frames
prevent overclaiming. Negative counting/co-presence/comparison results remain.
Separate stage status, manifests/code/source hashes, local-remote SHA verification.
Do not mark the full task complete while GPU inference or QA is unfinished.

## Execution record

A/B completed in 9.55 s: 6 dual-logistic fits, 2 ridge fits; 30 enhanced networks
and 3 legacy linear controls reused. Five unit tests pass. Read-only replay
verified all 41 controls and every 26-point curve/validation choice. The global
selection SHA256 is 575924e3eec234c6710593d433bb853baa45014edee0347e3f96b809c30113ad.
All 5454 Qwen2 validation/test JPEG/payload rows match; physical-lambda results
are explicitly under the old accounting model, not newly measured energy.
Canonical A/B report/figures: analysis_ab_v2; the initial analysis_ab is retained
and v2 fixes the shared figure legend to include ridge (absent for Qwen2).

The Qwen2 same-input linear models match the neural result. Qwen2.5 has a smaller
neural benefit and Smol a larger one; all controls remain. The common validation
74% target does not transfer as a guaranteed test threshold. In particular,
Qwen2.5 network policies use fewer images but have lower test accuracy at the
selected points, preventing a blanket dominance claim.

C launched as PID 1320874 in tmux tgcn-accuracy-followup-vlm-20260908, log
outputs/accuracy_followup_20260908/phase_c.log. Full 2646-key coverage and both
checkpoint/JPEG/prompt/reuse gates passed before launch. GPU budget is two hours;
phase_c/status.json is authoritative. No whole-task COMPLETE while C is running.

## Completed execution and interpretation

C completed naturally in 4396.28 s (73.27 min): 5052 new answers and 240 exact
reused answers, two complete 2646-key receiver runs. PID1320874 exited; GPU was
verified idle (no compute process, 9 MiB, 0%) before being handed to the queued
detector/decoder subtask. No further GPU work is part of this follow-up.

Independent CPU QA checked all5292 answers, source JPEG/prompt/grid/ground-truth
and scorer consistency, and every original field of the240 reused answers.
Eight unit tests pass. All41 A/B scoring controls and six final rendered figures
were inspected. C results: 3B NF4 63.152%, 7B NF4 65.873%, +2.721pp. The larger
model loses4.455pp on co-presence, while counting improves8.696pp. New-only mean
inference time is .6477s versus .9294s. Exploratory101-image-cluster interval for
the overall difference is [-.920,+6.263]pp, p=.1556; do not claim confirmed
stable superiority. The six-SNR curves are nonmonotonic and must not be smoothed
or described as a guaranteed accuracy increase with SNR.

The internal dated decision report also uses results-report. Its required
reference/template files were absent, so its available main-file ten-section
structure was used as a fallback. r00 is an explicit pending project-round
normalization placeholder. No Obsidian write-back or manuscript edit was made.
Local/remote bundle hashes must be verified after final packaging.
