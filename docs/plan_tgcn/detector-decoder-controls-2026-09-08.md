# Detector and receiver decoder controls

User authorization: compare a genuinely different detector and a lightweight decoder alternative; preserve original results/manuscript. Owner: detector_decoder_controls agent. GPU is queued behind crossreceiver_revision's 3B/7B validation, not preempted.

## Decoder protocol (frozen before fitting)

Existing crossreceiver_v2 VisDrone/Rician split and six SNRs. Baseline uses original training-only class/SNR count ratio on counting questions. Two candidates: nonnegative-slope affine count regression and one-dimensional monotone isotonic regression, class/SNR-specific. Train targets are official annotation counts on **QA training images only**. Each image/SNR/class enters once regardless of task repetition. Inputs are parsed received `detector_counts_by_class`, never sender count columns. Candidate count correction feeds the same deterministic presence/count/comparison/co-presence/threshold logic; zero counts remain zero. Validation overall task-weighted accuracy selects among baseline and two candidates; test evaluated once with all candidates retained. Test is previously exposed exploratory development, not a pristine holdout. No scoring tolerance changes.

Reception audit: evidence text contains counts of all received classes, so second-class queries are supported. Original comparison CSV `transmitted_detector_count` stores the SECOND class count. Parsing actual evidence avoids this trap. Old baseline correctness is reproduced on every common decision before fitting. Original confidence/boxes are not used by candidates. No full-frame outage indicator exists in these cached token records; original frame-loss model is preserved, empty received counts remain zero.

CPU trial completed with baseline selected (no candidate improvement). Initial import-only failed attempt preserved in decoder.log; retry1 completes. No GPU activity.

## Detector protocol (pending GPU slot)

Only existing VisDrone-trained checkpoint is YOLOv8n, 50 epochs. Its args use official validation for checkpoint selection, which overlaps current QA splits; this is a historical detector-selection exposure, not a new training overlap. Do not call current benchmark untouched detector test.

Planned fair pilot: YOLOv8n versus YOLOv8s, both COCO initialization then 20 equal epochs of VisDrone training at 640/batch4/seed0 with the same optimizer/settings. The official training set is internally split by stable hash (90% fit/10% detector validation); no official validation image used in detector fitting or checkpoint selection. Existing 11-category mapping including `others` is preserved; task evaluation uses 10 target classes. Ignored regions retain historical conversion (no ignored-region suppression); report it, no official AP comparison claim.

Check first-epoch cost and stop/report if two-model pilot exceeds approximately two GPU hours. Old 50-epoch n run took 4394 s; s cost remains to be measured. Both detectors need same completed epochs for any architecture attribution. The original trained detector is a contextual baseline, not same-budget comparator. Do not call undertrained pilot replacement superior/inferior architecture conclusively.

Freeze detector inference at conf .25, IoU .7, max_det300. Rebuild receiving counts using unchanged deterministic LDPC/Rician loss mechanism and original record format. Fit original ratio calibration on QA training only for each detector; hold decoding algorithm fixed. First validation, then one frozen test evaluation. No router retraining or manuscript edits in this stage. Runtime/model/payload bytes separately measured; no old detector Joules reused, no energy claims if GPU sensor unavailable.

Official model documentation: https://docs.ultralytics.com/models/yolov8/ and https://docs.ultralytics.com/modes/train/ (consulted 2026-09-08).

## Completion

Both detectors completed exactly 20 epochs and frozen QA evaluation. Runtime 4633.92 seconds (77.23 minutes); PID 1394423 exited normally. GPU released and explicitly handed to literature_baselines, which started its T-DeepSC task. No added training beyond the protocol.

Decoder alternatives did not improve validation, so original baseline retained. Test baseline/linear/isotonic accuracies: 67.0940/66.2393/66.9872 percent.

Same-budget detector n20/s20: validation QA 69.8035/70.5593 percent; test QA 67.7350/69.4801 percent. Test s-minus-n is +1.7450 points, with exploratory image-cluster interval [-1.718, 5.231], so no confirmatory superiority claim. Presence/counting/threshold improve, comparison/co-presence decline. Old n50 has higher validation (73.0537) than either new model; do not replace the manuscript based only on test gain. No router retraining, new VLM predictions, or manuscript edits.

Reports: paper/outputs/detector_decoder_controls_20260908/analysis/ and detector_analysis/. Weights and raw records retained in the matching remote output directory; local analysis sync excludes .pt files. Power sensor N/A, no J claims. Single-pass inference timing is descriptive and not a randomized/interleaved architecture speed benchmark.
