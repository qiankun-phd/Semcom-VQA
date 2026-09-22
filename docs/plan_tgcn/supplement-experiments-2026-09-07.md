# Frozen supplemental experiment protocol — 2026-09-07

Authorized by the user's request to start necessary supplemental experiments.
Original data, formal_v1 artifacts and paper TeX remain unchanged.

## Questions and fixed design

1. Does a linear router obtain comparable accuracy–energy behavior under the
   same price-selection procedure as the full MLP? Reuse saved linear weights,
   three channels, both train-only and no-count-calibration modes. Save all 26
   original prices, all validation/test predictions and branch selections.
   Select minimum validation energy with accuracy no more than 0.01 below each
   router's own unpriced validation accuracy; ties choose smaller price. This
   is not an equal absolute accuracy target or a test-accuracy guarantee.
2. Which safe inputs support the MLP result? Train-only calibration, three
   channels, seeds 0–9, three variants: remove detector count/nonzero inputs
   (indices 0–15); remove SNR (0–14,16,17); question-type-only (0–4).
   Thus 90 paired-branch fits, each using unchanged `fit_pair`, (32,16) hidden
   layers, original optimizer, maximum 300 epochs and image-validation early
   stopping. Full 18-feature MLP results are reused, not re-trained.

The original image split, labels, calibration, energy and price grid are frozen.
No hyperparameter search, new VLM inference, detector inference, codec replay,
test-driven selection, new task generation or test-set reshuffling is allowed.

## Accounting and comparison boundaries

All primary learned variants pay the detector cost on every query to isolate
feature effects. For no-detector-input and question-type-only variants, save a
separate deployment-cost sensitivity: same decisions and selected price, with
detector cost removed only on the image branch. Detection-evidence selections
still require the detector. Do not merge sensitivity costs into the main table.
The unchanged energy model excludes encoder/router/decoder, flight, sensors,
and radio circuitry; edge VLM energy remains a measured proxy.

## Reuse gates

Require COMPLETE formal_v1, exact base script hash and package versions; exact
input CSV audit/hash, image splits and test ordering; payload audit covers all
keys with every receiver JPEG matched and unchanged codec source hash; exact
reconstructed labels/energy, saved linear test probabilities and decisions.
Replay and verify every full-model validation/test sweep, prices, per-row picks
and selected index. Hash all reused JSON/NPZ artifacts and the power input.

## Outputs and execution

New-only remote output:
`/home/qiankun/phd_research/vqa_semcom/outputs/revision_20260907_independent/supplement_v1`

CPU, nice 10, BLAS/OMP/MKL/NumExpr thread count 1, CUDA disabled. A separate
one-channel/one-seed/two-epoch smoke run must pass before the formal tmux launch.
Save immutable configuration manifest before fitting, progress/completion state,
all seed models/history/probabilities/picks, all sweep points, and descriptive
means/sample SD across every declared seed. No best-seed reporting.

## Launch evidence

- Five regression tests passed remotely in RA_DI.
- Real-data smoke: AWGN, both linear calibration modes, three ablations,
  seed 0, two epochs; COMPLETE. All AWGN formal reuse checks passed.
- Formal launch: `tgcn-supplement-20260907`, PID `4146236`.
- Start UNIX time: `1788797177.7611237` (server clock).
- Script SHA256: `94e04c162784fe6ab8a81a9cdc9c6c98ea20d4bb767597e6f4e61a8a0e3a67ca`.
- Log: `outputs/revision_20260907_independent/supplement_v1.log`.
- Initial process check: CPU 100%, ~0.7% host memory, no GPU inference.
- Estimated queue duration at launch: 5–15 minutes, not a completion guarantee.

## Completion and integrity check

Formal status: **COMPLETE**, 90 paired-branch ablation fits and six linear sweeps.
Actual runtime: **182.333 seconds** (3 minutes 2 seconds), with no VLM or detector
inference. The original full MLP results were reused, not re-trained.

Post-run read-only checks passed for all 96 outputs: all 26 validation/test
prices, per-row picks, validation-only selected index, selected test metrics,
finite NPZ arrays and ZIP CRC, 90 saved models, all training curves within the
300-epoch budget, and the separately reported detector-free-image-path cost
sensitivity. Logs contain exactly 90 ablation and six linear completion markers,
and no Traceback, Warning or Error. These are integrity checks, not statistical
evidence for one router's superiority. Descriptive complete-seed results are in
`supplement_v1/summary.json`; the existing manuscript has not been changed here.
