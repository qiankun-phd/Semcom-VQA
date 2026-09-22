# EXP-012 launch verification — 2026-09-22

## Checks completed

|Check|Outcome|
|Python compile check|PASS|
|New selector tests, local|41 / 41 PASS|
|Existing EXP-011 grid tests, local|31 / 31 PASS|
|Existing v0 pipeline regression tests, local|39 / 39 PASS|
|Selector-training and supervisor tests, ML server|18 / 18 PASS; includes real tiny CPU fitting and checkpoint restoration|
|Separate read-only protocol/data audit|PASS after fixing route byte mapping|
|Six-image codec smoke|18 packets; budget and packet-only RGB round trips PASS|
|Six-image receiver smoke|54 answers; all three actual visual tiers PASS|
|Source privacy scan|No credential-pattern, private host, or absolute user/server path matches in new source/docs|
|Whitespace diff check|PASS|
|Static type check|No configured checker/build package for this standalone experiment; not claimed|
|Ruff|Two nonfunctional F401 unused imports (`copy` in test and `hashlib` in training); retained once run source hashes were frozen|
|Coverage|Not measured|

The routing bug found during review is fixed: a joint action is 0–8, but the
frame byte is its receiver tier 0–2. All nine actions now have frame consistency
tests. Nothing in EXP-011's frozen source/output was edited.

The data audit found 600 distinct image IDs and JPEG hashes, balanced 480/120
splits, no identity overlap with 6554 images enumerated from 87 historical
manifests, and no exact-byte overlap with the 2520 accessible historical
training/development JPEGs. Sealed test images were not opened or hash-compared.
Sealed truth files were not read. The parent TDIUC annotation source can contain
excluded identities' answer fields; these are not used for sample selection or
training. No claim is made about foundation-model pretraining overlap.

## Execution boundary

At 09:09 UTC the sequential eight-hour-bounded supervisor was started and actual
full encoding progress was observed. These checks establish that the experiment
is running, **not** that learned routing succeeds. Its subsequent 5400 supervision
records, 24 model fits, validation choices, and frozen legacy-dev check are still
pending at this launch record. No test300, wireless sweep, paper change, or energy
claim is part of this launch.

The full-cost gate intentionally stays `PENDING` until paired live selected-path
timing is measured; cached component sums cannot satisfy it. The learned policy
must be compared with fixed and learned single-axis baselines before any claim
of benefit. A negative learnability result ends this feature/controller version.
