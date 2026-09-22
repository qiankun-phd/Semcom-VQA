# EXP-011: image rate × receiver visual budget

Completed development result: 120 images × 9 settings. The 4k/medium setting
matches the 4k/high correct count (93/120), with 48.93% fewer mean visual tokens
and 17.71% lower median generation time. It does **not** pass the preregistered
20% time-saving screen. The joint answer-informed oracle reaches 100/120 versus
97/120 and 98/120 for the strongest one-axis bounds; no trained router or test
result is implied. See the [aggregate analysis](../../../../docs/experiments/EXP-011/analysis-report.md).

Development-only feasibility test. No encoder, receiver or controller training is
performed by the grid runner. No test labels are read. The protocol is frozen
before the first inference and hashed into every record.

## Three steps and stopping rules

1. Run existing ordinary neural **full packets** at 2000/4000/8000 B against three
   receiver pixel targets. Reuse the 360 audited packet/reconstructed-PNG pairs;
   run 120 × 9 fresh predictions to obtain comparable timings and actual visual
   tokens. Historical predictions lack token counts and comparable runtime
   telemetry, so they are an audit reference, not replacements for new timings.
2. Compare all fixed settings, sequential independent fixed selection, one-axis
   answer-informed oracles and a joint answer-informed oracle. Each one-axis
   oracle uses its **best fixed other axis**, not only the 4k/high reference slices.
   Oracles are bounds,
   not deployable controllers. The 120 development images have been used before;
   neither these results nor bootstrap intervals establish generalization.
3. Only on positive development evidence, freeze a **deployable** policy and the
   comparison before wireless evaluation and the independently held-out test.
   A positive oracle alone does not authorize reporting a trained controller or
   evaluating test with per-question answer-informed selection. If there is no
   benefit, stop and preserve the negative result.

Screening thresholds in `protocol.json` are exploratory engineering decisions,
not proven noninferiority margins. The primary reference is 4000 B/high; the
primary oracle tradeoff coefficient is 0.05. Cost coefficients are dimensionless
utility weights, **not an energy model**. At this stage measure image bytes,
actual visual tokens and latency. GPU energy is unavailable on the current
machine and must remain null, not zero.

## What is and is not changing

```
cached ordinary neural packet (2/4/8 kB)
  -> verified unchanged RGB reconstruction
  -> receiver-only resolution target (low/medium/high)
  -> frozen standard Qwen + LoRA -> answer
```

All tiers keep the full image view. Low/medium/high mean requested pixel areas
50176/100352/200704; Qwen rounds to a patch grid. Actual visual token counts must
be measured from `image_grid_thw` and cross-checked against image placeholders.
These are not hard token caps. High preserves the historical processor target;
it may upscale the 320-long-edge codec reconstruction. Thus a speedup can be
removal of receiver upsampling, not novel semantic token pruning. Resizing on the
receiver does **not** reduce transmitted bytes.

Each 2/4/8 kB packet is independent, not a prefix of another. Symbols are accounted
as `510 * ceil(actual_bytes / 48)` under the existing digital PHY. This stage does
not simulate noisy delivery. Clean packet roundtrip is not wireless delivery.
Timing excludes source encoding, bitstream decoding, transmission and model load;
it cannot stand in for total system latency or energy.

## Runtime and dependencies

The runner is a small adapter over the audited EXP-009 runtime. Supply the existing
artifact directory and standard receiver adapter explicitly; model weights,
dataset images, predictions and historical runtime dependencies are **not** in
this public Git repository. This is not a standalone pretrained-model release.
The current validated runtime uses Python 3.10, torch 2.10.0, transformers 4.57.3,
peft 0.18.0, bitsandbytes 0.49.1, CompressAI 1.2.8 and Pillow 12.0.0. Its historical
runtime import chain must be present. A missing dependency is an error, not a
reason to silently change models or preprocessing.

From this directory (replace paths with your artifact locations):

```bash
python -m unittest discover -s . -p 'test_*.py'
python launch.py --stage1-root /path/to/qwen_compression_joint_stage1_20260921 \
  --adapter /path/to/tdiuc_qlora_pilot_20260916/standard/final_adapter \
  --output /path/to/rgb_rate_visual_budget_run
```

The supervisor runs six images (one per type) as a full-grid technical smoke,
then resumes those exact records in the 120-image grid and analyzes development
labels. Smoke tests geometry, decoding and memory, not accuracy acceptance.
Detailed logs remain in `logs/`; `status.json` and `decision.json` are the compact
progress/decision interfaces. A two-hour default wall-clock limit bounds the
process. Each process invocation has three excluded warmups. Interrupted inference is resumable only if all frozen inputs and code
hashes still agree. No test phase or new model training starts automatically.

Use a **new output directory** after changing code/protocol. Preserve failed runs
and logs for diagnosis. Do not commit model weights, raw questions, images, host
details or credentials. Only code, protocol and aggregate reports belong in Git.

## Conditional third-step handoff

After complete analysis, `prepare_stage3.py` revalidates predictions and scoring,
then freezes a qualifying **fixed** candidate (most correct, then fewest bytes,
then lowest median latency). It refuses partial results, increased image traffic
and altered analysis. If only the oracle passes, it records that a deployable
controller must still be trained; it never relabels oracle choices as a policy.

```bash
python prepare_stage3.py --analysis /path/to/run/analysis \
  --records /path/to/run/records.json --protocol protocol.json \
  --output /path/to/run/stage3_freeze.json
```

The observed run returns `CONTROLLER_TRAINING_PROTOCOL_NEEDED`, not a frozen
deployable policy. This is the intended stop before test access, not a failed
inference job. The first two stages are complete; the third validation stage is
pending a trained, development-validated and frozen controller.

This file is a validation plan, not a completed wireless/test experiment. The
frozen reference remains 4k/high, including all failed deliveries in the final
accuracy denominator. The shared channel definition and seeds are in the frozen
protocol. Before expanding, repeat the selected timing comparison to rule out a
one-run clock/load artifact. New test encoding, PHY sweeps, deployed controller
training (if needed), and UAV energy instrumentation are not implemented or run
by this bounded development supervisor.
