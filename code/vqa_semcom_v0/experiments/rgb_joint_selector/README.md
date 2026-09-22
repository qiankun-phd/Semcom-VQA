# EXP-012: learned joint rate / receiver visual-budget selector

This is a bounded **learnability experiment**, not a new image codec, retrained
VLM, wireless result, or independent-test claim. It follows the EXP-011 grid.

## Registered question and network

Can an inexpensive trained selector choose one of nine `(raw byte cap, visual
pixel target)` actions, preserving answer accuracy while lowering transmitted
image bytes and receiver computation?

```text
question → stable hashed words/bigrams (256) ┐
source RGB → thumbnail/statistics (83)       ├→ MLP 339→128→64→action scores
                                           ┘            ↓ one action only
source RGB → frozen ordinary neural codec at 2/4/8 kB → one-byte tier + codec packet
                  → receiver RGB decode → full-view low/medium/high resize
                  → frozen standard Qwen3-VL-4B + rank-8 LoRA → answer
```

The selector is an ensemble of three seeded small MLPs. Each action head learns
its own probability of correctness with BCE; deployment maximizes predicted
correctness minus the preregistered nominal resource penalty. It does **not**
run nine codec/VLM candidates at deployment. Nine-way inference is offline
training-supervision collection only. The image codec remains question-agnostic;
the learned question/image conditioning changes the system's resource action.

The full-image frame adds the same one-byte receiver tier to every method.
Raw codec caps are 2000/4000/8000 B; full caps are 2001/4001/8001 B. The byte
contains `low=0`, `medium=1`, or `high=2`, **not** the nine-way action index.
The receiver strips this byte before invoking the unchanged codec decoder.
Pixel targets are not token caps: actual visual grids/tokens are measured.

## Data, freeze, and controls

- Fresh TDIUC/COCO: 480 train and 120 internal validation, six question types
  balanced. Exclude project-known receiver training, historical experiment
  manifests, legacy dev120, and sealed-test300 identities before selection.
- JPEG identity/hash checks prevent known duplicates. They do not establish
  absence from foundation-model pretraining. Test JPEGs/truth files stay sealed.
- No answers, ground-truth task types, identities, candidate VLM predictions,
  confidence values, or measured post-inference times enter deployment features.
- Fit image feature standardization on the fresh 480 training images only.
- 8 model variants × 3 seeds = 24 fits: joint nine-head, rate-only at each of
  three fixed tiers, compute-only at each of three fixed rates, question-only.
- New validation chooses minimum-BCE checkpoints, strong fixed action, and
  the fixed other axis for the two single-axis baselines. Freeze everything
  before the one exploratory evaluation on **reused** legacy dev120.
- Report all seeds, the probability ensemble, image-feature shuffle control,
  actual framed bytes, equivalent LDPC symbol accounting, actual visual tokens,
  paired answer changes, and component timings. No radio delivery is simulated.

See [protocol.json](protocol.json) for all thresholds and boundaries. The primary
accuracy/byte reference is 4k/medium; the strongest validation-selected fixed
action additionally participates in utility comparisons. Do not conflate these.
The EXP-011 20% fixed-latency criterion is unchanged.

## Running the bounded pipeline

Requires the audited stage-1 codec/runtime, standard receiver adapter, and
completed EXP-011 grid as external local artifacts; they are not bundled in this
public directory. Use the same pinned offline ML environment. No credentials or
machine addresses are needed in these source files.

1. `prepare_data.py --project-root PROJECT --stage1-root STAGE1 --protocol PROTOCOL --output RUN`
2. `features.py --output RUN --protocol PROTOCOL --legacy-stage1 STAGE1`
3. `encode_data.py --output RUN --protocol PROTOCOL --stage1-root STAGE1 --smoke`
4. `infer_supervision.py --output RUN --protocol PROTOCOL --stage1-root STAGE1 --adapter ADAPTER --grid-code EXP011/code --smoke`
5. `launch.py --output RUN --protocol PROTOCOL --stage1-root STAGE1 --adapter ADAPTER --legacy-grid EXP011`

Run each with the ML environment's Python interpreter. Once both smokes pass,
the locked supervisor sequentially completes 1800 independent codec packets,
5400 VQA supervision records, and 24 selector fits/evaluations. It shares one
absolute eight-hour deadline across resumes, stops on errors, validates hashes,
and writes verbose output to `supervisor_*.log`, not the conversation.
Do not edit hashed inputs/source files during a run. Restarting with unchanged
inputs reuses verified completed records/checkpoints.

Useful status artifacts: `supervisor_status.json`, `encoding_status.json`,
`inference_status.json`, `training_status.json`. Final artifacts:
`validation_report.json`, `controller_frozen.json`, `policy.json`,
`legacy_dev_report.json`, `training_complete.json`.

The complete cost gate remains **PENDING**, even if answer/byte screens pass:
paired live single-selected-path profiling must include feature extraction,
selector, source preprocessing/analysis transform/gain search/entropy coding,
decoder, receiver preprocessing, and VLM generation. Separately timed cached
components are not a measured end-to-end latency. Energy is unavailable, not
zero. Test300 and SNR experiments are never started automatically.

## Verification

```bash
python -m unittest discover -s code/vqa_semcom_v0/experiments/rgb_joint_selector -p 'test_*.py'
```

Tests cover data separation, budget framing, tier/grid verification, features,
actual small CPU model fitting/checkpoint restore, no-label deployment, frozen
legacy-truth access, source/hash guards, supervisor timeout and duplicate lock.
