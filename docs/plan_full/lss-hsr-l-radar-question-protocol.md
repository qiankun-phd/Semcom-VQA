# LSS-HSR-L Radar-to-Question Experiment Protocol

Date: 2026-05-30

Purpose: turn LSS-HSR-L from a low-altitude radar recognition dataset
into an auxiliary semantic-event source for UAV patrol-warning VQA
semantic communication.

Architecture note:
`plan/lss-hsr-l-radar-vqa-system-architecture.md` explains the UAV/UE
algorithm split and the narrative innovation behind this protocol.

## Core Task

Use radar to decide which VQA / confirmation question matters before
the UAV spends visual semantic bandwidth.

The experiment is not "radar replaces VQA." The experiment is:

1. Radar observes a low-altitude target.
2. Radar semantics produce target class, confidence, trajectory, and
   scene-risk features.
3. These features generate patrol-warning question types.
4. The semantic communication controller decides visual-symbol budget,
   transmit power, UAV confirmation priority, and risk fallback.

To run the whole interface before downloading real data, use the
synthetic placeholder demo:

```bash
python3 hppo-uav/run/lss_hsr_l_demo_pipeline.py
```

This writes a tiny fake LSS-style folder plus manifest, question-label,
and policy-simulation outputs. It is only a pipeline demonstration, not
real dataset evidence.

## Label Schema

### Target Superclass

| Superclass | LSS-HSR-L classes |
|---|---|
| `rotary_uav` | DJI Air3, DJI Mini3 Pro, DJI Mavic 3e, DJI Phantom 4 RTK |
| `biological` | small sparrow, bird flock, large migratory bird |
| `fixed_rotating` | ground fixed rotating target |
| `vehicle` | car |

### Derived Question Types

| Question type | Example question | Label rule |
|---|---|---|
| `target_identity` | What type of low-altitude target is present? | 9-way class label |
| `drone_or_bird` | Is this target a rotary-wing UAV rather than a bird? | `rotary_uav` vs `biological`; others map to `not_applicable` or `other` |
| `intrusion_risk` | Does the trajectory suggest patrol-warning risk? | trajectory speed/heading/range rule after README confirms fields |
| `visual_confirm` | Should a UAV camera be dispatched for visual confirmation? | high-risk scene OR low drone-bird margin OR unknown/ambiguous class |

## Features for the Semantic Controller

Add these features to the existing question-conditioned state:

```text
radar_target_group
radar_class_id
radar_class_confidence
drone_bird_margin
trajectory_speed_bin
trajectory_heading_change_bin
scene_type
scene_risk_weight
confirm_needed
```

If the dataset does not include per-sample scene labels, use target
class and trajectory only, and keep scene type as a dataset-level
description rather than a model feature.

## Semantic Price Extension

Current typed price:

```text
lambda[m, q]
```

Patrol-warning extension:

```text
lambda[m, q, c]
```

where:

- `m`: serving UAV or confirmation UAV
- `q`: question type
- `c`: radar target superclass

Optional higher-risk variant:

```text
lambda[m, q, c] += scene_risk_weight * floor_violation
```

This makes drone-vs-bird ambiguity near airport-like scenes more
expensive than a high-confidence benign target.

## Experiment Phases

### Phase A: Dataset Audit

Run:

```bash
python3 hppo-uav/run/lss_hsr_l_dataset_audit.py \
  --dataset-path /path/to/数据集及使用说明.zip
```

Proceed only if:

- README / usage instructions exist.
- license allows academic use and derived labels.
- class names or label files confirm the 9-way target mapping.
- Doppler waterfall and/or trajectory data are identifiable.

### Phase B: Radar Semantic Event Baseline

Build a simple radar classifier:

- Waterfall-only baseline
- Trajectory-only baseline
- Late-fusion baseline

Metrics:

- 9-way accuracy
- 4-way superclass macro-F1
- drone-vs-bird false alarm / miss rate
- calibration error if confidence is used by the risk gate

### Phase C: Radar-Question Label Builder

First build a sample manifest from the downloaded zip or extracted
directory:

```bash
python3 hppo-uav/run/lss_hsr_l_manifest_builder.py \
  --dataset-path /path/to/数据集及使用说明.zip
```

This produces:

```text
paper/data/lss_hsr_l_sample_manifest.csv
```

The manifest contains:

```text
sample_id,target_class,target_group,source_path,modality,scene_type,
radar_confidence,drone_bird_margin,trajectory_risk_score
```

The confidence and risk fields are intentionally blank unless they come
from a classifier or trajectory rule. After filling or joining those
fields, generate the long-form radar-question table:

```bash
python3 hppo-uav/run/lss_hsr_l_question_builder.py \
  --input-csv paper/data/lss_hsr_l_sample_manifest.csv
```

The generated table has one row per derived question:

```text
sample_id,target_class,target_group,question_type,question,answer,
radar_confidence,drone_bird_margin,trajectory_risk_score,scene_type,
confirm_needed,semantic_priority,label_source
```

This table becomes the bridge between LSS-HSR-L and the semantic
communication simulator. The `visual_confirm` rows decide whether
the UAV visual semantic link should spend confirmation bandwidth; the
`semantic_priority` field can be mapped to typed price or risk-threshold
settings.

### Phase D: Semantic-Control Feature Adapter

Convert the long-form radar-question table into the actual control
features used by the two-level architecture:

```bash
python3 hppo-uav/run/lss_hsr_l_feature_adapter.py \
  --input-csv paper/data/lss_hsr_l_radar_questions.csv
```

This produces:

```text
paper/data/lss_hsr_l_semantic_control_features.csv
```

The feature table adds:

```text
target_group_id,question_type_id,scene_risk_weight,
semantic_priority_weight,risk_gate_enabled,risk_threshold,
visual_symbol_floor,typed_price_seed,uav_controller_hint,
ue_selector_hint
```

This is the system-level innovation interface. The UAV side receives
`uav_controller_hint`, `risk_threshold`, and `visual_symbol_floor` for
slow patrol/confirmation control. The UE/query side receives
`ue_selector_hint`, `question_type_id`, and `typed_price_seed` for fast
question selection and target-aware semantic pricing.

Recommended algorithm split:

- UAV controller: constrained PPO / HPPO-style slow control for route,
  confirmation dispatch, transmit power, and shared symbol floor.
- UE/query controller: contextual bandit or PDQN-style selector for the
  current warning question and visual-confirmation action.
- Coupling variable: `typed_price_seed`, initialized from
  `lambda[m,q,c]` and refined by learned risk feedback.
- Safety fallback: `risk_gate_enabled` with a lower `risk_threshold` for
  critical airport or high-trajectory-risk UAV events.

### Phase E: Communication Policy Evaluation

Run the lightweight communication-layer policy probe:

```bash
python3 hppo-uav/run/lss_hsr_l_comm_policy_sim.py \
  --input-csv paper/data/lss_hsr_l_radar_questions.csv
```

Compare:

- no-radar baseline: uniform visual confirmation budget
- radar-triggered greedy: confirm only high-risk or ambiguous samples
- typed-price QRS: use `lambda[m,q,c]`
- LearnRisk-QRS: typed price plus risk fallback

Metrics:

- visual semantic-symbol cost
- warning answer floor rate
- drone-vs-bird miss rate
- false visual-confirm dispatch rate
- critical-priority floor rate
- average visual-symbol cost

This probe is not a final learned-controller result. It checks whether
the radar-triggered semantic interface has the right trade-off shape
before training: lower visual-symbol cost than uniform confirmation,
zero or near-zero UAV miss rate, and bounded false confirmation on
benign birds/cars.

## How This Strengthens the Research Idea

The dataset supports a more original system claim:

> Low-altitude UAV VQA semantic communication should be event-triggered
> by radar semantics, question-conditioned by patrol-warning need, and
> risk-gated before spending scarce visual semantic bandwidth.

This is stronger than a generic VQA pipeline because the system decides
whether and how to ask a visual question from radar context. The
algorithmic architecture then follows naturally:

- UAV algorithm: slow constrained control for patrol/confirmation route
  and shared visual-symbol budgets.
- UE/query algorithm: contextual bandit or PDQN-style selector for the
  current warning question.
- Interface: target-aware typed semantic price.
- Safety: risk head for ambiguous drone-vs-bird or high-risk trajectory
  events.

## Stop Conditions

Do not use LSS-HSR-L as evidence for the VQA system if:

- license forbids derived labels or redistribution of processed
  features;
- class labels are unavailable or ambiguous;
- trajectory data cannot be mapped to target samples;
- README shows the data is only for detection/recognition benchmarks
  and forbids secondary use;
- no defensible bridge exists from radar semantics to VQA confirmation.
