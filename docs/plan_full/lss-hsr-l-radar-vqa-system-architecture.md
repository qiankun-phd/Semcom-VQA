# LSS-HSR-L Radar-Triggered VQA Semantic Communication Architecture

Date: 2026-05-30

Purpose: define the system innovation opened by LSS-HSR-L for low-altitude
patrol-warning VQA semantic communication. This note focuses on narrative and
architecture, not manuscript prose.

Dataset anchor:

- Name: LSS-HSR-L, Low-Altitude target recognition dataset based on
  Holographic Staring Radar
- CSTR: `31253.11.sciencedb.radars.00063`
- DOI: `10.57760/sciencedb.radars.00063`
- Source page: ScienceDB / Science Data Bank dataset page
- Public file indicated by the page/user-provided metadata:
  `数据集及使用说明.zip`, `199.86 MB`

## Core Reframing

LSS-HSR-L should not be used as a replacement for VQA images. Its value is
that it can create a radar semantic event layer before visual semantic
transmission starts.

The system claim becomes:

> A low-altitude patrol-warning network should first infer radar target
> semantics, then ask only the visual questions that reduce warning risk under
> limited UAV communication resources.

This is stronger than a generic UAV-VQA semantic communication story because
the communication system decides which question is worth asking from an
external sensing event, rather than transmitting visual semantics uniformly.

## System Layers

| Layer | Role | LSS-HSR-L signal | Output to communication system |
|---|---|---|---|
| Radar event layer | Detect and classify low-altitude target | Doppler waterfall, trajectory | target group, confidence, trajectory risk |
| Question layer | Convert event into warning question | target class, target group, scene | target_identity, drone_or_bird, intrusion_risk, visual_confirm |
| Feature adapter | Convert questions into control features | long-form question table | risk gate, typed price seed, visual symbol floor |
| UAV slow controller | Patrol route and confirmation resource control | feature table, queue state, channel state | confirmation dispatch, power, symbol floor |
| UE/query fast selector | Current semantic question selection | feature table, answer risk, link state | question action, answer floor, VQA budget split |
| Risk fallback | Prevent unsafe under-communication | critical/high-risk features | forced confirmation or higher semantic budget |

## Algorithm Architecture

### UAV Side

Use a constrained slow controller. The natural candidates are:

- constrained PPO / HPPO when UAV movement, power, and confirmation dispatch
  are continuous-discrete hybrid actions;
- a Lyapunov or CMDP wrapper when warning-floor violations must be bounded;
- a high-level route allocator if multiple UAVs share patrol areas.

Inputs:

```text
uav_controller_hint
risk_threshold
visual_symbol_floor
scene_risk_weight
trajectory_risk_score
typed_price_seed
channel_state
remaining_symbol_budget
```

Outputs:

```text
confirmation_uav_id
patrol_or_intercept_action
transmit_power
minimum_visual_symbols
fallback_trigger
```

### UE / Query Side

Use a fast contextual selector for the current warning question. The natural
candidates are:

- contextual bandit for question choice when the action is mostly discrete;
- PDQN-style hybrid selector when question choice and semantic budget are
  jointly selected;
- risk-aware neural bandit when false negative cost differs sharply between
  UAV, bird, car, and fixed rotating targets.

Inputs:

```text
ue_selector_hint
question_type_id
target_group_id
semantic_priority_weight
typed_price_seed
radar_confidence
drone_bird_margin
```

Outputs:

```text
question_action
answer_floor
vqa_semantic_symbol_budget
whether_to_request_visual_confirmation
```

## Originality Points

1. Radar-triggered question formation

   The system does not assume a fixed VQA query stream. Radar semantics decide
   whether the next useful question is identity, drone-vs-bird, intrusion
   risk, or visual confirmation.

2. Target-aware typed semantic price

   The price extends from `lambda[m,q]` to `lambda[m,q,c]`, where `c` is the
   radar target superclass. A drone-vs-bird ambiguity near an airport becomes
   more expensive than a high-confidence car.

3. Split controller design

   UAVs solve slow constrained patrol/confirmation decisions, while UE/query
   agents solve fast semantic question selection. This avoids forcing one
   monolithic RL policy to learn both timescales.

4. Risk-gated visual confirmation

   Visual VQA bandwidth is spent only when radar uncertainty or trajectory risk
   justifies it. This creates an explicit low-altitude-economy patrol-warning
   story: conserve visual semantic bandwidth while protecting critical warning
   answers.

5. Modality-complementary evaluation

   The radar dataset can support communication experiments even without direct
   camera images by testing whether radar events reduce visual-confirmation
   cost, lower false dispatches, and preserve warning floors.

## What LSS-HSR-L Can Validate

Strong validation candidates:

- 9-way radar target recognition and 4-way superclass macro-F1;
- drone-vs-bird ambiguity detection;
- trajectory-risk to confirmation-priority mapping;
- radar-triggered reduction in visual confirmation cost;
- zero or near-zero rotary-UAV miss rate under risk fallback;
- lower false visual confirmation on birds/cars than uniform confirmation.

Weak or indirect candidates:

- final visual VQA answer accuracy, unless a visual dataset or simulator is
  joined;
- natural-language VQA quality;
- commercial low-altitude deployment claims before license and operational
  assumptions are verified.

## Minimum Experimental Gates

1. Dataset gate: README, license, label mapping, and sample pairing are
   verified from the downloaded archive.
2. Radar gate: radar classifier or rule provides confidence, target group, and
   trajectory risk.
3. Question gate: each sample maps to derived patrol-warning questions.
4. Feature gate: each question row maps to `typed_price_seed`,
   `risk_threshold`, and UAV/UE controller hints.
5. Communication gate: radar-triggered policies reduce visual symbol cost
   while preserving warning-floor and rotary-UAV miss constraints.

## Current Local Toolkit

Run the synthetic placeholder demo:

```bash
python3 hppo-uav/run/lss_hsr_l_demo_pipeline.py
```

Run each stage on downloaded data after license review:

```bash
python3 hppo-uav/run/lss_hsr_l_dataset_audit.py --dataset-path /path/to/数据集及使用说明.zip
python3 hppo-uav/run/lss_hsr_l_manifest_builder.py --dataset-path /path/to/数据集及使用说明.zip
python3 hppo-uav/run/lss_hsr_l_question_builder.py --input-csv paper/data/lss_hsr_l_sample_manifest.csv
python3 hppo-uav/run/lss_hsr_l_feature_adapter.py --input-csv paper/data/lss_hsr_l_radar_questions.csv
python3 hppo-uav/run/lss_hsr_l_comm_policy_sim.py --input-csv paper/data/lss_hsr_l_radar_questions.csv
```

The current demo output already checks the interface shape:

```text
question rows: 16
risk-gated rows: 8
max typed price seed: 10.2789
learnrisk_qrs visual symbol cost: 18
uniform visual symbol cost: 24
drone miss rate: 0.0
```

These numbers are only synthetic pipeline evidence. Real claims require the
downloaded LSS-HSR-L archive, license verification, and classifier or trajectory
rules derived from actual data.
