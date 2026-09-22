# LSS-HSR-L for Low-Altitude UAV VQA Semantic Communication

Date: 2026-05-30

Dataset under review:

- Name: LSS-HSR-L, Low-Altitude target recognition dataset based on
  Holographic Staring Radar
- CSTR: `31253.11.sciencedb.radars.00063`
- DOI: `10.57760/sciencedb.radars.00063`
- Stated downloadable file: `数据集及使用说明.zip`
- Stated size: `199.86 MB`
- Sensor: L-band holographic staring radar
- Modalities: Doppler waterfall plot and trajectory data
- Scenes: Shenzhen, Changsha, Chongqing; urban, airport, suburban, and
  related low-altitude environments
- Classes: DJI Air3, DJI Mini3 Pro, DJI Mavic 3e, DJI Phantom 4 RTK,
  small sparrow, bird flock, large migratory bird, fixed ground rotating
  target, and car

## Verdict

LSS-HSR-L is useful, but not as a direct replacement for a camera-VQA
dataset. It is useful as the low-altitude patrol-warning semantic layer
that tells the communication system what event is happening, what target
type is likely, and which question should be asked next.

For our direction, the best framing is:

> radar-triggered VQA semantic communication for low-altitude patrol
> warning, where radar supplies low-altitude target semantics and UAV
> visual sensing supplies confirmatory VQA evidence.

This lets the work move from a generic "UAV sends image semantics for
VQA" story to a low-altitude economy patrol-warning system:

1. A ground or infrastructure radar detects a low-slow-small target.
2. The system forms a semantic query such as "Is this target a UAV or a
   bird?", "Is it approaching the airport boundary?", or "Which UAV
   should be dispatched to confirm it?"
3. UAVs allocate visual-symbol budget, trajectory, channel, and power
   according to the warning question.
4. The semantic communication layer sends only the radar/VQA evidence
   needed to answer the warning question under latency and bandwidth
   limits.

## Why It Fits the Low-Altitude Economy Setting

| Need in our system | What LSS-HSR-L contributes | How to use it |
|---|---|---|
| Real low-altitude targets | Drone, bird, fixed rotating object, and car categories | Replace abstract question types with patrol-warning target questions |
| Urban/airport/suburban deployment | Multi-city, multi-scene collection | Create domain/regime labels for nominal and stress evaluation |
| Early warning semantics | Doppler waterfall plot and trajectory data | Generate target-class, motion, and threat-level semantic features |
| UAV tasking | Target class and trajectory can imply whether visual confirmation is needed | Add a radar-triggered UAV dispatch / symbol-budget policy |
| Safety gate | Bird-vs-UAV ambiguity and airport scenes create floor-sensitive questions | Train or evaluate risk gates on high-consequence question types |

## What It Cannot Do Alone

- It is not a natural-image VQA dataset.
- It likely does not contain free-form questions or text answers.
- It does not directly give onboard UAV camera views.
- It cannot by itself validate DeepSC-VQA image-question transmission.
- It should not be used for commercial or engineering deployment until
  the downloaded usage instructions and license are checked.

Therefore it should be treated as a radar semantic event dataset that
drives or complements VQA, not as the VQA dataset itself.

## New System Narrative Enabled by the Dataset

The dataset supports a stronger and more original narrative:

**Radar-triggered question-conditioned semantic communication.**
Low-altitude patrol warning is not a passive image-upload task. A radar
event first determines the question type and urgency. The UAV semantic
communication system then decides whether to spend visual symbols,
which UAV should confirm, and whether the answer must prioritize target
identity, trajectory, or threat boundary crossing.

This naturally creates three semantic-question groups:

1. **Identity questions:** "Is the target a UAV, bird, car, or fixed
   rotating object?"
2. **Trajectory / warning questions:** "Is it approaching a protected
   area?", "Is its motion consistent with a drone?"
3. **Confirmation questions:** "Should a UAV camera be dispatched for
   visual confirmation?", "Which UAV should send high-fidelity visual
   semantics?"

These questions match the role-split architecture:

- **UAV layer:** chooses patrol trajectory, confirmation route, channel
  reuse, and radar-triggered visual-symbol budget.
- **UE/query layer:** chooses local symbol/power for the specific
  warning question.
- **Typed price:** becomes target-type-by-question pressure, e.g.,
  higher price for drone-vs-bird ambiguity near airport scenes.
- **Risk gate:** prevents a cheap semantic shortcut when the radar class
  is ambiguous or the target enters a high-risk region.

## Experiment Designs

### E1. Radar Semantic Event Classifier

Train a simple classifier on Doppler waterfall plots and/or trajectory
features:

- Input: waterfall image, trajectory sequence, or both.
- Output: 9-way target class and 4-group superclass.
- Purpose: produce radar semantic confidence, not final VQA proof.
- Metrics: accuracy, macro-F1, confusion between drones and birds.

Why it helps: the confusion matrix defines which questions are
semantically high risk. Drone-vs-bird mistakes become the natural floor
violation events for the risk gate.

### E2. Radar-to-Question Generator

Convert each radar sample into structured warning questions:

- `identity`: "What type of low-altitude target is present?"
- `drone_alert`: "Is this target likely a rotary-wing UAV?"
- `bird_false_alarm`: "Is this warning likely caused by birds?"
- `trajectory_risk`: "Is the target motion consistent with intrusion?"
- `confirm_needed`: "Should UAV visual confirmation be requested?"

The answer labels can be derived from class and trajectory metadata.
This is not open-ended VQA; it is a controlled radar-question answering
task aligned with patrol warning.

### E3. Semantic Communication Budgeting

Use the radar classifier confidence and question type to set semantic
prices:

- High confidence benign bird/car: lower visual-symbol budget.
- Drone-like or ambiguous target: higher visual-symbol budget and lower
  tolerated answer floor.
- Airport or protected-area scene: higher typed price and stricter risk
  threshold.

This can be evaluated with existing QRS gates by replacing synthetic
question types with radar-derived warning question types.

### E4. Radar + UAV Visual Confirmation Simulation

If no paired camera data exists, simulate visual confirmation as a
second-stage semantic source:

- Radar creates the warning event and query.
- UAV visual VQA is invoked only for uncertain/high-risk events.
- The communication controller decides when to pay the additional
  visual semantic cost.

This keeps the claim honest: LSS-HSR-L provides radar warning semantics,
while the existing DeepSC-VQA LUT remains the visual-semantic
communication model.

## Concrete Changes to Our Architecture

Add these state features:

- `radar_target_group`: drone / biological / fixed-rotating / vehicle
- `radar_class_confidence`
- `drone_bird_margin`
- `trajectory_risk_score`
- `scene_type`: urban / airport / suburban
- `confirm_needed`

Add these question types:

- `target_identity`
- `drone_or_bird`
- `intrusion_risk`
- `visual_confirm`

Add typed price dimensions:

- UAV index `m`
- question type `q`
- target group `c`
- scene risk `r`

The original price $\lambda_{m,q}$ can be extended to
$\lambda_{m,q,c}$ or $\lambda_{m,q,r}$ for patrol-warning semantics.
This is a natural system innovation: semantic price now reflects both
the question and the low-altitude target context.

## Usefulness Score

| Criterion | Score | Reason |
|---|---:|---|
| Low-altitude economy relevance | 5/5 | Directly targets low-slow-small objects in real city / airport / suburban scenes |
| UAV patrol-warning relevance | 5/5 | Drone-vs-bird and trajectory warning are central patrol questions |
| Direct VQA compatibility | 2/5 | No free-form visual questions unless we construct them |
| Semantic communication relevance | 4/5 | Strong for semantic event-triggering and budget control |
| Immediate experiment effort | 3/5 | Requires downloading, parsing README/license, and building derived question labels |
| Paper novelty value | 5/5 | Shifts story from generic VQA transmission to radar-triggered patrol-warning semantic control |

Overall: **worth using as an auxiliary low-altitude semantic-event
dataset**, especially for narrative innovation and an additional
patrol-warning experiment. It should not replace the existing VQA
semantic communication pipeline.

## Required Download Checks

After downloading `数据集及使用说明.zip`, verify:

1. License and allowed use: academic only, non-commercial, derivative
   restrictions, required citation format.
2. File layout: whether waterfall plots are image files, MATLAB files,
   NumPy arrays, or mixed formats.
3. Labels: exact mapping for the 9 classes and whether superclass labels
   are already present.
4. Trajectory format: coordinates, timestamps, radar range/velocity, or
   processed tracks.
5. Scene metadata: whether Shenzhen/Changsha/Chongqing and
   urban/airport/suburban labels are per sample or only descriptive.
6. Train/test split: whether an official split exists.
7. Whether any paired optical/camera data exists; if not, do not call it
   a visual VQA dataset.

## Citation Plan

If license permits use, cite:

- Dataset name: LSS-HSR-L: Low-Altitude target recognition dataset
  based on Holographic Staring Radar
- DOI: `10.57760/sciencedb.radars.00063`
- CSTR: `31253.11.sciencedb.radars.00063`
- Repository: ScienceDB / Science Data Bank

Use the exact authors and recommended citation from the downloaded
README or ScienceDB citation tab.
