# The Evidence–Question Complementarity Principle (UAV-VQA SemCom)

> Principled core finding derived from the head-to-head comparison. Reuses the
> held-out-test predictions across 5 question types × 3 channels × 2 VLM backbones.
> Figure: `outputs/figures/comparison/F6_complementarity.pdf`. 2026-07-01.

---

## 1. The principle (one sentence)

> **In task-oriented VQA semantic communication over a lossy channel, the optimal
> transmit evidence modality is determined by the question's reasoning type, and is
> predictable from question semantics alone: route count-based (symbolic) questions to
> the discrete detector token, and existence-based (perceptual) questions to the image.**

## 2. Evidence: token-gain orders strictly by reasoning type

Δ = acc(detector-token, s1) − acc(image, s2), pooled over AWGN/Rayleigh/Rician, test set:

| question type | reasoning | token | image | Δ = t − i |
|---|---|---|---|---|
| counting (exact count) | symbolic | 0.463 | 0.269 | **+0.194** |
| co_presence (A>0 ∧ B>0) | symbolic | 0.673 | 0.551 | **+0.122** |
| comparison (count A vs B) | symbolic | 0.846 | 0.736 | **+0.111** |
| threshold (count ≥ N) | symbolic | 0.615 | 0.595 | **+0.021** |
| presence (∃ instance) | perceptual | 0.693 | 0.761 | **−0.068** |

All four **symbolic/count-based** questions have Δ>0 (token wins); the single
**perceptual/existence** question has Δ<0 (image wins). Sign is fully separated by
reasoning type. The magnitude also tracks *reliance on reliable discrete counting*:
exact counting (+0.19) ≫ borderline threshold (+0.02, where "≥N" is a close call for
both modalities).

## 3. A zero-parameter semantic rule matches the data-calibrated selector

| routing policy | test accuracy (n=5616) |
|---|---|
| fixed token (M3) | 0.6464 |
| fixed image (M1) | 0.6036 |
| **semantic rule (0-param): symbolic→token, perceptual→image** | **0.6761** |
| **data-calibrated selector (Wilson-LCB over question_type)** | **0.6761** |
| oracle (per-decision) | 0.7521 |

The hand-specified rule selects the **identical** service as the fully data-calibrated
selector for **every** question type (`semantic rule == calibrated policy: True`) and
reaches the **same** test accuracy. **Optimal evidence is predictable from question
semantics — no calibration data or multi-dimensional quality LUT is required for the
selection.** (An ablation separately shows SNR/view/freshness/risk and LCB-vs-mean add
≈0 to this selection under the current setup.)

## 4. Why — an information-sufficiency reading

Route to the modality that preserves the **question's sufficient statistic** under the channel:

- **Symbolic (count) questions.** The sufficient statistic is a discrete count *c*. The
  detector token encodes *c* as a compact symbol (~880 B) that traverses the channel
  essentially losslessly and robustly. The image path must ship pixels whose
  *countability* is destroyed by lossy source-coding + fading, so the count degrades.
  → the token preserves the task-relevant statistic; the image does not.
- **Perceptual (existence) questions.** `presence` hinges on **recall** — spotting even
  one small/occluded instance. A detector suffers missed detections → the token wrongly
  reports "no"; the VLM-on-image retains higher recall for "is there any X".
  → pixels preserve the existence statistic better than a lossy detector's token.

This is a goal-oriented (task-oriented) selection principle: *transmit the representation
that is a sufficient statistic for the specific question, given the channel.*

## 5. Robustness of the principle

- **Cross-channel**: sign of Δ holds on AWGN, Rayleigh, Rician independently.
- **Cross-VLM backbone**: routing (presence→image, counting/comparison→token) is
  identical for Qwen2-VL-2B and Qwen2.5-VL-3B — the effect is a property of the
  *task*, not of a particular VLM.
- **Held-out**: policy learned on train images, reported on disjoint test images.

## 6. How this reframes the contribution

- **Weak (drop or demote):** "a rich multi-dimensional quality LUT + Wilson-LCB risk
  selection" — the ablation does not support it for the selection task.
- **Strong (headline):** *the evidence–question complementarity principle* — a
  predictable, information-sufficiency-grounded, cross-VLM/cross-channel rule for
  goal-oriented evidence selection in VQA semantic communication, empirically validated
  against traditional / analog / token / oracle baselines with the "cliff-effect"
  graceful-degradation property.

### Paper-ready contribution sentence

> "We identify and validate an **evidence–question complementarity principle** for
> goal-oriented VQA semantic communication: the transmit representation that maximizes
> answer accuracy under a lossy air-to-ground channel is a *sufficient statistic for the
> question type* — compact detector tokens for count-based (symbolic) questions and the
> image for existence-based (perceptual) questions. A zero-parameter semantic rule
> derived from this principle matches a fully data-calibrated selector and is robust
> across three fading channels and two VLM backbones."

## 7. Open extensions (raise the ceiling further)

- **Make SNR live**: with a cliff-prone digital image path, the token↔image winner flips
  with SNR, so SNR-conditioning earns its place (and sharpens the cliff figure).
- **Make freshness/risk live**: time-varying cache/AoI scenarios (freshness gates
  cache-vs-transmit) and a risk-veto gate (risk gates a safety fallback).
- **Predict the criterion from raw questions**: a lightweight classifier that maps an
  arbitrary NL question → symbolic/perceptual, so the rule generalizes beyond templated
  types.
- **Resource-allocation leg**: where the multi-dim LUT + risk-aware LCB genuinely earn
  their place (constrained multi-stream allocation), turning the router into a learned
  semantic resource-allocation system.
