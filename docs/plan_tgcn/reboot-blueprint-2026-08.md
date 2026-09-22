# TGCN Reboot Blueprint — Chapter Plan + INSIGHT Collection

> Produced by ARS `academic-paper` plan mode (Socratic), 2026-08-05.
> Manuscript: "Ask Before You Transmit: An Energy-Frugal Evidence-Level Semantic Communication System for UAV Visual Question Answering" — IEEE TGCN.
> Spine decision: **A (energy-lever)**, with **B (complementarity-law → TCCN/TWC)** recorded as fallback (§7).
> **Amendment (same day, post-stress-test): selector strategy changed to PREDICTOR-PRIMARY** — the per-sample parametric predictor becomes the headline routing method; the Wilson-LCB LUT and the zero-parameter rule are retained but demoted to baselines/ablation. C2 is repositioned accordingly (§1a). Advisor recommended the three-selector-ladder alternative; user decided predictor-primary. Schedule extends 4 → 5 weeks.
> Companion document: `scoop-check-2026-08-05.md` (fresh collision check; no HIGH threats; T/②/⑥/⑧ all unclaimed as of 2026-08).

---

## 1. INSIGHT Collection

### [INSIGHT: thesis_statement]
Under the constraint that task accuracy is not below the status-quo (full-image transmission), question-type-conditioned multi-level evidence routing is the energy-optimal transmit policy for UAV VQA over wireless semantic links with real VLM receivers, because the measured per-answer energy frontier is compute-dominated: the highest-value transmit-side decision is whether/what evidence triggers or declines VLM inference (2.2× measured joules reduction at every SNR, accuracy raised; energy-price sweep exposes the full controllable frontier down to token-only at 75–79×).

*(Constraint-optimization form adopted in Step 3 to pre-empt the "token-only is greener" reversal attack.)*

### [INSIGHT: contribution_claim] — user's verbatim answers, Step 2.5

**L5-W1 (what citers will say this established):**
> Establishes that for UAV VQA over wireless semantic links with real VLM receivers, the energy frontier is often compute-dominated; the main transmit-side joules lever is whether/what evidence triggers or declines VLM inference, realized by type-conditioned multi-level evidence routing with measured large joules savings and non-degraded task accuracy under unified channel-use+joules accounting.

**L5-W2 (what is missing without it / who benefits):**
> Without this paper, literature still lacks a joint wireless+real-VLM system study of per-answer joules (VLM forward vs radio) with declining inference as a first-class routing action. Must fill because green semcom and goal-oriented VQA evolved in parallel and designers otherwise over-tune radio while missing the dominant inference energy term. Benefits: UAV/edge VQA and energy-aware semcom system designers.

**L5-W3 (who decides differently):**
> Designers budget joules for VLM invoke/decline and evidence tiers; operators adopt ask/route-before-heavy-inference under energy caps; stacks expose inference gates as control knobs; papers report answers-per-joule with real VLMs.

**Fencing note (user):**
> Will fence vs arXiv:2607.09520 (Seeing is Free...) — our lever is run/where/what-evidence, not on-device token count under always-on decoding.

### 1a. Selector-strategy amendment (2026-08-05, post-stress-test)

User decision: **predictor-primary**. Facts established during the exchange:
- All three selectors were already evaluated on the same grid (LUT vs rule head-to-head on 16,848+4,260 decisions; predictor via exact log replay — offline re-selection over complete fixed-token/fixed-image per-question logs is statistically equivalent to in-loop evaluation). The gap was narrative placement plus the predictor's missing joules number, not missing experiments.
- Campaign logs and energy data live on **server 160** (`~/phd_research/vqa_semcom/outputs/`, branch `codex/lut-semantic-utility-upgrade`), *not* 182 (182 holds only older v0/v1 pipeline outputs, though both VLM prediction sets exist there too). All new selector work is log replay: no GPU required.
- Advisor position (recorded dissent): recommended keeping the three-selector ladder with the equivalence result as C2's unclaimed finding; warned that predictor-primary moves the paper toward the crowded learned-routing space (INAR-VL, GO-SG) and weakens the deployability story. User accepted the trade-off; rule/LUT are retained as baselines so the equivalence evidence is not lost, only demoted.

New C2: **sender-side per-sample evidence routing** — a lightweight predictor over transmitter-available features decides per query which evidence level (hence which computation) runs; +0.7–1.1 pt over calibrated LUT at ~half payload (111→58 KB), gains concentrated on presence/counting; question type is the dominant feature (the zero-parameter rule ≈ calibrated LUT equivalence survives as the ablation showing this), and the receiver's verbalized confidence is anti-calibrated (AUC 0.406), justifying sender-side placement.

### Framing decisions (Steps 1–3)
- **Posture toward green-semcom literature: COMPOSES** ("a lever that stacks on existing power/bandwidth allocators"), not challenges. Strong version appears once, scoped, in §VII.B.
- **Contribution order: energy leads.** C1 = compute-dominated frontier + routing-as-lever (headline joules number to be recomputed for the predictor policy, W5; rule's 2.2× reported alongside as the zero-parameter floor); C2 = per-sample evidence routing as above (amended); C3 = unified channel-uses + joules methodology (21k decisions, 8-method bank). Old findings ⑥⑧ fold under C1/C2 as supporting evidence.
- **Conclusion takeaway: "ask before you transmit"** — the cheapest joule is the inference you never run; a green-networking primitive orthogonal to power/rate/bandwidth optimization.
- **Headline defense:** accuracy-constrained energy minimization phrasing (routing optimal under accuracy ≥ status quo; token-only is the relaxed-constraint extreme; energy-price sweep gives the controllable frontier).
- **Statistics fix:** replace "statistically indistinguishable (p=0.12)" with TOST equivalence test + difference CI ("equivalent within ±δ"). Local computation from existing per-item paired outcomes.

---

## 2. Chapter Plan (keep / rewrite / drop verdicts + word budgets)

Current total ≈ 8,790 words → target ≈ 9,700. **Page-budget check required in week 3** (TGCN ≤ 30 pp double-spaced; +900 words ≈ +3.5 pages — if over, first cuts come from §III PHY detail and §VI Act 3 prose, never from Act 1 or §VII.B).

| § | Section | Verdict | Words now → target |
|---|---------|---------|--------------------|
| I | Introduction | **REWRITE (contributions block + framing; C2 = per-sample routing)** | 878 → ~950 |
| II | Related Work | **KEEP structure + SHARPEN (add ~14 citations; harden learned-routing fence)** | 598 → ~880 |
| III | System Model | **REWRITE (reorder: energy accounting promoted)** | 1553 → ~1600 |
| IV | Evidence Routing | **REWRITE (predictor-first restructure)** | 529 → ~650 |
| V | Theory | **KEEP order + reposition intro (grounds the predictor's dominant feature)** | 1158 → ~1170 |
| VI | Experiments | **REWRITE (three-act restructure + predictor promoted into main comparison)** | 3285 → ~3550 |
| VII | Discussion | **REWRITE (expand: energy-scope defense)** | 383 → ~700 |
| VIII | Conclusion | **REWRITE (takeaway paragraph)** | 270 → ~280 |
| IX | Appendix | **KEEP + add TOST details** | 136 → ~200 |

### §I Introduction — REWRITE
- **Core argument:** the obstacle is no longer the answerer but the cost of feeding it; the dominant cost is the VLM forward pass; therefore the key transmit-time question is which evidence level this question needs.
- **Keep:** opening 2 paragraphs (already energy-led), the "different question asked at transmission time" paragraph.
- **Rewrite:** contributions block to C1(energy) → C2(mechanism) → C3(methodology); headline stated in accuracy-constrained form; composes-with sentence mirrors cover letter.
- **Evidence pointers:** 2.2× (F11/F13/F14), rule=selector equivalence (→TOST result), 21k decisions.

### §II Related Work — KEEP + SHARPEN
- Four streams unchanged (green semcom / goal-oriented VQA semcom / task-adaptive modality selection / adaptive coding & resource control); each stream's closing gap-sentence re-pointed at the energy thesis.
- **Integrate scoop-check citations** (`scoop-check-2026-08-05.md §c`): 10 must-add with one-clause differentiation each; 4 team-follow-up cites; optional list as space allows. Heaviest fences: 2607.09520 (energy-bottleneck locus), 2607.28276/2509.08913 (query-oriented coding), 2605.18853 (edge-cloud routing), DocPrune + 2509.09955 (⑧ adjacents), 2605.22883 (joules-per-goal metric).

### §III System Model — REWRITE (reorder)
- New order: Scenario/notation → Evidence levels → **Joint energy accounting (moved up)** → s1 PHY → How each level is answered → s2 PHY → Channels → Unified channel-use accounting → Tasks/data/split.
- Transition logic: "each level selects which computation runs; here is how we bill it" precedes PHY detail; PHY subsections instantiate the E_tx terms, answering subsection instantiates E_cmp.
- **Audit item (stress-test weak point 3):** energy-billing perimeter must be itemized — detector CPU energy on the s1 path, idle-power amortization policy, measurement window — so 2.2× is perimeter-auditable.

### §IV Evidence Routing — REWRITE (predictor-first)
- New order: **IV.A per-sample predictor (the method)** — features, BCE-OracleNet recipe, sender-side justification (anti-calibrated receiver confidence) → **IV.B baselines**: zero-parameter rule (deployability floor) + Wilson-LCB LUT (calibrated ceiling of type-only routing), with the equivalence result (rule ≈ LUT, TOST + CI) presented as the ablation establishing that question type is the predictor's dominant feature → **IV.C ROI** unchanged.
- All content retained; nothing deleted. The rule/LUT text compresses (~150 words) while the predictor subsection expands (features, training protocol, replay-evaluation methodology now stated as method-grade detail).
- **Change within IV.B:** equivalence claim re-stated via TOST + CI (details → §IX).

### §V Theory — KEEP
- Order unchanged: counting sufficiency + perception bound (support C2) → energy–accuracy Pareto dominance (climax, supports C1). Sharpen the two bridge sentences so the build from "safe" to "energy-optimal" is explicit.

### §VI Experiments — REWRITE (three-act, predictor in the main comparison)
- **Act 1 — ENERGY (the claim):** setup/baselines → measured power + per-answer frontier (was VI.G) with the **predictor policy added to the frontier (W5)** → energy-controllable routing / price sweep (was VI.H). Constrained-optimization headline stated here; token-only extreme reported in the same breath; rule's 2.2× reported as the zero-parameter floor next to the predictor's number.
- **Act 2 — MECHANISM (why declining inference is safe):** three-selector head-to-head table (accuracy / payload / joules per answer, per channel — was tab:persample, now promoted) → accuracy vs SNR incl. error-free win (②) → complementarity → who-decodes → token budget / less-is-more (⑧, fenced vs DocPrune + 2509.09955) → routing-dimension ablation + rule≈LUT equivalence (repositioned as "type is the dominant feature") → **learned-router ladder (W6: logistic vs small MLP)**.
- **Act 3 — ROBUSTNESS + SCOPE:** stats (incl. new TOST) → **predictor robustness replays (W7: second pool, cross-VLM, feature ablation)** → DJSCC → cross-VLM → CSI mismatch (⑥, fenced vs 2602.10482) → latency → link footprints.
- Figure mapping: Act 1 = F11, F13, F14, F5(energy); Act 2 = F1, F2, F4, F6, F8, F17; Act 3 = F7, F9, F10, F15, F16, F3. F11/F14 regenerate with the predictor series (naturefig scripts exist).
- No content dropped; new transitions + new table/series only.

### §VII Discussion — REWRITE (expand)
- VII.A safety bridge to UTM: **keep** (UAV-domain differentiator).
- **VII.B "When does compute dominate?" (NEW):** parametric E_cmp/E_tx ratio; state the claim boundary explicitly (Jetson-class NPUs, sub-Mbps vs Mbps-class links, VLM size scaling); the one place the strong version of the thesis appears, scoped. This is the pre-emption of the most predictable reviewer attack.
- VII.C limitations: **expand** — one GPU class, one VLM family + one cross-check, single-domain (aerial), perimeter of energy billing.

### §VIII Conclusion — REWRITE
- Built around the ask-before-transmit takeaway: evidence-level routing as a green primitive that composes with every existing allocator; 2.2× measured; answers-per-joule as the reporting metric (echoes L5-W3).

### §IX Appendix — KEEP + ADD
- Add TOST procedure + per-channel equivalence bounds table.

---

## 3. New work items (non-writing)

| # | Item | Cost | Blocking? |
|---|------|------|-----------|
| W1 | TOST equivalence test + difference CIs from existing per-item paired outcomes (rule vs LCB, both pools) | ~0.5 day, local | **DONE (primary pool, 2026-08-05)**: anchor reproduced exactly (n=16,848, 167 discordant, McNemar p=0.1217, AWGN bit-identical, fading deviates only presence@−5dB). TOST PASS: pooled 90% CI [−0.00002, +0.0025], δ*=0.0025 ≪ 0.01. Paper phrasing: "equivalent within ±0.0025 (TOST, α=0.05)". Data: `paper/data/w1_tost_equivalence.json`. Second pool in W7 run: anchor reproduced (p=0.2043, 224 discordant); TOST at δ=0.01 marginal (CI [−0.0011, +0.0105], δ*=0.0105) — report honestly as "equivalent within ±0.011" on pool 2 |
| W2 | Energy-billing perimeter itemization (detector CPU, idle amortization, window; **now incl. predictor feature-extraction + inference cost — must be billed since the predictor is the method**) | ~0.5 day, local/160 | Yes — feeds §III, §VII.C |
| W3 | Integrate 14 scoop-check citations into refs.bib + §II differentiation clauses; **harden the learned-routing fence (INAR-VL, GO-SG) now that the headline method is learned** | ~1 day | Yes — feeds §II |
| W4 | Cover letter: refresh positioning paragraph to constraint-optimization headline + new near-neighbor fences + predictor-primary wording | ~0.5 day | Final week |
| W5 | **Predictor energy billing** (log replay on 160): joules-per-answer for the predictor policy across channels/SNRs → new headline number; F11/F14 regenerated with predictor series | ~0.5–1 day, 160, no GPU | **Yes — headline (C1) depends on it** |
| W6 | **Learned-router ladder**: small MLP on same sender features, same replay protocol, vs logistic — mandatory now that the predictor is the method ("why only logistic?" is a certain reviewer question) | ~1 day, local/160, no GPU | **DONE (2026-08-05)**: MLP(32,16)×3 seeds = **real headroom, +1.9–2.4 pts over logistic** (0.703–0.705 vs 0.680–0.683, seed std ≤0.002; oracle 0.745). Energy comparable on AWGN/Rayleigh (8.1–8.5 J), noisier on Rician (9.0–12.6 J by seed, mean 10.4). **Framing decision (user, 2026-08-05): (b) MLP is the headline method.** Selector ladder becomes rule → LUT → logistic → **MLP (method)**; MLP reported mean±std over 3 seeds, fixed hyperparams (32,16). Follow-up W8 DONE (2026-08-05): per-SNR MLP series — ratio vs image 3.0–4.3× seed-mean at every SNR, ≥2.5× every seed; acc > rule in all 18 cells (+2.2 to +4.9 pts); MLP robustness: 2nd pool in-domain 0.762–0.766, transfer 0.752–0.757 (vs rule 0.7293), cross-VLM 0.698/0.709 (vs 0.673/0.674). Data: `paper/data/w6_router_ladder.json`, `w8_mlp_headline.json`, `w8_mlp_series.csv`. **Writing started same day: §I and §IV rewritten (compile clean); refs.bib 35→50 (15 verified adds, `paper/data/w3_new_refs.bib`)** |
| W7 | **Predictor robustness replays**: second pool (490-image) evaluation, cross-VLM replay (both VLM prediction sets exist on servers), feature ablation | ~1–1.5 days, 160, no GPU | **DONE (2026-08-05)**: second pool in-domain 0.7406 (+1.1 pt over rule 0.7293, f_img 0.22); **cross-dataset transfer** (v3-trained→dv) 0.7364 (+0.7 pt, f_img 0.064). Cross-VLM: v25 predictor 0.698 vs rule 0.673; v26 0.707 vs 0.674 — advantage grows on other receivers. Feature ablation (rician): drop-detector −0.008 (→ rule level), drop-qtype **+0.002** — the predictor's edge over the type rule IS the sender-side detector evidence; qtype redundant given it. Data: `paper/data/w7_robustness.json` |

| W11 | **MLP deepening** (post-decision-(b) gaps): paired McNemar MLP-vs-rule/linear; MLP feature ablation; MLP λ-frontier; per-type decomposition | ~40 min replay, 182 | **DONE (2026-08-05)**: McNemar decisive every seed (vs rule 639–660/53–115, vs linear 431–440/32–86, all p<10⁻⁵⁰); MLP ablation: detector −2.4 AND polarity −2.4 (each collapses to linear level), qtype −0.25 — polarity×detector interaction is the MLP's edge; per-type gains concentrate on presence (+6.4–7.1); MLP λ-frontier dominates linear everywhere (0.690@3.2 J, 0.701@6.6 J). Folded into §IV/§VI; F5 frontier swapped to MLP sweep. Data: `paper/data/w11_mlp_deepening.json`, `c1_frontier_rician_mlp.csv` |

Explicitly deferred (would exceed the time box): non-Qwen third VLM repeat, Jetson-class power datapoint, in-loop learned routing (bandit/RL — that is the TCCN v19 paper's territory; keep the boundary), DroneVehicle on AWGN/Rayleigh (needs GPU campaign). All listed as future work in §VII.C.

---

## 4. Five-week schedule (submission target: ~2026-09-09)

| Week | Work | Status |
|------|------|--------|
| 1 | Data tasks: W5/W1/W2 (+W6/W7/W8/W11 pulled forward) | **DONE 2026-08-05** |
| 2 | §I rewrite; W3 citations; §III/§IV/§VI/§VII/§VIII rewrites; figure regen (F5/F11/F14 with MLP series) | **DONE 2026-08-05** |
| 3 | Page trim 33→**30** (F17+F16 → `supplementary.tex` [3 pp, compiles]; abstract 331→245 w — also fixes IEEE ≤250 limit; refs scriptsize; float-sep tightening; ~1,500 words of prose compression) | **DONE 2026-08-05** — main.pdf 30 pp, 0 undefined refs, 0 TODO/CITE |
| 4 | Self-review or `/ars-reviewer` pass; fixes | next |
| 5 | W4 cover letter; final compile; submit (~2026-09-09 target, running ~2 weeks ahead) | pending |

**Gate result (2026-08-05): PASSED.** W5 ran on 182 (log replay, self-check vs published λ=0 frontier point: PASS). Predictor policy: 8.58–9.03 J/answer across all 18 (channel, SNR) cells; **3.61–3.84× vs rate-adaptive image at every SNR** (headline upgrades from 2.2× to ≈3.6×); 1.61–1.72× below the rule; accuracy ≥ rule in every cell; f_img 0.25–0.27 vs rule 0.438. Data: `paper/data/w5_ml_series.csv`, `paper/data/w5_persample_energy.json`. Headline phrasing: "≈3.6× at every SNR on all three channels, while raising accuracy" (conservative end of the 3.61–3.84 range).

---

## 5. Stress-test record (Step 3)

| Weak point | Attack | Adopted defense |
|------------|--------|-----------------|
| Headline baseline | "Token-only is 33× greener than routing — why route?" | Accuracy-constrained energy-minimization phrasing; token-only reported as relaxed-constraint extreme; price sweep = controllable frontier (§I, §VI Act 1) |
| Equivalence statistics | "p=0.12 non-significance ≠ equivalence" | TOST + difference CI (W1); "equivalent within ±δ" phrasing |
| Compute-dominance scope | "Does it hold on Jetson + fast links?" | §VII.B parametric E_cmp/E_tx boundary; billing perimeter itemized (W2); "often" retained in claim wording |
| ⑧ novelty | "Pruning-raises-accuracy already exists (DocPrune)" | Fence: non-monotone gain from confidence-ordered detector-token truncation *in a transmission system*; contrast monotone framing of 2509.09955 |
| T novelty | "Compute energy dominance is known (2607.09520)" | Fence: their locus is on-device decoding under always-on inference; our lever is run/where/what-evidence with radio-vs-compute measured across SNR |
| Learned-routing adjacency (NEW, opened by predictor-primary) | "This is INAR-VL/GO-SG-style learned routing" | Fence: evidence-*level* routing at the transmitter over a physical channel with measured joules, logistic-footprint sender-side model (they route between models / give manual guidelines, no channel, no energy axis); rule≈LUT ablation shows the signal is question-type, not model capacity |
| Learned-baseline sufficiency (NEW) | "Why only logistic regression?" | W6 MLP ladder: either MLP adds little (strengthens type-dominance) or adds more (reported honestly as headroom) |
| Predictor generalization (NEW) | "Trained and tested on one pool/VLM" | W7: second-pool + cross-VLM replays; training-data dependence stated in §VII.C; zero-parameter rule kept in-paper as the no-calibration fallback |

---

## 6. Fallback plan B (recorded, not active)

If TGCN desk-rejects or reviews kill the energy framing: pivot to **complementarity-law paper → TCCN (primary per 2026-07 assessment) / TWC (needs DPI formalization)**. Transfer cost estimate:
- §I contributions reorder back to rule-first (C2→C1); §VI re-arcs Act 2 first; §VII.B demotes to a paragraph.
- §V counting/perception lemmas promote to headline theory; add the DPI/sufficient-statistic proposition (July assessment §四 item 6, ~1 week derivation).
- Energy results remain as a strong secondary axis — no data discarded.
- Everything in W1–W3 (TOST, perimeter, citations) transfers unchanged.

---

## 7. Provenance

- Spine options A/B/C presented 2026-08-05; user chose "A now, B as fallback doc".
- All keep/rewrite/drop verdicts user-confirmed in three chapter-negotiation rounds (§I–II, §III–V, §VI–VIII).
- Contribution claims verbatim from user (Step 2.5, Kong L2 boundary respected).
- Collision check: `scoop-check-2026-08-05.md` (background agent, live-verified IDs).
- Selector-strategy amendment (same day): user challenged LUT-as-headline; advisor established that all three selectors share the same experimental grid (predictor via exact log replay) and recommended a three-selector-ladder presentation; user chose **predictor-primary, rule/LUT demoted to baselines** — advisor dissent recorded in §1a, trade-offs accepted by user. Original L5-W1/W2/W3 wording ("type-conditioned") predates this amendment and is preserved verbatim above as the Step 2.5 record.
- Next pipeline step options: full/revision drafting against this blueprint, or `/ars-reviewer` on the restructured draft before submission.
