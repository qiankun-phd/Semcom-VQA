# Simulated Review Panel — Editorial Decision Package (2026-08-05)

> ARS `academic-paper-reviewer` full mode. Panel: EIC (TGCN editorial) + R1 (ML-for-comms methodology) + R2 (semantic communications domain) + R3 (edge-AI systems / green computing) + DA (devil's advocate). Five independent seats, no cross-referencing; synthesis based solely on the five filed reports.

## Panel verdicts

| Seat | Recommendation | Headline finding |
|---|---|---|
| EIC | **Major Revision** | Fit 4/5, originality 4/5; "well-executed, unusually honest measurement paper… worth publishing"; abstract/scoping/companion-study issues |
| R1 Methodology | **Major Revision** | Design 4, stats 3, repro 4; cluster-naive inference + P2 post-hoc margin + 3 seeds |
| R2 Domain | **Major Revision** | Coverage 4, novelty 3, theory 3, significance 4; two-ledger energy + token-PHY inconsistency + method-of-record contradiction |
| R3 Systems | **Major Revision** | Measurement 4, realism 3; radiated-power billing + idle anomaly + single VLM energy point |
| DA | **1 CRITICAL + 5 MAJOR** | Workload-mix dependence of headline; also honestly reports 6 failed attack lines |

**No seat recommended Reject. All four scoring seats: the core routing result is genuine, TGCN-relevant, and the revision is achievable largely without new VLM campaigns.**

## DA CRITICAL — visible adjudication (Iron Rule #4)

**DA-C1: The headline ratio is a function of the author-chosen question-type prior $w_t$, on a benchmark whose four "symbolic" families are count-derived by construction; no mix-sensitivity analysis exists; abstract generalizes to "task-oriented aerial VQA".**

**Adjudication: VALIDATED (upheld).** Independently corroborated by EIC Major 5 (template-benchmark scope) — two seats converged without contact. The claim-construction link is real: §III-H's generator makes 4/5 families deterministic functions of per-class counts, and $w_t$ appears in Proposition 1 yet is never swept. This blocks any Accept outcome and anchors the decision at Major Revision. **Resolution path is cheap**: a $w_t$-sensitivity sweep is pure replay over existing logs (energy/accuracy as a function of the presence share and of a hypothetical "image-forced" question fraction), plus rescoping the abstract/§VIII claims to structured/templated aerial VQA. The finding itself survives — as a scoped claim with its dependence made explicit.

## Consensus clusters (≥2 seats independently)

**A. Energy-accounting credibility (all five seats touch it) — the revision's center of gravity.**
- A1. Radio billed at *radiated* power only; PA efficiency (20–40%) + RF chain + protocol overhead make DC radio energy 3–5×; the "order of magnitude above the radio" margin compresses to ~2–4×. (R1#4, R3#1)
- A2. UAV-battery joules and edge-server grid joules summed into one scalar under a battery-limited motivation; UAV-side saving is ~2 J/query, dominant 32.3 J is wall-socket energy. (R2#1, R3#4, DA#3)
- A3. 52.3 W idle baseline is 4–5× typical for an RTX 4060 — unexplained; incremental billing hinges on it (with ~12 W idle, incremental VLM would be ~56 J, i.e., the anomaly currently *understates* the compute term). Needs P-state explanation + one wall-meter cross-check. (R3#2)
- A4. Single VLM J/item (5 dB Rician) applied across a 14–215 KB payload range despite dynamic-resolution tokenization; prefill-dominated energy plausibly varies severalfold across SNR. Measure at SNR extremes or state fixed-resolution preprocessing. (R3#3)
- A5. Detector energy billed to token path only, but the per-sample router needs detector features on *every* query → routed-image path under-billed ~0.1 J; "conservative against token path" framing now wrong under MLP router. (R1-m4, R2#6, DA-partial)
- A6. Scope qualifiers (unbatched single-tenant, incremental billing, proxy hardware) live in §VI-B/§VII-B but not at claim sites (abstract, §I, §VIII); total-power ratio (~2.3×) absent from abstract. (EIC#2#3, R3#5, DA#5)

**B. Statistical inference (R1-led, DA-corroborated).**
- B1. McNemar p-values and TOST SEs are cluster-naive while the bootstrap is image-clustered — inconsistent; marginal inferences (rule-vs-LUT, P2) are anti-conservative. Recompute cluster-robust. (R1#1)
- B2. **P2 TOST failed at the pre-registered δ=0.01 (CI edge +0.0105) and was reported as "equivalence within ±0.011" — post-hoc margin widening.** Must be re-reported as "equivalence not established at δ=0.01 (δ*=0.0105, descriptive)". Abstract's unqualified "TOST-equivalent" must be scoped to P1. (R1#2, DA-cherry#2)
- B3. Three seeds cannot support "std ≤0.002" or tail claims; run ≥10 seeds (pure replay + cheap MLP fits, no GPU campaigns). (R1#5)
- B4. Per-type CIs uncorrected for multiplicity — mark exploratory or Holm-correct. (R1-m2)

**C. Method-of-record positioning (R2/DA convergence — revisits the predictor-primary decision).**
- C1. §VII-C(vi) concedes the LUT wins in preliminary *closed-loop* scheduling, while the MLP is crowned "method of record" — a positioning contradiction all headline numbers inherit (offline replay, simulated channels). (R2#3, DA#4)
- C2. Missing cascade baseline: token-first, escalate-on-symbolic-insufficiency — conditions on more information at ~zero token cost; the anti-calibration defense covers only VLM-confidence escalation. **Replay-computable from existing logs.** (DA#6)
- C3. Router training-protocol disclosure gaps: cross-receiver/DV router train-test splits unstated (implementation used train split — verified in `w8_mlp_headline.py`; must be *written*), MLP early-stopping monitor (sklearn internal validation_fraction from train — must be written), hyperparameter provenance. (R1#3, R1#6)

**D. Claim scoping beyond A/B (EIC/DA/R2).**
- D1. Weak-receiver artifact risk: counting image path 0.28→0.375 from 2B→3B against fixed token 0.45; all receivers ≤3B; "structural, not receiver-specific" overreaches. Soften to the tested regime + add trend note. (DA#2)
- D2. "Ladder"/"system" framing vs binary evaluated object; s0/s3 never in a headline experiment. (R2#5, DA#7-cosmetic)
- D3. Companion-study numbers (370.6/394.3/≈404 m; closed-loop claim) are quantitative but uncited — cite or delete. (EIC#4)

**E. Technical-correctness singletons (must verify).**
- E1. **Token-path outage (5.2% @ −5 dB Rayleigh) appears inconsistent with the billed 0.5 bit/use rate** (theoretical per-block outage ~73%); back-calculates to r_min≈0.024 bit/use ⇒ either unbilled time diversity or mismatched PHY assumptions. Verify in campaign code; correct either the billing or the outage model. If token airtime grows, energy conclusions survive but 265×/latency figures change. (R2#2)
- E2. Lemma 2's proof assumes what it must show (lossy $\hat I$ retaining sub-threshold evidence); restate retention as a named assumption or demote to remark; Lemma 1 near-tautological — demote. Keep Prop. 1 labeled as conditional accounting. (R2#4)
- E3. Missing verified references: Strinati & Barbarossa (goal-oriented 6G), Shao/Mao/Zhang (IB task-oriented edge inference, JSAC'22), Neurosurgeon (ASPLOS'17), Kountouris & Pappas (semantics-empowered), Qin et al. semantic-communications overview. (R2)

## Disagreement & arbitration

- **R3 wants "compute-dominated" demoted and the inversion-proof routing lever promoted; EIC wants compute-dominance retained as the central TGCN message with scope attached.** Arbitration: keep compute-dominance as the measured finding *with its scope traveling to every claim site*, and promote "the token path never inverts" as the robust companion claim (both seats endorse this line).
- **DA's "is this a communications paper?" challenge vs EIC fit 4/5.** Arbitration: EIC's call stands (energy accounting is the TGCN contribution; PSCom precedent), but the §VII-A composition-with-allocators claim must stop leaning on uncited companion results (D3).
- **Stale points (synthesizer note):** EIC's abstract-length figure (~340 w) predates the final trim (current abstract ≈245 w — recheck at revision); EIC page count (28) vs build (30) immaterial. The number-density critique stands regardless.

## Editorial Decision: **MAJOR REVISION**

The panel is unanimous that the core contribution — a measured per-answer joint-energy frontier for a real VLM receiver over coded links, with an automatic evidence router that is simultaneously cheaper and more accurate than image transmission — is genuine, TGCN-relevant, and publishable. It is equally unanimous that the manuscript cannot be accepted as framed: the headline multipliers rest on an idealized radio-energy model and a conflated two-node ledger; the equivalence statistics contain one post-hoc margin and cluster-naive SEs; the method of record contradicts the paper's own closed-loop evidence; and the headline generalizes over a question mix the benchmark itself constructs. Every blocking item has a repair path that requires no new VLM campaigns except two optional measurements (wall-meter cross-check; per-SNR VLM energy points).

## Revision Roadmap (prioritized; feeds revision mode directly)

**P0 — blocking (address before resubmission):**
| # | Item | Source | Cost |
|---|---|---|---|
| R-1 | $w_t$ mix-sensitivity sweep (replay) + rescope "task-oriented aerial VQA" → structured/templated aerial VQA at abstract/§I/§VIII | DA-C1, EIC#5 | replay, no GPU |
| R-2 | Energy re-billing: DC-power band (PA η∈{0.25,0.4,1}) in Eq.(1) + platform-separated ledgers (UAV vs edge columns in Table II/F13) + idle-baseline explanation (+ optional wall-meter point) + per-SNR VLM energy (measure extremes or state fixed-resolution) + detector billed per-query under learned routers | A1–A5 | mostly re-accounting; 2 optional measurements |
| R-3 | Cluster-robust McNemar + TOST recompute; P2 re-reported as *not established* at δ=0.01; ≥10 MLP seeds with quantiles | B1–B3 | local compute |
| R-4 | Protocol disclosure: router train splits (per-receiver/DV), early-stopping monitor, hyperparameter provenance, seeds-fixed-in-advance statement | C3 | writing |
| R-5 | Token-PHY consistency: specify r_min + diversity assumption; reconcile outage model with channel-use billing (verify in campaign code) | E1 | code check + writing |
| R-6 | Companion-study numbers: cite or delete | D3 | writing |

**P1 — strongly advised:**
| # | Item | Source |
|---|---|---|
| R-7 | Positioning repair: EITHER re-crown rule/LUT as deployable contribution (MLP = measured headroom) OR add the token-first cascade baseline (replay-computable) and confront the closed-loop admission head-on | C1, C2 |
| R-8 | §V: Lemma 2 retention property as named assumption; Lemma 1 → remark; keep Prop. 1 as conditional | E2 |
| R-9 | Scope clauses at claim sites (single-tenant, proxy-conditioned, incremental); total-power ~2.3× once in abstract; "structural" → tested-regime wording | A6, D1 |
| R-10 | Add 5 verified missing references with one-line positioning each | E3 |
| R-11 | Presentation: claim→evidence summary table; prune forward references; harmonize ≈33 J/32.3 J, F14 annotation vs range; fix `\today` stamp | EIC#1#6 + minors |

**P2 — polish:** mission-level context row (queries-per-Wh under declared hover power; latency-hover coupling — strengthens the paper), λ-vs-battery-state deployment recipe, per-type CI multiplicity note, "method of record" repetition, DeepSC mislabel in Table I, unique-question count alongside decision count.

**Attacks that failed (for the response letter):** symbolic-decoder artifact (who-decodes + paraphrase probe held), residual mock numbers (none), broad cherry-picking (disclosure rate praised by DA), train/test leakage (crc32 split held), CSI-error fragility (mismatch matrix held), energy micro-asymmetries (declared, mostly against the token path).

> Full per-seat reports: session task outputs of 2026-08-05 (EIC / R1 / R2 / R3 / DA). Next step per workflow: revision mode against this roadmap, then `re-review` for verification.

## P0 revision executed (2026-08-06)

Fork decisions (user): **R-7 Option A** (rule/LUT = deployable contribution, MLP = measured headroom, no cascade baseline); **optional measurements skipped** (parked for response letter). All six P0 items applied, no new VLM campaigns:

| P0 | What was done | Data/scripts | Sections touched |
|---|---|---|---|
| P0-1 energy | Radiated→DC declared (η∈{0.25,0.4,1}, ratios η-invariant, worst-link dominance 12×→6.3×→3.9×); platform-separated UAV/edge ledgers declared; detector billed per-query under learned routers (+≈0.1 J image path); idle-baseline sensitivity (12 W hypo ⇒ E_vlm_inc≈56 J, margins widen — measured idle is conservative); total-power ≈2.3× now in abstract | `w15_ledger.json` | §III-C, §VI-B, §VII-B, abstract |
| P0-2 stats | All paired inference image-clustered (B=5000): P1 pooled TOST δ*=0.0028 PASS; **P2 δ*=0.0158 — "not established at δ=0.01" stated in §IV-B, supplementary, honest wording**; MLP 10 seeds (min–max replaces std); clustered MLP-vs-rule CI [+0.022,+0.044]; per-type CIs marked exploratory | `w13_cluster_stats.json`, `w14_mlp_10seed.*` | §IV-B, §VI-J, supplementary S2 |
| P0-3 w_t | New mix-sensitivity paragraph (presence 10→90%: MLP 3.5×→2.8×; image-forced f: 2.3×@0.25, 1.6×@0.5, →1×); claims scoped to "structured aerial VQA" in abstract/§I/§VIII | `w12_wt_pertype.json`, `w15_ledger.json` | §VI (new ¶), abstract, §I, §VIII |
| P0-4 protocol | Train-split-only fitting stated (incl. per-receiver/DV retrains — matches implementation); MLP early stopping = internal validation from train; hyperparams fixed before test; 10 fixed seeds | — | §IV-B(new IV-B), §VI-L |
| P0-5 token PHY | **Confirmed inconsistency** (code: `fer_for` uses slot-spread r_min=b/(Bτ)≈0.024 vs billing 16 uses/byte); §III-D now declares slot-spreading design + range billing (1.4×10⁴ compact ↔ 3×10⁵ spread); 265×→"265× compact / ≥12× worst-case"; latency 49×→"14–49×" | `digital_link.py` L426-439 verified | §III-D, §III-G, §VI-N |
| P0-6 companion | Quantitative separation numbers deleted; qualitative bridge kept with "beyond this paper's scope" | — | §VII-A, §VII-C(vi) |
| R-7A | Deployable-first reframe: §IV reordered (type pair = contribution, learned = headroom); §VI readings (a)/(b) swapped; "method of record" phrasing removed globally; abstract/§I/§VIII lead with rule 2.2×, MLP as headroom | — | §I, §IV, §VI, §VII, §VIII, abstract, tab:baselines |

**Compile**: main 32 pp (grew +2 from revision text — needs a small re-trim pass), supplementary 3 pp, zero undefined refs/TODO. **Remaining**: P1 items (Lemma 2 restatement, 5 missing refs w/ verification, presentation polish), re-trim to ≤30, response letter, `re-review`.

## P1 revision executed (2026-08-06)

| Item | Done |
|---|---|
| R-8 theory | Lemma 2 retention → **Definition 1 (detectability-preserving coding)** with honest failure note; Lemma 1 labeled near-definitional; proofs → Supp. S4 |
| R-10 refs | 5 classics verified against primary sources (Crossref/dblp/arXiv/S2) and cited in §I/§II; refs.bib 50→55 |
| R-11 polish | ≈33 J caption harmonized; \today → fixed date; DeepSC mislabel fixed; unique-question counts added; F14 annotation 3.1×→3.5× (ten-seed); "structural" → tested-≤3B-regime wording (D1) |
| Ten-seed alignment | F5/F11/F14 regenerated from `w14_mlp_10seed_series.csv`; all text numbers realigned (3.4–4.1×, 24–29%, 8.2–10.1 J, +2.3–4.7 pts); tab:persample MLP column = ten-seed means |
| Page re-trim | **30 pages** restored: F1/F15 (+ earlier F13/F8/F16/F17) → supplementary (now 5 pp, incl. proofs); ~700 words tightened; bib howpublished shortened; abstract ~265 words |
| Response letter | `response-letter-2026-08.md` (10 summary items + per-seat prepared responses + parked-measurement offers) |
| Re-review | verification agent dispatched 2026-08-06 (item-by-item + 7 consistency spot-checks) |

## Re-review verdict + residual fixes (2026-08-06)

**Verifier verdict: "Another Minor round needed"** — all six P0 items + R-7A verified as substantively executed; 6 residuals found. All fixed same day:

1. Three-seed leftovers (§VI ll.192/374/387): λ-sweep declared as three-seed exception; MLP-vs-linear gain restated ten-seed min–max; ablation "std ≤0.002" removed (three-seed ablation, declared) ✔
2. §V stale "3.0–4.3×" → 3.4–4.1× ✔
3. §I stale transfer range "+2.3–2.8" → "+1.4–3.1" (ten-seed) ✔
4. 09_appendix.tex landmine (pre-revision cluster-naive TOST + post-hoc "±0.011" wording + duplicate label) → file retired to a pointer comment; content deleted ✔
5. Response-letter [R3-3] flag made true: §VII-C(vii) now states the 5 dB-workload / dynamic-resolution caveat ✔
6. Minors: §I "headline instantiation/router" → ladder/strongest-learned-rung wording; §VIII TOST "on the primary pool" qualifier; §VI "compact-end 265×"; §III duplicated-phrase typo; stale % comment headers ✔

Verifier's two stale observations (noted, no action): page count "32" came from the outdated execution-log line — the build is **30 pp**; claim→evidence table consciously dropped for the page budget (recorded here as the decision of record).

**State of record (2026-08-06):** main.pdf 30 pp / supplementary.pdf 5 pp, 0 undefined refs, 0 TODO, abstract ≈265 w, all statistics image-clustered & ten-seed, all residual-grep clean. Remaining before submission (~4 weeks ahead of the 2026-09-09 target): human read-through of the full PDF, W4 cover-letter refresh, upload package assembly.

---

## Condensation pass (2026-08-18, advisor page target)

**Trigger**: advisor target 25–26 pp (headroom for post-review curves); TGCN peer survey
(9 papers): figures median 11 (range 8–17), tables median 1, body prose median ~9.5k words,
results ≈24% of body.

**Executed** (user-confirmed plan):
- Prose: 10,474 → **8,613 words** (−1,861). Structural cuts, no claim removed:
  - §III fully rewritten: counting-calibration / judging-rule / detector-training /
    answer-prior protocol details → supplementary "Evaluation-Protocol Details" (new S-section),
    one-line declarations + pointers retained in main.
  - §V fully rewritten (864 raw words): corollary folded into Remark rem:trunc; proofs already
    in supplementary.
  - §VI: setup, energy-exp protocol + dominance, linkfootprint, m6, energyctl, ladder block,
    stats, routerrobust, wt-sensitivity all tightened.
  - §I C2/C3, §II green-stream + VideoQA-SC, §IV protocol para, §VII-A/-B, §VIII tightened.
- Figure/table rebalance (main = 7 figs + 3 tables; TGCN norm figs≫tables):
  - F1 (accuracy vs SNR) and F15 (cross-VLM) RESTORED to main with short captions.
  - tab:main and tab:whodecodes MOVED to supplementary; all cross-refs repointed
    (§V→"supplementary token-consumer table"; linkfootprint→Fig. fig:f1).
- Caption style: all \evfig captions cut to ≤1 sentence claim (detail in body text).

**Result**: main.pdf **26 pages** (675,822 bytes), supplementary.pdf 5 pages.
Zero undefined refs/citations, zero \TODO/\CITE. Composition: 7 figures + 3 tables main,
supplementary holds tab:main, tab:whodecodes, TOST table, F8/F13/F16, dim-ablation, proofs,
protocol details.

**Verify (human)**: read-through of condensed §III/§V for lost nuance; check F1/F15 render
at new widths; confirm supplementary protocol section matches code defaults.

---

## Structure rebalance (2026-08-23, TGCN section-structure survey)

**Survey basis** (11 papers, structure fully verified): TGCN n=6 — median 6 sections /
11.5 subsections, results section 2–4 subsections, standalone theory section 0/6;
semantic-comm systems (JSAC/TCSVT) n=5 — 5–7 sections, results 3–7 subsections (median 4),
standalone Discussion 0/5. Our pre-rebalance state: 8 sections / 38 subsections,
Experiments alone 15 (9 under 150 words).

**Executed** (user-confirmed; pure moves + run-in demotion, no sentence deleted):
- §V theory MERGED into §IV as subsection IV-D "Analytical support for the type split"
  (label sec:theory kept; 05_theory.tex retired, not \input; all lemma/def/prop
  environments intact; supplementary proof-section title updated).
- §III 8→4: levels+answering merged (III-A); s1-PHY/s2-PHY/channels/unified-billing
  merged into III-C with \emph run-ins; all labels preserved as run-in anchors.
- §VI(now §V) 15→6: A setup · B energy frontier (+energyctl run-in) · C grid accuracy +
  complementarity + who-decodes + token budget · D selector ladder (+dim-ablation run-in)
  · E statistical strength + cross-VLM + router robustness · F DJSCC + latency + link
  footprints. All subsec labels kept (run-in \label resolves to enclosing subsection).
- §VII(now §VI) Discussion: safety-bridge paragraph folded into section body; 2 subsections
  (boundary, limitations) — kept standalone despite 0/11 peers, as it carries
  panel-mandated content.
- Supplementary hardcoded section numbers updated (VI-H→V-D, VI-M→V-F, "Proofs of
  Section V"→"...(Section IV-D)").

**Result**: 7 sections / 21 subsections; main.pdf **25 pages** (673,144 bytes),
supplementary 5 pages. Zero undefined/multiply-defined refs. PDF heading map verified.

**Verify (human)**: run-in flow at merge seams (III-A answering para; V-C mechanism chain;
V-F triple); Section IV-D forward-references from IV-A read naturally.

## Discussion dissolve + figure restore (2026-08-23, round 2)

- §VI Discussion dissolved (0/11 peers have one): boundary analysis → V-B run-in
  (subsec:boundary), safety bridge → V-F run-in (subsec:safety/outlook), Limitations →
  subsection V-G (sec:limits). §V renamed "Experiments and Discussion" (AMC-paper pattern).
  07_discussion.tex retired. Now **6 sections** (= TGCN median) / V has 7 subsections.
- F13 energy stack restored to V-B (fig:estack), F8 token-budget sweep restored to V-C
  (fig:f8), short captions; their supplementary sections removed (supplementary now 4 pp).
  Main: **9 figures + 3 tables**, 26 pages, zero undefined refs.
- Self-audit of prose style (for style pass): experiments 9.1 numeric tokens/100 words;
  parenthetical asides ~20/500 words overall, §IV peaks at 29.5/500w. Peer benchmark
  survey dispatched.

## Prose-style pass: numbers + parentheticals (2026-08-23, round 3)

**Peer benchmark** (4 JSAC/TGCN results sections, hand-counted): 2–3.5 numeric tokens/100w
(mostly operating conditions, not results; setup paragraphs quarantine 8–14/100w); 0.5–2
prose asides/500w (EASE ceiling 5.1); 0–6 exact result values per results section, one
hedged headline number per paragraph; ZERO ±/CI/seeds in prose across all four.

**Executed** (peer-standard, with one deliberate deviation: clustered CIs/TOST/seed spreads
are panel-mandated and stay, concentrated in V-E "Statistical strength"):
- Exact values already carried by figures/tables removed from prose (F5/F11/F14 readings,
  tab:persample ladder values, F1 grid points, F6 sign-flip values, F8 budget points,
  supp whodecodes/TOST/dim-ablation values). Ranges rounded to hedged forms ("about a
  quarter of queries", "about one point") only where the exact range survives elsewhere.
- Multi-clause parentheticals (semicolon asides) converted to prose/em-dash across
  §I/§III/§IV/§V: TOST CI details → supplementary table pointer; tab:power workload
  declaration → prose; limitations (vii)/(viii) inner parens → em-dash; §III protocol
  triple-paren sentence → three declarative sentences.
- "104 unique test images" (dropped from §IV) restored to its authoritative home in V-E.
- Numbers with no other home (feature ablation, router robustness, latency operating
  point, boundary margins) kept verbatim.

**Result**: experiments numeric density 9.1→6.7/100w (peers 2–3.5 but ours is a
measurement paper; residual is headline ratios + only-home values); true multi-word
asides ≈6.1/500w overall (§V ≈ EASE's 5.1 ceiling; §III higher by design, setup-
sanctioned). Main 25 pages, zero undefined refs. Prose ~8,400 words.

**Verify (human)**: readings (a)-(d) paragraph and TOST paragraph read-through — the
heaviest rewrites; confirm no needed number lost (all removed values live in figures,
tables, or supplementary).

## Abstract trim (2026-08-23, round 4)

Peer abstracts: TGCN n=6 median 177 words (146–203), JSAC n=3 219–233; only 2/9 quote
any number (2–4 values max). Ours 272 → 217 (scale list → "more than 21,000 scored
decisions", defensive parentheticals out, 3 headline numbers kept: +0.11–0.17 / 2.2× /
3.4–4.1×) → **198** after user challenged the "measurement-paper" license for exceeding
TGCN norms (correctly: TGCN precedent 2206.10380 is a measurement paper yet keeps a
187-word number-free abstract). Final cuts: opener merged into one sentence,
"decoded symbolically" and "budget-tunable sweep" clauses dropped (both live in C2/V-B).
198 words = inside TGCN range. 25 pages, compile clean.

## Figure-size normalization (2026-08-23, round 5)

Diagnosis: naturefig scripts author figures at IEEE target sizes (single-column ~3.4in,
full-width ~7.1in, fonts 7-8pt at that size), but main.tex displayed them at 0.38-0.86×
native, pushing effective fonts to 3-6pt (IEEE floor ~6pt, recommended 8-10pt).
Worst: F8 at 0.38× (~3pt), F1 at 0.48×, F13 at 0.55×, F6 at 0.57×.

Fix — display at ≈ native (author-designed) width: F1/F8 → 2.15·evfigw (≈\linewidth),
F6 → \linewidth, F12 → 0.9\linewidth, F15 → 1.6·evfigw, F13 → 1.1·evfigw,
F14 → 1.05·evfigw, F5/F11 panels → 0.48\linewidth each. All scales now 0.82-1.0.
Side fix: energyctl argmax formula → display equation (killed an 18pt overfull the
larger figures exposed). Supplementary figures already ≥ native; untouched.

Result: 26 pages (was 25; within advisor's 25-26 target), zero undefined refs,
no overfull >10pt.

## Em-dash purge (2026-08-24, user style rule)

User rule: no em-dash asides, short declarative sentences, whole manuscript.
Rewrote ~100 prose `---` sites across abstract, §I–§V(+concl), and supplementary
into separate sentences / colon expansions / comma appositions / semicolons.
Table `---` empty-cell placeholders kept. Zero prose `---` remain (sweep-verified).
26 pages, zero undefined refs, zero overfull >10pt. Rule saved to memory
(writing-style-no-emdash) so future edits never reintroduce it.

## Citation added: manual-analysis current practice (2026-08-24)

§I claim "In current practice these questions are mostly answered by people watching
the feed" now cites (both Crossref-verified):
- pi2020convolutional (Adv. Eng. Informatics 43:101009, DOI 10.1016/j.aei.2019.101009).
  Verbatim abstract support: "The current process is resource-intensive (must be
  carried out manually) and requires offline computing (through post-processing of
  aerial videos)."
- ofli2016combining (Big Data 4(1):47-59, DOI 10.1089/big.2014.0064). Abstract-level
  support: FEMA/JRC aerial big-data warning; crowdsourced human annotation of aerial
  imagery as the operational mechanism.
Considered and REJECTED: Rakha & Gorodetsky 2018 (exists, but abstract has no manual-
review statement; wrong domain), Murphy "Disaster Robotics" 2014 (exists, but no
verifiable supporting passage retrievable — do not cite without checking the book).
refs.bib now 57 entries.

## Introduction rewrite to TGCN skeleton (2026-08-24)

Per 6-paper intro survey (median ~1,080w; contributions 3 bullets x 40-80w; organization
paragraph 6/6 and always last; separate-RW-mode intros run 714-1,040w with 8-12 refs):
- P1 motivation: measurement spoiler removed ("In our measurements it costs more...
  (Section V-B)" deleted); VLM energy-hunger now qualitative + cite zhan2026seeing.
  User directive: intro storyline = proposed system's superiority, not measurement.
- P3 approach: explicit "we propose an evidence-level semantic communication system"
  + one qualitative superiority sentence; s0/s3-scoping sentence dropped (lives in V-A/IV-C).
- Contributions: bold paragraphs (130-160w each) -> 3 itemize bullets (~70-90w each),
  reordered system -> mechanism -> methodology (system first per user), lead-in switched
  to "The main contributions of this paper are summarized as follows:". Numbers kept:
  2.2x / 3.4-4.1x (b1), 21,000+ (b3); all deleted numbers verified to live in body
  (32.3J/2.1J in V-B, MLP accs in tab:persample, +-0.003 in IV-A, 75-79x in V-B,
  7.7pt in V-C, transfer points in V-E, scale details in III/V-E).
- Organization paragraph added as final element; lone \subsection{Contributions} removed.
- First-credit fence to jiang2025lmmvn dropped from intro, verified retained in §II.
Result: intro 731 words (matches separate-RW peer mode: 714-1,040), 26 pages, clean compile.

## Related-work alignment to TGCN norms (2026-08-24)

Per 4-paper RW survey (forward-refs to own sections: 1 total across 4 papers; own
numbers in RW prose: 0; subsections inside RW: 0/4; closing self-positioning
paragraph: 4/4):
- Removed all 6 self-references from §II (energyacct/energy-exp/tab:power/persample/
  mismatch pointers + "cf. Limitations") and both own-result numbers (t=3 vs t=48,
  +7.7 -> qualitative; all live in §V).
- Flattened the 4 subsections into thematic paragraphs with scoping topic sentences
  ("A second stream makes VQA itself the receiver task." etc.).
- Added a closing positioning paragraph ("In summary... In contrast, this paper
  asks, per answered question, whether the dominant energy consumer needs to run
  at all..."). Qualitative differentiation sentences (EASE-style interleaved)
  retained. 26 pages, clean compile. Structure now: §II = 6 thematic paragraphs,
  0 subsections.

## Advisor critique round (2026-08-24): flat SNR curves + terminology

**Critique 1 (flat lines vs SNR).** Verified against energy_summary.json: genuine
physics in the declared regime (radio term 11.7x swing but <= 6% of budget at
B = 1 MHz, P_tx = 0.5 W, E_vlm = 32.3 J), regime-dependent (20% at eta = 0.25,
39% at B = 100 kHz, 72% both), and 4/6 plotted series were flat BY CONSTRUCTION
(fixed symbol counts). Fix: Fig. 2 redesigned as (a) radio term vs SNR with compute
references + token airtime band, (b) regime map of radio share under (B, eta);
F11 (answers/joule vs SNR) moved to supplementary; Fig. 3 = frontier only; Fig. 5
caption states link-independence as the claim; V-B gained a quantitative sentence
(11.7x / 6% / 20-39-72%) and boundary item (b) now matches Fig. 2(b).
**Critique 2 (results have no spine).** Diagnosed (7 subsections, 16 run-ins,
topical titles, V-B overloaded). Restructure into a question-driven spine proposed
(A setup / B energy / C accuracy / D mechanism / E learned headroom / F robustness
/ G limitations) — NOT yet executed; awaiting user go-ahead.
**Terminology.** "billing/billed/charged/priced" metaphor (28 sites) replaced by
accounting/incurs/counted/expressed; "complex channel uses" replaced paper-wide by
airtime T_s = u_s/B in seconds (B = 1 MHz declared), F8 panel (c) axis relabelled.
"energy price lambda" kept (Lagrangian usage).
**Page budget.** Held at 26 pages after the full-width Fig. 2 via trims: readings
(c)+(d) merged, roadmap sentence, sanity-check sentence, sensitivity paragraph
compressed (also removed a stale shaded-band reference), reproducibility,
whodecodes tail, limitations (iii)/(vi)/(vii), conclusion redundancies.

## §V question-driven spine executed + PA-efficiency citations (2026-08-24)

**§V restructure (pure moves, run-in labels preserved, no sentence deleted):**
A Setup (+ one spine sentence naming the five questions) · B Energy: the deployable
saving and the learned headroom (power table, readings, Figs. 2-4, sensitivity,
boundary) · C Accuracy is not traded: the fixed-policy grid (Fig. 5 + statistics/TOST
run-in, label subsec:stats moved onto the run-in) · D Mechanism: evidence-question
complementarity (Fig. 6, who-decodes, token budget Fig. 7, dimension ablation +
mismatch) · E Learned headroom: the selector ladder (Table 3, feature ablation, AUC,
lambda sweep moved here from B) · F Robustness and scope (cross-VLM Fig. 8, router
robustness, DJSCC run-in with label subsec:m6, latency/safety/link Fig. 9, question-mix
sensitivity moved here, reproducibility) · G Limitations. New labels: subsec:mechanism,
subsec:robust. Headings now carry the claim; one figure family per question.

**PA-efficiency band now cited** (both Crossref-verified): cui2005energy (TWC 4(5):
2349-2360, DOI 10.1109/TWC.2005.853882; P_amp = alpha P_t, alpha = xi/eta - 1, eta = 0.35
class-A, 0.75 class-B+) and auer2011howmuch (IEEE Wireless Commun. 18(5):40-49, DOI
10.1109/MWC.2011.6056691; P_PA = P_out/(eta_PA(1-sigma_feed)); eta_PA macro 31.1%,
micro 22.8%, pico 6.7%). §III-C sentence: "0.25-0.4 spans the linear-amplifier and
base-station efficiencies reported in [Cui, Auer], eta = 1 is the lossless reference".
Caveat kept in mind: Auer's values are base-station PAs, not UAV terminals.

**Page budget:** 26 pages after shrinking Figs. 3/5/7/8/9 slightly (all scales >= 0.85
of native) and trimming limitations (ii)/(viii). refs.bib now 59 entries.

## Venue-idiom terminology pass (2026-08-24)

Two independent audits (§I–III, §IV–VI+supp) -> ~340 substitutions in two batches.
User decisions: keep "router / evidence routing" (defined; title/keywords; figure
legends); conclusion aphorism kept but de-colloquialized ("The lowest-energy inference
is the one that is never executed.").
Replaced (collisions with comms meanings): cell -> (type, SNR) bin / decisions;
band -> range; spread/slot-spread -> full-slot transmission; compute floor -> compute
lower bound; seed noise/jitter -> variability; backbone -> basis; image rate -> fraction
routed; no-LoS -> NLoS; VLM "tokens" -> output/visual input tokens.
Replaced (ML/finance/colloquial): rung/ladder -> selector/router, selector family;
headroom -> attainable gain; exact offline replay -> trace-driven evaluation; pool ->
dataset (TOST rows D1/D2); beats -> outperforms; cheap -> lower-energy; declines ->
bypasses/omits; headline -> nominal/reported/primary; forward pass -> VLM inference;
answerer -> decoder / receiver inference; lever -> energy-saving mechanism; knob ->
parameter; joules (noun) -> energy; communication-plus-computation -> joint
communication and computation; commercial/real VLM -> off-the-shelf/pretrained;
declared -> stated/used; plus ~60 single-site idioms (honest readings, load-bearing,
sits at, falls off a cliff, Bottom chips, recipe, seat, stark, ...). Headings:
"No accuracy penalty: the fixed-policy grid", "Learned routers: attainable gain",
"Decoder choice for the token path", "Relation to UAV traffic management",
Fig. 2 caption "SNR dependence of the per-answer energy budget".
Kept deliberately: energy accounting, airtime, measurement campaign, frozen (gloss),
oracle (genie-aided gloss), energy price lambda, digital cliff / cliff effect.
Page budget held at 26 via trims (batching duplicate sentence, paraphrase probe,
dim-ablation tail, scheduling-interface sentence). Backups in scratchpad
sections_before_terms/.

## Method-name unification (2026-08-24, advisor critique 3)

Found 4 names for the type-level series (Evidence routing / Type routing / LUT /
Routing) and 3 for the learned one (Per-sample router / MLP / Per-sample λ point).
Canonical set fixed (see figures/MANIFEST.md same date); user rule: parallel scheme
names "Type routing (ours)" / "Linear routing" / "Per-sample routing (ours)" /
"Budget-tunable routing"; component words only in prose. 8 figures regenerated,
3 tables renamed, prose series references aligned. 26 pages, clean compile.
Caption/header pass (same day): all \caption and table headers scanned against the
canonical set; aligned Alg. 1 caption ("linear routing and per-sample routing (MLP)"),
Fig. 5 ("Type routing and Fixed token accuracy are link-independent ... Fixed-rate image
collapses"), Fig. 6 ("Per-type accuracy of Fixed token and Rate-adaptive image"), Fig. 8
("Fixed token (hatched) vs. Rate-adaptive image per receiver VLM"), Fig. 9 ("Fixed token
outage against measured Fixed-rate image FER"), supp. DJSCC ("coverage of Uncoded
analog"). Mechanism-level captions (TOST "rule vs. calibrated LUT", "LCB selector",
tab:power component rows) intentionally keep component names. 26 pages.

## Budget-tunable routing promoted to the method family (2026-08-24)

- §IV-B: new Eq. (argmax-lambda) right after Eq. (argmax): s_lambda(x) = argmax_s
  [P(correct|x,s) - lambda E_s(gamma)], lambda = 0 recovers per-sample routing; V-E now
  references it instead of defining it inline.
- Table 1: row "Budget-tunable routing (lambda sweep)" added (all figure-legend names now
  have a Table 1 row).
- §I bullet 2: "an energy-price extension makes the operating point budget-tunable".
- Fig. 4 lambda point switched to the MLP sweep (3.24 J / 0.690), consistent with Fig. 3.
Positioning unchanged: Type routing = deployable headline (2.2x); Per-sample routing
(= lambda 0) = attainable gain (3.4-4.1x); budget-tunable = controllability extension.
Page budget recovered by trims (idle-sensitivity sentence, crossover sentence, whodecodes
rhetorical question, companion-study aside; Fig. 6 at 0.95 linewidth).

## Writing Quality Check pass (2026-08-24)

Checklist: academic-paper skill references/writing_quality_check.md (A flagged terms,
B punctuation, C throat-clearing, D structure, E burstiness).
Findings before: A clean (only domain-standard "robust"); B em-dash 0 but semicolons
100 = 14.2/1000w (limit 2); C clean; D paragraph variation healthy, conclusion one
271-word paragraph, synonym cycling (11 names for the type-level method, 8 for the
learned one); E 47 sentences >= 40 words (longest 76).
Fix (5 parallel per-file rewrites, prose only; envs/captions/cites/refs/labels/numbers
byte-preserved and snapshot-verified): semicolons 100 -> 6 (0.8/1000w); >= 40-word
sentences 47 -> 0; conclusion kept as ONE paragraph (user decision; a trial split was reverted); count announcements
("Three observations") removed; synonyms consolidated to "the type rule"/"type routing"
and "the per-sample router"/"per-sample routing" (+ "linear router/routing"), LUT only
in the TOST passage; §V-E heading -> "Per-sample routing: attainable gain"; Table 1 row
text aligned. Abstract: "learned selector" -> "per-sample router" (2). Page budget held
at 26 (Fig. 4 -> 1.0 evfigw, Fig. 3 -> 0.44 linewidth, duplicate protocol sentence in
§IV removed). Backups: scratchpad/before_wqc/.

## Gain notation: percentages first in headline sentences (2026-08-24)

Peer survey: TGCN/JSAC abstracts and results state gains as percentages or dB ("up to
77%", "5.17%", "99.5% bandwidth", EASE "10 folds"); no "x" notation in 9 abstracts.
Converted (values unchanged, percent = 1 - 1/k): 2.2x -> "by 55% (a factor of 2.2)"
[abstract, §I bullet 3, §VI, V-B reading (a)]; 3.4-4.1x -> "71-76% (3.4-4.1x)" [same four
sites]; ">= 2.5x for every seed" -> "no seed saving less than 60%"; 75x -> "nearly two
orders of magnitude"; 265x / >= 12x -> "exceeds two orders of magnitude ... above an
order of magnitude" (factors kept in parentheses); 11.7x radio swing -> "more than an
order of magnitude (11.7x)"; 14-49x latency -> "reduction of more than an order of
magnitude (14-49x)". Kept as-is: ratio trajectories (w_t sweep 3.5x -> 2.8x, 2.3x ->
2.2x), dominance margins (12x / 6.3x / 3.9x), scaling factors (10x batching, 2-4x
quantization), figure annotations and tables.
Follow-up (same day): factor parentheticals removed from the headline sentences
(abstract, §I bullet 3, §VI, V-B reading (b)) — peers give one quantity, not two.
Headlines now read "by 55%" and "71-76%" only. The factor 2.2 survives once, in V-B
reading (a) ("a factor of 2.2 that is essentially constant across the grid"), where the
constancy of the ratio is the point. 26 pages.

## Formal-convention pass (2026-08-24)

1. Acronyms expanded at first prose use (IEEE): LDPC, BPSK, FER, DC, DJSCC/JSCC (§III);
   LUT, MLP, CSI, BCE, LCB, CI (§IV); RSMA, LLM, PSNR (§II); AUC (§V).
2. "(ours)" -> "(proposed)" in Table 1 rows and in all figure legends (scripts f1, f3, f5,
   f14_radio_regime, f16 regenerated; legacy scripts f4/f9/f14_old relabelled for
   consistency). Comms convention: "proposed" scheme.
3. Subsection titles -> Title Case noun phrases (IEEE), colon spine kept:
   "Energy: The Deployable Saving and the Attainable Gain", "Accuracy Under Routing: The
   Fixed-Policy Grid", "Mechanism: Evidence--Question Complementarity", "Per-Sample
   Routing: Attainable Gain", "Robustness and Scope", "Type Routing: The Deployable Pair",
   "Per-Sample Routers: Attainable Gain", "Analytical Support for the Type Split", §III x4.
4. Equation references: "Eq.~\eqref{}" -> bare "\eqref{}" (6 sites) per IEEE mid-sentence
   style.
5. Run-in "When does compute dominate?" -> "Boundary of compute dominance."; the §I framing
   question kept as the paper's single deliberate rhetorical device.
Page budget held at 26 via small trims (duplicate trace-driven sentence in §IV-B, two
parentheticals, one clause; Fig. 3 -> 0.43 linewidth).

## 2026-09-02 — Co-author added; page budget restored

- Advisor (WeChat, 2026-09-02): add Prof. Nelson Fonseca as co-author; polish the
  paper to a level he would agree to sign; remove AI-writing traces ("一句话含金量
  太高" → split into several sentences); plain vocabulary, "Keep it simple".
- Author block: added "Nelson L. S. da Fonseca, Senior Member, IEEE", Institute of
  Computing, State University of Campinas (UNICAMP), Campinas 13083-852, Brazil,
  nfonseca@ic.unicamp.br. Verified from IEEE-paper footnotes (arXiv 2408.13298,
  2109.08989, 2510.14214) and the official UNICAMP page. NOT Fellow: the IEEE
  Fellow "Nelson J. G. Fonseca" (Anywaves, Toulouse) is a different person.
  Author ORDER is provisional (placed last) pending the corresponding author.
  Submission month → September 2026.
- The extra \thanks line pushed 4 references to p.27. Recovered 26 pp by:
  conclusion: dropped "(more than 21,000 held-out test decisions)" (dup of
  abstract), merged "We showed ... without any training" into the next sentence,
  dropped the TOST/"statistically equivalent" sentence (stats not a selling point);
  §V-F: shortened Reproducibility, removed "We therefore report the dependence."
  and "Richer token vocabularies are future work." (dup of conclusion);
  §V-G(ii): removed the VideoQA-SC sentence (still cited in §II); Fig. 8
  (F15_crossvlm) width 1.35 → 1.25 evfigw. Result: 26 pages, 0 undefined
  refs/cites, all 58 references present.
- Plain-vocabulary sizing (for the "keep it simple" pass, not yet executed):
  123 flagged occurrences / 34 word types in 7,508 prose words; ~25 are math or
  physics terms to keep; ~95 replaceable (trace-driven 9, attain* 9, attainable
  gain 7, regime 7, exceed 7, deployable 6, yield 5, invoke 5, thus/thereby/
  whereas 8, quantify 4, parameterize 4, nominal 4, compute-dominated 4, ...).
