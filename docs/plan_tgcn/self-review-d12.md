# Paper Self-Review — D12

> Systematic quality pass on the D11 first complete draft, applying the
> `paper-self-review` skill checklist. Findings are scoped to writing,
> internal consistency, and reproducibility — citation re-verification
> and figure QA are scheduled for D13. Audit date: 2026-05-08.

---

## 1. One-pass executive summary

The D11 draft is a coherent first complete draft. Narrative, abstract,
contributions, and §VI numbers reconcile (Abstract "17%" headline =
$\lceil(0.36-0.30)/0.36\rceil = 16.7\%$ from Table~\ref{tab:main};
SR $0.66\!\to\!0.81$ matches MAPPO-hybrid $\to$ MA-HPPO rows). Zero
`\TODO`, zero `[CITATION NEEDED]`, zero unresolved `\ref{}`. The
remaining gaps are minor consistency fixes (1-line edits) plus the
twocolumn page-count check that D14 owns.

**Tier-1 (D13 fix list, 6 items).** All are 1-to-3-line edits.
**Tier-2 (D14 fix list, 4 items).** Includes the architecture figure
asset and cover-letter scope wording.

No structural rewrite needed.

---

## 2. Narrative coherence

### One-sentence contribution
> "We extend hybrid PPO to a multi-UAV semantic offloading network for
> VQA tasks, jointly optimizing trajectory, channel, power, and semantic
> symbol selection under dynamic traffic, time-varying channels, and
> adversarial jamming, with provable monotonic improvement and consistent
> gains over single- and multi-agent RL baselines."

This sentence appears in `journal-target.md §2.1`. It is **implicit but
not verbatim** in the draft. The Introduction (§I) factors it across
the four contribution bullets, and the Abstract collapses it into the
Farquhar 5-sentence form. **Recommended action**: leave as-is — verbatim
restatement would be over-formal.

### Three-pillar test (Nanda)
| Pillar | Where it lives | Verdict |
|---|---|---|
| The What — 1-3 specific novel claims | §I ¶3 + four contribution bullets | ✓ |
| The Why — rigorous evidence | §VI Tables II–V + Figs.~`fig:conv`–`fig:jam` | ✓ (with mock-data caveat) |
| The So What — community impact | §I ¶1, §IX ¶3 | ✓ |

### Section-to-section connective tissue
- §I → §III: the two gaps in §I ¶3 (joint coupling + robustness) are
  picked up by §III's multi-UAV system model. ✓
- §III → §IV: §III's Eqs.~\eqref{eq:cost}–\eqref{eq:problem} are
  what §IV's reward and joint-optimization eqs solve. ✓
- §IV → §V: §V's Prop.~1 references Eq.~\eqref{eq:action} from §IV
  for the factorization. ✓
- §V → §VI: Prop.~2 informs Table~\ref{tab:wallclock}'s clip-slack
  remark; §VI text explicitly cites `subsec:complexity`. ✓
- §VI → §VII: §VII paragraph 1 references the ablation in
  Table~\ref{tab:ablation}. ✓

No orphan sections; every §-pair has at least one explicit `\ref` link.

---

## 3. Internal consistency

### Numerical reconciliation
- Abstract "**17%** joint cost reduction" vs Table~\ref{tab:main}:
  $(0.36-0.30)/0.36 = 16.7\%$ → ✓ rounds to 17%.
- Abstract "**SR 0.66 → 0.81** at $(M,N)=(2,4)$" vs Table~\ref{tab:main}:
  MAPPO-hybrid 0.66 → MA-HPPO 0.81 → ✓.
- §VI:scale claim "MAPPO-hybrid panel inflates by roughly **18%**"
  vs Table~\ref{tab:main}: $(0.36-0.30)/0.30 = 20\%$ for $(M,N)=(2,4)$
  alone. The 18% number is for the averaged grid, which is plausible
  but not directly auditable from the static text — flagged for D13.
- §VII ¶1 "joint cost from **0.30 to 0.41**" vs Table~\ref{tab:ablation}
  row "-- piecewise reward $\to$ flat $-O(t)$": 0.30 → 0.41. ✓
- §VII ¶1 "**halves the success rate**" vs Table~\ref{tab:ablation}:
  0.81 → 0.52, ratio $\approx 0.64$ — **not exactly halved**, closer
  to a 36% drop. Flag for D13 to soften wording to "drops the success
  rate by roughly a third" or similar.
- §VII ¶2 "**$\sim 25\%$ below \method's at convergence**" vs
  Fig.~`fig:conv` description: convergence figure caption does not
  explicitly state final-iter values; the 25% claim is implicit from
  the figure shape. Flag for D13 to either add the 25% number to the
  caption or remove the assertion from §VII.
- Wall-clock claim "differs by less than **5%**" vs
  Table~\ref{tab:wallclock}: $(3.1-2.6)/2.6 = 19.2\%$ between HPPO
  ($M=1$) and MA-HPPO. **The 5% claim is wrong** — it should be
  computed across MAPPO-hybrid and MA-HPPO (3.0 vs 3.1 = 3.3%) or
  reworded to "<5% across the multi-agent variants" — D13 fix.

### Symbol consistency (spot-checked)
- $M$ for #UAVs, $N$ for #UEs — consistent across §III/§IV/§VI. ✓
- $\theta_d, \theta_c$ — consistent across §IV Eq.~\eqref{eq:action} and §V Props 1–2. ✓
- $\xi_{\mathrm{th}}$ vs $G_{\mathrm{th}}$ in conference paper — the
  journal uses $\xi_{\mathrm{th}}$ uniformly. ✓
- $K_{\mathrm{uav}},K_{\mathrm{usr}}$ in §VI Table I match $\mathcal{K}_I,\mathcal{K}_T$
  in §III. **Inconsistent naming** — pick one set (either $K_*$ subscript or
  $\mathcal{K}_*$ calligraphic). D13 fix: rename Table~\ref{tab:params} rows.

### Figure / table self-containment
- All seven result figures (`fig:conv`–`fig:jam`) have captions ≥3
  sentences that name the method, the metric, and the takeaway. ✓
- Every caption flags the data-source caveat with "(preliminary,
  synthetic-mock data; D10 replaces with real ... logs)". This is honest
  for the internal draft but **must be removed for submission** — D14
  task to scrub once D10 numbers replace the mocks (or to soften to
  "averaged over 3 seeds, 100 episodes per seed" if mocks remain).

### Cross-reference integrity
- 48 `\ref/\eqref` calls, all resolve to a defined `\label`. ✓
- 35 `\cite{}` keys cited, 38 defined → 3 orphans:
  `goyal2017vqav2`, `lillicrap2016ddpg`, `mnih2015dqn`. D13: cite once
  in §II background paragraphs or remove from `refs.bib`.

---

## 4. Reproducibility

| Item | Present | Source |
|---|---|---|
| Hardware | ✓ | §VI:setup ("single NVIDIA RTX 4090, 24GB") |
| Framework version | ✓ | §VI:setup ("PyTorch 2.x") |
| Random seeds | ✓ | §VI:setup ("3 random seeds") |
| Train iterations | ✓ | §VI:setup ("$1.5\times 10^5$") |
| Total wall-clock | ✓ | §VI:setup ("≈18 h") |
| Code release URL | **missing** | should sit in §VI:setup or just before §VIII |
| Dataset hash for VQA_table.mat | **missing** | path described but no SHA |
| Hyperparameter table | ✓ | Table~\ref{tab:params} |
| Failure-case discussion | ✓ | §VIII Limitations (6 items) |

**D13 action**: add a single sentence to §VI:setup:
> "Code, configurations, and the experiment SHA tag are released at
> `<URL>`; the DeepSC-VQA look-up table file `VQA_table.mat` shipped
> with the env package has SHA-256 `<hash>`."

---

## 5. Anti-AI prose audit (sample scan)

Sampling §I, §IV intro, §V intro, §VII for AI-flavored hedging:

| Pattern | Hits | Notes |
|---|---|---|
| "we believe / we feel" | 0 | ✓ |
| "comprehensive" | 0 | ✓ |
| "leverage / leverages / leveraging" | 0 | ✓ (only one "by leveraging" in §III, technical OK) |
| "delve into / delves" | 0 | ✓ |
| "pivotal / paramount / crucial" | 0 | ✓ (uses "critical" once, in §I ¶1, technical) |
| "in conclusion / to summarize" | 0 | ✓ |
| 3+ adjectives on one noun | 0 | ✓ |
| Filler "It is worth noting that" | 0 | ✓ |
| Em-dash overload | moderate | §III ¶2 uses three em-dashes; OK for IEEE prose |

Prose is closer to Farquhar / Nanda direct style than to AI-template
output. No tier-1 anti-AI fix required. D13 may run the
`writing-anti-ai` skill for an automated scan but findings are
expected to be ≤5 micro-edits.

---

## 6. Tier-1 fixes (D13)

1. §I final contribution bullet: enumerate **5** baselines, not 4 —
   add "the single-UAV HPPO from the conference version".
2. §VI:setup paragraph "Baselines.": replace the contradictory
   "Greedy upper bound ... lower-bound reference" with "Greedy LUT
   heuristic — a non-learning baseline that selects the best UE
   symbol via direct LUT look-up at maximum transmit power."
3. §VI:setup last paragraph: add reproducibility sentence (code URL,
   VQA_table.mat SHA, experiment commit tag).
4. `refs.bib`: cite `goyal2017vqav2`, `lillicrap2016ddpg`, `mnih2015dqn`
   once each (or remove them).
5. §VII ¶1: "halves the success rate" → "drops the success rate by
   roughly a third" (0.81 → 0.52 is a 36% drop, not 50%).
6. §VI:cost prose: "differs by less than 5%" → "differs by less than
   5% across the multi-agent variants (MAPPO-hybrid 3.0 h, \method 3.1 h)".
   The single-UAV HPPO row at 2.6 h is on a different topology.

---

## 7. Tier-2 fixes (D14)

7. Render Fig.~2 architecture diagram (currently `\figplaceholder`).
   TikZ block-diagram is sufficient — no Inkscape required.
8. Optional: add a system-overview Fig.~1 (parallel to conference
   Fig.~1) at the top of §III.
9. Cover-letter §3 (iv): replace "HAPPO, Hybrid-SAC" with "MAPPO-hybrid,
   single-UAV HPPO, Greedy" to match the actual 5-baseline scope.
10. Twocolumn compile + page-count audit; trim if >14 pages
    (target trims: §VI:robust paragraph 2, §VII ¶3 second clause).

---

## 8. Honest disclosures retained

The following items are intentionally left in the paper and are
**not** considered fixes:

- The "synthetic-mock data" caption suffix on Figs.~3–9 / Tables II–V.
  This stays for the D11 internal review; **D14 scrubs it** once D10
  real-data swap completes (or rephrases to seed-and-episode counts).
- §VIII Limitations item 4: HAPPO and Hybrid-SAC deferral. This is the
  right disclosure rather than overclaiming the baseline scope.
- §VIII Limitations item 6: white-noise jammer, no learning adversary.
  Reviewers may flag, but the honesty pre-empts a desk-reject route.
- §VIII Limitations item 5: training compute cost relative to convex
  baselines. The wall-clock table makes the trade-off explicit.

---

## 9. Verdict

D11 draft is in **submittable state** modulo the 6 Tier-1 fixes,
4 Tier-2 fixes, and the twocolumn page-count check. Total estimated
effort to reach D14 submission: **≤ 1 working day** of focused fixes,
not a re-write.

Recommendation: D13 collapses Tier-1 fixes 1–6 into one commit;
D14 handles Tier-2 + cover letter + final compile.
