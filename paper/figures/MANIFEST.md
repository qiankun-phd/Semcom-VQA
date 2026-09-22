# Paper 1 -- Figure MANIFEST

Regenerated **2026-07-09** from the current prediction logs on the compute
server (`lab-s2:~/phd_research/vqa_semcom`, branch `main`; commit
`9707835`). Python `~/.conda/envs/uav_semcom/bin/python`.

All nine PDFs are produced from the CURRENT 5-question-type merged logs
`outputs/vlm/v3_0_{awgn,rayleigh,rician}_predictions.csv` (via the tidy report
`outputs/reports/comparison_v3_5qt.csv`). Test split = `crc32(image_id) % 100 < 20`,
with `image_id` always read as a **string** (DroneVehicle ids are zero-padded and
must not be coerced to int). Filenames match the paper's `\evfig{...}` macro.

**IEEE styling (all figures):** in-figure titles removed (IEEE places the caption
below); channel / operating point kept only as a small in-axes label; error
bands/bars added from the Wilson 95% `lcb`/`ucb` columns (no fabricated
uncertainty); legend labels use the descriptive method names (see 2026-07-10 note below).

| Fig | file | generating script | data source CSV | source logs |
|-----|------|-------------------|-----------------|-------------|
| F1 | `F1_accuracy_snr.pdf` | `make_comparison_figures_v2.py` | `comparison_v3_5qt.csv` | `v3_0_{awgn,rayleigh,rician}` (+`m2_analog_*`, `v2_0_*_naive`, `v3_0_clean`) |
| F2 | `F2_cliff.pdf` | `make_comparison_figures_v2.py` | `comparison_v3_5qt.csv` | as F1 |
| F3 | `F3_latency.pdf` | `build_latency_breakdown.py` | `comparison_v3_5qt.csv` + `latency_breakdown.csv` | `v2_0_rayleigh`, `m2_analog_rayleigh`, `v2_0_rayleigh_naive` (measured `latency_sec`) |
| F4 | `F4_bytype.pdf` | `naturefig_f4_bytype.py` (faithful bar chart from `make_comparison_figures_v2.py` F4 block) | `comparison_v3_5qt.csv` | **2026-07-23:** keep original **grouped bars** style; high-res PNG via PDF@400 dpi only (no line-plot redesign). |
| F5 | `F5_pareto.pdf` | `make_comparison_figures_v2.py` | `comparison_v3_5qt.csv` | as F1 |
| F6 | `F6_complementarity.pdf` | `make_complementarity_fig.py` | computed in-script from pooled `v3_0_*` | `v3_0_{awgn,rayleigh,rician}` |
| F7 | `F7_mismatch.pdf` | `build_mismatch_matrix.py --prefix v3_0` | `mismatch_matrix.csv` | `v3_0_{awgn,rayleigh,rician}` |
| F8 | `F8_token_budget.pdf` | `build_token_budget_sweep.py` | `token_budget_full.csv` | `v2_0_snr` detections + task CSVs |
| F9 | `F9_separation.pdf` | `make_p1_figures.py` (P1 batch, commit `d95c507`) | `outputs/reports/separation_v2/sepcap_peak_shared.csv` | separation_v2 M/G/1 rerun (Rayleigh, peak load, shared C2, 2-sigma) |
| F10 | `F10_fer.pdf` | `make_p1_figures.py` | `outputs/reports/p1_fer_payload.json` | token outage (exact code path) + measured LDPC calibration (`link_calibration_v2_0_*_naive.json`) |

## One-line description

- **F1** accuracy vs SNR, 3 panels (AWGN/Rayleigh/Rician); M4 best non-oracle at every
  **2026-07-23 restyle:** `naturefig_f1_accuracy_snr.py` (TNR, Type-42, clean spines,
  panel tags clip_on=False, PNG@400 dpi); data `comparison_v3_5qt.csv` unchanged.
  SNR (~0.68 @20 dB); Wilson-95% bands on M4/M1/M3.
- **F2** cliff effect (Rayleigh): fixed-rate M0_naive collapses (~0.41 @-5 dB) while
  adaptive routing degrades gracefully.
- **F3** end-to-end latency breakdown (Rayleigh): stacked upload / detector / VLM
  **2026-07-23 restyle:** `naturefig_f3_latency.py` (TNR, Type-42, PNG@400 dpi); data `latency_breakdown.csv`.
  inference; token path orders below image.
- **F4** per-type accuracy (Rician @5 dB) with Wilson error bars; symbolic->token,
  presence->image; M4 tracks the winner.
- **F5** efficiency Pareto (Rician): accuracy vs mean complex channel uses/query (log x);
  M4 on the frontier; Wilson vbars on M4.
- **F6** evidence-question complementarity (pooled 3 ch): Delta=acc(token)-acc(image) per
  type, Wilson error bars.
- **F7** CSI-mismatch heatmaps (assumed x true SNR): off-diagonal ~ matched diagonal.
  **2026-07-23 restyle:** `naturefig_f7_mismatch.py` (TNR, Type-42, shared viridis scale, PNG@400 dpi); data `mismatch_matrix.csv`.
  scheduler robust to stale SNR.
- **F8** top-t token-budget sweep (Rician): non-monotone "less is more" on threshold.
  **2026-07-23 restyle:** `naturefig_f8_token_budget.py` aligned with F4/F14 (Times New Roman,
  Type-42, no top/right spines, high-res PNG@400 dpi); data still `token_budget_full.csv`.
- **F9** safety bridge at the corrected M/G/1-queue caliber (Rayleigh, peak load, shared
  C2, 2-sigma): token holds the BUBBLES baseline (370.6 vs 370.5 m); image pays up to
  +23.7 m / ~6% capacity at -5 dB. REPLACES the old "token buys 48.8 m / 1.14x" asset
  (that model wrongly credited tokens with shrinking the official 1.8 s C2 term).
  **2026-07-23 restyle:** `naturefig_f9_separation.py` (TNR, Type-42, PNG@400 dpi); data
  `sepcap_peak_shared.csv`.
- **F10** residual frame-loss curves: token information-outage (measured mean payload)
  vs measured fixed-rate LDPC FER; explains both the token flatness and the naive
  Rayleigh(0.979) > AWGN(1.000 FER) inversion at -5 dB.
  **2026-07-23 restyle:** `naturefig_f10_fer.py` (TNR, Type-42, clean spines, PNG@400 dpi);
  data `p1_fer_payload.json`.

## OLD -> NEW (bugs fixed this revision)

- **F6:** the previous asset hard-coded stale pilot-server deltas
  `[0.194, 0.122, 0.111, 0.021, -0.068]` with the **threshold sign wrong (+0.021)**.
  NEW computes the deltas in-script from the current pooled `v3_0` logs (string
  `image_id`, same split as `build_evidence_complementarity.py`): counting **+0.174**
  (tallest), comparison +0.151, co_presence +0.119, threshold **-0.014 (negative bar)**,
  presence -0.087. Wilson error bars added.
- **F4:** the previous asset showed the **M4 presence bar below M1**, contradicting the
  presence->image rule. NEW reads the fresh `comparison_v3_5qt.csv`: M4 presence bar
  equals the image accuracy (**0.7805**), above the token 0.678. Asymmetric Wilson error
  bars added.

Server staging: `outputs/figures/paper1_final/` (also `outputs/figures/comparison/`).

## 2026-07-10 — descriptive method names + IEEE final-size regeneration

All method legends now use the descriptive names of the paper's
Table `tab:baselines` (no internal `M0..M6` codes anywhere in the
manuscript):

| internal code | paper-facing name |
|---------------|-------------------|
| M0_errorfree  | Error-free image (ideal) |
| M0_naive      | Fixed-rate image |
| M1_image      | Rate-adaptive image |
| M2_analog     | Uncoded analog |
| M6_djscc      | DJSCC (learned) |
| M3_token      | Fixed token |
| M4_adaptive   | Evidence routing (ours) |
| M5_oracle     | Oracle (upper bound) |

Figures were regenerated (server commit `6b7695b`) at true IEEE final
sizes: F1 (and the TGCN F8) at 7.16 in for `figure*`/one-column layouts,
F2--F7 + the TCCN F8 at 3.5 in single-column, energy panels at 3.4 in;
fonts sized for the final layout, Type-42 embedded Liberation Sans,
vector PDF + 300 dpi PNG previews. Colors/markers/line styles unchanged
(grayscale-safe). Data pipelines untouched: `latency_breakdown.csv`,
`mismatch_matrix.csv`, `energy_summary.json` byte-identical after
regeneration; F6 deltas and F7 matrices reproduce the logged values
exactly. TCCN uses `F8_token_budget_col` (3.5 in) as `F8_token_budget.pdf`;
TGCN uses the 7.16 in variant under the same name.

## 2026-07-10 — publication-figure redraw of F1 / F5 / F8 / F11 (naturefig)

F1, F5 (`F5_pareto_energy`), F8 and F11 are now produced by dedicated
plot-only redraw scripts (server `scripts/naturefig_f1_accuracy_snr.py`,
`naturefig_f5_energy.py` for both energy panels, and
`naturefig_f8_token_budget.py`; outputs staged under
`outputs/figures/naturefig/` in PDF+SVG+PNG, `svg.fonttype=none` /
Type-42 editable text, 600 dpi PNG). Same data sources as before
(`comparison_v3_5qt.csv`, `token_budget_full.csv`,
`gpu_power_phases.json`, `c1_frontier_rician.csv` — all git-clean at
regeneration); no number changed.

Content of record preserved (verified against the 1d2f042 assets):

- **F5**: the C1 budget-tunable per-sample routing frontier (lambda
  sweep, drawn point-for-point: all-token endpoint `0.435 J / 0.634`,
  lambda=0 endpoint `8.87 J / 0.680`, peak `0.691` near `4.7 J`), the
  "energy price lambda sweep" tag, the fixed-rate digital-cliff note,
  Wilson bars, the Jetson detector-energy band, grayscale-safe
  linestyles.
- **F11**: the three direct plateau labels `1.46` /
  `0.043--0.047` / `~0.018 (all full-image pipelines)`
  (cross-checked against `energy_summary.json`), same linestyles.

Presentation-layer deltas only:

- **F1**: hero emphasis on the routing curve; direct labels for the
  error-free reference (panel c) and the fixed-rate digital cliff
  (panel a); panel letters (a)(b)(c); softer Wilson-band alpha.
- **F5/F11**: panel letters (a)/(b) (caption switched from Left/Right
  to (a)/(b)); F5 adds a dotted vertical guide at the measured VLM
  compute floor and leader-line SNR endpoint labels; the earlier grey
  empirical-Pareto staircase idea was dropped in favor of the lambda
  frontier (the frontier object of record). F11 gains y-headroom so
  the fixed-token line is off the frame edge.
- **F8**: panel letters (a)(b)(c); claim annotations at the sweep
  points they describe (comparison saturates at t=3, threshold peak
  ringed at t=32, counting still climbing at t=48); legend display
  name `co_presence` -> `co-presence` (CSV keys unchanged).

## 2026-07-10 — TGCN figure count 3 -> 5: new Fig. 1 architecture + restored complementarity figure

Two new naturefig assets (server commit `8c87366`, scripts
`scripts/naturefig_fig1_architecture.py` and
`scripts/naturefig_f6_complementarity.py`; SVG+PDF+PNG under
`outputs/figures/naturefig/`, Type-42 fonts, grayscale-safe):

| Fig (TGCN) | file | generating script | data source |
|-----------|------|-------------------|-------------|
| Fig. 1 | `F12_architecture.pdf` | `naturefig_fig1_architecture.py` | pure vector schematic; every number quoted verbatim from the manuscript (token 0.7–1.2 KB / 0.9 KB mean; image 14–215 KB; token 0.435 J/answer = detector 0.43 J + radio ≲0.01 J; image ≈33 J = VLM 32.3 J + radio ≤2.7 J; ≈75× gap) |
| Fig. 4 | `F6_complementarity.pdf` | `naturefig_f6_complementarity.py` | panel (a) per-arm k/n recomputed from `v3_0_{ch}_predictions.csv` (same dedup as `paper1_stats.py`, runtime-asserted against `paper1_stats.json`); panel (b) deltas + image-clustered bootstrap CIs read from `outputs/reports/paper1_stats.json` (`table3_delta_{visdrone,dronevehicle}`) — matches `tab:comp` exactly |

TGCN figure order is now: Fig. 1 architecture (§III), Fig. 2
`F1_accuracy_snr`, Fig. 3 `F5_pareto_energy`+`F11_answers_per_joule`,
Fig. 4 `F6_complementarity` (two-panel, §VI complementarity), Fig. 5
`F8_token_budget`. All cross-references are `\ref`-based; no hardcoded
figure numbers exist in the prose. Page budget held at 30 pp via §II/§V/§VI
connective-prose trims (no numeric conclusion or disclosure removed) and
tighter float separations (`\textfloatsep` 14pt, `\floatsep` 10pt,
`\abovecaptionskip` 6pt).

## 2026-07-11 -- TGCN table-to-figure batch + Times-family fonts (160 commit `ab42b2a`)

- **Fonts:** every naturefig script now renders with `font.family=serif`,
  serif stack `Times New Roman -> Nimbus Roman -> Liberation Serif`, and
  `mathtext.fontset=stix`. On the render host (160) the resolved face is
  **genuine Times New Roman (macOS system font, installed to
  `~/.local/share/fonts` on 160) + STIX math**; Type-42 embedding
  (`pdffonts` shows `TimesNewRomanPS*MT`, no Nimbus). All 11 assets
  re-rendered with true TNR on 2026-07-11: F1, F5, F6, F8, F11, F12,
  F13, F14, F15, F16, F17 (font swap only, zero data/layout change;
  F13 lambda working point unchanged at 0.680 @ 2.4 J).
- **New figures (table conversions, data of record unchanged):**

| Fig | file | generating script | converts | data source |
|-----|------|-------------------|----------|-------------|
| F13 | `F13_energy_stack.pdf` | `naturefig_f13_energy_stack.py` | new (energy decomposition) | `energy_summary.json` + `gpu_power_phases.json` + `c1_frontier_rician.csv` |
| F14 | `F14_energy_vs_snr.pdf` | `naturefig_f14_energy_vs_snr.py` (in `figures/` + server `vqa_semcom/scripts/`) | `tab:jpa` | `energy_summary.json`. **2026-07-23 restyle:** match F1/F5/F17 naturefig (serif Type-42, 3.35×2.65, hero lw 1.4); P_tx 0.1–1 W as point-wise error bars on token/routing/rate-adaptive only; original annotations restored; PNG from PDF@400 dpi for slides. |
| F15 | `F15_crossvlm.pdf` | `naturefig_f15_crossvlm.py` | `tab:crossvlm` | `paper1_stats.json:table5_crossvlm` |
| F16 | `F16_djscc_snr.pdf` | `naturefig_f16_djscc_snr.py` | `tab:m6` | `comparison_v3_5qt.csv` + `p1_m6_results.json` |
| F17 | `F17_dim_ablation.pdf` | `naturefig_f17_dim_ablation.py` | `tab:dimablation` | `ablation_mechanism_v3.csv` |

- **F6 enhanced:** panel (b) carries the numeric Delta of record per point
  plus dagger marks for CIs crossing zero -- it absorbs the former
  `tab:comp` (dropped from the manuscript).

## 2026-07-11 -- axis-style unification: all data figures use closed (boxed) axes

F6, F13, F14, F15, F16, F17 previously used half-open axes
(`spines["top"/"right"].set_visible(False)`); the six naturefig scripts now
keep all four spines visible (same 0.6 pt `axes.linewidth` as the left/bottom
spines), matching F1/F5/F8/F11 which were already boxed by default. Axis
style only -- data, labels, fonts (true TNR), sizes, grids, and tick
directions unchanged; all six re-rendered in PDF+SVG+PNG on 160 (SVG staged
server-side as before). F12 is a pure vector schematic with no axes (not
touched). F6(b) Delta labels verified post-regeneration via pdftotext:
VisDrone +0.171 / +0.148 / +0.108 / -0.015† / -0.081 (the true-pooled
values of record from `paper1_stats.json`).

## 2026-08-05 — MLP router series added to F5 / F11 / F14 (predictor-primary reboot)

`F5_pareto_energy`, `F11_answers_per_joule`, `F14_energy_vs_snr` regenerated
locally by `naturefig_f5_energy_mlp.py` and `naturefig_f14_energy_vs_snr_mlp.py`
(patched copies of the 160-server naturefig scripts, kept in this directory).
New series "Per-sample router (ours)" (`M8_mlp`, #1f7a53, triangle) from
`../data/w8_mlp_series.csv` (three-seed mean; bars = per-seed min–max energy);
M4 label renamed "Type routing (ours)". λ-frontier legend entry renamed
"Budget-tunable routing (linear)". Data of record: `../data/w8_mlp_headline.json`
(built on 182 from v3_0 log replay; see docs/plan_tgcn/reboot-blueprint-2026-08.md W8).
F1 still carries the old "Evidence routing (ours)" legend label — regenerate on
next server pass for label consistency.

## 2026-08-23 — unified style pass (closed boxes + one palette), local rebuild

All nine result figures regenerated locally (Mac, mpl 3.10.8) with a unified
style, per user request:
- **Closed axes everywhere**: F8 switched from half-open (top/right hidden)
  to full box; all others verified already closed.
- **One palette across figure families** (anchors = F6's token/image colors):
  token slate `#5a6b7c` (was `#9aa7b4` in F1/F5/F14), image orange `#e8962f`
  (was `#ffb454`), muted red `#d1495b` (was coral `#ff6b6b` / `#e06c75`),
  muted purple `#8e6bb5` (was lilac `#c678dd`), blue `#4ea1ff`,
  greens for *ours* unchanged (`#5ad19a` type rule, `#1f7a53` MLP).
  F15 receiver oranges aligned to the image-orange family
  (`#f4b860`/`#e8962f`/`#9c5f28`).
- Scripts fetched back from 182 (`naturefig_f13/f15/f16/f17`) into `figures/`;
  data staged under `paper/outputs/{reports,energy,vlm}` (v3_0 prediction CSVs
  from 160 `~/phd_research/vqa_semcom/outputs/vlm/`, ~170MB each — NOT to be
  committed; delete after rebuilds).
- Zero data change: same inputs of record, style-only diff.

## 2026-08-24 — F3 + F10 promoted into main text as Fig. "linklevel"

- `F3_latency.pdf` (latency stacked bars, Rayleigh) and `F10_fer.pdf`
  (token outage vs measured fixed-rate LDPC FER, three channels) rebuilt
  locally with the unified style pass: closed axes (both were half-open),
  palette aligned (`#ff6b6b`->`#d1495b`, `#ffb454`->`#e8962f`,
  `#c678dd`->`#8e6bb5`, `#9aa7b4`->`#5a6b7c`; F10 rayleigh
  `#e06c75`->`#d1495b`). Data: `latency_breakdown.csv`,
  `p1_fer_payload.json` (both in figures/, unchanged).
- Placed as a two-panel figure in §V-F (labels fig:linklevel); latency
  prose trimmed to the 14-49x headline, exact decomposition values now
  carried by panel (a) only. Main text: 10 figure environments + 3 tables.

## 2026-08-24 — unit change: complex channel uses -> airtime (seconds)

Paper-wide, transmission cost is now expressed as airtime T_s = u_s/B
(B = 1 MHz declared in §III), with "complex symbols" mentioned once for the
information-theoretic reader. Numbers unchanged (uses / 1e6 = seconds):
3.72e6 uses -> 3.72 s, 1.4e4 -> 14 ms, slot 3e5 -> 0.3 s, DJSCC 9.2e4 ->
92 ms. F8 panel (c) x-axis relabelled "airtime per query (ms)" (formatter
v/1e3), regenerated; data unchanged.

## 2026-08-24 — Fig. 2 redesigned (F14_energy_vs_snr, new script naturefig_f14_radio_regime.py)

Advisor critique: SNR-axis plots were flat lines. Diagnosis (energy_summary.json):
the radio term of the rate-adaptive image swings 11.7x (2.06 -> 0.18 J, -5 -> 20 dB)
but is <= 6% of the per-answer budget at B = 1 MHz, so totals are flat; 4/6 series
were flat by construction (fixed uses). Redesign, same output filename:
- (a) radio energy vs SNR (log) with measured compute references (VLM 32.3 J,
  detector 0.43 J) and the token airtime range band (compact frame -> full slot).
- (b) regime map: radio share of the image path vs SNR for (B, eta) in
  {1 MHz, 100 kHz} x {1, 0.25}; worst-link shares 6.0 / 20.3 / 39.0 / 71.8 %.
  Pure accounting (E_tx scales with 1/B, 1/eta); no new measurements.
- F11_answers_per_joule moved from main Fig. 3 to the supplementary (flat, redundant
  with the frontier). Main Fig. 3 is now the single frontier panel (F5).
- Fig. 5 (F1) caption states the flatness as the claim (link-independent by design).
Old naturefig_f14_energy_vs_snr_mlp.py kept for reference; no longer the source of Fig. 2.

## 2026-08-24 — method-name unification across all figures and tables

Advisor critique: method names differed across figures/tables. Canonical names (Table 1
is the authority; its caption "the same names appear in all figures and tables" is now
true): Error-free image · Fixed-rate image · Rate-adaptive image · Uncoded analog ·
DJSCC (learned) · Fixed token · Type routing (ours) · Linear routing · Per-sample
routing (ours) · Budget-tunable routing (λ sweep / λ point) · Oracle (upper bound).
Rule: scheme names use "X routing"; "router / rule / LUT / MLP" appear only in prose
when describing the mechanism. Script edits + regeneration: F1 ("Error-free image",
"Type routing (ours)"), F3 ("Type routing (ours)"), F5 ("Per-sample routing (ours)",
panel tag "(a)" removed, "VLM compute lower bound"), F6 ("Fixed token (symbolic
decoder)" / "Rate-adaptive image (frozen VLM)"), F10 ("Fixed token outage (AWGN)" /
"Fixed-rate image FER (…)"), F13 ("Type routing", "Budget-tunable routing (λ point)",
"VLM compute (s2 inference)"), F14 ("VLM inference, measured"), F15 ("Fixed token
(symbolic decoder)" / "Rate-adaptive image, <VLM>"), F16 ("Type routing (ours)").
Tables: tab:baselines rows renamed (+ Linear routing row), tab:persample and
supplementary tab:main headers renamed. Data unchanged.

## 2026-08-24 — Fig. 4 (F13) lambda point switched to the MLP sweep

The "Budget-tunable routing (λ point)" bar now comes from data/c1_frontier_rician_mlp.csv
(copied to outputs/energy/), row nearest 3.2 J: λ = 7.94e-3 J^-1, acc 0.690, 3.237 J,
frac_image 0.086 (split: radio 0.061 J, detector 0.391 J, VLM 2.785 J). Previously it used
the LINEAR sweep (0.680 @ 2.43 J), inconsistent with Fig. 3's dashed MLP curve. Text
("about 0.69 at a third of the router's energy") already matched the MLP point.

## 2026-08-24 — legend suffix "(ours)" -> "(proposed)"

Communications convention ("the proposed scheme"). Relabelled and regenerated:
F1 (Fig. 5), F14_energy_vs_snr via naturefig_f14_radio_regime.py (Fig. 2),
F5 + F11 (Fig. 3 / supp), F3 (Fig. 9a), F16 (supp). Legacy scripts f4/f9/f14_old
relabelled for consistency (not used in the paper). Table 1 rows: "Type routing
(proposed)", "Per-sample routing (proposed)". Data unchanged.

### 2026-09-02 — Fig. 8 width
- Fig. 8 (`F15_crossvlm.pdf`) `\evfig` width 1.35 → 1.25 evfigw (page budget after a
  fourth author footnote was added). PDF itself unchanged.
