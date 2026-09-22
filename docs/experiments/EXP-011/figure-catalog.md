# Figure catalog

## Figure 1 — accuracy grid

- Files: `figures/figure-01-accuracy-grid.pdf` and `.png` (600 dpi).
- Source: validated aggregate counts in `summary.json`; 120 paired development questions in every cell.
- Purpose: inspect whether image bytes and receiver visual budget interact sufficiently to motivate selection.
- Caption: Exact-match accuracy against actual mean image-bitstream kB for three visual tiers. Left uses 0–100%; right explicitly zooms to 65–85%. Markers and counts represent one frozen-model run, not repeated seeds. Orange circles/dashes denote low; blue squares/solid denotes medium; black triangles/dots denote high. No smoothing or significance marks; no fabricated error bars.
- Observation: 4k/medium and 4k/high each obtain 93/120; higher communication/visual budgets do not uniformly improve accuracy.
- Implication: investigate selection headroom while retaining fixed baselines; the figure itself does not validate a router.
- Checklist: verify all nine counts against the numeric table; read full-scale and zoom jointly; do not call equal counts equivalent accuracy; do not interpret connected lines as unseen-budget interpolation evidence.

## Figure 2 — observed generation latency and visual tokens

- Files: `figures/figure-02-latency-tokens.pdf` and `.png` (600 dpi).
- Source: `scored.json`, grouped by the same nine cells; 120 measurements per box.
- Purpose: verify actual visual-budget intervention and compare its token effect with measured latency.
- Caption: Per-question generation latency and actual visual-token distributions. Boxes show median and interquartile range; whiskers extend to 1.5 IQR; points outside are retained. Groups use the same tier colors as Figure 1. The 120 observations are different questions/images, not independent retraining runs; no bootstrap uncertainty is encoded in boxes.
- Observation: 4k/medium mean tokens decrease 48.93% relative to 4k/high, while median measured latency decreases 17.71%.
- Implication: budget knobs work, but measured speed and tokens are not interchangeable with energy or end-to-end costs. The 20% fixed-efficiency threshold is not met by this pair.
- Checklist: verify unchanged image bytes within each rate; distinguish input variability from random-seed uncertainty; retain outliers; report timing scope and missing energy; do not infer linear latency scaling with tokens.
