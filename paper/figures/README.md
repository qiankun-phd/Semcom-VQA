# figures/ — figure assets for Paper 1 (evidence routing)

The `main.tex` `\evfig` macro guards every figure with `\IfFileExists`, so
the document compiles **with or without** the PDF assets (a red placeholder
box appears where an asset is missing). To render the real figures, copy the
following PDFs from the 160 server into this directory:

Source on 160: `lab-s2:~/phd_research/vqa_semcom/outputs/figures/comparison/`
(branch `codex/lut-semantic-utility-upgrade`, tip `08aa141`;
built by `build_comparison_v2 / build_mismatch_matrix / build_persample_policy /
build_token_budget_sweep / analyze_crossvlm`).

| Filename expected here     | Source figure | Content |
|----------------------------|---------------|---------|
| `F1_accuracy_snr.pdf`      | F1 | accuracy vs. SNR, 3 channels |
| `F2_cliff.pdf`             | F2 | cliff effect (Rayleigh) |
| `F3_latency.pdf`           | F3 | end-to-end latency decomposition |
| `F4_bytype.pdf`            | F4 | per-type accuracy (Rician @5 dB) |
| `F5_pareto.pdf`            | F5 | accuracy vs. complex channel uses |
| `F6_complementarity.pdf`   | F6 | token gain Δ by question type |
| `F7_mismatch.pdf`          | F7 | CSI-mismatch matrix |
| `F8_token_budget.pdf`      | F8 | top-t token-budget sweep |

Rename the source PDFs to the filenames in the left column (or edit the
`\evfig{...}` first argument in the relevant `sections/*.tex`).

Example fetch:
```
scp lab-s2:'~/phd_research/vqa_semcom/outputs/figures/comparison/F1*.pdf' F1_accuracy_snr.pdf
```
