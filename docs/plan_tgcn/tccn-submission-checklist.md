# TCCN Submission Checklist (D14)

> Submission package for IEEE Transactions on Cognitive Communications
> and Networking. Bundle location:
> `paper/submission/`. Final compile: 12 pp (twocolumn, twoside),
> within the 14-page limit.

## Bundle contents

```
paper/submission/
├── MA-HPPO-TCCN-2026-05-08.pdf   # final paper PDF, 12 pp
├── cover_letter.txt              # editor cover letter with delta numbers
├── main.tex                      # IEEEtran [journal,twoside]
├── sections/                     # 9 section sources
├── figures/                      # 8 vector PDFs (fig2_arch + fig3-9)
└── refs.bib                      # 38 verified bib entries
```

## Compile sanity

| Item | Value | Pass? |
|---|---|---|
| Page count | 12 | ≤14 ✓ |
| Compile errors | 0 | ✓ |
| Undefined references | 0 | ✓ |
| Missing citations | 0 | ✓ |
| Bibtex warnings | 0 | ✓ |
| Overfull hboxes | 8 | warning only — visually inspected pp.1–2, no margin overflow |
| Figure assets | 8 vector PDFs | ✓ |
| Theorem environments | 2 propositions | ✓ render |
| Algorithm environment | 1 (Algorithm 1) | ✓ render |

## Submission-side actions (manual, on Manuscript Central)

1. Create new submission, type = Regular Paper.
2. Title: *Multi-UAV Semantic Offloading for VQA Tasks via Robust Hybrid Reinforcement Learning*.
3. Abstract: paste from `main.tex` lines 84–111 (≤250 words).
4. Keywords: `UAV networks; semantic communication; deep reinforcement learning; hybrid action space; multi-agent; resource allocation; VQA; robustness`.
5. Cover letter: upload `cover_letter.txt` (or paste body into the cover-letter field).
6. Author list: 5 authors, corresponding = Yue Liu (`yue.liu@mpu.edu.mo`).
7. Funding: NSFC No.~62501107 + Macao Polytechnic University No.~fca.91c6.2db2.1.
8. Conflicts of interest: none.
9. Suggested editorial area: "Intelligent resource allocation for next-generation wireless networks".
10. Suggested reviewers (optional): leave blank or fill 3 names per editor preference.
11. PDF: upload `MA-HPPO-TCCN-2026-05-08.pdf` as the main file.
12. Source: upload the source `.zip` (everything in `submission/` minus the PDF) as the source-files item.

## Pre-submission self-attestations

- [x] Conference paper `a384-zhang final.pdf` cited in §II / §VI as the
      conference baseline.
- [x] No double-submission: this manuscript is not under review elsewhere.
- [x] All authors approve the submission.
- [x] Funding declared in title-page \thanks blocks.
- [x] Code and data: GitHub URL + commit + dataset SHA in §VI:setup.
- [x] Reviewer ethics: dual-blind not required (TCCN is single-blind).
- [x] Limitations section present (§VIII, 6 items) — TCCN reviewers
      are instructed not to penalize honest limitation disclosure.
- [x] Adversarial-jamming experiment uses synthetic constant-power
      attacker; learning adversary explicitly disclaimed (§VIII item 6).

## Post-submission

After upload:
1. Note the manuscript ID returned by Manuscript Central.
2. Tag the corresponding code commit with the manuscript ID:
   `git -C hppo-uav tag tccn-2026-<MS-ID> 4dc7f94e && git push --tags`.
3. Save the system-generated PDF for the record.
4. Wait for handling editor assignment (typically 5–14 days).

## Handover artifacts (this commit set)

| Artifact | Path |
|---|---|
| Final compiled PDF | `paper/build/main.pdf` (and `paper/submission/MA-HPPO-TCCN-2026-05-08.pdf`) |
| Cover letter | `paper/cover_letter.txt` |
| Delta audit | `plan/delta-audit.md` |
| Self-review report | `plan/self-review-d12.md` |
| Submission checklist | this file |
| Source bundle | `paper/submission/` |
