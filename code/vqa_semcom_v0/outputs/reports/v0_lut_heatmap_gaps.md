# V0.5 LUT Heatmap and Service-Level Gaps

Cells aggregate fresh-cache, medium/good-channel LUT accuracy. The gap columns quantify how much semantic evidence improves over cache reuse.

| question type | view | s0 cache | s1 light | s2 full | s3 ROI | s1-s0 | s2-s1 | s3-s2 | best level |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| attribute | poor | 0.514 | 0.347 | 0.442 | 0.416 | -0.167 | 0.095 | -0.026 | 0 |
| attribute | medium | 0.536 | 0.527 | 0.620 | 0.592 | -0.009 | 0.093 | -0.028 | 2 |
| attribute | good | 0.550 | 0.639 | 0.751 | 0.716 | 0.089 | 0.112 | -0.035 | 2 |
| counting | poor | 0.550 | 0.374 | 0.475 | 0.446 | -0.175 | 0.101 | -0.030 | 0 |
| counting | medium | 0.576 | 0.592 | 0.663 | 0.627 | 0.016 | 0.071 | -0.036 | 2 |
| counting | good | 0.597 | 0.730 | 0.806 | 0.761 | 0.133 | 0.076 | -0.045 | 2 |
| presence | poor | 0.630 | 0.424 | 0.527 | 0.494 | -0.206 | 0.103 | -0.033 | 0 |
| presence | medium | 0.649 | 0.679 | 0.734 | 0.710 | 0.030 | 0.055 | -0.024 | 2 |
| presence | good | 0.673 | 0.830 | 0.896 | 0.854 | 0.156 | 0.067 | -0.042 | 2 |
| relation | poor | 0.486 | 0.329 | 0.426 | 0.378 | -0.158 | 0.097 | -0.048 | 0 |
| relation | medium | 0.507 | 0.487 | 0.603 | 0.548 | -0.021 | 0.116 | -0.055 | 2 |
| relation | good | 0.514 | 0.607 | 0.702 | 0.671 | 0.093 | 0.095 | -0.031 | 2 |
| risk | poor | 0.494 | 0.334 | 0.436 | 0.388 | -0.160 | 0.102 | -0.047 | 0 |
| risk | medium | 0.517 | 0.491 | 0.606 | 0.561 | -0.026 | 0.115 | -0.045 | 2 |
| risk | good | 0.536 | 0.605 | 0.732 | 0.683 | 0.069 | 0.127 | -0.049 | 2 |
