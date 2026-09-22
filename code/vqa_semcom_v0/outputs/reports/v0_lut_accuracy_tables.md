# V0.5 LUT Accuracy Tables

Mean expected accuracy aggregated over channel, freshness, and risk-level cells.

| question type | service level | poor view | medium view | good view |
|---|---:|---:|---:|---:|
| attribute | 0 | 0.391 | 0.413 | 0.426 |
| attribute | 1 | 0.284 | 0.430 | 0.527 |
| attribute | 2 | 0.407 | 0.569 | 0.688 |
| attribute | 3 | 0.382 | 0.539 | 0.656 |
| counting | 0 | 0.419 | 0.443 | 0.460 |
| counting | 1 | 0.307 | 0.479 | 0.595 |
| counting | 2 | 0.436 | 0.609 | 0.737 |
| counting | 3 | 0.407 | 0.574 | 0.701 |
| presence | 0 | 0.477 | 0.503 | 0.520 |
| presence | 1 | 0.347 | 0.548 | 0.684 |
| presence | 2 | 0.487 | 0.681 | 0.821 |
| presence | 3 | 0.457 | 0.648 | 0.784 |
| relation | 0 | 0.368 | 0.392 | 0.405 |
| relation | 1 | 0.266 | 0.404 | 0.499 |
| relation | 2 | 0.389 | 0.547 | 0.662 |
| relation | 3 | 0.351 | 0.502 | 0.612 |
| risk | 0 | 0.375 | 0.402 | 0.414 |
| risk | 1 | 0.276 | 0.407 | 0.497 |
| risk | 2 | 0.398 | 0.557 | 0.675 |
| risk | 3 | 0.355 | 0.514 | 0.625 |

## Fresh + Good/Medium Sanity Check

| question type | view | cache s=0 | light s=1 | full image s=2 | ROI/crop s=3 |
|---|---|---:|---:|---:|---:|
| attribute | medium | 0.536 | 0.527 | 0.620 | 0.592 |
| attribute | good | 0.550 | 0.639 | 0.751 | 0.716 |
| counting | medium | 0.576 | 0.592 | 0.663 | 0.627 |
| counting | good | 0.597 | 0.730 | 0.806 | 0.761 |
| presence | medium | 0.649 | 0.679 | 0.734 | 0.710 |
| presence | good | 0.673 | 0.830 | 0.896 | 0.854 |
| relation | medium | 0.507 | 0.487 | 0.603 | 0.548 |
| relation | good | 0.514 | 0.607 | 0.702 | 0.671 |
| risk | medium | 0.517 | 0.491 | 0.606 | 0.561 |
| risk | good | 0.536 | 0.605 | 0.732 | 0.683 |
