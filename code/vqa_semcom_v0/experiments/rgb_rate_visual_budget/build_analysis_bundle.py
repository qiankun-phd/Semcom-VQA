"""Render an aggregate-only exploratory bundle from completed dev-grid analysis.

Does not change inference, rescore answers, select new thresholds, or open test
data. Matplotlib is required only for producing the two real scientific figures.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TIERS = ("low", "medium", "high")
BUDGETS = (2000, 4000, 8000)
COLORS = {"low": "#E69F00", "medium": "#0072B2", "high": "#000000"}
MARKERS = {"low": "o", "medium": "s", "high": "^"}
STYLES = {"low": "--", "medium": "-", "high": ":"}
Json = dict[str, Any]


def load_inputs(folder: Path) -> tuple[Json, Json, list[Json], Json]:
    summary = json.loads((folder / "summary.json").read_text())
    decision = json.loads((folder / "decision.json").read_text())
    scored = json.loads((folder / "scored.json").read_text())
    if summary["audit"]["scope"] != "development" or not decision["development_screen_only"]:
        raise ValueError("This renderer accepts development-only analysis")
    if not summary["audit"]["decision_eligible"]:
        raise ValueError("This full bundle requires the completed paired development grid")
    expected = {f"{budget}_{tier}" for budget in BUDGETS for tier in TIERS}
    if set(summary["cells"]) != expected or len(scored) != summary["audit"]["n_questions"] * 9:
        raise ValueError("Incomplete or unexpected grid")
    seen = Counter((row["id"], row["budget"], row["tier"]) for row in scored)
    if any(count != 1 for count in seen.values()):
        raise ValueError("Duplicate scored pairs")
    for cell, metric in summary["cells"].items():
        budget, tier = cell.split("_")
        rows = [row for row in scored if row["budget"] == int(budget) and row["tier"] == tier]
        if len(rows) != metric["n"] or sum(row["correct"] for row in rows) != metric["correct"]:
            raise ValueError(f"Scored answers and aggregate counts disagree: {cell}")
        for key, function, stat in (("image_bytes", statistics.mean, "mean"),
                                    ("actual_visual_tokens", statistics.mean, "mean"),
                                    ("receiver_seconds", statistics.median, "median")):
            if not math.isclose(function(row[key] for row in rows), metric[key][stat], rel_tol=1e-10):
                raise ValueError(f"Scored resources and aggregate disagree: {cell}/{key}")
    provenance = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                  for name in ("summary.json", "decision.json", "scored.json", "report.md")}
    return summary, decision, scored, provenance


def save_figure(figure: Any, folder: Path, stem: str) -> None:
    figure.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(folder / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(figure)


def figures(summary: Json, scored: list[Json], folder: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), gridspec_kw={"width_ratios": [1, 1.2]})
    for axis, limits, label in ((axes[0], (0, 100), "(a) Full accuracy scale"),
                                (axes[1], (65, 85), "(b) Detail: 65–85% scale")):
        for tier in TIERS:
            cells = [summary["cells"][f"{budget}_{tier}"] for budget in BUDGETS]
            xs = [cell["image_bytes"]["mean"] / 1000 for cell in cells]
            ys = [cell["accuracy"] * 100 for cell in cells]
            axis.plot(xs, ys, color=COLORS[tier], marker=MARKERS[tier], linestyle=STYLES[tier],
                      linewidth=1.7, markersize=5, label=f"{tier.capitalize()} visual tier")
            if axis is axes[1]:
                offsets = {"low": (0, -13), "medium": (0, 10), "high": (0, -5)}
                for x, y, cell in zip(xs, ys, cells):
                    dx, dy = offsets[tier]
                    # Keep coincident medium/high points legible without moving data.
                    if tier == "high" and cell["accuracy"] >= .76:
                        dx, dy = 0, -14
                    if tier == "high" and x < 3:
                        dx, dy = 24, -12
                    if tier == "low" and x < 3:
                        dx, dy = 8, -30
                    leader = {"arrowstyle": "-", "color": COLORS[tier], "linewidth": .7} if tier == "low" and x < 3 else None
                    axis.annotate(f"{cell['correct']}/{cell['n']}", (x, y), xytext=(dx, dy),
                                  textcoords="offset points", ha="center", fontsize=8, color=COLORS[tier],
                                  arrowprops=leader)
        axis.set(xlabel="Mean image bitstream (kB, 1 kB = 1000 B)", ylabel="Exact-match accuracy (%)",
                 ylim=limits, xlim=(1.3, 8.7), xticks=[2, 4, 8])
        axis.grid(axis="y", alpha=.22)
        axis.text(.03, .05, label, transform=axis.transAxes, fontsize=9)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, .9))
    save_figure(figure, folder, "figure-01-accuracy-grid")

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    order = [(budget, tier) for budget in BUDGETS for tier in TIERS]
    labels = [f"{budget // 1000}k\n{tier[0].upper()}" for budget, tier in order]
    for axis, key, ylabel in ((axes[0], "receiver_seconds", "Receiver generate time (s)"),
                               (axes[1], "actual_visual_tokens", "Actual visual tokens")):
        values = [[row[key] for row in scored if row["budget"] == budget and row["tier"] == tier]
                  for budget, tier in order]
        boxes = axis.boxplot(values, patch_artist=True, widths=.58,
                             medianprops={"color": "#FFFFFF", "linewidth": 1.4},
                             flierprops={"marker": ".", "markersize": 3, "alpha": .35},
                             whiskerprops={"linewidth": .8}, capprops={"linewidth": .8})
        for box, (_, tier) in zip(boxes["boxes"], order):
            box.set(facecolor=COLORS[tier], edgecolor=COLORS[tier], alpha=.9)
        for divider in (3.5, 6.5):
            axis.axvline(divider, color="#CCCCCC", linewidth=.7)
        axis.set_xticks(range(1, len(labels) + 1))
        axis.set_xticklabels(labels)
        axis.set(xlabel="Image byte cap × visual tier (L / M / H)", ylabel=ylabel, ylim=(0, None))
        axis.grid(axis="y", alpha=.2)
    handles = [Line2D([], [], color=COLORS[tier], marker=MARKERS[tier], linestyle="none",
                      label=f"{tier.capitalize()} visual tier") for tier in TIERS]
    figure.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, .9))
    save_figure(figure, folder, "figure-02-latency-tokens")


def numeric_table(summary: Json) -> str:
    lines = ["| Byte cap | Tier | Correct / N | Accuracy | Mean image B | Mean symbols | Mean tokens | Median generate s | Latency Q1–Q3 s | Energy J |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for budget in BUDGETS:
        for tier in TIERS:
            metric = summary["cells"][f"{budget}_{tier}"]
            time = metric["receiver_seconds"]
            energy = metric["energy_j_all_rows_mean"]
            lines.append(f"| {budget} | {tier} | {metric['correct']}/{metric['n']} | {metric['accuracy']:.2%} | "
                         f"{metric['image_bytes']['mean']:.2f} | {metric['ldpc_complex_symbols']['mean']:.2f} | "
                         f"{metric['actual_visual_tokens']['mean']:.2f} | {time['median']:.6f} | "
                         f"{time['q25']:.6f}–{time['q75']:.6f} | {'unavailable' if energy is None else f'{energy:.6f}'} |")
    return "\n".join(lines)


def write_reports(summary: Json, decision: Json, provenance: Json, folder: Path) -> None:
    n = summary["audit"]["n_questions"]
    primary = summary["cells"]["4000_high"]
    medium = summary["cells"]["4000_medium"]
    saving = 1 - medium["receiver_seconds"]["median"] / primary["receiver_seconds"]["median"]
    token_saving = 1 - medium["actual_visual_tokens"]["mean"] / primary["actual_visual_tokens"]["mean"]
    oracle = summary["oracle_upper_bounds"][str(summary["primary_lambda"])]
    table = numeric_table(summary)
    (folder / "numeric-summary.md").write_text("# Exact aggregate summary\n\n" + table + "\n", encoding="utf-8")
    report = f"""# RGB rate × receiver visual-budget: development analysis

## Question and comparison unit

Can independently varying actual image bitstream size and receiver visual preprocessing uncover an accuracy–communication–compute tradeoff worth learning? This is a completed, paired **{n}-question / {summary['audit']['n_images']}-image development screen**, not an independent test. Each image/question has nine evaluations. The frozen receiver and codec are unchanged; no question gating or internal token pruning is introduced.

Primary reference: 4,000-byte image cap, high visual tier. The byte caps are 2,000/4,000/8,000 B, not nested packet prefixes. Visual tiers are verified by actual token counts rather than nominal pixel settings. Accuracy is project-normalized exact match, not VQA soft accuracy.

## Main findings

The 4,000-byte medium tier scores {medium['correct']}/{n}, versus {primary['correct']}/{n} for the high tier, while mean visual-token count falls {token_saving:.2%} and median generation latency falls {saving:.2%}. Equal total correct counts do not imply identical per-question predictions. The measured latency reduction is not an energy measurement and does not establish accuracy equivalence on unseen data.

The preregistered efficient-fixed screen requires at most one lost answer, no increase in mean image bytes, and at least 20% median-latency reduction. Its outcome is **{decision['efficient_fixed_candidate']}**. The reported 4k/medium improvement must not be relabeled as satisfying the 20% threshold.

At λ={summary['primary_lambda']}, the answer-informed joint oracle reaches {oracle['joint_oracle']['correct']}/{n}; the strongest rate-only oracle reaches {oracle['rate_only_oracle_best_fixed_tier']['correct']}/{n}, and the strongest compute-only oracle reaches {oracle['compute_only_oracle_best_fixed_rate']['correct']}/{n}. Joint routing-headroom screen: **{decision['potential_routing_headroom']}**. These are optimistic upper bounds using known answers; no trained router achieves these figures yet.

## Exact numeric summary

{table}

## Figures and interpretation

![Accuracy grid](figures/figure-01-accuracy-grid.png)

Figure 1 tests whether more bytes or visual tokens reliably improve accuracy. Notice the full-scale panel and the explicitly labeled zoom, not only the small plotted differences. Medium at 4k matches high at 4k; larger byte or visual budgets need not improve exact-match answers. This supports investigating resource selection, but does not demonstrate a generalizable learned selector. Point estimates have no run-to-run error bars because only one frozen-model inference run exists; descriptive paired resampling appears in the appendix.

![Latency and actual tokens](figures/figure-02-latency-tokens.png)

Figure 2 checks that the visual-budget intervention changes actual processing rather than only a configuration flag. Each box summarizes {n} individual question/image evaluations, not {n} training runs; medians, quartiles, 1.5-IQR whiskers, and outlying observations are shown. The token distributions separate strongly while latency savings are more modest. Therefore token reduction must not be reported as proportional energy reduction, and a latency screen should use measured generation time.

## Decision and limitations

Frozen screen recommendation: `{decision['recommendation']}`. Any next controller must use deployable features, be trained without sealed-test answers, and be frozen before wireless and independent-test evaluation. The oracle is not a policy eligible for deployment.

All results are from a reused development set and a single checkpoint/runtime. Generation timing includes vision and language generation, but excludes image decoding and preprocessing; variability also reflects different answers and generated lengths. Aggregate timing is not complete end-to-end latency. Question/downlink communication is excluded by scope. LDPC complex-symbol counts are analytical payload accounting, not a measured channel-delivery curve. Energy is unavailable when telemetry is null. No statistical significance, low-SNR robustness, energy saving, or independent-test generalization is claimed.
"""
    (folder / "analysis-report.md").write_text(report, encoding="utf-8")
    appendix = ["# Statistical appendix", "", "## Design and estimands", "",
                f"{n} questions, {summary['audit']['n_images']} image clusters, nine paired configurations, one inference run. Accuracy differences are candidate minus reference. Resources are actual image bytes, actual visual tokens, and measured generate latency. No independent training-seed variation is available.",
                "", "No t-test, normality test, or confirmatory p-value is appropriate for the present reused-development/model-selection screen. Reported 95% percentile ranges use the existing paired image-cluster bootstrap, 2,000 resamples, seed 20260922. Selection is held fixed: the intervals omit policy/configuration-selection uncertainty. They are exploratory descriptive resampling intervals, unadjusted for multiple contrasts, and are not simultaneous or generalization guarantees.",
                "", "## Paired answer gains and losses at the primary utility weight", "",
                "| Comparison | Gained | Lost | Net correct | Accuracy difference pp | Descriptive 95% range pp | Utility difference | Descriptive 95% utility range |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, comparison in summary["paired_comparisons_primary_lambda"].items():
        intervals = comparison["paired_image_bootstrap"]
        lo, hi = intervals["accuracy_difference_percentile_95"]
        ulo, uhi = intervals["utility_difference_percentile_95"]
        appendix.append(f"| {name} | {comparison['gained_correct']} | {comparison['lost_correct']} | "
                        f"{comparison['net_correct']} | {100 * comparison['accuracy_difference']:.3f} | "
                        f"[{100 * lo:.3f}, {100 * hi:.3f}] | {comparison['utility_difference']:.6f} | [{ulo:.6f}, {uhi:.6f}] |")
    appendix.extend(["", "## Fixed configuration descriptives", "", table, "", "## Sensitivity across the frozen λ grid", "",
                     "| λ | Best fixed | Joint oracle correct | Strongest rate-only correct | Strongest compute-only correct | Joint mean utility |",
                     "|---:|---|---:|---:|---:|---:|"])
    for values in summary["oracle_upper_bounds"].values():
        appendix.append(f"| {values['weight']} | {values['best_fixed_cell']} | {values['joint_oracle']['correct']}/{n} | "
                        f"{values['rate_only_oracle_best_fixed_tier']['correct']}/{n} | "
                        f"{values['compute_only_oracle_best_fixed_rate']['correct']}/{n} | {values['joint_oracle']['utility_mean']:.6f} |")
    appendix.extend(["", "Utility is correctness − λ × (actual bytes / 8000 + actual visual tokens / same-image high-tier tokens). Accuracy/cost λ tradeoffs are descriptive; no λ is selected from sealed-test outcomes.",
                     "", "## Missingness and provenance", "",
                     "Missing energy is never converted to zero. The aggregate mean energy is unavailable if any required observation is missing. No per-question answers, IDs, images, or prompts are included in this bundle.", ""])
    for cell, values in summary["cells"].items():
        appendix.append(f"- {cell}: energy missing {values['energy_missing_count']}/{values['n']}.")
    appendix.extend(["", "Input SHA-256 hashes (filenames only; no private host paths):", ""])
    appendix.extend(f"- `{name}`: `{value}`" for name, value in provenance.items())
    (folder / "stats-appendix.md").write_text("\n".join(appendix) + "\n", encoding="utf-8")
    catalog = f"""# Figure catalog

## Figure 1 — accuracy grid

- Files: `figures/figure-01-accuracy-grid.pdf` and `.png` (600 dpi).
- Source: validated aggregate counts in `summary.json`; {n} paired development questions in every cell.
- Purpose: inspect whether image bytes and receiver visual budget interact sufficiently to motivate selection.
- Caption: Exact-match accuracy against actual mean image-bitstream kB for three visual tiers. Left uses 0–100%; right explicitly zooms to 65–85%. Markers and counts represent one frozen-model run, not repeated seeds. Orange circles/dashes denote low; blue squares/solid denotes medium; black triangles/dots denote high. No smoothing or significance marks; no fabricated error bars.
- Observation: 4k/medium and 4k/high each obtain {medium['correct']}/{n}; higher communication/visual budgets do not uniformly improve accuracy.
- Implication: investigate selection headroom while retaining fixed baselines; the figure itself does not validate a router.
- Checklist: verify all nine counts against the numeric table; read full-scale and zoom jointly; do not call equal counts equivalent accuracy; do not interpret connected lines as unseen-budget interpolation evidence.

## Figure 2 — observed generation latency and visual tokens

- Files: `figures/figure-02-latency-tokens.pdf` and `.png` (600 dpi).
- Source: `scored.json`, grouped by the same nine cells; {n} measurements per box.
- Purpose: verify actual visual-budget intervention and compare its token effect with measured latency.
- Caption: Per-question generation latency and actual visual-token distributions. Boxes show median and interquartile range; whiskers extend to 1.5 IQR; points outside are retained. Groups use the same tier colors as Figure 1. The {n} observations are different questions/images, not independent retraining runs; no bootstrap uncertainty is encoded in boxes.
- Observation: 4k/medium mean tokens decrease {token_saving:.2%} relative to 4k/high, while median measured latency decreases {saving:.2%}.
- Implication: budget knobs work, but measured speed and tokens are not interchangeable with energy or end-to-end costs. The 20% fixed-efficiency threshold is not met by this pair.
- Checklist: verify unchanged image bytes within each rate; distinguish input variability from random-seed uncertainty; retain outliers; report timing scope and missing energy; do not infer linear latency scaling with tokens.
"""
    (folder / "figure-catalog.md").write_text(catalog, encoding="utf-8")
    (folder / "bundle-provenance.json").write_text(json.dumps({"input_sha256": provenance,
        "scope": "development_descriptive", "contains_private_question_content": False,
        "figures": ["figure-01-accuracy-grid", "figure-02-latency-tokens"]}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, decision, scored, provenance = load_inputs(args.analysis)
    figure_folder = args.output / "figures"
    figure_folder.mkdir(parents=True, exist_ok=True)
    figures(summary, scored, figure_folder)
    write_reports(summary, decision, provenance, args.output)
    print(json.dumps({"output": str(args.output), "questions": summary["audit"]["n_questions"],
                      "figures": 2, "scope": "development_descriptive"}))


if __name__ == "__main__":
    main()
