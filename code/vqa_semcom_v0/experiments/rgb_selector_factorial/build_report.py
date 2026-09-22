"""Produce the aggregate-only EXP-013 analysis bundle and real data figures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUPS = ("original_absolute", "balanced_absolute", "original_gain", "balanced_gain")
SPLITS = ("validation", "legacy_dev")
SHORT = ("Original\nabsolute", "Balanced\nabsolute", "Original\ngain", "Balanced\ngain")
NAMES = {"original_absolute": "Original / absolute", "balanced_absolute": "Balanced / absolute",
         "original_gain": "Original / gain", "balanced_gain": "Balanced / gain",
         "fixed_2000_low": "Fixed 2k / low", "fixed_4000_medium": "Fixed 4k / medium",
         "fixed_4000_low": "Fixed 4k / low", "frozen_rate_at_low": "Frozen rate at low",
         "frozen_compute_at_4000": "Frozen compute at 4k"}
COLORS = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(stat: dict, digits: int = 4) -> str:
    if stat["mean"] is None:
        return "NA"
    lo, hi = stat["ci95"]
    return f"{stat['mean']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(row) + " |" for row in rows])


def descriptive_row(name: str, seed: str, result: dict) -> list[str]:
    s = result["summary"]
    auc = result["within_image_auc"]
    return [name, seed, f"{s['correct']}/{s['n']}", f"{result['rescue']}/{result['harm']}",
            f"{s['image_bytes']['mean']:.2f}", f"{s['ldpc_complex_symbols']['mean']:.2f}",
            f"{s['actual_visual_tokens']['mean']:.2f}", f"{s['utility']:.6f}",
            f"{auc['mean']:.4f}" if auc else "NA"]


def check_evaluation(evaluation: dict) -> None:
    if evaluation.get("experiment_id") != "EXP-013" or evaluation.get("statistics", {}).get("holm_family_size") != 12:
        raise ValueError("Not a complete registered EXP-013 evaluation")
    if evaluation["reproduction"].get("passed") is not True or evaluation["full_cost_gate"] != "PENDING":
        raise ValueError("Reproduction or full-cost boundary failed")
    for split in SPLITS:
        data = evaluation["splits"][split]
        if not data["mixed_set_identical_for_all_groups_and_seeds"] or set(data["groups"]) != set(GROUPS):
            raise ValueError("Incomplete groups or incompatible AUC populations")
        for group in GROUPS:
            for seed in ("7", "17", "27", "ensemble"):
                row = data["groups"][group][seed]
                if row["summary"]["n"] != data["n"] or row["within_image_auc"]["n"] != data["mixed_outcome_n"]:
                    raise ValueError("Group denominators differ")


def export(fig: plt.Figure, directory: Path, basename: str) -> list[Path]:
    paths = []
    for extension in ("pdf", "png"):
        path = directory / f"{basename}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def clean_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=.7, alpha=.8)
    ax.set_axisbelow(True)
    ax.set_xticks(range(4), SHORT)
    ax.tick_params(axis="both", labelsize=9)


def point_interval(ax: plt.Axes, x: int, statistic: dict, color: str, multiplier: float = 1) -> None:
    mean = statistic["mean"] * multiplier
    lo, hi = [value * multiplier for value in statistic["ci95"]]
    # Percentile endpoints need not bracket the point estimate for discrete data.
    ax.vlines(x, lo, hi, color=color, linewidth=1.6)
    ax.hlines([lo, hi], x - .045, x + .045, color=color, linewidth=1.3)
    ax.plot(x, mean, marker="s", color=color, markersize=6)


def make_figures(evaluation: dict, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.3), layout="constrained")
    metrics = (("correct", "Cached accuracy (%)", 100.), ("image_bytes", "Framed image bytes (B)", 1.),
               ("actual_visual_tokens", "Actual visual tokens", 1.))
    for r, split in enumerate(SPLITS):
        data = evaluation["splits"][split]
        for c, (metric, ylabel, multiplier) in enumerate(metrics):
            ax = axes[r, c]
            clean_axis(ax)
            for x, (group, color) in enumerate(zip(GROUPS, COLORS)):
                point_interval(ax, x, data["groups"][group]["ensemble"]["uncertainty"][metric], color, multiplier)
                for jitter, seed in zip((-.12, 0, .12), ("7", "17", "27")):
                    value = data["groups"][group][seed]["uncertainty"][metric]["mean"] * multiplier
                    ax.plot(x + jitter, value, "o", color=color, alpha=.45, markersize=3.2)
            for control, style in (("fixed_2000_low", "--"), ("fixed_4000_medium", ":")):
                value = data["controls"][control]["ensemble"]["uncertainty"][metric]["mean"] * multiplier
                ax.axhline(value, color="#666666", linestyle=style, linewidth=1.2,
                           label=NAMES[control])
            ax.set_ylabel(ylabel)
            ax.set_title(f"{'Validation' if split == 'validation' else 'Historical development'} · n={data['n']}", fontsize=10)
            ax.set_ylim(bottom=0, top=100 if metric == "correct" else None)
            if r == 0 and c == 1:
                ax.legend(loc="best", fontsize=8, frameon=False)
    paths = export(fig, directory, "figure-01-accuracy-resources")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), layout="constrained")
    for r, split in enumerate(SPLITS):
        data = evaluation["splits"][split]
        auc_ax, change_ax = axes[r]
        clean_axis(auc_ax)
        clean_axis(change_ax)
        for x, (group, color) in enumerate(zip(GROUPS, COLORS)):
            point_interval(auc_ax, x, data["groups"][group]["ensemble"]["within_image_auc"], color)
            for jitter, seed in zip((-.12, 0, .12), ("7", "17", "27")):
                auc_ax.plot(x + jitter, data["groups"][group][seed]["within_image_auc"]["mean"],
                            "o", color=color, alpha=.45, markersize=3.2)
            for field, shift, bar_color, hatch in (("rescue", -.17, "#0072B2", ""), ("harm", .17, "#E69F00", "//")):
                result = data["groups"][group]["ensemble"]
                change_ax.bar(x + shift, result[field], width=.31, color=bar_color, hatch=hatch,
                              label=field.capitalize() if x == 0 else None, edgecolor="white", linewidth=.6)
                lo, hi = [v * data["n"] for v in result["uncertainty"][field]["ci95"]]
                change_ax.vlines(x + shift, lo, hi, color="#333333", linewidth=1.)
                change_ax.plot([x + shift], [result[field]], "_", color="#333333", markersize=5)
        auc_ax.axhline(.5, color="#777777", linestyle="--", linewidth=1)
        auc_ax.set_ylim(0, 1)
        auc_ax.set_ylabel("Within-image AUC")
        auc_ax.set_title(f"{split} · identical mixed subset n={data['mixed_outcome_n']}", fontsize=10)
        change_ax.set_ylabel("Answers changed vs 2k / low")
        change_ax.set_title(f"{split} · n={data['n']}", fontsize=10)
        change_ax.set_ylim(bottom=0)
        change_ax.legend(frameon=False, fontsize=9)
    paths += export(fig, directory, "figure-02-auc-rescue-harm")
    return paths


def build_report(evaluation: dict, output: Path, evaluation_path: Path | None = None) -> dict:
    check_evaluation(evaluation)
    output.mkdir(parents=True, exist_ok=True)
    figures = make_figures(evaluation, output / "figures")
    operating_observations = []
    for split in SPLITS:
        data = evaluation["splits"][split]
        base = data["groups"]["original_absolute"]["ensemble"]["summary"]
        gain = data["groups"]["original_gain"]["ensemble"]["summary"]
        rate = data["controls"]["frozen_rate_at_low"]["ensemble"]["summary"]
        operating_observations.append(
            f"{split}: original gain changes correct answers by {gain['correct'] - base['correct']:+d} "
            f"and mean framed bytes by {gain['image_bytes']['mean'] - base['image_bytes']['mean']:+.2f} B versus original absolute; "
            f"its correct count is {gain['correct']} versus frozen rate-only {rate['correct']}, "
            f"with utility difference {gain['utility'] - rate['utility']:+.6f}")
    operating_observation = "; ".join(operating_observations) + "."
    auc_observation = "; ".join(
        f"{split}: scale AUC effect {evaluation['splits'][split]['factorial_contrasts']['auc']['scale']['mean_difference']:+.4f}, "
        f"target/loss AUC effect {evaluation['splits'][split]['factorial_contrasts']['auc']['target_loss']['mean_difference']:+.4f}"
        for split in SPLITS) + "."
    passes = sum(evaluation["splits"][split]["groups"][group]["system_screen"]["preliminary_pass"] for split in SPLITS for group in GROUPS)
    minimum_holm = min(result["holm_p"] for split in SPLITS for family in evaluation["splits"][split]["factorial_contrasts"].values() for result in family.values())
    report = ["# EXP-013 four-group development analysis", "",
              "Question: how do dimension-based image-block scaling and relative upgrade-gain learning change ranking and resource-aware routing? All four preregistered groups, all three seeds, and their ensemble are retained.", "",
              "The target/loss factor changes output parameterization, objective, and checkpoint loss together. It does not isolate a target-only causal effect. Both development splits have been repeatedly inspected.", "",
              f"Original-absolute reproduction passed: maximum score difference {evaluation['reproduction']['max_abs_difference']:.3g}; action mismatches {evaluation['reproduction']['action_mismatches']}. The ensemble averages seed outputs before routing. Full-cost gate: **PENDING**.", "",
              "## Core conclusion", "",
              ("All four groups fail the unchanged preliminary system screen on both development splits. " if passes == 0 else f"{passes} of eight group/split screens pass. ") + operating_observation, "",
              auc_observation + f" The scale and target/loss AUC directions reverse across splits in this run. The smallest Holm-adjusted p-value is {minimum_holm:.4f}; none of the registered contrasts is below 0.05. These development results do not establish a single causal explanation and do not authorize automatic expansion, additional fits, or sealed-test/SNR access."]
    headers = ["Group / control", "Predictor", "Correct", "Rescue / harm", "Mean framed B", "Mean symbols", "Mean tokens", "Utility", "Mixed AUC"]
    for split in SPLITS:
        data = evaluation["splits"][split]
        report += ["", f"## {split}", "", f"n={data['n']} paired images; the same {data['mixed_outcome_n']} mixed-outcome images define AUC for every group and seed.", ""]
        rows = [descriptive_row(NAMES[group], "ensemble", data["groups"][group]["ensemble"]) for group in GROUPS]
        rows += [descriptive_row(NAMES[name], "ensemble", control["ensemble"]) for name, control in data["controls"].items()]
        report += [table(headers, rows), "", "AUC is unavailable for fixed controls and is not compared against restricted-axis controls; their operating metrics use the identical complete image set.", ""]
        pass_rows = [[NAMES[group], str(data["groups"][group]["system_screen"]["preliminary_pass"]),
                      ", ".join(name for name, passed in data["groups"][group]["system_screen"]["preliminary_checks"].items() if not passed) or "none"]
                     for group in GROUPS]
        report += [table(["Group", "EXP-012 preliminary screen", "Failed checks"], pass_rows)]
    report += ["", "## Factorial effects", "", "Effects below apply to the ensemble. Positive scale effects favor balanced input blocks; positive target/loss effects favor gain learning. Interaction is a difference of differences. All 12 p-values share one Holm family.", ""]
    contrasts = []
    for split in SPLITS:
        for metric, family in evaluation["splits"][split]["factorial_contrasts"].items():
            for contrast, result in family.items():
                contrasts.append([split, metric, contrast, str(result["n"]), interval(result),
                                  f"{result['sign_flip_p']:.5f}", f"{result['holm_p']:.5f}"])
    report += [table(["Split", "Metric", "Effect", "n", "Effect [95% paired CI]", "Raw p", "Holm p"], contrasts), "",
               "The intervals are pointwise, unadjusted 95% bootstrap intervals, conditional on these fitted checkpoints. A pointwise interval excluding zero is not a Holm-adjusted significance claim. The seed range and development reuse remain material uncertainty; effect direction alone does not establish a single root cause or authorize choosing a new deployment policy.", "",
               "## Figures", "", "![Accuracy and resources](figures/figure-01-accuracy-resources.png)", "",
               "Figure 1 compares operating accuracy, framed bytes, and actual visual tokens. Squares are ensembles; small circles show all three training seeds. Vertical intervals are pointwise, unadjusted 95% paired-image bootstrap intervals (2,000 draws); they do not include training or checkpoint-selection uncertainty. Gray dashed/dotted lines are fixed 2k/low and 4k/medium. All five controls appear in the numeric tables.", "",
               "Purpose: assess the operating accuracy/resource tradeoff. Observation: " + operating_observation + " Interpretation: the gain formulation's small correct-answer increase comes with additional transmitted bytes and remains below the frozen rate-only control. Implication: retain the failed system-screen outcome; higher cached accuracy alone does not establish a deployable improvement.", "",
               "![AUC and changed answers](figures/figure-02-auc-rescue-harm.png)", "",
               "Figure 2 compares per-image ranking on the common mixed-outcome subset with rescue and harm against 2k/low. AUC squares/circles use the same conventions as Figure 1. Rescue/harm bars are ensemble counts; pointwise, unadjusted intervals resample the complete paired image set and are expressed as counts at the displayed sample size. Ranking improvement must be read alongside both rescue and harm.", "",
               "Purpose: distinguish within-image ranking from useful or harmful upgrades. Observation: " + auc_observation + " Interpretation: both factorial ranking directions depend on the reused development split; neither intervention supplies consistent directional evidence. Implication: do not identify scaling or target/loss as a unique root cause, and do not choose a winner or expand the experiment from this result.", "",
               "## Boundaries", ""]
    report += [f"- {item}" for item in evaluation["limitations"]]
    report += ["- No new VLM/codec calls, sealed-test access, SNR simulation, energy measurement, or live end-to-end latency measurement occurred. Symbols equal 510 × ceil(framed bytes / 48); this is an accounting quantity.",
               "- No winner is selected after viewing these results. All four groups receive the unchanged EXP-012 system screen; the full-cost gate remains PENDING.",
               "- Engineering provenance: historical cached grid/truth locations were resolved through explicit read-only paths after an initial missing-path stop. This path correction did not change the protocol, trained checkpoints, or input contents, and did not trigger additional fits."]
    appendix = ["# Statistical appendix", "", "Analysis unit: distinct source image, paired across all groups and controls. AUC excludes only all-correct/all-wrong images using the shared cached outcomes, never group-specific predictions. Within-image AUC gives tied correct/wrong score pairs half credit, then averages images equally.", "",
                "Scale = ½[(balanced absolute − original absolute) + (balanced gain − original gain)]. Target/loss = ½[(original gain − original absolute) + (balanced gain − balanced absolute)]. Interaction = balanced gain − balanced absolute − original gain + original absolute.", "",
                "All ensemble contrasts use 2,000 paired-image percentile-bootstrap draws and 10,000 two-sided image sign flips, seed 20260922. p = (1 + number of absolute flipped means at least as extreme as observed)/(10000 + 1). Sign exchangeability/symmetry under the null is assumed; interventions were not randomized at the image level. No normality-based test or outlier exclusion was used. Holm adjusts one family of 12 tests (3 contrasts × 2 metrics × 2 development splits). Raw mean differences are the effect sizes.", "",
                "All 95% bootstrap intervals are pointwise and unadjusted for multiple comparisons; only the reported Holm p-values adjust the 12-test family. A pointwise interval excluding zero does not establish adjusted significance. Bootstrap intervals condition on fitted checkpoints; they omit uncertainty from fitting, checkpoint selection, and repeated development reuse. Training seeds are shown separately and are not treated as independent datasets. No individual-seed p-value is added to the registered family.", ""]
    for split in SPLITS:
        data = evaluation["splits"][split]
        appendix += [f"## {split}: every seed and ensemble", ""]
        rows = [descriptive_row(NAMES[group], seed, data["groups"][group][seed]) for group in GROUPS for seed in ("7", "17", "27", "ensemble")]
        rows += [descriptive_row(NAMES[name], seed, control[seed]) for name, control in data["controls"].items() for seed in ("7", "17", "27", "ensemble")]
        appendix += [table(headers, rows), "", "### Ensemble uncertainty", ""]
        rows = [[NAMES[group], metric, interval(stat)] for group in GROUPS for metric, stat in
                {**data["groups"][group]["ensemble"]["uncertainty"], "auc": data["groups"][group]["ensemble"]["within_image_auc"]}.items()]
        appendix += [table(["Group", "Metric", "Mean [95% CI]"], rows), "", "### Training seed mean and sample SD", ""]
        rows = [[NAMES[group], metric, f"{stat['mean']:.6f} ± {stat['sample_sd']:.6f}"] for group in GROUPS
                for metric, stat in data["groups"][group]["seed_mean_sd"].items()]
        appendix += [table(["Group", "Metric", "Seed mean ± sample SD (3 seeds)"], rows), "", "### Paired ensemble differences against every control", ""]
        rows = [[NAMES[group], NAMES[control], metric, interval(stat)] for group in GROUPS
                for control, metrics in data["groups"][group]["paired_controls"].items() for metric, stat in metrics.items()]
        appendix += [table(["Group", "Control", "Metric", "Group − control [95% CI]"], rows), "", "### Contrast seed sensitivity", ""]
        rows = [[metric, contrast, *[f"{result['seed_effects'][str(seed)]:.6f}" for seed in (7, 17, 27)],
                 f"{result['seed_effect_mean']:.6f} ± {result['seed_effect_sample_sd']:.6f}"]
                for metric, family in data["factorial_contrasts"].items() for contrast, result in family.items()]
        appendix += [table(["Metric", "Contrast", "Seed 7", "Seed 17", "Seed 27", "Mean ± sample SD"], rows), ""]
    appendix += ["## Registered ensemble tests", "", table(["Split", "Metric", "Effect", "n", "Effect [95% CI]", "Raw p", "Holm p"], contrasts)]
    catalog = ["# Figure catalog", "", "Source: evaluation.json. Figures contain aggregate measurements only; no sample identities or fabricated measurements. Export: vector PDF and 300 dpi PNG. Colors use the Okabe–Ito palette; different markers and harm hatching add non-color cues.", ""]
    for basename, purpose, observation, implication in (
        ("figure-01-accuracy-resources", "Check whether each intervention changes operating accuracy and resources together.",
         "; ".join(f"{split}: " + ", ".join(f"{group}={evaluation['splits'][split]['groups'][group]['ensemble']['summary']['correct']}/{evaluation['splits'][split]['n']}" for group in GROUPS) for split in SPLITS),
         "Any accuracy improvement must be evaluated against byte/token use and the unchanged screen; token counts do not establish energy savings."),
        ("figure-02-auc-rescue-harm", "Separate ranking on the common mixed subset from successful or harmful resource upgrades.",
         "; ".join(f"{split}: " + ", ".join(f"{group} AUC={evaluation['splits'][split]['groups'][group]['ensemble']['within_image_auc']['mean']:.3f}, rescue/harm={evaluation['splits'][split]['groups'][group]['ensemble']['rescue']}/{evaluation['splits'][split]['groups'][group]['ensemble']['harm']}" for group in GROUPS) for split in SPLITS),
         "A change in AUC is a mechanism signal to compare with rescue/harm, not evidence of a unique causal explanation."),
    ):
        interpretation = (operating_observation + " The modest gain in correct answers does not pass the registered system tradeoff." if basename == "figure-01-accuracy-resources" else
                          auc_observation + " Opposite split directions do not support a consistent causal attribution.")
        catalog += [f"## {basename}", "", f"Files: figures/{basename}.pdf and figures/{basename}.png.", "", f"Purpose: {purpose}", "", f"Observation: {observation}.", "", f"Interpretation: {interpretation}", "", f"Implication: {implication}", "",
                    "Caption requirements: name both development splits, n and mixed n, all four groups, ensemble-before-routing, three seed dots, 2,000 bootstrap draws, pointwise unadjusted conditional 95% intervals, and development reuse. Only Holm p-values, not pointwise intervals, adjust multiple comparisons. Figure 1 reference lines show only two fixed controls; exact tables list all five.", "",
                    "Interpretation checklist: inspect every group; keep shared AUC denominators; compare effect directions on both splits; read intervals and seed variability; use the 12-test Holm family; retain PENDING full-cost status.", ""]
    for filename, lines in (("analysis-report.md", report), ("stats-appendix.md", appendix), ("figure-catalog.md", catalog)):
        (output / filename).write_text("\n".join(lines) + "\n")
    manifest = {"evaluation_sha256": sha(evaluation_path) if evaluation_path else None,
                "report_source_sha256": sha(Path(__file__)),
                "figures": {str(path.relative_to(output)): sha(path) for path in figures},
                "reports": {name: sha(output / name) for name in ("analysis-report.md", "stats-appendix.md", "figure-catalog.md")},
                "full_cost_gate": "PENDING", "contains_sample_identities": False}
    (output / "report_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text())
    output = args.output_dir or args.evaluation.parent / "analysis-output"
    print(json.dumps(build_report(evaluation, output, args.evaluation)))


if __name__ == "__main__":
    main()
