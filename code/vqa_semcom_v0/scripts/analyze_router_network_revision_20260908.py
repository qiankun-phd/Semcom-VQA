#!/usr/bin/env python3
"""Read-only prediction QA plus descriptive/exploratory analysis in a new folder."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import zipfile
import joblib
import numpy as np
import router_network_revision_20260908 as run


def cluster_contrast(difference: np.ndarray, images: np.ndarray, rng: np.random.Generator) -> dict:
    unique = np.unique(images)
    sums = np.array([difference[images == i].sum() for i in unique])
    sizes = np.array([(images == i).sum() for i in unique])
    draws = rng.integers(0, len(unique), size=(10000, len(unique)))
    boot = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    signs = rng.choice([-1., 1.], size=(10000, len(unique)))
    perm = signs @ sums / sizes.sum()
    effect = float(difference.mean())
    return {"images": len(unique), "decision_n": len(difference), "effect_pp": effect * 100,
            "cluster_bootstrap_95_percentile_ci_pp": (np.quantile(boot, [.025, .975]) * 100).tolist(),
            "two_sided_cluster_sign_flip_p": float((1 + (np.abs(perm) >= abs(effect) - 1e-15).sum()) / 10001),
            "replications": 10000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert run.read(args.run / "status.json")["state"] == "COMPLETE"
    args.out.mkdir(parents=True, exist_ok=False)
    selection = run.read(args.run / "validation_selection.json")
    selection_hash = run.sha(args.run / "validation_selection.json")
    assert selection["protocol_sha256"] == run.sha(args.run / "protocol.json")
    assert run.read(args.run / "protocol.json")["script_sha256"] == run.sha(Path(run.__file__))
    keys = [k for k in run.read(args.source / "common_keys.json") if k["split"] == "test"]
    images = np.array([k["image"] for k in keys])
    summary = run.read(args.run / "summary.json")
    checked, baseline_reproduced, contrasts, rows, cost = 0, 0, {}, {}, {}
    provenance = {}
    rng = np.random.default_rng(20260908)
    for receiver in run.RECEIVERS:
        path = args.source / receiver / "evaluation_inputs.npz"
        provenance[str(path)] = run.sha(path)
        with np.load(path) as z:
            data = {k: z[k].copy() for k in ("train_x", "train_y", "validation_x", "validation_y", "test_x", "test_y", "linear")}
        screened = {}
        for name, cfg in run.CONFIGS.items():
            screened[name] = []
            for seed in range(3):
                dest = args.run / "screen" / receiver / name / f"seed_{seed}"
                record = run.read(dest / "record.json")
                models = joblib.load(dest / "models.joblib")
                with np.load(dest / "validation_outcomes.npz") as z:
                    score = run.route_score(models, cfg, data["validation_x"])
                    np.testing.assert_array_equal(score, z["score"])
                    np.testing.assert_array_equal(score > 0, z["pick"])
                    assert run.accuracy(data["validation_y"], score) == record["validation_accuracy"]
                if name == "original_bce":
                    oldpath = args.source / receiver / f"seed_{seed}_outcomes.npz"
                    with np.load(oldpath) as z:
                        p = z["validation_probabilities"]
                        np.testing.assert_array_equal(p[:, 1] - p[:, 0], score)
                    baseline_reproduced += 1
                screened[name].append(record)
                checked += 1
        assert run.select(screened) == selection["selected"][receiver]
        baseline = run.read(args.source / receiver / "baselines.json")
        new_correct, old_correct, allrecords = [], [], []
        config = run.CONFIGS[selection["selected"][receiver]]
        for seed in range(10):
            dest = args.run / "final" / receiver / f"seed_{seed}"
            record = run.read(dest / "record.json")
            assert record["selection_sha256"] == selection_hash
            assert (dest / "record.json").stat().st_mtime >= (args.run / "validation_selection.json").stat().st_mtime
            models = joblib.load(dest / "models.joblib")
            score = run.route_score(models, config, data["test_x"])
            with np.load(dest / "test_outcomes.npz") as z:
                np.testing.assert_array_equal(score, z["score"])
                np.testing.assert_array_equal(score > 0, z["pick"])
                np.testing.assert_array_equal(data["test_y"], z["y"])
            assert run.test_summary(keys, data["test_y"], score) == record["test"]
            assert all(0 < h["epochs"] <= 300 and len(h["loss_history"]) == h["epochs"] for h in record["training"])
            new_correct.append(data["test_y"][np.arange(len(keys)), (score > 0).astype(int)])
            oldpath = args.source / receiver / f"seed_{seed}_outcomes.npz"
            provenance[str(oldpath)] = run.sha(oldpath)
            with np.load(oldpath) as z:
                old_correct.append(data["test_y"][np.arange(len(keys)), z["pick"]])
            allrecords.append(record)
            checked += 1
        new, old = np.array(new_correct).mean(axis=0), np.array(old_correct).mean(axis=0)
        linear = data["test_y"][np.arange(len(keys)), data["linear"]]
        np.testing.assert_allclose(np.mean(new), summary[receiver]["accuracy_mean"], atol=1e-15)
        for label, reference in (("original_mlp", old), ("linear", linear)):
            contrasts[f"{receiver}_vs_{label}"] = cluster_contrast(new - reference, images, rng)
        rows[receiver] = {"selected": summary[receiver], "per_type": {}}
        for qt in run.QTYPES:
            mask = np.array([k["qt"] == qt for k in keys])
            vals = np.array(new_correct)[:, mask].mean(axis=1)
            rows[receiver]["per_type"][qt] = {"n": int(mask.sum()), "selected_mean": float(vals.mean()),
                "selected_sample_sd": float(vals.std(ddof=1)), "original_mean": float(old[mask].mean()),
                "linear": float(linear[mask].mean())}
        cost[receiver] = {}
        for name, records in (("original_bce_screen", screened["original_bce"]), ("selected_final", allrecords)):
            cost[receiver][name] = {"n_runs": len(records), "parameters": records[0]["parameter_count"],
                "training_seconds_mean": float(np.mean([r["training_seconds"] for r in records])),
                "single_query_median_us_mean": float(np.mean([r["inference"]["single_query_median_us"] for r in records])),
                "batch_us_per_query_mean": float(np.mean([r["inference"]["batch_us_per_query"] for r in records]))}
    ordered = sorted(contrasts, key=lambda key: contrasts[key]["two_sided_cluster_sign_flip_p"])
    running = 0.
    for rank, key in enumerate(ordered):
        running = max(running, min(1., (len(ordered) - rank) * contrasts[key]["two_sided_cluster_sign_flip_p"]))
        contrasts[key]["holm_adjusted_p_six_contrasts"] = running
    for path in args.run.rglob("*.npz"):
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
        with np.load(path) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
    run.dump(args.out / "qa.json", {"verified_models": checked, "original_validation_exact_reproduction": baseline_reproduced,
              "selection_sha256": selection_hash, "all_npz_crc_finite": True, "protocol_frozen": True})
    run.dump(args.out / "comparisons.json", rows)
    run.dump(args.out / "cost_comparison.json", cost)
    run.dump(args.out / "exploratory_stats.json", contrasts)
    run.dump(args.out / "additional_source_sha256.json", provenance)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    for method, offset, color, marker in (("linear", -.18, "#CC79A7", "s"), ("original", 0, "#0072B2", "o"), ("selected", .18, "#D55E00", "D")):
        mean, sd = [], []
        for receiver in run.RECEIVERS:
            s = summary[receiver]
            mean.append(100 * s[{"linear": "linear_accuracy", "original": "original_mlp_mean", "selected": "accuracy_mean"}[method]])
            sd.append(0 if method == "linear" else 100 * s["original_mlp_sample_sd" if method == "original" else "accuracy_sample_sd"])
        ax.errorbar(np.arange(3) + offset, mean, yerr=sd, fmt=marker, linestyle="none", color=color, capsize=3, label=method)
    ax.set_xticks(range(3), ["Qwen2-VL", "Qwen2.5-VL", "SmolVLM"])
    ax.set_ylabel("Test answer accuracy (%)")
    ax.set_ylim(67.5, 71)
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(args.out / f"test_comparison.{ext}", dpi=600)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    for i, (name, color, marker) in enumerate(zip(run.CONFIGS, ["#0072B2", "#E69F00", "#009E73", "#CC79A7"], ["o", "s", "D", "^"])):
        values = [selection["validation_candidates"][r][name] for r in run.RECEIVERS]
        ax.errorbar(np.arange(3) + (i - 1.5) * .14, [v["mean"] * 100 for v in values],
                    yerr=[v["sample_sd"] * 100 for v in values], fmt=marker, linestyle="none", color=color, capsize=3, label=name)
    ax.set_xticks(range(3), ["Qwen2-VL", "Qwen2.5-VL", "SmolVLM"])
    ax.set_ylabel("Screening validation accuracy (%)")
    ax.set_ylim(71.5, 77)
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(args.out / f"all_candidate_validation.{ext}", dpi=600)
    plt.close(fig)
    lines = ["# Router network experiment analysis", "", "Existing matched VisDrone/Rician benchmark; lambda=0; 104 test images and 2808 decisions. Configurations selected only on validation. Prior test exposure means this is follow-up development, not pristine holdout confirmation.", "", "## Complete final comparison", "", "Accuracy in percent; ± is sample SD across all ten training seeds.", "", "| Receiver | Selected configuration | Selected | Original MLP | Linear | Delta vs original (pp) |", "|---|---|---:|---:|---:|---:|"]
    for receiver, s in summary.items():
        lines.append(f"| {receiver} | {s['selected_config']} | {s['accuracy_mean']*100:.4f} ± {s['accuracy_sample_sd']*100:.4f} | {s['original_mlp_mean']*100:.4f} ± {s['original_mlp_sample_sd']*100:.4f} | {s['linear_accuracy']*100:.4f} | {(s['accuracy_mean']-s['original_mlp_mean'])*100:+.4f} |")
    lines += ["", "Observation: validation-chosen configurations show small mean gains on all three receivers, not a large jump in overall accuracy. The larger BCE wins validation on Qwen2; advantage regression wins on the other receivers. Disagreement-only classification did not win any receiver and all its validation results remain available.", "", "No test-guided follow-up changes were made. Per-type and every-seed values are retained in comparisons.json and final/*/record.json. Linear baselines are not removed. CPU-only training and model-call timings are in cost_comparison.json; larger networks cost more and the single-head regression has fewer parameters than the original two-head pair. These timings exclude answering, transmission, detector work, and hardware power.", "", "## Scope and decision", "", "The findings justify discussing network/target choice as a modest routing refinement, not a general solution to the main accuracy limit. Small improvements on this already-studied benchmark are insufficient by themselves to require a DV rerun or replace the manuscript. Further independent dataset evaluation would be a separate authorization and should retain this frozen selection. Avoid claiming statistical superiority from seed SD or the exploratory appendix.", "", "## Validation screening", "", "| Receiver | Candidate | Validation mean ± sample SD (%) |", "|---|---|---:|"]
    for receiver, candidates in selection["validation_candidates"].items():
        for name, values in candidates.items():
            lines.append(f"| {receiver} | {name} | {values['mean']*100:.4f} ± {values['sample_sd']*100:.4f} |")
    (args.out / "analysis-report.md").write_text("\n".join(lines) + "\n")
    stats = ["# Exploratory statistical appendix", "", "These are post-selection descriptive sensitivity calculations, not independent confirmatory tests. Average each method's ten seed outcomes per decision before comparing; seeds are not independent datasets. Resample 104 image clusters with replacement 10,000 times (RNG20260908), retaining every question/SNR within an image; use ratio of total correct-difference sums to total decisions, preserving pooled weighting. Report percentile 95% CIs. Scene dependence beyond image remains unmodeled.", "", "Two-sided cluster sign-flip randomization uses 10,000 sign patterns and the pooled difference as statistic; assumes independent image clusters and symmetry/exchangeability of paired cluster differences under the null. No normality-based t test is used. Holm adjustment covers all six receiver×baseline contrasts. Neither interval nor p-value adjusts for prior development/test exposure or repeated historical comparisons. Interpret cautiously and do not assert universal gains.", "", "| Contrast | Effect (pp) | Image-bootstrap 95% CI (pp) | Sign-flip p | Holm p |", "|---|---:|---:|---:|---:|"]
    for name, v in contrasts.items():
        ci = v["cluster_bootstrap_95_percentile_ci_pp"]
        stats.append(f"| {name} | {v['effect_pp']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {v['two_sided_cluster_sign_flip_p']:.5f} | {v['holm_adjusted_p_six_contrasts']:.5f} |")
    (args.out / "stats-appendix.md").write_text("\n".join(stats) + "\n")
    (args.out / "figure-catalog.md").write_text("# Figure catalog\n\n## test_comparison\n\nPurpose: compare selected routers against retained original MLP and linear baseline. Points are pooled six-SNR test accuracy on 2808 matched decisions; errors are ten-seed sample SD (linear deterministic). Deliberately zoomed percentage axis, dot plot rather than truncated bars. Observation: gains are small and seed distributions overlap. Implication: network refinement is modest, not a large universal advantage. No significance stars or prior-exposure-adjusted inference.\n\n## all_candidate_validation\n\nPurpose: disclose all four candidate configurations, including non-winners. Points/errors are mean/sample SD of three screening seeds on 2646 validation decisions. All receivers use their validation-selected configuration, before test loading. Observation: preferred objective varies by receiver and margins are small. Implication: no single larger architecture dominates; do not drop unsuccessful candidates or use the plot as independent test evidence.\n")
    print(json.dumps({"qa_models": checked, "baseline_reproduction": baseline_reproduced, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
