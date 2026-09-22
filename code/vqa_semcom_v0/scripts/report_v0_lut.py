#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import ensure_parent, load_config, resolve_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _counter_table(title: str, counter: Counter[str]) -> list[str]:
    total = sum(counter.values()) or 1
    lines = [f"## {title}", "", "| value | count | ratio |", "|---|---:|---:|"]
    for key, value in sorted(counter.items()):
        lines.append(f"| {key} | {value} | {value / total:.3f} |")
    lines.append("")
    return lines


def _policy_table(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["## Simulation Results", "", "No simulation result file found.", ""]
    lines = [
        "## Simulation Results",
        "",
        "| policy | success | accuracy | delay | energy | quality violation | deadline violation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {float(row['task_success_rate']):.3f} | "
            f"{float(row['average_accuracy']):.3f} | {float(row['average_delay']):.3f} | "
            f"{float(row['average_energy']):.3f} | {float(row['quality_violation_rate']):.3f} | "
            f"{float(row['deadline_violation_rate']):.3f} |"
        )
    lines.append("")
    return lines


def _write_distribution(tasks: list[dict[str, str]], lut: list[dict[str, str]], sim: list[dict[str, str]], out_path: Path) -> None:
    ensure_parent(out_path)
    acc_values = [float(row["expected_accuracy"]) for row in lut]
    ci_values = [float(row["std_or_ci"]) for row in lut]
    lines = [
        "# V0.5 LUT Distribution Report",
        "",
        "This report treats the LUT as a task-conditioned semantic accuracy model, not as an image-quality score.",
        "",
        f"- task rows: `{len(tasks)}`",
        f"- LUT rows: `{len(lut)}`",
        f"- accuracy range: `{min(acc_values):.3f}` to `{max(acc_values):.3f}`",
        f"- accuracy mean: `{statistics.mean(acc_values):.3f}`",
        f"- nonzero CI cells: `{sum(1 for v in ci_values if v > 0)}/{len(ci_values)}`",
        "",
    ]
    for col in ["question_type", "view_quality_bin", "risk_level", "target_class"]:
        if tasks and col in tasks[0]:
            lines.extend(_counter_table(f"Task Distribution: {col}", Counter(row[col] for row in tasks)))
    for col in ["service_level", "channel_bin", "freshness_bin"]:
        if lut and col in lut[0]:
            lines.extend(_counter_table(f"LUT Coverage: {col}", Counter(row[col] for row in lut)))
    lines.extend(_policy_table(sim))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_accuracy_tables(lut: list[dict[str, str]], out_path: Path) -> None:
    ensure_parent(out_path)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in lut:
        key = (row["question_type"], row["service_level"], row["view_quality_bin"])
        grouped[key].append(float(row["expected_accuracy"]))
    lines = [
        "# V0.5 LUT Accuracy Tables",
        "",
        "Mean expected accuracy aggregated over channel, freshness, and risk-level cells.",
        "",
        "| question type | service level | poor view | medium view | good view |",
        "|---|---:|---:|---:|---:|",
    ]
    qtypes = sorted({key[0] for key in grouped})
    levels = sorted({key[1] for key in grouped}, key=int)
    for qtype in qtypes:
        for level in levels:
            cells = []
            for view in ["poor", "medium", "good"]:
                values = grouped.get((qtype, level, view), [])
                cells.append(f"{statistics.mean(values):.3f}" if values else "-")
            lines.append(f"| {qtype} | {level} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")
    lines.append("## Fresh + Good/Medium Sanity Check")
    lines.append("")
    lines.append("| question type | view | cache s=0 | light s=1 | full image s=2 | ROI/crop s=3 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    sanity: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in lut:
        if row["freshness_bin"] == "fresh" and row["channel_bin"] in {"medium", "good"} and row["view_quality_bin"] in {"medium", "good"}:
            sanity[(row["question_type"], row["view_quality_bin"], row["service_level"])].append(float(row["expected_accuracy"]))
    for qtype in qtypes:
        for view in ["medium", "good"]:
            vals = []
            for level in ["0", "1", "2", "3"]:
                data = sanity.get((qtype, view, level), [])
                vals.append(f"{statistics.mean(data):.3f}" if data else "-")
            lines.append(f"| {qtype} | {view} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_lut_heatmap_and_gaps(lut: list[dict[str, str]], out_path: Path) -> None:
    ensure_parent(out_path)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in lut:
        if row["freshness_bin"] == "fresh" and row["channel_bin"] in {"medium", "good"}:
            grouped[(row["question_type"], row["view_quality_bin"], row["service_level"])].append(float(row["expected_accuracy"]))
    qtypes = sorted({row["question_type"] for row in lut})
    views = ["poor", "medium", "good"]
    lines = [
        "# V0.5 LUT Heatmap and Service-Level Gaps",
        "",
        "Cells aggregate fresh-cache, medium/good-channel LUT accuracy. The gap columns quantify how much semantic evidence improves over cache reuse.",
        "",
        "| question type | view | s0 cache | s1 light | s2 full | s3 ROI | s1-s0 | s2-s1 | s3-s2 | best level |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for qtype in qtypes:
        for view in views:
            means: dict[str, float] = {}
            for level in ["0", "1", "2", "3"]:
                values = grouped.get((qtype, view, level), [])
                means[level] = statistics.mean(values) if values else 0.0
            best_level = max(["0", "1", "2", "3"], key=lambda level: means[level])
            lines.append(
                f"| {qtype} | {view} | {means['0']:.3f} | {means['1']:.3f} | {means['2']:.3f} | {means['3']:.3f} | "
                f"{means['1'] - means['0']:.3f} | {means['2'] - means['1']:.3f} | {means['3'] - means['2']:.3f} | {best_level} |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sparse_cells(lut: list[dict[str, str]], out_path: Path, sparse_threshold: int = 5) -> None:
    ensure_parent(out_path)
    fieldnames = list(lut[0].keys()) if lut else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in lut:
            if int(row["sample_count"]) < sparse_threshold:
                writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tasks = _read_csv(resolve_path(cfg["paths"]["tasks_csv"]))
    lut = _read_csv(resolve_path(cfg["paths"]["lut_csv"]))
    sim_path = resolve_path(cfg["paths"]["sim_results_csv"])
    sim = _read_csv(sim_path) if sim_path.exists() else []
    report_dir = resolve_path("outputs/reports")
    _write_distribution(tasks, lut, sim, report_dir / "v0_lut_distribution.md")
    _write_accuracy_tables(lut, report_dir / "v0_lut_accuracy_tables.md")
    _write_lut_heatmap_and_gaps(lut, report_dir / "v0_lut_heatmap_gaps.md")
    _write_sparse_cells(lut, report_dir / "v0_sparse_cells.csv")
    print(f"reports_dir={report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
