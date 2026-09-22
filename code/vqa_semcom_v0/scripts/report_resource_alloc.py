#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import ensure_parent, load_config, resolve_path


def _bool(value: str) -> bool:
    return value.lower() in {"true", "1", "yes"}


def _mean(rows: list[dict[str, str]], key: str) -> float:
    if not rows or key not in rows[0]:
        return 0.0
    return sum(float(row[key]) for row in rows) / max(1, len(rows))


def _rate(rows: list[dict[str, str]], key: str, value: str | None = None) -> float:
    if value is None:
        return sum(float(_bool(row[key])) for row in rows) / max(1, len(rows))
    return sum(float(row[key] == value) for row in rows) / max(1, len(rows))


def _group(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["policy"], []).append(row)
    return grouped


def _task_level(rows: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["policy"], row["episode"], row["task_id"]), []).append(row)
    denom = max(1, len(grouped))
    return {
        "unique_task_success": sum(float(any(_bool(r["success"]) for r in task_rows)) for task_rows in grouped.values()) / denom,
        "completion": sum(float(any(_bool(r.get("task_completed", r["success"])) for r in task_rows)) for task_rows in grouped.values()) / denom,
        "attempts_per_task": sum(len(task_rows) for task_rows in grouped.values()) / denom,
    }


def _bar_svg(labels: list[str], values: list[float], title: str, path: Path) -> None:
    ensure_parent(path)
    width = 720
    height = 260
    margin = 48
    bar_w = max(12, (width - 2 * margin) // max(1, len(values)) - 12)
    max_v = max(1e-9, max(values, default=1.0))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin}" y="28" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{margin}" y2="{margin}" stroke="#333"/>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        x = margin + 12 + i * (bar_w + 12)
        h = int((height - 2 * margin - 20) * value / max_v)
        y = height - margin - h
        lines.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="#2f6f8f"/>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6}" text-anchor="middle" font-family="Arial" font-size="11">{value:.2f}</text>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - margin + 16}" text-anchor="middle" font-family="Arial" font-size="10">{label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--results", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--fig-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_path = resolve_path(args.results or cfg["paths"].get("resource_results_csv", "outputs/resource_alloc/v0_resource_results.csv"))
    out_md = resolve_path(args.out_md or cfg["paths"].get("resource_report_md", "outputs/resource_alloc/v0_resource_report.md"))
    fig_dir = resolve_path(args.fig_dir or cfg["paths"].get("resource_fig_dir", "outputs/resource_alloc/figures"))
    with results_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grouped = _group(rows)
    policies = sorted(grouped)

    ensure_parent(out_md)
    lines = ["# UAV-VQA Goal-oriented Resource Allocation Report", ""]
    lines.append("Batch mode is one-shot allocation. MDP mode is multi-slot rollout; use `unique task success` and `completion` when comparing MDP policies.")
    lines.append("")
    lines.append("## Policy Comparison")
    lines.append("")
    lines.append("| policy | attempt success | unique task success | completion | attempts/task | semantic utility | delay | energy | rate | SNR | SINR | quality sat. | deadline sat. | feasible | conflict |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for policy in policies:
        r = grouped[policy]
        task_level = _task_level(r)
        lines.append(
            f"| {policy} | {_rate(r, 'success'):.3f} | {task_level['unique_task_success']:.3f} | "
            f"{task_level['completion']:.3f} | {task_level['attempts_per_task']:.3f} | {_mean(r, 'semantic_utility'):.3f} | "
            f"{_mean(r, 'delay'):.3f} | {_mean(r, 'energy'):.3f} | {_mean(r, 'rate_mbps'):.3f} | {_mean(r, 'snr_db'):.3f} | {_mean(r, 'sinr_db'):.3f} | {_rate(r, 'quality_ok'):.3f} | "
            f"{_rate(r, 'deadline_ok'):.3f} | {_rate(r, 'feasibility_class', 'feasible'):.3f} | {_rate(r, 'airspace_conflict'):.3f} |"
        )

    lines.extend(["", "## Channel Consistency", ""])
    lines.append("| policy | effective bad | effective medium | effective good | impairment dB | dist m | elev deg | LoS prob. | path loss | interference | fading |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for policy in policies:
        r = grouped[policy]
        lines.append(
            f"| {policy} | {_rate(r, 'effective_channel_bin', 'bad'):.3f} | {_rate(r, 'effective_channel_bin', 'medium'):.3f} | "
            f"{_rate(r, 'effective_channel_bin', 'good'):.3f} | {_mean(r, 'channel_impairment_db'):.3f} | "
            f"{_mean(r, 'distance_3d_m'):.3f} | {_mean(r, 'elevation_deg'):.3f} | {_mean(r, 'los_probability'):.3f} | "
            f"{_mean(r, 'path_loss_db'):.3f} | {_mean(r, 'interference_dbm'):.3f} | {_mean(r, 'fading_db'):.3f} |"
        )

    lines.extend(["", "## Delay Breakdown", ""])
    lines.append("| policy | travel | sensing | upload | queue | model load | inference |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for policy in policies:
        r = grouped[policy]
        lines.append(
            f"| {policy} | {_mean(r, 'travel_delay'):.3f} | {_mean(r, 'sensing_delay'):.3f} | {_mean(r, 'upload_delay'):.3f} | "
            f"{_mean(r, 'queue_delay'):.3f} | {_mean(r, 'model_load_delay'):.3f} | {_mean(r, 'inference_delay'):.3f} |"
        )

    lines.extend(["", "## Semantic Cache, MEC, and UAV State", ""])
    lines.append("| policy | cache hit prob. | semantic cache hit | acc. gain | payload MB | semantic eff. | LUT missing | utility/latency | GPU mem ok | battery ok | battery remaining | UAV util. |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for policy in policies:
        r = grouped[policy]
        lines.append(
            f"| {policy} | {_mean(r, 'cache_hit_probability'):.3f} | {_rate(r, 'semantic_cache_hit'):.3f} | "
            f"{_mean(r, 'accuracy_gain'):.3f} | {_mean(r, 'payload_mb'):.3f} | {_mean(r, 'semantic_efficiency'):.3f} | "
            f"{_rate(r, 'lut_missing'):.3f} | {_mean(r, 'utility_per_latency'):.3f} | "
            f"{_rate(r, 'gpu_memory_ok'):.3f} | {_rate(r, 'battery_ok'):.3f} | {_mean(r, 'battery_remaining'):.3f} | {_mean(r, 'uav_utilization'):.3f} |"
        )

    lines.extend(["", "## Violation Breakdown", ""])
    lines.append("| policy | quality violation | deadline violation | resource violation | airspace conflict |")
    lines.append("|---|---:|---:|---:|---:|")
    for policy in policies:
        r = grouped[policy]
        lines.append(
            f"| {policy} | {1.0 - _rate(r, 'quality_ok'):.3f} | {1.0 - _rate(r, 'deadline_ok'):.3f} | "
            f"{1.0 - _rate(r, 'resource_ok'):.3f} | {_rate(r, 'airspace_conflict'):.3f} |"
        )

    lines.extend(["", "## Service Level Distribution", ""])
    lines.append("| policy | cache s=0 | tags/tokens s=1 | full image s=2 | ROI/crop s=3 |")
    lines.append("|---|---:|---:|---:|---:|")
    for policy in policies:
        r = grouped[policy]
        lines.append(
            f"| {policy} | {_rate(r, 'service_level', '0'):.3f} | {_rate(r, 'service_level', '1'):.3f} | "
            f"{_rate(r, 'service_level', '2'):.3f} | {_rate(r, 'service_level', '3'):.3f} |"
        )

    lines.extend(["", "## Representative Task Trace", ""])
    lines.append("| policy | task | attempt | sensing | s | acc | gain | payload MB | cache p | delay | upload | infer | channel | SNR | SINR | LoS | path loss | intf | GPU | mem ok | batt ok | LUT miss | feasible | success | completed |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for row in rows[: min(12, len(rows))]:
        lines.append(
            f"| {row['policy']} | {row['task_id']} | {int(float(row.get('attempt_index', 1)))} | {row['sensing_decision']} | {row['service_level']} | "
            f"{float(row['expected_accuracy']):.3f} | {float(row.get('accuracy_gain', 0.0)):.3f} | {float(row.get('payload_mb', 0.0)):.3f} | "
            f"{float(row.get('cache_hit_probability', 0.0)):.2f} | "
            f"{float(row['delay']):.3f} | {float(row.get('upload_delay', 0.0)):.3f} | "
            f"{float(row.get('inference_delay', 0.0)):.3f} | {row.get('effective_channel_bin', '')} | {float(row.get('snr_db', 0.0)):.2f} | "
            f"{float(row.get('sinr_db', row.get('snr_db', 0.0))):.2f} | {float(row.get('los_probability', 0.0)):.2f} | "
            f"{float(row.get('path_loss_db', 0.0)):.1f} | {float(row.get('interference_dbm', -120.0)):.1f} | {float(row.get('gpu_share', 0.0)):.2f} | "
            f"{int(_bool(row.get('gpu_memory_ok', 'true')))} | {int(_bool(row.get('battery_ok', 'true')))} | {int(_bool(row.get('lut_missing', 'false')))} | "
            f"{row['feasibility_class']} | {int(_bool(row['success']))} | {int(_bool(row.get('task_completed', row['success'])))} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _bar_svg(policies, [_task_level(grouped[p])["unique_task_success"] for p in policies], "Unique Task Success Rate", fig_dir / "success_rate.svg")
    _bar_svg(policies, [_mean(grouped[p], "average" if False else "delay") for p in policies], "Average Delay", fig_dir / "average_delay.svg")
    _bar_svg(policies, [_mean(grouped[p], "semantic_utility") for p in policies], "Semantic Utility", fig_dir / "semantic_utility.svg")
    _bar_svg(policies, [_mean(grouped[p], "upload_delay") for p in policies], "Average Upload Delay", fig_dir / "upload_delay.svg")
    _bar_svg(policies, [_mean(grouped[p], "inference_delay") for p in policies], "Average Inference Delay", fig_dir / "inference_delay.svg")
    print(f"report={out_md}")
    print(f"figures={fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
