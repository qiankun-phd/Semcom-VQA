#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import ensure_parent, load_config, resolve_path
from vqa_semcom.data.visdrone import VISDRONE_CATEGORY


COLORS = ["red", "lime", "cyan", "yellow", "magenta", "orange", "deepskyblue", "white"]


def _read_tasks(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _annotation_counts(ann_dir: Path) -> list[tuple[int, str]]:
    counts = []
    for path in sorted(ann_dir.glob("*.txt")):
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split(",")
            if len(parts) >= 6 and parts[5].strip() != "0":
                count += 1
        counts.append((count, path.stem))
    return counts


def _select_examples(counts: list[tuple[int, str]], num_images: int) -> list[tuple[str, str]]:
    if not counts:
        return []
    sorted_counts = sorted(counts)
    picks = [
        ("low", sorted_counts[min(len(sorted_counts) - 1, max(0, len(sorted_counts) // 5))][1]),
        ("medium", sorted_counts[min(len(sorted_counts) - 1, len(sorted_counts) // 2)][1]),
        ("dense", sorted_counts[min(len(sorted_counts) - 1, (4 * len(sorted_counts)) // 5)][1]),
    ]
    dedup: list[tuple[str, str]] = []
    seen = set()
    for label, stem in picks:
        if stem not in seen:
            dedup.append((label, stem))
            seen.add(stem)
    return dedup[:num_images]


def _draw_one(stem: str, label: str, root: Path, out_dir: Path) -> tuple[Path, int]:
    img_path = root / "images" / f"{stem}.jpg"
    ann_path = root / "annotations" / f"{stem}.txt"
    out_path = out_dir / f"{stem}_{label}_qa_annotated.jpg"
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    count = 0
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            continue
        x, y, w, h = map(float, parts[:4])
        cid = int(float(parts[5]))
        trunc = int(float(parts[6]))
        occ = int(float(parts[7]))
        if cid == 0:
            continue
        category = VISDRONE_CATEGORY.get(cid, str(cid))
        box_label = f"{category} o{occ} t{trunc}"
        color = COLORS[cid % len(COLORS)]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        text_width = draw.textlength(box_label, font=font)
        draw.rectangle([x, max(0, y - 16), x + text_width + 4, y], fill=color)
        draw.text((x + 2, max(0, y - 15)), box_label, fill="black", font=font)
        count += 1
    title = f"VisDrone {label}: {stem} | objects: {count}"
    draw.rectangle([0, 0, img.width, 28], fill="black")
    draw.text((8, 6), title, fill="white", font=font)
    ensure_parent(out_path)
    img.save(out_path, quality=92)
    return out_path, count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--num-images", type=int, default=3)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolve_path(cfg["paths"]["visdrone_val"])
    tasks = _read_tasks(resolve_path(cfg["paths"]["tasks_csv"]))
    tasks_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for task in tasks:
        tasks_by_image[task["image_id"]].append(task)
    out_dir = resolve_path("outputs/figures")
    examples = _select_examples(_annotation_counts(root / "annotations"), args.num_images)
    lines = [
        "# V0.5 VisDrone Q/A Examples",
        "",
        "Each image shows VisDrone object boxes and the generated VQA-style questions derived from annotations.",
        "",
    ]
    for label, stem in examples:
        out_path, count = _draw_one(stem, label, root, out_dir)
        rel = out_path.relative_to(resolve_path("."))
        lines.extend([f"## {label.title()} density example: `{stem}`", "", f"- annotated objects: `{count}`", f"- figure: `{rel}`", ""])
        image_tasks = tasks_by_image.get(stem, [])
        for task in image_tasks[:8]:
            lines.append(f"- **{task['question_type']}** Q: {task['question']} A: `{task['answer']}`")
        if len(image_tasks) > 8:
            lines.append(f"- ... {len(image_tasks) - 8} more generated Q/A tasks")
        lines.append("")
    md_path = out_dir / "v0_qa_examples.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"examples_md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
