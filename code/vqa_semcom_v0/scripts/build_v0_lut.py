#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.data.visdrone import demo_objects, load_visdrone_annotations
from vqa_semcom.quality.lut_builder import build_lut, write_lut_csv, write_summary
from vqa_semcom.tasks.generate_tasks import generate_tasks, write_tasks_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.demo:
        objects = demo_objects()
    else:
        objects = load_visdrone_annotations(resolve_path(cfg["paths"]["visdrone_val"]), limit_images=args.limit_images)
    if not objects:
        raise RuntimeError("No VisDrone objects found; use --demo or download/extract the dataset.")
    tasks = generate_tasks(objects, cfg)
    if not tasks:
        raise RuntimeError("No VQA-style tasks generated.")
    task_path = resolve_path(cfg["paths"]["tasks_csv"])
    write_tasks_csv(tasks, task_path)
    task_dicts = [dict((k, str(v)) for k, v in task.__dict__.items()) for task in tasks]
    lut_rows = build_lut(task_dicts, cfg)
    write_lut_csv(lut_rows, resolve_path(cfg["paths"]["lut_csv"]))
    write_summary(lut_rows, task_dicts, resolve_path(cfg["paths"]["lut_json"]), resolve_path(cfg["paths"]["lut_md"]), cfg)
    print(f"objects={len(objects)} tasks={len(tasks)} lut_rows={len(lut_rows)}")
    print(f"tasks_csv={task_path}")
    print(f"lut_csv={resolve_path(cfg['paths']['lut_csv'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
