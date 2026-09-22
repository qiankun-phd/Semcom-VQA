#!/usr/bin/env python3
"""S2: configs for the cross-FAMILY VLM check -- SmolVLM-Instruct (Idefics3/SmolLM2,
fully non-Qwen) on the rician channel, mirroring the v25 (Qwen2.5-VL-3B) protocol."""
import json
import sys

sys.path.insert(0, "src")
from vqa_semcom.config import load_config

MODEL = "HuggingFaceTB/SmolVLM-Instruct"
specs = {
    "main": ("outputs/tasks/v1_7_tasks.csv", ["presence", "counting"]),
    "cmp": ("outputs/tasks/v2_comparison_tasks.csv", ["comparison"]),
    "extra": ("outputs/tasks/v2_extra_tasks.csv", ["co_presence", "threshold"]),
}
for tag, (tasks_csv, qtypes) in specs.items():
    c = load_config("configs/v2_0_ldpc_channel.yaml")  # rician
    c["vlm"]["model_name"] = MODEL
    c["vlm"]["model_local_path"] = MODEL
    c["vlm"]["fallback_model_name"] = MODEL
    c["vlm"]["question_types"] = qtypes
    c["paths"]["tasks_csv"] = tasks_csv
    c["paths"]["vlm_predictions_csv"] = f"outputs/vlm/v26_rician_{tag}_predictions.csv"
    c["paths"]["degraded_image_dir"] = f"outputs/vlm/degraded_images_v26_rician_{tag}"
    c.setdefault("detector", {})["detections_csv"] = f"outputs/detector/v26_rician_{tag}_detections.csv"
    out = f"configs/v26_rician_{tag}.json"
    json.dump(c, open(out, "w"), indent=2)
    print("wrote", out, "| model=", MODEL, "| qtypes=", qtypes)
