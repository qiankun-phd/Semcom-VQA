import json, sys
sys.path.insert(0, "src")
from vqa_semcom.config import load_config

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
# main = presence+counting (v1_7 tasks); cmp = comparison tasks. Rician channel, 2nd VLM.
specs = {
    "main": ("outputs/tasks/v1_7_tasks.csv", ["presence", "counting"]),
    "cmp":  ("outputs/tasks/v2_comparison_tasks.csv", ["comparison"]),
}
for tag, (tasks_csv, qtypes) in specs.items():
    c = load_config("configs/v2_0_ldpc_channel.yaml")  # rician
    c["vlm"]["model_name"] = MODEL
    c["vlm"]["model_local_path"] = MODEL
    c["vlm"]["fallback_model_name"] = MODEL
    c["vlm"]["question_types"] = qtypes
    c["paths"]["tasks_csv"] = tasks_csv
    c["paths"]["vlm_predictions_csv"] = f"outputs/vlm/v25_rician_{tag}_predictions.csv"
    c["paths"]["degraded_image_dir"] = f"outputs/vlm/degraded_images_v25_rician_{tag}"
    c.setdefault("detector", {})["detections_csv"] = f"outputs/detector/v25_rician_{tag}_detections.csv"
    out = f"configs/v25_rician_{tag}.json"
    json.dump(c, open(out, "w"), indent=1)
    print("wrote", out, "| model=", c["vlm"]["model_name"], "| qtypes=", qtypes)
