import json, sys
sys.path.insert(0, "src")
from vqa_semcom.config import load_config

bases = {"rician": "configs/v2_0_ldpc_channel.yaml", "awgn": "configs/v2_0_awgn.json", "rayleigh": "configs/v2_0_rayleigh.json"}
for ch, bp in bases.items():
    c = load_config(bp)
    c["vlm"]["question_types"] = ["co_presence", "threshold"]
    c["paths"]["tasks_csv"] = "outputs/tasks/v2_extra_tasks.csv"
    c["paths"]["vlm_predictions_csv"] = f"outputs/vlm/v2_0_{ch}_extra_predictions.csv"
    c["paths"]["degraded_image_dir"] = f"outputs/vlm/degraded_images_v2_0_{ch}_extra"
    c.setdefault("detector", {})["detections_csv"] = f"outputs/detector/v2_0_{ch}_extra_detections.csv"
    out = f"configs/v2_0_{ch}_extra.json"
    json.dump(c, open(out, "w"), indent=1)
    print("wrote", out)
