import json, sys, copy
sys.path.insert(0, "src")
from vqa_semcom.config import load_config

base = load_config("configs/v2_0_ldpc_channel.yaml")
for kind in ["awgn", "rayleigh"]:
    c = copy.deepcopy(base)
    c["vlm"]["fading_link"]["fading"]["kind"] = kind
    p = c["paths"]
    p["vlm_predictions_csv"] = f"outputs/vlm/v2_0_{kind}_predictions.csv"
    p["vlm_lut_csv"] = f"outputs/lut/v2_0_{kind}_semantic_quality_lut.csv"
    p["vlm_report_md"] = f"outputs/reports/v2_0_{kind}_eval_report.md"
    p["degraded_image_dir"] = f"outputs/vlm/degraded_images_v2_0_{kind}"
    p["link_calibration_json"] = f"outputs/lut/link_calibration_v2_0_{kind}.json"
    p["sim_results_csv"] = f"outputs/sim/v2_0_{kind}_resource_results.csv"
    p["sim_summary_md"] = f"outputs/sim/v2_0_{kind}_resource_summary.md"
    c.setdefault("detector", {})["detections_csv"] = f"outputs/detector/v2_0_{kind}_detections.csv"
    out = f"configs/v2_0_{kind}.json"
    json.dump(c, open(out, "w"), indent=1)
    print("wrote", out, "| kind=", c["vlm"]["fading_link"]["fading"]["kind"],
          "| preds=", p["vlm_predictions_csv"])
