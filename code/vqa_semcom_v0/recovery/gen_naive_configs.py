import json, sys, copy
sys.path.insert(0, "src")
from vqa_semcom.config import load_config

# base configs per channel
bases = {
    "rician": "configs/v2_0_ldpc_channel.yaml",
    "awgn": "configs/v2_0_awgn.json",
    "rayleigh": "configs/v2_0_rayleigh.json",
}
for kind, basepath in bases.items():
    c = copy.deepcopy(load_config(basepath))
    # naive = fixed-rate LDPC block erasure (cliffs), instead of rate-adaptive
    c["vlm"]["fading_link"]["channel_mode"] = "ldpc_erasure"
    # ensure fading kind correct (rician base already; awgn/rayleigh json already set)
    c["vlm"]["fading_link"]["fading"]["kind"] = kind
    p = c["paths"]
    p["vlm_predictions_csv"] = f"outputs/vlm/v2_0_{kind}_naive_predictions.csv"
    p["vlm_lut_csv"] = f"outputs/lut/v2_0_{kind}_naive_lut.csv"
    p["vlm_report_md"] = f"outputs/reports/v2_0_{kind}_naive_report.md"
    p["degraded_image_dir"] = f"outputs/vlm/degraded_images_v2_0_{kind}_naive"
    p["link_calibration_json"] = f"outputs/lut/link_calibration_v2_0_{kind}_naive.json"
    c.setdefault("detector", {})["detections_csv"] = f"outputs/detector/v2_0_{kind}_naive_detections.csv"
    out = f"configs/v2_0_{kind}_naive.json"
    json.dump(c, open(out, "w"), indent=1)
    print("wrote", out, "| kind=", kind, "| mode=", c["vlm"]["fading_link"]["channel_mode"])
