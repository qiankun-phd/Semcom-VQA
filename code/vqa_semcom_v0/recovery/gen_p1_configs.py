#!/usr/bin/env python3
"""Generate P1 configs: clean (error-free) main/cmp/extra + v25 (Qwen2.5-VL-3B) extra."""
import json


def load(p):
    return json.load(open(p))


def dump(cfg, p):
    json.dump(cfg, open(p, "w"), indent=2)
    print("wrote", p)


def diff_keys(a, b, path=""):
    out = []
    for k in sorted(set(a) | set(b)):
        pa, pb = a.get(k), b.get(k)
        if isinstance(pa, dict) and isinstance(pb, dict):
            out += diff_keys(pa, pb, path + "/" + str(k))
        elif pa != pb:
            out.append((path + "/" + str(k), pa, pb))
    return out


v2cmp = load("configs/v2_0_rician_cmp.json")
v25cmp = load("configs/v25_rician_cmp.json")
print("--- v2_0_rician_cmp vs v25_rician_cmp diffs:")
for p, a, b in diff_keys(v2cmp, v25cmp):
    print(f"  {p}\n    v2_0: {json.dumps(a)[:120]}\n    v25 : {json.dumps(b)[:120]}")

# ---- clean configs: base on the awgn family, one very high SNR = error-free ----
for suffix, base in (("", "configs/v2_0_awgn.json"),
                     ("_cmp", "configs/v2_0_awgn_cmp.json"),
                     ("_extra", "configs/v2_0_awgn_extra.json")):
    cfg = load(base)
    cfg["paths"]["vlm_predictions_csv"] = f"outputs/vlm/v2_0_clean{suffix or '_main'}_predictions.csv"
    cfg["paths"]["degraded_image_dir"] = f"outputs/vlm/degraded_images_v2_0_clean{suffix or '_main'}"
    dump(cfg, f"configs/v2_0_clean{suffix or '_main'}.json")

# ---- v25 extra: v2_0 extra tasks evaluated by the 2nd VLM ----
cfg = load("configs/v2_0_rician_extra.json")
overrides = diff_keys(load("configs/v2_0_rician_cmp.json"), load("configs/v25_rician_cmp.json"))
for path, _, v25_val in overrides:
    parts = [p for p in path.split("/") if p]
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    leaf = parts[-1]
    old = node.get(leaf)
    if isinstance(old, str) and "v2_0_rician_cmp" in str(old):
        node[leaf] = str(v25_val).replace("cmp", "extra") if v25_val else old
    else:
        node[leaf] = v25_val
cfg["paths"]["vlm_predictions_csv"] = "outputs/vlm/v25_rician_extra_predictions.csv"
cfg["paths"]["degraded_image_dir"] = "outputs/vlm/degraded_images_v25_rician_extra"
dump(cfg, "configs/v25_rician_extra.json")
print("--- v25_rician_extra sanity:")
v = load("configs/v25_rician_extra.json")
print("  tasks_csv:", v["paths"]["tasks_csv"])
print("  predictions:", v["paths"]["vlm_predictions_csv"])
print("  model:", v.get("vlm", {}).get("model_local_path", "?"))
