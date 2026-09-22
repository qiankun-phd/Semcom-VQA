#!/usr/bin/env python3
"""Independent-calibration, leakage-free routing experiments on cached outcomes.

Never mutates input logs or images. Does not invoke detectors, VLMs or GPUs.
Payload reconstruction replays the original CPU codec and checks saved JPEGs.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier

QTYPES = ("presence", "counting", "comparison", "co_presence", "threshold")
CLASSES = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
           "awning-tricycle", "bus", "motor")
SNRS = (-5, 0, 5, 10, 15, 20)
FEATURE_NAMES = [f"question_type:{v}" for v in QTYPES] + [f"target_class:{v}" for v in CLASSES] + [
    "snr_div20", "raw_detector_count_clipped60", "raw_detector_nonzero"]
PRICES = [0.0] + np.logspace(-5, -1, 25).tolist()
ENERGY_DET = 0.4275


def dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_for(image_id: str) -> str:
    bucket = zlib.crc32(image_id.encode()) % 100
    return "test" if bucket < 20 else ("validation" if bucket >= 80 else "train")


def truth(value: str) -> bool:
    if str(value).lower() not in ("true", "false", "1", "0", "yes", "no"):
        raise ValueError(f"Invalid correctness: {value!r}")
    return str(value).lower() in ("true", "1", "yes")


def features(g: dict) -> list[float]:
    # Explicit allowlist: no answer polarity, annotation-derived risk/view,
    # ground truth, calibrated counts or receiver outputs enter the predictors.
    return [float(g["qt"] == q) for q in QTYPES] + [float(g["class"] == c) for c in CLASSES] + [
        g["snr"] / 20, min(g["raw"], 60) / 60, float(g["raw"] > 0)]


def load_groups(path: Path) -> tuple[list[dict], dict]:
    csv.field_size_limit(sys.maxsize)
    groups: dict = {}
    duplicates = 0
    with path.open(newline="") as stream:
        for r in csv.DictReader(stream):
            b = r["service_level"]
            if b not in ("1", "2"):
                continue
            snr = int(float(r["snr_bin"].replace("dB", "")))
            key = (r["image_id"], r["question"], snr)
            g = groups.setdefault(key, {"image": key[0], "question": key[1], "snr": snr,
                "qt": r["question_type"], "class": r["target_class"], "split": split_for(key[0])})
            branch = {"correct": truth(r["correct"]), "bytes": int(r["payload_bytes"])}
            if b == "1":
                branch.update(raw=int(float(r["raw_detector_count"])),
                    transmitted=int(float(r["transmitted_detector_count"])),
                    calibrated=int(float(r["calibrated_detector_count"])),
                    gt=int(float(r["object_count"])), polarity=r.get("presence_polarity", ""),
                    answer=r["ground_truth_answer"])
            else:
                branch["path"] = r["image_path"]
            if b in g:
                duplicates += 1
                if g[b] != branch:
                    raise ValueError(f"Conflicting duplicate for {key}, branch {b}")
            g[b] = branch
    rows = []
    for key, g in sorted(groups.items()):
        if "1" not in g or "2" not in g:
            raise ValueError(f"Unpaired decision: {key}")
        if g["qt"] not in QTYPES or g["snr"] not in SNRS:
            raise ValueError("Unexpected task type or SNR")
        g["raw"] = g["1"]["raw"]
        rows.append(g)
    images = {s: sorted({g["image"] for g in rows if g["split"] == s})
              for s in ("train", "validation", "test")}
    assert all(images.values())
    assert not any(set(images[a]) & set(images[b]) for a,b in
                   (("train","validation"),("train","test"),("validation","test")))
    presence = [g for g in rows if g["qt"] == "presence"]
    audit = {"path": str(path), "sha256": sha(path), "decisions": len(rows),
        "duplicates_identical_on_checked_fields": duplicates, "images_by_split": images,
        "decisions_by_split": dict(Counter(g["split"] for g in rows)),
        "presence_polarity_matches_answer": sum((g["1"]["polarity"] == "positive") ==
                                                (g["1"]["answer"].lower() == "yes") for g in presence),
        "presence_decisions": len(presence), "feature_names": FEATURE_NAMES}
    return rows, audit


def count_correct(pred: int, gt: int) -> bool:
    return abs(pred - gt) <= max(1, round(0.1 * gt))


def apply_ratio(raw: int, ratio: float) -> int:
    return raw if raw < 3 else max(0, round(raw * ratio))


def fit_calibration(groups: list[dict]) -> dict:
    if any(g["split"] != "train" for g in groups):
        raise ValueError("Calibration fit accepts training images only")
    buckets = defaultdict(list)
    for g in groups:
        if g["qt"] == "counting":
            buckets[(g["class"], g["snr"])].append(g)
    ratios = {}
    for key, rows in buckets.items():
        d = sum(g["1"]["transmitted"] for g in rows)
        ratio = min(4.0, max(0.5, sum(g["1"]["gt"] for g in rows) / d)) if d > 0 else 1.0
        old = sum(count_correct(g["1"]["transmitted"], g["1"]["gt"]) for g in rows)
        new = sum(count_correct(apply_ratio(g["1"]["transmitted"], ratio), g["1"]["gt"]) for g in rows)
        ratios[f"{key[0]}|{key[1]}"] = ratio if new >= old else 1.0
    return ratios


def labels(groups: list[dict], ratios: dict) -> np.ndarray:
    pairs = []
    for g in groups:
        det = g["1"]["correct"]
        if g["qt"] == "counting":
            pred = apply_ratio(g["1"]["transmitted"], ratios.get(f'{g["class"]}|{g["snr"]}', 1.0))
            det = count_correct(pred, g["1"]["gt"])
        pairs.append([det, g["2"]["correct"]])
    return np.asarray(pairs, dtype=np.int8)


def fit_lut(groups: list[dict], y: np.ndarray) -> dict:
    if any(g["split"] != "train" for g in groups):
        raise ValueError("LUT fit accepts training images only")
    buckets = defaultdict(list)
    for i,g in enumerate(groups):
        buckets[(g["qt"],g["snr"])].append(i)
    policy = {}
    for key, inds in buckets.items():
        # Equal n on the paired branches makes Wilson-LCB ranking identical
        # to accuracy ranking. Ties choose detection, as in the original loop.
        policy[f"{key[0]}|{key[1]}"] = int(y[inds, 1].sum() > y[inds, 0].sum())
    return policy


def probabilities(model, x: np.ndarray) -> np.ndarray:
    p = model.predict_proba(x)
    return p[:, list(model.classes_).index(1)]


def fit_pair(xtrain: np.ndarray, ytrain: np.ndarray, xval: np.ndarray,
             yval: np.ndarray, seed: int, max_epochs: int) -> tuple[list, list]:
    models, records = [], []
    for branch in range(2):
        model = MLPClassifier(hidden_layer_sizes=(32,16), activation="relu", solver="adam",
            alpha=0.0001, batch_size=200, learning_rate_init=0.001, early_stopping=False,
            random_state=seed, shuffle=True)
        best, best_loss, bad, history = None, float("inf"), 0, []
        for epoch in range(max_epochs):
            model.partial_fit(xtrain, ytrain[:,branch], classes=np.array([0,1]))
            p = np.clip(probabilities(model,xval), 1e-12, 1-1e-12)
            loss = float(-np.mean(yval[:,branch]*np.log(p)+(1-yval[:,branch])*np.log(1-p)))
            history.append(loss)
            if loss < best_loss - 1e-4:
                best, best_loss, bad = copy.deepcopy(model), loss, 0
            else:
                bad += 1
            if bad >= 10:
                break
        assert best is not None
        models.append(best)
        records.append({"branch": branch, "epochs_run": len(history), "best_validation_bce": best_loss,
                        "validation_bce": history})
    return models, records


def fit_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xb = np.column_stack([x, np.ones(len(x))])
    w = np.zeros((xb.shape[1], 2))
    for _ in range(400):
        p = 1/(1+np.exp(-np.clip(xb@w, -50, 50)))
        w -= 0.5 * (xb.T@(p-y)/len(x) + 0.001*w)
    return w


def linear_predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-np.clip(np.column_stack([x,np.ones(len(x))])@w,-50,50)))


def choose(p: np.ndarray, energy: np.ndarray, price: float) -> np.ndarray:
    return (p[:,1]-p[:,0] > price*(energy[:,1]-energy[:,0])).astype(np.int8)


def metrics(y: np.ndarray, pick: np.ndarray, energy: np.ndarray) -> dict:
    ix = np.arange(len(pick))
    return {"n": len(pick), "accuracy": float(y[ix,pick].mean()),
            "image_fraction": float(pick.mean()), "energy_j": float(energy[ix,pick].mean())}


def reconstruct_payload(groups: list[dict], repo: Path, channel: str, limit: int | None = None) -> dict:
    import cv2
    sys.path.insert(0,str(repo/"src"))
    from vqa_semcom.degradation.digital_link import (FadingConfig, LinkConfig, _seed_from,
        transmit_image_rate_adaptive, ergodic_spectral_efficiency)
    cfg = LinkConfig(fading=FadingConfig(kind=channel, k_factor_db=6.0))
    unique = {(g["image"],g["snr"]): g for g in groups}
    records = {}
    source_hashes = {}
    image_cache = None
    prev_image = None
    for j, ((iid,snr), g) in enumerate(sorted(unique.items())):
        if limit is not None and j >= limit:
            break
        source = repo/f"data/raw/visdrone/DET/val/images/{iid}.jpg"
        if iid != prev_image:
            image_cache = cv2.imread(str(source))
            if image_cache is None:
                raise FileNotFoundError(source)
            source_hashes[iid] = sha(source)
            prev_image = iid
        rng = np.random.default_rng(_seed_from(iid,f"{snr}dB"))
        decoded, meta = transmit_image_rate_adaptive(image_cache,snr,cfg,rng)
        ok, jpeg = cv2.imencode(".jpg",decoded,[int(cv2.IMWRITE_JPEG_QUALITY),95])
        assert ok
        saved = Path(g["2"]["path"])
        file_match = saved.is_file() and sha(saved) == hashlib.sha256(jpeg.tobytes()).hexdigest()
        # On an outage the old simulator returns before encoding and stores no
        # transmitted bytes. Charge a full slot explicitly, never zero energy.
        se = ergodic_spectral_efficiency(snr,cfg.fading)
        tx_time = cfg.tx_time_budget_s if meta["outage"] else meta["bytes"]*8/se/cfg.bandwidth_hz
        records[f"{iid}|{snr}"] = {"meta":meta,"airtime_s":tx_time,
            "saved_jpeg_matches": bool(file_match), "logged_file_bytes":g["2"]["bytes"],
            "wire_payload_bytes":meta.get("bytes"), "logged_path":str(saved)}
        if (j+1)%100 == 0:
            print(f"[payload {channel}] {j+1}/{len(unique)}",flush=True)
    return {"channel":channel,"records":records,"source_image_sha256":source_hashes,
        "n":len(records),"matched_saved_images":sum(r["saved_jpeg_matches"] for r in records.values()),
        "outage_cost":"full 0.3 s slot, because transmitted bytes are unrecorded on outage",
        "cv2_version":cv2.__version__,
        "link_source_sha256":sha(repo/"src/vqa_semcom/degradation/digital_link.py")}


def energy_matrix(groups: list[dict], payload: dict, evlm: float, learned: bool) -> np.ndarray:
    result = []
    for g in groups:
        # Detection packet text byte count is the existing compact-airtime
        # model. Its separate outage convention is not a measured PHY latency.
        det = ENERGY_DET + 0.5 * g["1"]["bytes"] * 8 / 0.5 / 1e6
        im = evlm + 0.5 * payload["records"][f'{g["image"]}|{g["snr"]}']["airtime_s"]
        result.append([det, im + (ENERGY_DET if learned else 0)])
    return np.asarray(result)


def summarize(groups: list[dict], y: np.ndarray, pick: np.ndarray, energy: np.ndarray) -> dict:
    out = {"pooled": metrics(y,pick,energy), "per_snr":{},"per_type":{}}
    for key,values in (("snr",SNRS),("qt",QTYPES)):
        for v in values:
            mask = np.array([g[key]==v for g in groups])
            if mask.any():
                out["per_snr" if key=="snr" else "per_type"][str(v)] = metrics(y[mask],pick[mask],energy[mask])
    return out


def run_channel(repo: Path, out: Path, channel: str, seeds: list[int], max_epochs: int) -> None:
    groups,audit = load_groups(repo/f"outputs/vlm/v3_0_{channel}_predictions.csv")
    dest = out/channel
    dest.mkdir(exist_ok=True)
    dump(dest/"data_audit.json",audit)
    payload = reconstruct_payload(groups,repo,channel)
    dump(dest/"payload_audit.json",payload)
    if payload["matched_saved_images"] != payload["n"]:
        raise ValueError(f"{channel}: reconstruction differs from saved receiver images; do not pair new cost with old outcomes")
    parts = {s:[g for g in groups if g["split"]==s] for s in ("train","validation","test")}
    x = {s:np.array([features(g) for g in rows]) for s,rows in parts.items()}
    power = json.loads((repo/"outputs/energy/gpu_power_phases.json").read_text())
    evlm = power["phases"]["vlm"]["joule_per_item_incremental"]
    energy = {s:energy_matrix(rows,payload,evlm,True) for s,rows in parts.items()}
    baseline_energy = energy_matrix(parts["test"],payload,evlm,False)
    dump(dest/"test_keys.json",[{k:g[k] for k in ("image","question","qt","snr","class")} for g in parts["test"]])
    for mode in ("train_only","none"):
        md = dest/mode
        md.mkdir(exist_ok=True)
        ratios = fit_calibration(parts["train"]) if mode=="train_only" else {}
        dump(md/"calibration.json",ratios)
        y = {s:labels(rows,ratios) for s,rows in parts.items()}
        policy = fit_lut(parts["train"],y["train"])
        dump(md/"lut.json",policy)
        test = parts["test"]
        picks = {"image":np.ones(len(test),dtype=int),"detection":np.zeros(len(test),dtype=int),
            "rule":np.array([g["qt"]=="presence" for g in test],dtype=int),
            "lut":np.array([policy.get(f'{g["qt"]}|{g["snr"]}',0) for g in test])}
        baselines = {name:summarize(test,y["test"],pk,baseline_energy) for name,pk in picks.items()}
        baselines["oracle_accuracy"] = float(y["test"].max(axis=1).mean())
        w = fit_linear(x["train"],y["train"])
        lp = linear_predict(w,x["test"])
        picks["linear"] = choose(lp,energy["test"],0)
        baselines["linear"] = summarize(test,y["test"],picks["linear"],energy["test"])
        dump(md/"baselines.json",baselines)
        np.savez_compressed(md/"baseline_outcomes.npz", y=y["test"],energy=energy["test"],
            baseline_energy=baseline_energy,linear_weights=w,linear_probabilities=lp,**picks)
        for seed in seeds:
            started = time.time()
            models,history = fit_pair(x["train"],y["train"],x["validation"],y["validation"],seed,max_epochs)
            pv = np.column_stack([probabilities(m,x["validation"]) for m in models])
            pt = np.column_stack([probabilities(m,x["test"]) for m in models])
            pick = choose(pt,energy["test"],0)
            sweep,validation_sweep = [],[]
            all_picks=[]
            for price in PRICES:
                pk = choose(pt,energy["test"],price)
                all_picks.append(pk)
                sweep.append({"lambda":price,**metrics(y["test"],pk,energy["test"])})
                vp = choose(pv,energy["validation"],price)
                validation_sweep.append({"lambda":price,**metrics(y["validation"],vp,energy["validation"])})
            # Predeclared selection: min validation cost within 1 pp of lambda=0
            # validation accuracy; test labels do not participate in selection.
            threshold = validation_sweep[0]["accuracy"]-0.01
            selected = min((i for i,r in enumerate(validation_sweep) if r["accuracy"]>=threshold),
                           key=lambda i:(validation_sweep[i]["energy_j"],PRICES[i]))
            record = {"seed":seed,"mode":mode,"features":FEATURE_NAMES,
                "main":summarize(test,y["test"],pick,energy["test"]),"sweep":sweep,
                "validation_sweep":validation_sweep,"validation_selected_index":selected,
                "selected_test":sweep[selected],"training":history,"seconds":time.time()-started}
            np.savez_compressed(md/f"seed_{seed}_outcomes.npz",probabilities=pt,pick=pick,
                validation_probabilities=pv,sweep_picks=np.asarray(all_picks),prices=np.array(PRICES))
            import joblib
            joblib.dump(models,md/f"seed_{seed}_models.joblib")
            dump(md/f"seed_{seed}.json",record)
            print(f"[done {channel} {mode} seed={seed}] elapsed={record['seconds']:.1f}s",flush=True)
            dump(out/"status.json",{"state":"RUNNING","channel":channel,"mode":mode,"last_seed":seed})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--stage",choices=("preflight","run"),required=True)
    ap.add_argument("--channels",nargs="+",default=["awgn","rayleigh","rician"])
    ap.add_argument("--seeds",nargs="+",type=int,default=list(range(10)))
    ap.add_argument("--max-epochs",type=int,default=300)
    args = ap.parse_args()
    # New directories only: prevents mixing results, silent resume or overwrite.
    args.out.mkdir(parents=True,exist_ok=False)
    import sklearn
    manifest={"stage":args.stage,"channels":args.channels,"seeds":args.seeds,"max_epochs":args.max_epochs,
        "script_sha256":sha(Path(__file__)),"python":sys.version,"numpy":np.__version__,"sklearn":sklearn.__version__,
        "split":"crc32(image)%100: test <20; validation >=80; train otherwise",
        "features":FEATURE_NAMES,"excluded_features":["presence_polarity","risk_level","view_quality_bin","freshness_bin"],
        "prices":PRICES,"selection":"min validation energy within 0.01 accuracy of unpriced validation result",
        "calibration_modes":["train_only","none"],"pid":os.getpid(),
        "energy_exclusions":["source encoder","router","symbolic decoder","flight","sensor","radio circuitry"],
        "scope":"new CPU router fitting and codec replay on existing branch outcomes; no VLM/detector inference"}
    dump(args.out/"manifest.json",manifest)
    try:
        for channel in args.channels:
            if args.stage=="preflight":
                groups,audit=load_groups(args.repo/f"outputs/vlm/v3_0_{channel}_predictions.csv")
                dump(args.out/f"{channel}_data.json",audit)
                payload=reconstruct_payload(groups,args.repo,channel,limit=6)
                dump(args.out/f"{channel}_payload.json",payload)
                print(f"[preflight {channel}] matched={payload['matched_saved_images']}/{payload['n']} splits={audit['decisions_by_split']}",flush=True)
                if payload["matched_saved_images"]!=payload["n"]:
                    raise ValueError("Payload reconstruction failed receiver-image identity gate")
            else:
                run_channel(args.repo,args.out,channel,args.seeds,args.max_epochs)
        dump(args.out/"status.json",{"state":"COMPLETE","stage":args.stage})
    except Exception as exc:
        dump(args.out/"status.json",{"state":"FAILED","error":repr(exc)})
        raise


if __name__=="__main__":
    main()
