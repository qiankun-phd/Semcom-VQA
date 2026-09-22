#!/usr/bin/env python3
"""Audit existing logs and re-account energy, without fitting or inference.

Raw sources are immutable. Derived records live under outputs/revision_20260907.
Learned-router airtime retains the source's branch/SNR mean-cost approximation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from statistics import mean

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "outputs/revision_20260907"
CHANNELS = ("awgn", "rayleigh", "rician")
SNRS = (-5, 0, 5, 10, 15, 20)
SOURCES: set[Path] = set()


def read_json(path: Path) -> dict:
    SOURCES.add(path)
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict]:
    SOURCES.add(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = read_json(PAPER / "outputs/energy/energy_summary.json")
    ev = summary["params"]["e_vlm_incremental_j"]
    ed = summary["params"]["e_det_mid_j"]
    comparison = read_csv(PAPER / "figures/comparison_v3_5qt.csv")
    base = {(r["channel"], int(float(r["snr_db"])), r["method"]): r
            for r in comparison if r["qtype"] == "all" and r["split"] == "test"}
    stats = read_json(PAPER / "outputs/reports/paper1_stats.json")
    mlp = read_json(PAPER / "data/w14_mlp_10seed.json")
    assert mlp["params"]["seeds"] == list(range(10))
    linear = read_json(PAPER / "data/w5_persample_energy.json")
    djscc = read_json(PAPER / "outputs/reports/p1_m6_results.json")
    result: dict = {"energy": {"detector_j": ed, "vlm_j": ev,
        "bandwidth_hz": 1e6, "p_tx_w": 0.5, "eta": 1,
        "airtime": "branch/SNR mean costs, not selected-sample payload measurements"},
        "series": {}, "audit": {}, "source_policy_note":
        "LUT replays the recorded per-channel policy in paper1_stats; the old comparison CSV LUT is a distinct reference."}
    gaps = []
    savings = []
    worst_seed = []
    for channel in CHANNELS:
        path = PAPER / f"outputs/vlm/v3_0_{channel}_predictions.csv"
        SOURCES.add(path)
        groups: dict = {}
        duplicates = conflicts = 0
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                branch = row["service_level"]
                if branch not in ("1", "2") or zlib.crc32(row["image_id"].encode()) % 100 >= 20:
                    continue
                snr = int(float(row["snr_bin"].replace("dB", "")))
                key = (row["image_id"], row["question"], snr)
                g = groups.setdefault(key, {"type": row["question_type"]})
                value = row["correct"].lower() in ("true", "1", "yes")
                if branch in g:
                    duplicates += 1
                    conflicts += int(g[branch] != value)
                g[branch] = value
        assert all("1" in g and "2" in g for g in groups.values())
        assert conflicts == 0, (channel, conflicts)
        assert len(groups) == 5616
        assert len({key[0] for key in groups}) == 104
        result["audit"][channel] = {"paired_decisions": len(groups), "images": 104,
                                       "duplicate_rows": duplicates, "conflicting_duplicates": conflicts}
        series = defaultdict(list)
        policy = stats["rule_vs_lcb_visdrone"]["policy_per_channel"][channel]
        for snr in SNRS:
            sample = [g for k, g in groups.items() if k[2] == snr]
            assert len(sample) == 936
            img_tx = float(base[channel, snr, "M1_image"]["mean_channel_uses"]) * 0.5e-6
            det_tx = float(base[channel, snr, "M3_token"]["mean_channel_uses"]) * 0.5e-6
            eimg, edet = img_tx + ev, det_tx + ed
            for method in ("image", "token", "rule", "lut", "oracle"):
                choice = []
                accuracy = []
                for g in sample:
                    pick = {"image": "2", "token": "1", "rule": "2" if g["type"] == "presence" else "1",
                            "lut": policy[f'{g["type"]}|{snr}dB'],
                            "oracle": "2" if g["2"] and not g["1"] else "1"}[method]
                    choice.append(pick == "2")
                    accuracy.append(g[pick])
                f = mean(choice)
                tx = f * img_tx + (1-f) * det_tx
                detector = (1-f) * ed
                r = {"snr_db": snr, "accuracy": mean(accuracy), "f_img": f,
                     "radio_j": tx, "detector_j": detector, "vlm_j": f * ev,
                     "energy_j": tx + detector + f * ev, "n": 936}
                series[method].append(r)
                if method in ("image", "token"):
                    legacy = base[channel, snr, "M1_image" if method == "image" else "M3_token"]
                    assert abs(r["accuracy"] - float(legacy["accuracy"])) < 0.00006
            for method in ("mlp", "linear"):
                if method == "mlp":
                    r0 = mlp["per_channel"][channel][f"{snr:.1f}"]
                    f, old, acc = r0["f_img_mean"], r0["j_mean"], r0["acc_mean"]
                else:
                    r0 = linear["per_channel"][channel]["per_snr"][f"{snr:.1f}"]
                    f, old, acc = r0["f_img"], r0["j_per_answer"], r0["acc"]
                # Verified against original W14/W5 source: image choices omitted detection.
                assert abs(old - (f * eimg + (1-f) * edet)) < 0.003
                tx = f * img_tx + (1-f) * det_tx
                total = old + f * ed
                r = {"snr_db": snr, "accuracy": acc, "f_img": f, "radio_j": tx,
                     "detector_j": ed, "vlm_j": f * ev, "energy_j": total,
                     "source_energy_j": old, "added_detection_j": f * ed, "n": 936}
                assert abs(total - tx - ed - f * ev) < 0.003
                if method == "mlp":
                    r.update({"acc_min": r0["acc_min"], "acc_max": r0["acc_max"]})
                    # Energy is monotone affine in f in this source account; infer
                    # extrema from the rounded stored energies, with rounding caveat.
                    for suffix in ("min", "max"):
                        old_extreme = r0[f"j_{suffix}"]
                        f_extreme = (old_extreme - edet) / (eimg - edet)
                        assert 0 <= f_extreme <= 1
                        r[f"energy_{suffix}"] = old_extreme + f_extreme * ed
                    savings.append(100 * (1-total/eimg))
                    worst_seed.append(100 * (1-r["energy_max"]/eimg))
                    gaps.append(100*(acc-series["rule"][-1]["accuracy"]))
                series[method].append(r)
            if channel == "rician":
                d = djscc["per_snr"][str(snr)]
                assert d["n"] == 936
                tx = d["mean_channel_uses"] * 0.5e-6
                series["djscc"].append({"snr_db": snr, "accuracy": d["acc"], "energy_j": tx+ev,
                                       "radio_j": tx, "detector_j": 0, "vlm_j": ev, "f_img": 1, "n": 936})
        result["series"][channel] = dict(series)
    result["headline"] = {"mlp_saving_pct_range": [min(savings), max(savings)],
                           "worst_individual_seed_saving_pct": min(worst_seed),
                           "mlp_minus_rule_pp_range": [min(gaps), max(gaps)]}
    frontier = read_csv(PAPER / "data/c1_frontier_rician_mlp.csv")
    result["recorded_sweep"] = [{"legacy_lambda": float(r["lambda"]), "accuracy": float(r["acc"]),
        "source_energy_j": float(r["E_j"]), "energy_j": float(r["E_j"]) + ed*float(r["frac_image"]),
        "f_img": float(r["frac_image"])} for r in frontier]
    # Per-type arithmetic only. No new policies or evaluation outcomes.
    wt = read_json(PAPER / "data/w12_wt_pertype.json")["per_type"]
    mlp_key = next(k for k in wt if "mlp" in k.lower())
    counts = {t: d["n"] for t,d in wt["rule"].items()}
    other_total = sum(n for t,n in counts.items() if t != "presence")
    mix = []
    for presence in (0.1, 0.2, 0.3, counts["presence"]/sum(counts.values()), 0.5, 0.6, 0.7, 0.8, 0.9):
        weights = {t: presence if t == "presence" else (1-presence)*n/other_total for t,n in counts.items()}
        record = {"presence_share": presence}
        for source, target in (("rule", "rule"), (mlp_key, "mlp"), ("all_image", "image")):
            data = wt[source]
            energy = sum(weights[t]*(data[t]["e_j"] + (data[t]["f_img"]*ed if target == "mlp" else 0)) for t in weights)
            record[target] = {"accuracy": sum(weights[t]*data[t]["acc"] for t in weights), "energy_j": energy}
        mix.append(record)
    result["question_mix_3seed"] = mix
    result["pooled"] = {ch: {m: {key: mean(r[key] for r in rows)
                         for key in ("accuracy", "energy_j", "f_img", "radio_j", "detector_j", "vlm_j")}
                         for m,rows in methods.items()} for ch,methods in result["series"].items()}
    # Prefer the source's pooled accuracy to a mean of rounded per-SNR values.
    for channel in CHANNELS:
        result["pooled"][channel]["linear"]["accuracy"] = linear["per_channel"][channel]["pooled"]["acc"]
    (OUT / "audited_results.json").write_text(json.dumps(result, indent=2)+"\n")
    (OUT / "source_sha256.json").write_text(json.dumps({str(p.relative_to(PAPER)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}, indent=2)+"\n")
    print(json.dumps({"headline":result["headline"], "pooled":result["pooled"], "audit":result["audit"]}, indent=2))


if __name__ == "__main__":
    main()
