"""Export detailed TGCN publication metrics across SNRs, Modes, and Tasks.

Runs evaluation on the 2,400 test images for:
- Mode 1: Constant Symbol Power
- Mode 2: Constant Transmission Energy Budget per Query
- Task breakdowns across the 6 TDIUC question types
- Action routing histograms
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

CELLS = [(b, t) for b in (2000, 4000, 8000) for t in ("low", "medium", "high")]
BUDGETS = (2000, 4000, 8000)
TIERS = ("low", "medium", "high")
CELL_NAMES = [f"{b}_{t}" for b, t in CELLS]
SNRS = [-5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]

BUDGET_SYMBOLS = {2000: 21420, 4000: 42840, 8000: 85170}
BUDGET_BYTES = {2000: 1998, 4000: 3997, 8000: 7982}


def normalize_answer(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").split())


class ChannelAwareSelector(nn.Module):
    def __init__(self, in_features: int = 340, hidden1: int = 128, hidden2: int = 64, num_actions: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_ensemble(checkpoints_dir: Path, seeds: List[int], device: torch.device) -> List[nn.Module]:
    models = []
    for s in seeds:
        ckpt = checkpoints_dir / f"selector_seed_{s}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        m = ChannelAwareSelector(340, 128, 64, 9).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m)
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--delivery-cache", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--channel-seed", type=int, default=1701)
    args = parser.parse_args()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print("Loading test data...")
    manifest = json.loads((args.test_data_dir / "test_manifest.json").read_text())
    truth_raw = json.loads((args.test_data_dir / "test_truth.sealed.json").read_text())
    truth = {r["id"]: r["answer"] for r in truth_raw}
    records = json.loads((args.test_data_dir / "test_records.json").read_text())
    features_raw = json.loads((args.test_data_dir / "test_features.json").read_text())
    features = {r["id"]: r for r in features_raw}
    scalers = json.loads((args.test_data_dir / "scalers.json").read_text())
    primary_joint_actions = json.loads((args.test_data_dir / "primary_joint_actions.json").read_text())

    print("Loading delivery cache...")
    raw_cache = json.loads(args.delivery_cache.read_text())
    delivery_map = {}
    for k, v in raw_cache.items():
        im, sd, snr, plen = k.split("|")
        delivery_map[(im, int(sd), float(snr), int(plen))] = v

    print("Loading selector ensemble...")
    models = load_ensemble(args.checkpoints_dir, [7, 17, 27], device)

    # organize samples
    samples = {}
    for r in manifest:
        qid = r["id"]
        samples[qid] = {
            "image_id": str(r["image_id"]),
            "question_type": r["question_type"],
            "truth": truth[qid],
            "exp014_primary_joint_action": primary_joint_actions[qid],
            "actions": {},
        }

    for r in records:
        qid = r["id"]
        cell_name = f"{r['budget']}_{r['tier']}"
        action_idx = CELL_NAMES.index(cell_name)
        is_corr = int(normalize_answer(r["prediction"]) == normalize_answer(truth[qid]))
        samples[qid]["actions"][action_idx] = {
            "cell": cell_name,
            "budget": r["budget"],
            "tier": r["tier"],
            "correct": is_corr,
            "image_bytes": r["image_bytes"],
            "actual_visual_tokens": r["actual_visual_tokens"],
            "ldpc_complex_symbols": 510 * math.ceil(r["image_bytes"] / 48),
        }

    # policy inference across SNRs
    mean = np.array(scalers["scalers"]["large"]["mean"], dtype=np.float32)
    std = np.array(scalers["scalers"]["large"]["std"], dtype=np.float32)
    img_mult = 1.0 / math.sqrt(83.0)

    policy_choices: Dict[float, Dict[str, int]] = {snr: {} for snr in SNRS}
    routing_histograms: Dict[float, Counter] = {snr: Counter() for snr in SNRS}

    print("Running cross-layer policy network inference...")
    for snr in SNRS:
        snr_norm = (snr - (-5.0)) / 25.0
        x_rows, qids = [], []
        for r in manifest:
            qid = r["id"]
            feat = features[qid]
            q_feat = np.array(feat["question_features"], dtype=np.float32)
            im_feat = (np.array(feat["image_features"], dtype=np.float32) - mean) / std * img_mult
            x_vec = np.concatenate([q_feat, im_feat, [snr_norm]])
            x_rows.append(x_vec)
            qids.append(qid)
        x_tensor = torch.tensor(np.array(x_rows), dtype=torch.float32)
        with torch.no_grad():
            preds = [m(x_tensor) for m in models]
            ensemble_logits = torch.stack(preds).mean(dim=0)
            actions = ensemble_logits.argmax(dim=-1).tolist()
        for qid, a in zip(qids, actions):
            policy_choices[snr][qid] = a
            routing_histograms[snr][CELL_NAMES[a]] += 1

    methods = [
        "fixed_2000_low", "fixed_2000_medium", "fixed_2000_high",
        "fixed_4000_low", "fixed_4000_medium", "fixed_4000_high",
        "fixed_8000_low", "fixed_8000_medium", "fixed_8000_high",
        "exp014_blind_joint", "tgcn_cross_layer"
    ]
    question_types = sorted(list(set(r["question_type"] for r in manifest)))

    results: Dict[str, Any] = {
        "snrs": SNRS,
        "question_types": question_types,
        "mode1_constant_power": {},
        "mode2_constant_energy": {},
        "by_task_mode2": {},
        "routing_histograms": {str(k): dict(v) for k, v in routing_histograms.items()},
    }

    n_images = len(samples)

    for snr in SNRS:
        s_snr = str(snr)
        results["mode1_constant_power"][s_snr] = {}
        results["mode2_constant_energy"][s_snr] = {}
        results["by_task_mode2"][s_snr] = {qt: {} for qt in question_types}

        snr_round = round(snr, 2)
        snr_4k = round(snr - 10.0 * math.log10(BUDGET_SYMBOLS[4000] / BUDGET_SYMBOLS[2000]), 2)
        snr_8k = round(snr - 10.0 * math.log10(BUDGET_SYMBOLS[8000] / BUDGET_SYMBOLS[2000]), 2)

        for m_name in methods:
            m1_corr, m1_deliv, m1_syms, m1_toks = 0, 0, 0, 0
            m2_corr, m2_deliv, m2_syms, m2_toks = 0, 0, 0, 0
            task_m2 = {qt: {"corr": 0, "deliv": 0, "syms": 0, "toks": 0, "n": 0} for qt in question_types}

            for qid, s in samples.items():
                qt = s["question_type"]
                task_m2[qt]["n"] += 1

                if m_name.startswith("fixed_"):
                    c_name = m_name.replace("fixed_", "")
                    a_idx = CELL_NAMES.index(c_name)
                elif m_name == "exp014_blind_joint":
                    a_idx = s["exp014_primary_joint_action"]
                elif m_name == "tgcn_cross_layer":
                    a_idx = policy_choices[snr][qid]
                else:
                    raise ValueError(m_name)

                act = s["actions"][a_idx]
                b = act["budget"]
                plen = BUDGET_BYTES[b]
                syms = act["ldpc_complex_symbols"]
                toks = act["actual_visual_tokens"]

                # Mode 1 delivery: eff_snr is constant nominal SNR
                ok_m1 = delivery_map[(s["image_id"], args.channel_seed, snr_round, plen)]
                m1_syms += syms
                m1_toks += toks
                if ok_m1:
                    m1_deliv += 1
                    m1_corr += act["correct"]

                # Mode 2 delivery: eff_snr depends on budget
                eff_snr_m2 = snr_round if b == 2000 else (snr_4k if b == 4000 else snr_8k)
                ok_m2 = delivery_map[(s["image_id"], args.channel_seed, eff_snr_m2, plen)]
                m2_syms += syms
                m2_toks += toks
                task_m2[qt]["syms"] += syms
                task_m2[qt]["toks"] += toks
                if ok_m2:
                    m2_deliv += 1
                    m2_corr += act["correct"]
                    task_m2[qt]["deliv"] += 1
                    task_m2[qt]["corr"] += act["correct"]

            results["mode1_constant_power"][s_snr][m_name] = {
                "strict_acc": m1_corr / n_images,
                "pdr": m1_deliv / n_images,
                "mean_symbols": m1_syms / n_images,
                "mean_tokens": m1_toks / n_images,
            }
            results["mode2_constant_energy"][s_snr][m_name] = {
                "strict_acc": m2_corr / n_images,
                "pdr": m2_deliv / n_images,
                "mean_symbols": m2_syms / n_images,
                "mean_tokens": m2_toks / n_images,
            }

            for qt in question_types:
                tn = task_m2[qt]["n"]
                results["by_task_mode2"][s_snr][qt][m_name] = {
                    "strict_acc": task_m2[qt]["corr"] / tn,
                    "pdr": task_m2[qt]["deliv"] / tn,
                    "mean_tokens": task_m2[qt]["toks"] / tn,
                    "mean_symbols": task_m2[qt]["syms"] / tn,
                }

    args.output_json.write_text(json.dumps(results, indent=2))
    print(f"Metrics successfully written to {args.output_json}")


if __name__ == "__main__":
    main()
