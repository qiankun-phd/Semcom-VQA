"""EXP-015: Independent Blind Test Evaluation of Cross-Layer Policy Network.

Evaluates the trained 340-dimensional channel-aware policy ensemble on all 2,400
blind test images across the full SNR span [-5.0 to 20.0 dB].
Uses precomputed physical channel realizations from Step 1 delivery cache.
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


def build_question_prior(records: List[Dict], truth: Dict[str, str]) -> Dict[str, str]:
    by_type: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        qid = r["id"]
        if qid in truth:
            qt = r.get("question_type", "unknown")
            by_type[qt][normalize_answer(truth[qid])] += 1
    prior = {}
    for qt, counts in by_type.items():
        prior[qt] = counts.most_common(1)[0][0]
    return prior


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
    parser.add_argument("--test-data-dir", type=Path, required=True,
                        help="Path containing test_manifest.json, test_records.json, test_truth.sealed.json, test_features.json")
    parser.add_argument("--checkpoints-dir", type=Path, required=True,
                        help="Path to trained selector checkpoints")
    parser.add_argument("--delivery-cache", type=Path, required=True,
                        help="Path to delivery_cache.json from Step 1")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to save test evaluation report")
    parser.add_argument("--channel-seed", type=int, default=1701)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 80)
    print("EXP-015: INDEPENDENT BLIND TEST EVALUATION")
    print("=" * 80)

    # 1. Load test dataset
    print(f"Loading test inputs from {args.test_data_dir}...")
    manifest = json.loads((args.test_data_dir / "test_manifest.json").read_text())
    truth_raw = json.loads((args.test_data_dir / "test_truth.sealed.json").read_text())
    truth = {r["id"]: r["answer"] for r in truth_raw}
    records = json.loads((args.test_data_dir / "test_records.json").read_text())
    features_raw = json.loads((args.test_data_dir / "test_features.json").read_text())
    features = {r["id"]: r for r in features_raw}
    scalers = json.loads((args.test_data_dir / "scalers.json").read_text())
    primary_joint_actions = json.loads((args.test_data_dir / "primary_joint_actions.json").read_text())

    n_images = len(manifest)
    print(f"Loaded {n_images} test images, {len(records)} test records.")

    # 2. Load delivery cache
    print(f"Loading delivery cache from {args.delivery_cache}...")
    raw_cache = json.loads(args.delivery_cache.read_text())
    delivery_map = {}
    for k, v in raw_cache.items():
        im, sd, snr, plen = k.split("|")
        delivery_map[(im, int(sd), float(snr), int(plen))] = v
    print(f"Loaded {len(delivery_map):,} channel transmission records.")

    # 3. Load trained model ensemble
    seeds = [7, 17, 27]
    models = load_ensemble(args.checkpoints_dir, seeds, device)
    print(f"Loaded {len(models)} selector models for ensemble inference.")

    # 4. Question prior fallback
    prior_answers = build_question_prior(records, truth)

    # 5. Organize test samples
    samples = {}
    for r in manifest:
        qid = r["id"]
        prior_pred = prior_answers.get(r["question_type"], "yes")
        samples[qid] = {
            "image_id": str(r["image_id"]),
            "question_type": r["question_type"],
            "truth": truth[qid],
            "prior_answer": prior_pred,
            "prior_correct": int(normalize_answer(prior_pred) == normalize_answer(truth[qid])),
            "exp014_primary_joint_action": primary_joint_actions[qid],
            "actions": {},
        }

    for r in records:
        qid = r["id"]
        cell_name = f"{r['budget']}_{r['tier']}"
        action_idx = CELL_NAMES.index(cell_name)
        is_correct = int(normalize_answer(r["prediction"]) == normalize_answer(truth[qid]))
        samples[qid]["actions"][action_idx] = {
            "cell": cell_name,
            "budget": r["budget"],
            "tier": r["tier"],
            "correct": is_correct,
            "image_bytes": r["image_bytes"],
            "actual_visual_tokens": r["actual_visual_tokens"],
            "ldpc_complex_symbols": 510 * math.ceil(r["image_bytes"] / 48),
        }

    # 6. Run forward pass of EXP-015 policy across all test images and SNRs
    mean = np.array(scalers["scalers"]["large"]["mean"], dtype=np.float32)
    std = np.array(scalers["scalers"]["large"]["std"], dtype=np.float32)
    img_mult = 1.0 / math.sqrt(83.0)

    print("Running cross-layer policy network inference across test set...")
    policy_choices: Dict[float, Dict[str, int]] = {snr: {} for snr in SNRS}
    routing_histograms: Dict[float, Dict[str, int]] = {snr: Counter() for snr in SNRS}

    for snr in SNRS:
        snr_norm = (snr - (-5.0)) / 25.0
        x_rows = []
        qids = []
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

        for qid, a_idx in zip(qids, actions):
            policy_choices[snr][qid] = a_idx
            routing_histograms[snr][CELL_NAMES[a_idx]] += 1

    print("\nLearned Policy Routing Behavior across SNR:")
    for snr in SNRS:
        hist_str = ", ".join(f"{k}: {v}" for k, v in sorted(routing_histograms[snr].items()))
        print(f"  SNR {snr:5.1f} dB -> {hist_str}")

    # 7. Evaluate Performance under Mode 2 (Constant Energy Budget)
    mode2_results = {}
    print("\nAggregating test metrics under Mode 2 (Constant Energy Budget)...")

    methods = [
        "fixed_2000_low", "fixed_2000_medium", "fixed_4000_medium", "fixed_8000_high",
        "exp014_blind_joint", "exp015_learned_cross_layer"
    ]

    for snr in SNRS:
        snr_round = round(snr, 2)
        snr_4k = round(snr - 10.0 * math.log10(BUDGET_SYMBOLS[4000] / BUDGET_SYMBOLS[2000]), 2)
        snr_8k = round(snr - 10.0 * math.log10(BUDGET_SYMBOLS[8000] / BUDGET_SYMBOLS[2000]), 2)
        mode2_results[str(snr)] = {}

        for m_name in methods:
            strict_correct = 0
            prior_correct = 0
            delivered_cnt = 0
            total_symbols = 0
            total_tokens = 0

            for qid, s in samples.items():
                if m_name == "fixed_2000_low":
                    a_idx = CELL_NAMES.index("2000_low")
                elif m_name == "fixed_2000_medium":
                    a_idx = CELL_NAMES.index("2000_medium")
                elif m_name == "fixed_4000_medium":
                    a_idx = CELL_NAMES.index("4000_medium")
                elif m_name == "fixed_8000_high":
                    a_idx = CELL_NAMES.index("8000_high")
                elif m_name == "exp014_blind_joint":
                    a_idx = s["exp014_primary_joint_action"]
                elif m_name == "exp015_learned_cross_layer":
                    a_idx = policy_choices[snr][qid]
                else:
                    raise ValueError(m_name)

                act = s["actions"][a_idx]
                b = act["budget"]
                plen = BUDGET_BYTES[b]
                eff_snr = snr_round if b == 2000 else (snr_4k if b == 4000 else snr_8k)

                ok = delivery_map[(s["image_id"], args.channel_seed, eff_snr, plen)]
                total_symbols += act["ldpc_complex_symbols"]
                total_tokens += act["actual_visual_tokens"]

                if ok:
                    delivered_cnt += 1
                    strict_correct += act["correct"]
                    prior_correct += act["correct"]
                else:
                    prior_correct += s["prior_correct"]

            mode2_results[str(snr)][m_name] = {
                "pdr": delivered_cnt / n_images,
                "strict_accuracy": strict_correct / n_images,
                "prior_accuracy": prior_correct / n_images,
                "mean_symbols": total_symbols / n_images,
                "mean_tokens": total_tokens / n_images,
            }

    # 8. Save test report
    report = {
        "experiment_id": "EXP-015",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_test_images": n_images,
        "snrs": SNRS,
        "routing_histograms": {str(k): dict(v) for k, v in routing_histograms.items()},
        "mode2_constant_energy": mode2_results,
    }
    out_file = args.output_dir / "test_evaluation_report.json"
    out_file.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {out_file}")

    # 9. Print formatted comparative table
    print("\n" + "=" * 105)
    print("MODE 2: TEST SET END-TO-END VQA STRICT ACCURACY (%) COMPARISON")
    print("=" * 105)
    header = f"{'SNR (dB)':<9} | {'Fixed 2k_med':<13} | {'Fixed 4k_med':<13} | {'Fixed 8k_high':<13} | {'EXP-014 Blind':<13} | {'EXP-015 Learned':<15} | {'Gain vs 4k':<12}"
    print(header)
    print("-" * 105)
    for snr in SNRS:
        r = mode2_results[str(snr)]
        a2 = r["fixed_2000_medium"]["strict_accuracy"] * 100
        a4 = r["fixed_4000_medium"]["strict_accuracy"] * 100
        a8 = r["fixed_8000_high"]["strict_accuracy"] * 100
        a_exp14 = r["exp014_blind_joint"]["strict_accuracy"] * 100
        a_exp15 = r["exp015_learned_cross_layer"]["strict_accuracy"] * 100
        diff = a_exp15 - a4
        print(f"{snr:<9.1f} | {a2:<13.2f} | {a4:<13.2f} | {a8:<13.2f} | {a_exp14:<13.2f} | {a_exp15:<15.2f} | {diff:>+11.2f}%")
    print("=" * 105)


if __name__ == "__main__":
    main()
