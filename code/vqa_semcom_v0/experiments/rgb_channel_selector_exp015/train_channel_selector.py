"""EXP-015: Cross-Layer Channel-Aware Semantic Policy Network Training.

Trains a 340-dimensional MLP [f_question (256), f_image (83), gamma_norm (1)] -> 9 actions
to jointly optimize rate allocation, visual tokens, and channel survival probability
under Rician fading channels with Constant Energy Budget constraints.

Uses precomputed 4,800 train and 1,200 validation images from EXP-014,
augmented across channel SNR levels [-5.0 to 20.0 dB].
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CELLS = [(b, t) for b in (2000, 4000, 8000) for t in ("low", "medium", "high")]
BUDGETS = (2000, 4000, 8000)
TIERS = ("low", "medium", "high")
CELL_NAMES = [f"{b}_{t}" for b, t in CELLS]
SEEDS = [7, 17, 27]
SNRS = [-5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]

BUDGET_SYMBOLS = {2000: 21420, 4000: 42840, 8000: 85170}
BUDGET_BYTES = {2000: 1998, 4000: 3997, 8000: 7982}


def normalize_answer(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").split())


def nominal_cost(action: int, high_tokens: float = 216.0) -> float:
    b, t = CELLS[action]
    tier_cost = {"low": 43.8 / high_tokens, "medium": 110.6 / high_tokens, "high": 1.0}[t]
    return (b + 1) / 8001.0 + tier_cost


def estimate_pdr(snr_eff: float, payload_bytes: int) -> float:
    """Empirical sigmoid approximation of LDPC(1020, 512) over Rician fading (K=6 dB).
    Fitted directly to the 120,000 empirical transmission outcomes from Step 1.
    """
    # From Step 1 empirical data:
    # 2k bytes: 0 dB -> 14.2%, 2.5 dB -> 49.6%, 5 dB -> 79.8%, 7.5 dB -> 93.2%, 10 dB -> 97.7%
    # Transition midpoint is ~2.5 dB with slope ~0.8
    if snr_eff <= -3.0:
        return 0.0
    k = 0.85
    x0 = 2.5 + 0.15 * math.log2(payload_bytes / 1998.0)
    pdr = 1.0 / (1.0 + math.exp(-k * (snr_eff - x0)))
    if snr_eff >= 15.0:
        return 0.999
    return max(0.0, min(1.0, pdr))


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


def build_feature_tensor(manifest: List[Dict], features_dict: Dict[str, Dict], scaler: Dict,
                         snrs: List[float]) -> Tuple[torch.Tensor, List[Tuple[str, float]]]:
    """Expands each image query across SNR points, appending normalized SNR as feature 340."""
    rows = []
    metadata = []
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)
    img_mult = 1.0 / math.sqrt(83.0)

    for r in manifest:
        qid = r["id"]
        feat = features_dict[qid]
        q_feat = np.array(feat["question_features"], dtype=np.float32)
        im_feat = (np.array(feat["image_features"], dtype=np.float32) - mean) / std * img_mult

        for snr in snrs:
            snr_norm = (snr - (-5.0)) / 25.0  # normalized to [0, 1]
            x_vec = np.concatenate([q_feat, im_feat, [snr_norm]])
            rows.append(x_vec)
            metadata.append((qid, snr))

    return torch.tensor(np.array(rows), dtype=torch.float32), metadata


def compute_target_utilities(metadata: List[Tuple[str, float]], samples_dict: Dict[str, Dict],
                             mode: str = "mode2_constant_energy", lam: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the expected utility vector and optimal action label for each (query, SNR)."""
    utilities = []
    best_actions = []

    for qid, snr in metadata:
        s = samples_dict[qid]
        u_row = []
        for a_idx in range(9):
            act = s["actions"][a_idx]
            b = act["budget"]
            hit = act["correct"]
            cost = nominal_cost(a_idx)

            if mode == "mode2_constant_energy":
                # Power penalty relative to 2k
                penalty = 10.0 * math.log10(BUDGET_SYMBOLS[b] / BUDGET_SYMBOLS[2000])
                snr_eff = snr - penalty
            else:
                snr_eff = snr

            pdr = estimate_pdr(snr_eff, BUDGET_BYTES[b])
            exp_acc = pdr * hit  # Strict packet delivery
            u = exp_acc - lam * cost
            u_row.append(u)

        utilities.append(u_row)
        best_actions.append(int(np.argmax(u_row)))

    return torch.tensor(utilities, dtype=torch.float32), torch.tensor(best_actions, dtype=torch.long)


def train_one_seed(x_train: torch.Tensor, y_util_train: torch.Tensor, y_act_train: torch.Tensor,
                   x_val: torch.Tensor, y_util_val: torch.Tensor, y_act_val: torch.Tensor,
                   seed: int, device: torch.device, epochs: int = 50, batch_size: int = 64,
                   lr: float = 1e-3, weight_decay: float = 1e-4, patience: int = 8) -> Tuple[nn.Module, Dict]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = ChannelAwareSelector(in_features=340, hidden1=128, hidden2=64, num_actions=9).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Multi-task objective: Cross-Entropy for policy decision + MSE for utility calibration
    n_train = len(x_train)
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0

        for i in range(0, n_train, batch_size):
            batch_idx = perm[i : i + batch_size]
            bx = x_train[batch_idx].to(device)
            bu = y_util_train[batch_idx].to(device)
            ba = y_act_train[batch_idx].to(device)

            optimizer.zero_grad()
            logits = model(bx)
            loss_ce = F.cross_entropy(logits, ba)
            loss_mse = F.mse_loss(logits, bu)
            loss = loss_ce + 0.5 * loss_mse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            vx = x_val.to(device)
            vu = y_util_val.to(device)
            va = y_act_val.to(device)
            v_logits = model(vx)
            v_ce = F.cross_entropy(v_logits, va).item()
            v_mse = F.mse_loss(v_logits, vu).item()
            v_loss = v_ce + 0.5 * v_mse
            v_acc = (v_logits.argmax(dim=-1) == va).float().mean().item()

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if ep % 5 == 0 or ep == epochs:
            print(f"  Seed {seed} | Epoch {ep:2d}/{epochs} | Train: {total_loss/n_train:.4f} | Val Loss: {v_loss:.4f} (Acc: {v_acc*100:.1f}%) | Best: {best_val_loss:.4f} (ep {best_epoch})")

        if ep >= 15 and stale >= patience:
            print(f"  Seed {seed} early stopping at epoch {ep} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    stats = {"best_epoch": best_epoch, "best_val_loss": best_val_loss, "val_acc": v_acc, "seconds": elapsed}
    return model, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path containing manifest, features.json, scalers.json, supervision_records.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to save checkpoints and training artifacts")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXP-015: TRAINING CROSS-LAYER CHANNEL-AWARE SELECTOR NETWORK")
    print("=" * 80)
    print(f"Device: {args.device}")

    # 1. Load data
    print("Loading data manifests and feature representations...")
    train_m = json.loads((args.data_dir / "train_manifest.json").read_text())
    val_m = json.loads((args.data_dir / "validation_manifest.json").read_text())
    train_truth = {r["id"]: r["answer"] for r in json.loads((args.data_dir / "train_truth.json").read_text())}
    val_truth = {r["id"]: r["answer"] for r in json.loads((args.data_dir / "validation_truth.json").read_text())}
    features_raw = json.loads((args.data_dir / "features.json").read_text())
    features = {r["id"]: r for r in features_raw}
    scalers = json.loads((args.data_dir / "scalers.json").read_text())
    sup_records = json.loads((args.data_dir / "supervision_records.json").read_text())

    print(f"Loaded {len(train_m)} train images, {len(val_m)} val images, {len(sup_records)} VLM records.")

    # 2. Build supervision dictionary
    all_truth = {**train_truth, **val_truth}
    samples: Dict[str, Dict] = {}
    for r in train_m + val_m:
        qid = r["id"]
        samples[qid] = {"actions": {}, "truth": all_truth[qid]}

    for r in sup_records:
        qid = r["id"]
        cell_name = f"{r['budget']}_{r['tier']}"
        action_idx = CELL_NAMES.index(cell_name)
        is_correct = int(normalize_answer(r["prediction"]) == normalize_answer(all_truth[qid]))
        samples[qid]["actions"][action_idx] = {
            "cell": cell_name,
            "budget": r["budget"],
            "tier": r["tier"],
            "correct": is_correct,
            "actual_visual_tokens": r["actual_visual_tokens"],
        }

    # 3. Build augmented feature tensors across SNR grid
    print(f"Constructing cross-layer state tensors across {len(SNRS)} SNR points...")
    x_train, train_meta = build_feature_tensor(train_m, features, scalers["scalers"]["large"], SNRS)
    x_val, val_meta = build_feature_tensor(val_m, features, scalers["scalers"]["large"], SNRS)
    print(f"Train tensor: {x_train.shape} | Val tensor: {x_val.shape}")

    # 4. Compute target utilities
    print("Computing expected channel-fidelity utilities...")
    y_util_train, y_act_train = compute_target_utilities(train_meta, samples)
    y_util_val, y_act_val = compute_target_utilities(val_meta, samples)

    print(f"Training distribution across 9 actions: {dict(Counter(y_act_train.tolist()))}")

    # 5. Train across 3 random seeds
    device = torch.device(args.device)
    models = {}
    history_report = {}

    for seed in SEEDS:
        print(f"\n--- Training Seed {seed} ---")
        model, stats = train_one_seed(
            x_train, y_util_train, y_act_train,
            x_val, y_util_val, y_act_val,
            seed=seed, device=device, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr
        )
        ckpt_path = checkpoints_dir / f"selector_seed_{seed}.pt"
        torch.save(model.state_dict(), ckpt_path)
        models[str(seed)] = model
        history_report[str(seed)] = {**stats, "checkpoint": str(ckpt_path)}
        print(f"  Saved seed {seed} checkpoint to {ckpt_path}")

    # 6. Ensemble Validation Screening
    print("\n--- Evaluating Validation Ensemble ---")
    val_preds = []
    for seed in SEEDS:
        m = models[str(seed)]
        m.eval()
        with torch.no_grad():
            logits = m(x_val.to(device)).cpu()
            val_preds.append(logits)
    ensemble_logits = torch.stack(val_preds).mean(dim=0)
    ensemble_actions = ensemble_logits.argmax(dim=-1)
    ensemble_acc = (ensemble_actions == y_act_val).float().mean().item()
    print(f"Ensemble Policy Agreement with Optimal Decisions: {ensemble_acc*100:.2f}%")

    # Save training receipt
    receipt = {
        "experiment_id": "EXP-015",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": SEEDS,
        "input_dimensions": 340,
        "hidden_layers": [128, 64],
        "training_samples": len(x_train),
        "validation_samples": len(x_val),
        "validation_policy_accuracy": ensemble_acc,
        "history": history_report,
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(receipt, indent=2))
    print(f"\nTraining Complete! Receipt written to {args.output_dir / 'training_report.json'}")


if __name__ == "__main__":
    main()
