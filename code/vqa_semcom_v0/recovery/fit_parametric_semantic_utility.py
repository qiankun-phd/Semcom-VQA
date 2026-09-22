"""Fit and evaluate the parametric semantic-utility model.

Fits the Bayesian logistic surface on the V1.9 LUT cells and evaluates it
against the discrete look-up table in two regimes:

1. **Random k-fold** -- in-grid interpolation.  Here the table is hard to beat
   on point accuracy because the fully-crossed 6-level SNR grid leaves a near
   twin of every held-out cell in the training fold (it memorises).  We report
   it for completeness and to calibrate the lower-confidence bound (LCB).
2. **Leave-one-SNR-out** -- SNR extrapolation.  An entire SNR level is removed,
   so the table can only nearest-neighbour to an adjacent level while the
   parametric model interpolates.  This is the regime the parametric model is
   built for.

The LCB is made *honest* with a structural residual variance estimated from
out-of-fold residuals (it absorbs the latent ``channel_bin`` the public API
marginalises out) on top of the quasi-binomial parameter variance.

Outputs:
* ``outputs/lut/v1_9_parametric_utility_model.json`` -- the fitted model,
* ``outputs/reports/parametric_semantic_utility.md`` -- the evaluation report.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from vqa_semcom.snr import snr_db_from_label
from vqa_semcom.semantic.utility import read_semantic_utility_csv
from vqa_semcom.semantic.parametric_utility import (
    BayesianLogistic,
    FeatureScaler,
    ParametricConfig,
    ParametricSemanticUtilityModel,
    encode_key,
    _design_from_cells,
)


def _clip01(x: float) -> float:
    return max(1e-4, min(1.0 - 1e-4, x))


def _logit(p: float) -> float:
    p = _clip01(p)
    return math.log(p / (1.0 - p))


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, x))))


def _logloss(k: float, n: float, p: float) -> float:
    p = _clip01(p)
    return -(k * math.log(p) + (n - k) * math.log(1.0 - p))


def _nearest_table_pred(train_cells, cell):
    """Discrete-LUT baseline: nearest-SNR cell in the same non-SNR group."""
    target = snr_db_from_label(cell.snr_bin)
    group = [
        c for c in train_cells
        if c.question_type == cell.question_type and c.service_level == cell.service_level
        and c.view_quality_bin == cell.view_quality_bin and c.freshness_bin == cell.freshness_bin
        and c.risk_level == cell.risk_level
    ]
    if group:
        best = min(group, key=lambda c: abs(snr_db_from_label(c.snr_bin) - target))
        return best.accuracy_mean, best.accuracy_lcb
    same = [c for c in train_cells if c.service_level == cell.service_level]
    if same:
        return float(np.mean([c.accuracy_mean for c in same])), 0.0
    return 0.5, 0.0


def _fit_fold(train, cfg):
    snr_values = [snr_db_from_label(c.snr_bin) for c in train]
    scaler = FeatureScaler(float(np.mean(snr_values)), float(np.std(snr_values)) or 1.0)
    X, k, n = _design_from_cells(train, scaler, use_raw=True)
    model = BayesianLogistic.fit(X, k, n, prior_var=cfg.prior_var)
    return model, scaler


def _record(model, scaler, cell, cfg):
    n_i = float(cell.sample_count)
    obs = float(cell.raw_accuracy_mean)
    k_i = round(obs * n_i)
    snr_db = scaler.snr_mean if (cell.service_level == 0 and cfg.cache_snr_invariant) else snr_db_from_label(cell.snr_bin)
    x = encode_key(cell.question_type, cell.service_level, snr_db, cell.view_quality_bin, cell.freshness_bin, cell.risk_level, scaler)
    eta, pvar = model.predict_logit(x)  # pvar already includes residual if set
    p = _sig(eta)
    return {"eta": eta, "pvar": pvar, "p": p, "obs": obs, "n": n_i, "k": k_i}


def random_kfold(cells, cfg, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cells)); rng.shuffle(idx)
    fold_of = {int(i): int(f) for f, ch in enumerate(np.array_split(idx, folds)) for i in ch}
    recs, table = [], []
    for f in range(folds):
        train = [c for i, c in enumerate(cells) if fold_of[i] != f]
        test = [c for i, c in enumerate(cells) if fold_of[i] == f]
        if not train or not test:
            continue
        model, scaler = _fit_fold(train, cfg)  # residual_logit_var = 0 here
        for cell in test:
            if cell.sample_count <= 0:
                continue
            recs.append(_record(model, scaler, cell, cfg))
            tm, tl = _nearest_table_pred(train, cell)
            table.append((tm, tl))
    return recs, table


def estimate_residual_logit_var(recs):
    """Structural logit variance = E[residual^2 - binomial sampling var]_+."""
    num = den = 0.0
    for r in recs:
        resid = _logit(r["obs"]) - r["eta"]
        samp = 1.0 / max(1e-6, r["n"] * r["p"] * (1.0 - r["p"]))
        struct = resid * resid - samp
        num += r["n"] * struct
        den += r["n"]
    return max(0.0, num / max(1e-9, den))


def metrics(recs, table, resid_var, z=1.96):
    pb = pl = cov = w = 0.0
    tb = tl = tcov = 0.0
    rel_p, rel_o, rel_w = [], [], []
    for r, (tm, tl_lcb) in zip(recs, table):
        var = r["pvar"] + resid_var
        std = math.sqrt(var)
        eta = r["eta"]
        kappa = 1.0 / math.sqrt(1.0 + math.pi * var / 8.0)
        pmean = _sig(eta * kappa)
        lcb = _sig(eta - z * std)
        n, obs, k = r["n"], r["obs"], r["k"]
        pb += n * (pmean - obs) ** 2; pl += _logloss(k, n, pmean); cov += n * (1 if obs >= lcb else 0); w += n
        tb += n * (tm - obs) ** 2; tl += _logloss(k, n, tm); tcov += n * (1 if obs >= tl_lcb else 0)
        rel_p.append(pmean); rel_o.append(obs); rel_w.append(n)
    return {
        "param_brier": pb / w, "param_logloss": pl / w, "param_cov": cov / w,
        "table_brier": tb / w, "table_logloss": tl / w, "table_cov": tcov / w,
        "rel": (np.array(rel_p), np.array(rel_o), np.array(rel_w)),
    }


def leave_one_snr_out(cells, cfg, resid_var, z=1.96):
    snr_levels = sorted({c.snr_bin for c in cells}, key=snr_db_from_label)
    out = []
    for L in snr_levels:
        train = [c for c in cells if c.snr_bin != L]
        test = [c for c in cells if c.snr_bin == L and c.service_level != 0]  # cache is SNR-invariant
        if not test:
            continue
        model, scaler = _fit_fold(train, cfg)
        model.residual_logit_var = resid_var
        pb = tb = w = pcov = 0.0
        for cell in test:
            r = _record(model, scaler, cell, cfg)
            var = r["pvar"] + resid_var; std = math.sqrt(var)
            kappa = 1.0 / math.sqrt(1.0 + math.pi * var / 8.0)
            pmean = _sig(r["eta"] * kappa); lcb = _sig(r["eta"] - z * std)
            tm, _tl = _nearest_table_pred(train, cell)
            n, obs = r["n"], r["obs"]
            pb += n * (pmean - obs) ** 2; tb += n * (tm - obs) ** 2; w += n
            pcov += n * (1 if obs >= lcb else 0)
        out.append((L, pb / w, tb / w, pcov / w))
    return out


def reliability_table(pred, obs, w, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1); rows = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (pred >= lo) & (pred < hi if b < bins - 1 else pred <= hi)
        if not np.any(m):
            continue
        ww = w[m]
        rows.append((lo, hi, float(np.average(pred[m], weights=ww)), float(np.average(obs[m], weights=ww)), int(ww.sum())))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lut", default="outputs/lut/v1_9_semantic_utility_with_ci.csv")
    ap.add_argument("--model-out", default="outputs/lut/v1_9_parametric_utility_model.json")
    ap.add_argument("--report-out", default="outputs/reports/parametric_semantic_utility.md")
    ap.add_argument("--prior-var", type=float, default=25.0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    cells = read_semantic_utility_csv(Path(args.lut))
    cfg = ParametricConfig(prior_var=args.prior_var)

    # 1) Out-of-fold pass to estimate honest structural residual variance.
    recs, table = random_kfold(cells, cfg, folds=args.folds)
    resid_var = estimate_residual_logit_var(recs)

    # 2) Fit the shipped model and bake in the calibrated residual variance.
    model = ParametricSemanticUtilityModel.fit_from_cells(cells, config=cfg, use_raw=True)
    model.model.residual_logit_var = resid_var
    model.save_json(Path(args.model_out))

    # 3) Metrics with the calibrated bound.
    m = metrics(recs, table, resid_var, z=cfg.z)
    rel = reliability_table(*m["rel"])
    snr_out = leave_one_snr_out(cells, cfg, resid_var, z=cfg.z)

    L = []
    L.append("# Parametric Semantic-Utility Model -- Calibration Report\n")
    L.append("Bayesian logistic regression (IRLS MAP + Laplace covariance, quasi-binomial\n"
             "dispersion + out-of-fold structural residual variance) fitted to the V1.9 LUT\n"
             "binomial cells, replacing the discrete table + nearest-SNR fallback.\n")
    L.append(f"- LUT cells: `{len(cells)}`  |  total VQA samples: `{int(sum(c.sample_count for c in cells))}`")
    L.append(f"- Features ({len(model.model.weights)}): `{', '.join(model.coefficients().keys())}`")
    L.append(f"- Prior `w~N(0,{args.prior_var})` (intercept free); dispersion `phi={model.model.dispersion:.2f}`; "
             f"structural residual logit-var `s2={resid_var:.4f}`; z=`{cfg.z}`\n")

    L.append(f"## A. Random {args.folds}-fold (in-grid interpolation)\n")
    L.append("Table memorises the dense grid here -- reported for calibration, not as the win.\n")
    L.append("| metric | parametric | discrete LUT |")
    L.append("|---|---|---|")
    L.append(f"| Brier (mean) | {m['param_brier']:.4f} | {m['table_brier']:.4f} |")
    L.append(f"| log-loss / sample | {m['param_logloss']:.4f} | {m['table_logloss']:.4f} |")
    L.append(f"| LCB coverage (target {cfg.z:.2f}sd ~ 0.975) | {m['param_cov']:.3f} | {m['table_cov']:.3f} |")
    L.append("")

    L.append("## B. Leave-one-SNR-out (SNR extrapolation -- the intended regime)\n")
    L.append("Entire SNR level removed from training; table can only nearest-neighbour.\n")
    L.append("| held-out SNR | parametric Brier | table Brier | better | param LCB cov |")
    L.append("|---|---|---|---|---|")
    pim = tim = 0.0
    for lvl, pbr, tbr, pcov in snr_out:
        pim += pbr; tim += tbr
        L.append(f"| {lvl} | {pbr:.4f} | {tbr:.4f} | {'parametric' if pbr < tbr else 'table'} | {pcov:.3f} |")
    if snr_out:
        L.append(f"| **mean** | **{pim/len(snr_out):.4f}** | **{tim/len(snr_out):.4f}** | "
                 f"**{'parametric' if pim < tim else 'table'}** | -- |")
    L.append("")

    L.append("## C. Reliability (predicted vs. observed, sample-weighted)\n")
    L.append("| pred bin | mean predicted | mean observed | samples |")
    L.append("|---|---|---|---|")
    for lo, hi, mp, mo, ww in rel:
        L.append(f"| [{lo:.1f},{hi:.1f}] | {mp:.3f} | {mo:.3f} | {ww} |")
    L.append("")

    L.append("## D. Fitted coefficients (log-odds)\n")
    L.append("| feature | weight |")
    L.append("|---|---|")
    for name, wv in model.coefficients().items():
        L.append(f"| `{name}` | {wv:+.3f} |")
    L.append("")

    L.append("## E. Off-grid SNR generalisation (the table cannot interpolate)\n")
    L.append("| question | service | SNR(dB) | mean | LCB | uncertainty |")
    L.append("|---|---|---|---|---|---|")
    for q in ("presence", "counting"):
        for sl in (1, 2):
            for snr in (-2.0, 7.0, 12.0, 18.0):
                est = model.U_sem(q, sl, snr, "medium", "fresh", "normal")
                L.append(f"| {q} | s{sl} | {snr:+.0f} | {est.accuracy_mean:.3f} | {est.accuracy_lcb:.3f} | {est.uncertainty:.3f} |")
    L.append("")
    L.append(f"Model written to `{args.model_out}`.\n")

    out = Path(args.report_out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[ok] model -> {args.model_out}")
    print(f"[ok] report -> {args.report_out}")
    print(f"[resid] dispersion={model.model.dispersion:.2f} struct_logit_var={resid_var:.4f}")
    print(f"[A in-grid] param Brier={m['param_brier']:.4f} cov={m['param_cov']:.3f} | table Brier={m['table_brier']:.4f} cov={m['table_cov']:.3f}")
    if snr_out:
        print(f"[B snr-extrap] param Brier={pim/len(snr_out):.4f} | table Brier={tim/len(snr_out):.4f}")


if __name__ == "__main__":
    main()
