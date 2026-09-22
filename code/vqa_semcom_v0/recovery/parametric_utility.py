"""Parametric, learnable semantic-utility model.

The discrete look-up table (:class:`SemanticUtilityModel`) stores one Wilson
interval per (question, service, snr, view, freshness, risk) cell.  That makes
sparse cells (n=4) very wide and forces nearest-neighbour fall-back for any SNR
that was not enumerated on the calibration grid.

This module replaces the table with a **Bayesian logistic-regression** surface
fitted to the same per-cell binomial observations:

    logit P(correct) = w . phi(key)

* ``phi`` encodes the key with an *ordinal/continuous* SNR feature, so the model
  generalises to off-grid SNR (e.g. 7 dB) and interpolates smoothly.
* The posterior is approximated with a Laplace (Gaussian) approximation around
  the MAP weights, giving a predictive logit variance per query.  We turn that
  into a principled lower-confidence bound (LCB) on accuracy that *borrows
  strength* across cells -- tight where data is dense, wide where it is sparse.

:class:`ParametricSemanticUtilityModel` subclasses :class:`SemanticUtilityModel`
and only overrides the per-cell estimators, so ``path_utility`` /
``get_service_candidates`` and the rest of the RL-facing API work unchanged.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from vqa_semcom.config import ensure_parent
from vqa_semcom.snr import snr_bin_label, snr_db_from_label
from vqa_semcom.semantic.utility import (
    SemanticUtilityCell,
    SemanticUtilityEstimate,
    SemanticUtilityModel,
    read_semantic_utility_csv,
)


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

VIEW_ORDINAL = {"poor": -1.0, "medium": 0.0, "good": 1.0}
FRESH_ORDINAL = {"expired": -1.0, "stale": 0.0, "fresh": 1.0}

FEATURE_NAMES = [
    "intercept",
    "q_counting",
    "svc_token",
    "svc_image",
    "snr_z",
    "snr_z:token",
    "snr_z:image",
    "view_ord",
    "fresh_ord",
    "risk_critical",
    "q_counting:token",
    "q_counting:image",
    "view_ord:image",
]


@dataclass(frozen=True)
class FeatureScaler:
    """Standardisation parameters for the continuous SNR feature."""

    snr_mean: float
    snr_std: float

    def z(self, snr_db: float) -> float:
        if self.snr_std <= 1e-9:
            return 0.0
        return (snr_db - self.snr_mean) / self.snr_std


def encode_key(
    question_type: str,
    service_level: int,
    snr_db: float,
    view_quality_bin: str,
    freshness_bin: str,
    risk_level: str,
    scaler: FeatureScaler,
) -> np.ndarray:
    """Map a LUT key to the fixed-length feature vector ``phi``."""

    q_counting = 1.0 if question_type == "counting" else 0.0
    svc_token = 1.0 if int(service_level) == 1 else 0.0
    svc_image = 1.0 if int(service_level) >= 2 else 0.0
    snr_z = scaler.z(float(snr_db))
    view_ord = VIEW_ORDINAL.get(view_quality_bin, 0.0)
    fresh_ord = FRESH_ORDINAL.get(freshness_bin, 0.0)
    risk_critical = 1.0 if risk_level == "critical" else 0.0
    return np.array(
        [
            1.0,
            q_counting,
            svc_token,
            svc_image,
            snr_z,
            snr_z * svc_token,
            snr_z * svc_image,
            view_ord,
            fresh_ord,
            risk_critical,
            q_counting * svc_token,
            q_counting * svc_image,
            view_ord * svc_image,
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Bayesian logistic regression (IRLS MAP + Laplace covariance)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


@dataclass
class BayesianLogistic:
    """MAP weights + Laplace posterior covariance for binomial observations."""

    weights: np.ndarray
    covariance: np.ndarray
    prior_var: float
    n_features: int

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        successes: np.ndarray,
        totals: np.ndarray,
        prior_var: float = 25.0,
        max_iter: int = 200,
        tol: float = 1e-8,
    ) -> "BayesianLogistic":
        """Fit ``logit p = X w`` with a Gaussian prior ``w ~ N(0, prior_var I)``.

        ``successes`` / ``totals`` are per-row binomial counts.  Optimisation is
        Newton/IRLS; the Hessian at the optimum is reused as the Laplace
        precision so the posterior covariance is ``(X^T S X + Lambda)^-1``.
        """

        X = np.asarray(X, dtype=float)
        k = np.asarray(successes, dtype=float)
        n = np.asarray(totals, dtype=float)
        n_rows, n_feat = X.shape
        # Do not shrink the intercept towards zero.
        prior_prec = np.eye(n_feat) / float(prior_var)
        prior_prec[0, 0] = 0.0
        w = np.zeros(n_feat, dtype=float)

        for _ in range(max_iter):
            eta = X @ w
            p = _sigmoid(eta)
            # Gradient of (log-lik - 0.5 w^T Lambda w)
            grad = X.T @ (k - n * p) - prior_prec @ w
            s = n * p * (1.0 - p)
            # Hessian (negative): X^T diag(s) X + Lambda
            hessian = (X.T * s) @ X + prior_prec
            # Guard against singular Hessian from all-saturated weights.
            hessian += np.eye(n_feat) * 1e-9
            step = np.linalg.solve(hessian, grad)
            w = w + step
            if np.max(np.abs(step)) < tol:
                break

        eta = X @ w
        p = _sigmoid(eta)
        s = n * p * (1.0 - p)
        hessian = (X.T * s) @ X + prior_prec + np.eye(n_feat) * 1e-9
        covariance = np.linalg.inv(hessian)
        return cls(weights=w, covariance=covariance, prior_var=float(prior_var), n_features=n_feat)

    def predict_logit(self, x: np.ndarray) -> tuple[float, float]:
        """Return (logit mean, logit variance) for a single feature row."""

        x = np.asarray(x, dtype=float)
        eta = float(x @ self.weights)
        var = float(x @ self.covariance @ x)
        return eta, max(0.0, var)

    def predict(self, x: np.ndarray, z: float = 1.96) -> tuple[float, float, float, float]:
        """Return (mean, lcb, ucb, logit_std) probabilities for one query.

        The mean marginalises the logit-normal posterior via MacKay's probit
        approximation; the bounds propagate +/- z logit-std through the link.
        """

        eta, var = self.predict_logit(x)
        std = math.sqrt(var)
        kappa = 1.0 / math.sqrt(1.0 + math.pi * var / 8.0)
        mean = float(_sigmoid(eta * kappa))
        lcb = float(_sigmoid(eta - z * std))
        ucb = float(_sigmoid(eta + z * std))
        return mean, lcb, ucb, std


# ---------------------------------------------------------------------------
# Parametric utility model (drop-in for SemanticUtilityModel)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParametricConfig:
    prior_var: float = 25.0
    z: float = 1.96
    cache_snr_invariant: bool = True


class ParametricSemanticUtilityModel(SemanticUtilityModel):
    """Learnable semantic-utility surface with Laplace predictive uncertainty.

    Keeps the discrete cells for payload look-up and effective sample counts,
    but replaces the accuracy / LCB / uncertainty estimates with the fitted
    Bayesian logistic-regression surface.
    """

    def __init__(
        self,
        cells: Iterable[SemanticUtilityCell],
        model: BayesianLogistic,
        scaler: FeatureScaler,
        config: ParametricConfig | None = None,
    ):
        super().__init__(cells)
        self.model = model
        self.scaler = scaler
        self.config = config or ParametricConfig()
        # Effective sample count per non-SNR group, for RL down-weighting.
        self._group_samples: dict[tuple[str, int, str, str, str], int] = {}
        for cell in self.cells:
            g = (cell.question_type, cell.service_level, cell.view_quality_bin, cell.freshness_bin, cell.risk_level)
            self._group_samples[g] = self._group_samples.get(g, 0) + cell.sample_count

    # -- fitting ---------------------------------------------------------
    @classmethod
    def fit_from_cells(
        cls,
        cells: Sequence[SemanticUtilityCell],
        config: ParametricConfig | None = None,
        use_raw: bool = True,
    ) -> "ParametricSemanticUtilityModel":
        cfg = config or ParametricConfig()
        cells = list(cells)
        snr_values = [snr_db_from_label(c.snr_bin) for c in cells]
        scaler = FeatureScaler(
            snr_mean=float(np.mean(snr_values)) if snr_values else 0.0,
            snr_std=float(np.std(snr_values)) if snr_values else 1.0,
        )
        X, k, n = _design_from_cells(cells, scaler, use_raw=use_raw)
        model = BayesianLogistic.fit(X, k, n, prior_var=cfg.prior_var)
        return cls(cells, model, scaler, cfg)

    @classmethod
    def fit_from_csv(
        cls,
        path: Path,
        config: ParametricConfig | None = None,
        use_raw: bool = True,
    ) -> "ParametricSemanticUtilityModel":
        return cls.fit_from_cells(read_semantic_utility_csv(path), config=config, use_raw=use_raw)

    # -- prediction ------------------------------------------------------
    def _predict(
        self,
        question_type: str,
        service_level: int,
        snr_db: float,
        view_quality_bin: str,
        freshness_bin: str,
        risk_level: str,
    ) -> tuple[float, float, float, float]:
        x = encode_key(
            question_type,
            int(service_level),
            float(snr_db),
            view_quality_bin,
            freshness_bin,
            risk_level,
            self.scaler,
        )
        return self.model.predict(x, z=self.config.z)

    def _effective_samples(self, question_type: str, service_level: int, view: str, fresh: str, risk: str) -> int:
        return self._group_samples.get((question_type, int(service_level), view, fresh, risk), 0)

    def _estimate(
        self,
        task_type: str,
        service_level: int,
        snr_bin: str | int | float,
        view_quality_bin: str,
        freshness_bin: str,
        risk_level: str,
    ) -> SemanticUtilityEstimate:
        if isinstance(snr_bin, (int, float)):
            snr_db = float(snr_bin)
            label = snr_bin_label(snr_bin)
        else:
            label = str(snr_bin)
            try:
                snr_db = snr_db_from_label(label)
            except ValueError:
                snr_db = self.scaler.snr_mean
        sl = int(service_level)
        if sl == 0 and self.config.cache_snr_invariant:
            snr_db = self.scaler.snr_mean  # cache answers are SNR-invariant
        mean, lcb, ucb, _std = self._predict(task_type, sl, snr_db, view_quality_bin, freshness_bin, risk_level)
        # Payload is deterministic per cell; reuse the empirical table.
        cell = self.table.get((task_type, sl, label, view_quality_bin, freshness_bin, risk_level))
        payload_kb = cell.payload_kb if cell is not None else self.by_service_payload.get(sl, 0.0)
        uncertainty = round(min(1.0, max(0.0, (ucb - lcb) / 2.0)), 6)
        samples = cell.sample_count if cell is not None else self._effective_samples(
            task_type, sl, view_quality_bin, freshness_bin, risk_level
        )
        return SemanticUtilityEstimate(
            accuracy_mean=round(mean, 6),
            accuracy_lcb=round(lcb, 6),
            payload_kb=round(payload_kb, 6),
            uncertainty=uncertainty,
            sample_count=int(samples),
        )

    # -- overrides used by the base RL-facing API ------------------------
    def U_sem(
        self,
        task_type: str,
        service_level: int,
        snr_bin: str | int | float,
        view_quality_bin: str,
        freshness_bin: str,
        risk_level: str,
    ) -> SemanticUtilityEstimate:
        return self._estimate(task_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level)

    def _cache_estimate(
        self,
        task_type: str,
        snr_bin: str | int | float,
        view_quality_bin: str,
        freshness_bin: str,
        risk_level: str,
    ) -> SemanticUtilityEstimate:
        return self._estimate(task_type, 0, snr_bin, view_quality_bin, freshness_bin, risk_level)

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "bayesian_logistic_semantic_utility",
            "feature_names": FEATURE_NAMES,
            "weights": self.model.weights.tolist(),
            "covariance": self.model.covariance.tolist(),
            "prior_var": self.model.prior_var,
            "scaler": {"snr_mean": self.scaler.snr_mean, "snr_std": self.scaler.snr_std},
            "config": {
                "prior_var": self.config.prior_var,
                "z": self.config.z,
                "cache_snr_invariant": self.config.cache_snr_invariant,
            },
        }

    def save_json(self, path: Path) -> None:
        ensure_parent(path)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, model_path: Path, cells_path: Path) -> "ParametricSemanticUtilityModel":
        data = json.loads(Path(model_path).read_text(encoding="utf-8"))
        model = BayesianLogistic(
            weights=np.array(data["weights"], dtype=float),
            covariance=np.array(data["covariance"], dtype=float),
            prior_var=float(data.get("prior_var", 25.0)),
            n_features=len(data["weights"]),
        )
        scaler = FeatureScaler(**data["scaler"])
        cfg_raw = data.get("config", {})
        cfg = ParametricConfig(
            prior_var=float(cfg_raw.get("prior_var", 25.0)),
            z=float(cfg_raw.get("z", 1.96)),
            cache_snr_invariant=bool(cfg_raw.get("cache_snr_invariant", True)),
        )
        return cls(read_semantic_utility_csv(cells_path), model, scaler, cfg)

    def coefficients(self) -> dict[str, float]:
        return {name: float(w) for name, w in zip(FEATURE_NAMES, self.model.weights)}


def _design_from_cells(
    cells: Sequence[SemanticUtilityCell],
    scaler: FeatureScaler,
    use_raw: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, successes, totals) binomial design from LUT cells."""

    rows: list[np.ndarray] = []
    successes: list[float] = []
    totals: list[float] = []
    for cell in cells:
        n = int(cell.sample_count)
        if n <= 0:
            continue
        mean = cell.raw_accuracy_mean if use_raw else cell.accuracy_mean
        k = round(float(mean) * n)
        snr_db = snr_db_from_label(cell.snr_bin)
        rows.append(
            encode_key(
                cell.question_type,
                cell.service_level,
                snr_db,
                cell.view_quality_bin,
                cell.freshness_bin,
                cell.risk_level,
                scaler,
            )
        )
        successes.append(float(k))
        totals.append(float(n))
    if not rows:
        raise ValueError("no usable cells to fit the parametric utility model")
    return np.vstack(rows), np.array(successes), np.array(totals)
