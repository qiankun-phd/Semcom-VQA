"""M2 baseline: DeepSC-style **analog** image transmission (uncoded JSCC-lite).

Contrast with the digital rate-adaptive path in ``digital_link.py``:

* Digital (M1, = service s2): source-code image to a byte budget set by the
  achievable rate, protect with LDPC.  Below a threshold SNR the code cannot be
  decoded -> *cliff effect* (image lost, VQA falls to chance).
* Analog (M2, this module): map the image pixels directly to channel uses,
  transmit as analog symbols, equalize with CSI at the receiver.  Reconstruction
  noise scales smoothly with 1/(SNR*|h|^2) -> *graceful degradation*, no cliff.

This is the classic "uncoded analog image transmission" baseline used to motivate
DeepJSCC; it needs no training and shares the same per-slot bandwidth budget
(B * tau channel uses) as the digital path, so the comparison is bandwidth-fair.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from vqa_semcom.degradation.digital_link import LinkConfig, build_link_config, sample_power_gain


def analog_bandwidth_compression(h: int, w: int, link_cfg: LinkConfig) -> tuple[float, float]:
    """Return (rho, scale).

    rho = available real dimensions / source real dimensions, where the slot
    offers ``B * tau`` complex channel uses (2 real dims each).  ``scale`` is the
    per-axis resize factor that makes the (downscaled) source fit the channel:
    scale = min(1, sqrt(rho)).
    """
    n_uses = float(link_cfg.bandwidth_hz) * float(link_cfg.tx_time_budget_s)  # complex symbols
    capacity_reals = 2.0 * n_uses
    source_reals = float(h) * float(w) * 3.0
    rho = capacity_reals / max(source_reals, 1.0)
    scale = min(1.0, math.sqrt(rho))
    return rho, scale


def transmit_image_analog(image_bgr: np.ndarray, snr_db: float, link_cfg: LinkConfig,
                          rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Analog (uncoded) transmission of ``image_bgr`` at ``snr_db``.

    Block fading: one power gain |h|^2 per image (slow-fading slot), matching the
    digital path's block-fading slot model.  Returns (reconstructed_image, meta).
    """
    import cv2

    h, w = image_bgr.shape[:2]
    rho, scale = analog_bandwidth_compression(h, w, link_cfg)

    # Analog *source* compression: downscale so #real-values <= channel capacity.
    if scale < 1.0:
        sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        src = cv2.resize(image_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        src = image_bgr

    x = src.astype(np.float64)
    mu = float(x.mean())
    sigma = float(x.std()) + 1e-8
    s = (x - mu) / sigma  # unit-power source symbols

    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    g = float(sample_power_gain(link_cfg.fading.kind, link_cfg.fading.k_factor_db, 1, rng)[0])
    eff = snr_lin * g  # post-equalization (ZF) effective SNR
    noise_var = 1.0 / max(eff, 1e-12)

    noisy = s + rng.standard_normal(s.shape) * math.sqrt(noise_var)
    rec = noisy * sigma + mu
    rec = np.clip(rec, 0.0, 255.0).astype(np.uint8)

    if scale < 1.0:
        rec = cv2.resize(rec, (w, h), interpolation=cv2.INTER_LINEAR)

    eff_snr_db = 10.0 * math.log10(eff + 1e-12)
    meta = {
        "method": "analog",
        "snr_db": float(snr_db),
        "scale": round(scale, 4),
        "rho": round(rho, 5),
        "h_gain": round(g, 4),
        "eff_snr_db": round(eff_snr_db, 2),
        "noise_var": round(noise_var, 5),
        # analog symbols actually used (complex): downscaled pixels / 2
        "channel_uses": int(round((rec.shape[0] * rec.shape[1] * 3 * (scale ** 2)) / 2.0)),
    }
    return rec, meta


def transmit_image_analog_to_path(image_path, out_path, snr_db: float, cfg: dict,
                                  rng: np.random.Generator | None = None) -> dict:
    """Read image at ``image_path``, transmit via analog link, write to ``out_path``."""
    import cv2

    link_cfg = build_link_config(cfg)
    rng = rng or np.random.default_rng(0)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    rec, meta = transmit_image_analog(img, snr_db, link_cfg, rng)
    import os
    os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)
    cv2.imwrite(str(out_path), rec)
    meta["out_path"] = str(out_path)
    return meta
