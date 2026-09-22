"""Validate the digital LDPC link: BER/FER table + visual image degradation."""
from __future__ import annotations
import sys, glob, json
from pathlib import Path
import numpy as np
import cv2

from vqa_semcom.degradation.digital_link import (
    LinkConfig, FadingConfig, LDPCConfig, calibrate_link, transmit_image,
)

SNRS = [-5, 0, 5, 10, 15, 20]
OUT = Path("outputs/vlm/link_demo")
OUT.mkdir(parents=True, exist_ok=True)

def run(kind, kdb, blocks):
    cfg = LinkConfig(fading=FadingConfig(kind=kind, k_factor_db=kdb),
                     ldpc=LDPCConfig(n=96, d_v=3, d_c=6, maxiter=50),
                     packet_payload_bits=1024, calib_blocks=blocks)
    calib = calibrate_link(SNRS, cfg, seed=0)
    label = f"{kind}" + (f"_K{kdb:g}dB" if kind == "rician" else "")
    print(f"\n=== {label}  (LDPC rate={list(calib.values())[0]['code_rate']:.2f}, blocks={blocks}) ===")
    print("SNR | post-LDPC BER | FER (block-erasure)")
    for s in SNRS:
        r = calib[f"{s}dB"]
        print(f"{s:>3}dB | {r['ber']:.4e}   | {r['fer']:.3f}")
    return cfg, calib, label

def main():
    blocks = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    img_path = sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob("data/raw/visdrone/DET/val/images/*.jpg"))[0]
    img = cv2.imread(img_path)
    print("image:", img_path, "shape:", None if img is None else img.shape)

    summary = {}
    for kind, kdb in [("rayleigh", 0.0), ("rician", 6.0)]:
        cfg, calib, label = run(kind, kdb, blocks)
        summary[label] = calib
        rng = np.random.default_rng(42)
        print(f"  -- transmitting image over {label} --")
        for s in SNRS:
            dec, meta = transmit_image(img, s, calib, cfg, rng, jpeg_quality=90)
            outp = OUT / f"{label}_{s}dB.png"
            cv2.imwrite(str(outp), dec)
            print(f"   {s:>3}dB: loss_frac={meta['loss_frac']:.3f} decode_ok={meta['decode_ok']} -> {outp.name}")
    (OUT / "calibration.json").write_text(json.dumps(summary, indent=2))
    print("\ncalibration + demo images saved to", OUT)

if __name__ == "__main__":
    main()
