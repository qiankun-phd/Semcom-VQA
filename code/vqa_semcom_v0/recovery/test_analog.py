import sys, glob, math
sys.path.insert(0, "src")
import numpy as np, cv2
from vqa_semcom.config import load_config
from vqa_semcom.degradation.analog_link import transmit_image_analog
from vqa_semcom.degradation.digital_link import build_link_config, transmit_image_rate_adaptive

cfg = load_config("configs/v2_0_ldpc_channel.yaml")
lc = build_link_config(cfg)
imgp = sorted(glob.glob("data/raw/visdrone/DET/val/images/*.jpg"))[0]
img = cv2.imread(imgp)
print("img", imgp.split("/")[-1], img.shape)


def psnr(a, b):
    m = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 99.0 if m < 1e-9 else 10 * math.log10(255 ** 2 / m)


print("%5s | %12s %7s | %12s %8s %5s" % ("SNR", "ANALOG_psnr", "scale", "DIG_psnr", "outage", "qual"))
for snr in [-5, 0, 5, 10, 15, 20]:
    # average over a few fading draws for stability
    ap = []
    for s in range(5):
        rng = np.random.default_rng(1000 + snr * 10 + s)
        a, am = transmit_image_analog(img, snr, lc, rng)
        ap.append(psnr(img, a))
    rng2 = np.random.default_rng(7 + snr)
    d, dm = transmit_image_rate_adaptive(img, snr, lc, rng2)
    print("%5d | %12.2f %7s | %12.2f %8s %5s" % (
        snr, float(np.mean(ap)), am["scale"], psnr(img, d), str(dm.get("outage")), str(dm.get("quality", "-"))))
