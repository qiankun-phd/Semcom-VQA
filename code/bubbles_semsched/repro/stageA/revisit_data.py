"""Synthetic revisit pairs for TRAINING (VisDrone train split), randomised version of the ref_gain.py generator:
current = image at long side 640, random crop; reference = same view with a random subset of objects removed
(OpenCV Telea inpainting) and a random revisit perturbation (shift / rotation / scale / photometric / tint)."""
import os, glob
import numpy as np
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset

VIS = os.path.expanduser('~/phd_research/uav-vqa-semantic-rl/data/processed/visdrone_yolo')


def load_boxes(split, stem):
    p = f'{VIS}/labels/{split}/{stem}.txt'
    rows = [l.split() for l in open(p).read().strip().splitlines() if l.strip()] if os.path.exists(p) else []
    if not rows:
        return np.zeros((0, 4))
    a = np.array(rows, float)
    return np.clip(np.stack([a[:, 1] - a[:, 3] / 2, a[:, 2] - a[:, 4] / 2, a[:, 1] + a[:, 3] / 2, a[:, 2] + a[:, 4] / 2], 1), 0, 1)


def box_mask(h, w, boxes, pad=2):
    m = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        m[max(0, int(y1 * h) - pad):min(h, int(np.ceil(y2 * h)) + pad), max(0, int(x1 * w) - pad):min(w, int(np.ceil(x2 * w)) + pad)] = 255
    return m


def perturb(ref, rng):
    h, w = ref.shape[:2]
    if rng.random() < 0.2:
        return ref
    sh = rng.uniform(0, 16); ang = rng.uniform(0, 2 * np.pi)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-2, 2), rng.uniform(0.98, 1.02))
    M[:, 2] += (sh * np.cos(ang), sh * np.sin(ang))
    ref = cv2.warpAffine(ref, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    gamma, gain = rng.uniform(0.85, 1.15), rng.uniform(0.9, 1.1)
    tint = 1 + rng.uniform(-0.03, 0.03, 3)
    return np.clip(255 * (ref / 255.) ** gamma * gain * tint, 0, 255).astype(np.uint8)


class RevisitPairs(Dataset):
    def __init__(self, split='train', crop=256):
        self.split, self.crop = split, crop
        self.paths = sorted(glob.glob(f'{VIS}/images/{split}/*.jpg'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rng = np.random.default_rng((torch.initial_seed() + i * 7919) % 2 ** 32)
        p = self.paths[i]
        im = Image.open(p).convert('RGB')
        w, h = im.size
        s = 640 / max(w, h)
        w1, h1 = round(w * s) // 2 * 2, round(h * s) // 2 * 2
        cur = np.asarray(im.resize((w1, h1), Image.LANCZOS))
        boxes = load_boxes(self.split, os.path.splitext(os.path.basename(p))[0])
        n = len(boxes)
        rc = rng.choice([0.1, 0.3, 0.5, 1.0])
        changed = rng.permutation(n)[:max(1, int(round(rc * n)))] if n else np.zeros(0, int)
        mask = box_mask(h1, w1, boxes[changed])
        ref = cv2.inpaint(cur, mask, 3, cv2.INPAINT_TELEA) if mask.any() else cur.copy()
        ref = perturb(ref, rng)
        c = self.crop
        y0 = rng.integers(0, h1 - c + 1); x0 = rng.integers(0, w1 - c + 1)
        sl = (slice(y0, y0 + c), slice(x0, x0 + c))
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div(255.)
        return t(cur[sl]), t(ref[sl]), torch.from_numpy(mask[sl] > 0).float()[None]
