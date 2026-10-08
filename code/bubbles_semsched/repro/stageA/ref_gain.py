#!/usr/bin/env python3
"""Cheap test for the reference-aided (revisit) inspection paradigm, with a DIGITAL codec, before any JSCC training.

Question: if the receiver (and the UAV) already hold the previous image of the same station, how many bytes does it
save to code the current image with that image as a reference (HEVC P-frame) vs on its own (HEVC intra), at equal
task quality? And how fast does the saving erode with misregistration / appearance change?

Synthetic revisit pairs on the E1a VisDrone test split (468 images, same detector G):
  current   = image resized to long side 640 (G's input scale)
  reference = current with all annotated objects removed (OpenCV Telea inpainting) -> "the change" = every object,
              then perturbed to emulate a revisit: aligned / rtk / gnss / poor (shift, rotation, scale, photometric).
Task metric: G's class-aware F1 on the decoded current image (E1a evaluate). Rate: bytes of the VCL NAL units of the
current picture only (parameter sets and SEI excluded for both arms).
"""
import os, sys, json, glob, random, time, subprocess, argparse, zlib
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.expanduser('~/phd_research/BUBBLES_equalbyte_20260929/scripts'))
import eq_byte as E

FF = os.path.expanduser('~/.local/share/mamba/envs/ff/bin/ffmpeg')
WD = os.path.expanduser('~/phd_research/BUBBLES_hoverreport_20260930')
QPS = [22, 27, 32, 37, 42, 47]
CHANGE = [1.0, 0.3, 0.1]   # fraction of annotated objects that are new since the last visit (removed from the reference)
CONDS = {  # shift px (at 640 scale), rotation deg, scale, gamma, gain
    'aligned': (0, 0.0, 1.00, 1.00, 1.00),
    'rtk': (2, 0.2, 1.00, 0.95, 1.05),
    'gnss': (8, 1.0, 1.01, 0.90, 1.10),
    'poor': (32, 3.0, 1.03, 0.90, 1.10),
}


def load_cur(path):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    s = 640 / max(w, h)
    w1, h1 = round(w * s) // 2 * 2, round(h * s) // 2 * 2
    return np.asarray(im.resize((w1, h1), Image.LANCZOS))


def make_ref(cur, boxes, cond, seed, changed):
    h, w = cur.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes[changed]:
        mask[max(0, int(y1 * h) - 2):min(h, int(np.ceil(y2 * h)) + 2), max(0, int(x1 * w) - 2):min(w, int(np.ceil(x2 * w)) + 2)] = 255
    ref = cv2.inpaint(cur, mask, 3, cv2.INPAINT_TELEA) if mask.any() else cur.copy()
    sh, rot, sc, gamma, gain = CONDS[cond]
    rng = np.random.default_rng(seed)
    if sh or rot or sc != 1:
        ang = rng.uniform(0, 2 * np.pi)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rot * rng.choice([-1, 1]), sc)
        M[:, 2] += (sh * np.cos(ang), sh * np.sin(ang))
        ref = cv2.warpAffine(ref, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if gamma != 1 or gain != 1:
        tint = 1 + rng.uniform(-0.03, 0.03, 3) if cond in ('gnss', 'poor') else np.ones(3)
        ref = np.clip(255 * (ref / 255.) ** gamma * gain * tint, 0, 255).astype(np.uint8)
    return ref, float(mask.mean() / 255)


def vcl_sizes(bs):
    """sizes of VCL NAL units (one slice per picture) in an Annex-B HEVC stream, parameter sets / SEI excluded."""
    starts, i = [], 0
    while True:
        j = bs.find(b'\x00\x00\x01', i)
        if j < 0:
            break
        starts.append(j + 3); i = j + 3
    out = []
    for k, s in enumerate(starts):
        e = starts[k + 1] - 3 if k + 1 < len(starts) else len(bs)
        while e > s and bs[e - 1] == 0:   # 4-byte start code of the next NAL
            e -= 1
        if ((bs[s] >> 1) & 0x3F) < 32:
            out.append(e - s + 3)
    return out


def encode(frames, qp):
    h, w = frames[0].shape[:2]
    cmd = [FF, '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '1', '-i', '-',
           '-pix_fmt', 'yuv420p', '-c:v', 'libx265', '-preset', 'medium',
           '-x265-params', f'qp={qp}:keyint=250:min-keyint=250:scenecut=0:bframes=0:info=0:log-level=error', '-f', 'hevc', '-']
    bs = subprocess.run(cmd, input=b''.join(f.tobytes() for f in frames), capture_output=True, check=True).stdout
    dec = subprocess.run([FF, '-loglevel', 'error', '-f', 'hevc', '-i', '-', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         input=bs, capture_output=True, check=True).stdout
    dec = np.frombuffer(dec, np.uint8).reshape(-1, h, w, 3)
    return vcl_sizes(bs), dec


def changed_sets(n, seed):
    rng = np.random.default_rng(seed + 7)
    order = rng.permutation(n)
    return {rc: np.sort(order[:max(1, int(round(rc * n)))]) if n else np.zeros(0, int) for rc in CHANGE}


def job(args):
    path, boxes = args
    cur = load_cur(path)
    res = {'bytes': {}, 'psnr': {}, 'dec': {}}
    seed = zlib.crc32(os.path.basename(path).encode())
    chg = changed_sets(len(boxes), seed)
    refs = {f'{c}@{rc}': make_ref(cur, boxes, c, seed, chg[rc]) for c in CONDS for rc in CHANGE}
    res['change_frac'] = {rc: refs[f'aligned@{rc}'][1] for rc in CHANGE}
    res['changed'] = {rc: chg[rc].tolist() for rc in CHANGE}
    for qp in QPS:
        sz, dec = encode([cur], qp)
        res['bytes'][f'intra|{qp}'] = sz[0]; res['dec'][f'intra|{qp}'] = dec[0]
        for c, (ref, _) in refs.items():
            sz, dec = encode([ref, cur], qp)
            res['bytes'][f'{c}|{qp}'] = sz[1]; res['dec'][f'{c}|{qp}'] = dec[1]
    for k, d in res['dec'].items():
        mse = np.mean((d.astype(np.float64) - cur) ** 2)
        res['psnr'][k] = float(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
    res['cur'] = cur
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ground', default=os.path.expanduser('~/phd_research/BUBBLES_equalbyte_20260929/runs/v8m_visdrone/weights/best.pt'))
    ap.add_argument('--out', default=f'{WD}/results/ref_gain.json')
    ap.add_argument('--ncal', type=int, default=80); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    allp = sorted(glob.glob(f'{E.VIS}/images/val/*.jpg'))
    random.Random(0).shuffle(allp)
    test = allp[a.ncal:]
    if a.limit:
        test = test[:a.limit]
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    gts = [E.load_gt(stem(p)) for p in test]
    G = E.Det(a.ground)
    keys = [f'intra|{q}' for q in QPS] + [f'{c}@{rc}|{q}' for c in CONDS for rc in CHANGE for q in QPS]
    dets = {k: [] for k in keys}; dets['raw640'] = []
    by = {k: [] for k in keys}; ps = {k: [] for k in keys}; frac = []; chg_all = []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        for i, r in enumerate(pool.map(job, [(p, g[0]) for p, g in zip(test, gts)], chunksize=1)):
            out = G([r['cur']] + [r['dec'][k] for k in keys])
            dets['raw640'].append(out[0])
            for k, d in zip(keys, out[1:]):
                dets[k].append(d); by[k].append(r['bytes'][k]); ps[k].append(r['psnr'][k])
            frac.append(r['change_frac']); chg_all.append(r['changed'])
            if (i + 1) % 50 == 0:
                print(f'{i + 1}/{len(test)} {time.time() - t0:.0f}s', flush=True)
    res = dict(config=vars(a), n=len(test), qps=QPS, conds=CONDS, change_frac=frac,
               raw640=E.evaluate(dets['raw640'], gts), arms={})
    def rec_changed(dl, rc):   # class-aware recall@0.5 restricted to the objects that are new since the last visit
        tp = n = 0
        for (db, dc, ds), (gb, gc), ch in zip(dl, gts, chg_all):
            idx = np.array(ch[rc], int)
            if len(idx) == 0:
                continue
            t, _ = E._match(db, dc, ds, gb[idx], gc[idx], True)
            tp += t; n += len(idx)
        return tp / max(n, 1)
    res['raw640_rec_changed'] = {rc: rec_changed(dets['raw640'], rc) for rc in CHANGE}
    for k in keys:
        m = E.evaluate(dets[k], gts)
        rcs = [rc for rc in CHANGE if k.split('|')[0].endswith(f'@{rc}')]
        res['arms'][k] = dict(f1=m['f1'], rec=m['rec'], prec=m['prec'], mean_bytes=float(np.mean(by[k])),
                              mean_psnr=float(np.mean(ps[k])), bytes=[int(b) for b in by[k]],
                              rec_changed={rc: rec_changed(dets[k], rc) for rc in (rcs or CHANGE)})
        print(f'{k:>18}: bytes {np.mean(by[k]):8.0f}  PSNR {np.mean(ps[k]):5.2f}  F1 {m["f1"]:.3f}', flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print('saved', a.out, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
