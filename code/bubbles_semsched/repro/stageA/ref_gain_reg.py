#!/usr/bin/env python3
"""Strong digital baseline for the reference-aided test: encoder-side registration + photometric correction.
The UAV holds both the reference and the current image, so it can register the reference to the current view
(ORB + RANSAC homography) and fit a per-channel gain/offset, then use the corrected reference for HEVC inter coding.
The decoder needs the same correction: 8 homography + 6 photometric parameters as float32 = 56 bytes, added to the
P-frame bytes. Same pairs, QPs and metrics as ref_gain.py (imports its functions)."""
import os, sys, json, glob, random, time, zlib, argparse
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_gain as RG
E = RG.E
SIDE = 56
REG_CONDS = ['rtk', 'gnss', 'poor']


def register(ref, cur):
    orb = cv2.ORB_create(4000)
    g1, g2 = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
    k1, d1 = orb.detectAndCompute(g1, None); k2, d2 = orb.detectAndCompute(g2, None)
    ok = d1 is not None and d2 is not None and len(k1) >= 8 and len(k2) >= 8
    H = None
    if ok:
        m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
        if len(m) >= 8:
            p1 = np.float32([k1[x.queryIdx].pt for x in m]); p2 = np.float32([k2[x.trainIdx].pt for x in m])
            H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    h, w = cur.shape[:2]
    warped = cv2.warpPerspective(ref, H, (w, h), borderMode=cv2.BORDER_REFLECT) if H is not None else ref.copy()
    out = np.empty_like(warped)
    for ch in range(3):   # per-channel gain/offset by least squares
        x = warped[..., ch].reshape(-1).astype(np.float64); y = cur[..., ch].reshape(-1).astype(np.float64)
        A = np.stack([x, np.ones_like(x)], 1)
        g, o = np.linalg.lstsq(A, y, rcond=None)[0]
        out[..., ch] = np.clip(g * warped[..., ch] + o, 0, 255).astype(np.uint8)
    return out, H is not None


def job(args):
    path, boxes = args
    cur = RG.load_cur(path)
    seed = zlib.crc32(os.path.basename(path).encode())
    chg = RG.changed_sets(len(boxes), seed)
    res = {'bytes': {}, 'dec': {}, 'reg_ok': {}}
    for c in REG_CONDS:
        for rc in RG.CHANGE:
            ref, _ = RG.make_ref(cur, boxes, c, seed, chg[rc])
            rref, ok = register(ref, cur)
            res['reg_ok'][f'{c}@{rc}'] = ok
            for qp in RG.QPS:
                sz, dec = RG.encode([rref, cur], qp)
                res['bytes'][f'{c}+reg@{rc}|{qp}'] = sz[1] + SIDE
                res['dec'][f'{c}+reg@{rc}|{qp}'] = dec[1]
    res['changed'] = {rc: chg[rc].tolist() for rc in RG.CHANGE}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ground', default=os.path.expanduser('~/phd_research/BUBBLES_equalbyte_20260929/runs/v8m_visdrone/weights/best.pt'))
    ap.add_argument('--base', default=f'{RG.WD}/results/ref_gain.json')
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
    keys = [f'{c}+reg@{rc}|{q}' for c in REG_CONDS for rc in RG.CHANGE for q in RG.QPS]
    dets = {k: [] for k in keys}; by = {k: [] for k in keys}; chg_all = []; regok = []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        for i, r in enumerate(pool.map(job, [(p, g[0]) for p, g in zip(test, gts)], chunksize=1)):
            out = G([r['dec'][k] for k in keys])
            for k, d in zip(keys, out):
                dets[k].append(d); by[k].append(r['bytes'][k])
            chg_all.append(r['changed']); regok.append(r['reg_ok'])
            if (i + 1) % 50 == 0:
                print(f'{i + 1}/{len(test)} {time.time() - t0:.0f}s', flush=True)
    base = json.load(open(a.base))

    def rec_changed(dl, rc):
        tp = n = 0
        for (db, dc, ds), (gb, gc), ch in zip(dl, gts, chg_all):
            idx = np.array(ch[rc], int)
            if len(idx):
                t, _ = E._match(db, dc, ds, gb[idx], gc[idx], True); tp += t; n += len(idx)
        return tp / max(n, 1)
    for k in keys:
        m = E.evaluate(dets[k], gts)
        rc = float(k.split('@')[1].split('|')[0])
        base['arms'][k] = dict(f1=m['f1'], rec=m['rec'], prec=m['prec'], mean_bytes=float(np.mean(by[k])),
                               bytes=[int(b) for b in by[k]], rec_changed={str(rc): rec_changed(dets[k], rc)})
        print(f'{k:>22}: bytes {np.mean(by[k]):8.0f}  F1 {m["f1"]:.3f}', flush=True)
    base['reg_success'] = {kk: float(np.mean([r[kk] for r in regok])) for kk in regok[0]}
    json.dump(base, open(a.base.replace('.json', '_reg.json'), 'w'), indent=1)
    print('saved', a.base.replace('.json', '_reg.json'), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
