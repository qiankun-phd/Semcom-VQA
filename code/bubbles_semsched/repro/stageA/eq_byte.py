#!/usr/bin/env python3
"""E1a equal-byte comparison (ideal channel). Protocol frozen 2026-09-29.

Consumer G = ground detector (frozen). Onboard O = tiny detector (frozen).
Arms (all capped at B payload bytes/image, 2 B params overhead included):
  jpeg/webp/avif : resize to long side L, per-image rate control on quality, headers stripped
  symbolic       : O detections on the full-res image, top-k by conf, 38 bits each (no entropy coding)
  hybrid         : symbolic (share rho of B) + AVIF (rest) -> G on decoded image, merged with symbolic boxes
L (and rho) are chosen per budget on a calibration split disjoint from the test split.
Metric: class-aware F1 / recall @IoU0.5, class-agnostic recall, vehicle recall, at conf>=0.25.
"""
import os, io, sys, json, time, random, argparse, glob
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

VIS = os.path.expanduser('~/phd_research/uav-vqa-semantic-rl/data/processed/visdrone_yolo')
BUDGETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
LS = [32, 48, 64, 96, 128, 192, 256, 384, 640]
QS = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 8, 6, 4, 2, 1]
RHOS = [0.25, 0.5, 0.75]
OVERHEAD = 2
SYM_BITS = 38
VEH = {3, 4, 5, 8}
CODECS = ['jpeg', 'webp', 'avif']


# ---------------- codecs (payload = entropy-coded bitstream, container headers stripped) ----------------
def _resize(img, L):
    w, h = img.size
    s = L / max(w, h)
    if s >= 1:
        return img
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)


def _enc_jpeg(img, q):
    b = io.BytesIO()
    img.save(b, 'JPEG', quality=q, subsampling=2, optimize=False)
    d = b.getvalue()
    i = 2
    while True:
        assert d[i] == 0xFF
        m = d[i + 1]
        seglen = (d[i + 2] << 8) | d[i + 3]
        if m == 0xDA:
            start = i + 2 + seglen
            break
        i += 2 + seglen
    return len(d) - start - 2, Image.open(io.BytesIO(d)).convert('RGB')


def _enc_webp(img, q):
    b = io.BytesIO()
    img.save(b, 'WEBP', quality=q, method=4)
    d = b.getvalue()
    assert d[:4] == b'RIFF' and d[8:16] == b'WEBPVP8 ', d[:16]
    return len(d) - 20, Image.open(io.BytesIO(d)).convert('RGB')


def _enc_avif(img, q):
    b = io.BytesIO()
    img.save(b, 'AVIF', quality=q, speed=7)
    d = b.getvalue()
    i, pay = 0, None
    while i + 8 <= len(d):
        sz = int.from_bytes(d[i:i + 4], 'big')
        typ = d[i + 4:i + 8]
        if sz == 1:
            sz = int.from_bytes(d[i + 8:i + 16], 'big')
        if typ == b'mdat':
            pay = sz - 8
            break
        i += sz
    assert pay is not None
    return pay, Image.open(io.BytesIO(d)).convert('RGB')


ENC = {'jpeg': _enc_jpeg, 'webp': _enc_webp, 'avif': _enc_avif}


def rate_control(args):
    path, codec, L, B = args
    img = Image.open(path).convert('RGB')
    r = _resize(img, L)
    cap = B - OVERHEAD
    p, dec = ENC[codec](r, QS[-1])
    if p > cap:
        return None
    best = (p, dec)
    lo, hi = 0, len(QS) - 2
    while lo <= hi:
        mid = (lo + hi) // 2
        p, dec = ENC[codec](r, QS[mid])
        if p <= cap:
            best = (p, dec)
            hi = mid - 1
        else:
            lo = mid + 1
    return best[0] + OVERHEAD, np.asarray(best[1])


# ---------------- detector ----------------
class Det:
    def __init__(self, path, conf=0.25):
        import torch
        from ultralytics import YOLO
        self.torch = torch
        try:
            from ultralytics.utils.nms import non_max_suppression
        except Exception:
            from ultralytics.utils.ops import non_max_suppression
        self.nms = non_max_suppression
        self.net = YOLO(path).model.cuda().eval().half()
        self.conf = conf

    def __call__(self, imgs, bs=32):
        import cv2
        torch = self.torch
        out = []
        for s in range(0, len(imgs), bs):
            chunk = imgs[s:s + bs]
            xs, meta = [], []
            for im in chunk:
                h, w = im.shape[:2]
                r = 640 / max(h, w)
                nw, nh = round(w * r), round(h * r)
                rs = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
                canvas = np.full((640, 640, 3), 114, np.uint8)
                px, py = (640 - nw) // 2, (640 - nh) // 2
                canvas[py:py + nh, px:px + nw] = rs
                xs.append(canvas)
                meta.append((px, py, nw, nh))
            x = torch.from_numpy(np.stack(xs)).cuda().permute(0, 3, 1, 2).half() / 255.0
            with torch.no_grad():
                pred = self.net(x)
            res = self.nms(pred, conf_thres=self.conf, iou_thres=0.7, max_det=300, max_time_img=5.0)
            for r_, (px, py, nw, nh) in zip(res, meta):
                a = r_.float().cpu().numpy()
                b = a[:, :4].copy()
                b[:, [0, 2]] = np.clip((b[:, [0, 2]] - px) / nw, 0, 1)
                b[:, [1, 3]] = np.clip((b[:, [1, 3]] - py) / nh, 0, 1)
                out.append((b, a[:, 5].astype(int), a[:, 4]))
        return out


# ---------------- metrics ----------------
def _iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


def _match(db, dc, ds, gb, gc, class_aware):
    order = np.argsort(-ds)
    iou = _iou(db, gb)
    used = np.zeros(len(gb), bool)
    tp = 0
    for i in order:
        best, bj = 0.5, -1
        for j in range(len(gb)):
            if used[j] or (class_aware and gc[j] != dc[i]):
                continue
            if iou[i, j] >= best:
                best, bj = iou[i, j], j
        if bj >= 0:
            used[bj] = True
            tp += 1
    return tp, used


def evaluate(dets, gts):
    tp = fp = fn = tpa = ngt = vt = vn = 0
    for (db, dc, ds), (gb, gc) in zip(dets, gts):
        ngt += len(gb)
        t, used = _match(db, dc, ds, gb, gc, True)
        tp += t; fp += len(db) - t; fn += len(gb) - t
        vt += int(sum(used[j] and gc[j] in VEH for j in range(len(gb))))
        vn += int(sum(gc[j] in VEH for j in range(len(gb))))
        ta, _ = _match(db, dc, ds, gb, gc, False)
        tpa += ta
    p = tp / max(tp + fp, 1); r = tp / max(ngt, 1)
    return dict(f1=2 * p * r / max(p + r, 1e-9), prec=p, rec=r, rec_agn=tpa / max(ngt, 1),
                rec_veh=vt / max(vn, 1))


def load_gt(stem):
    rows = [l.split() for l in open(f'{VIS}/labels/val/{stem}.txt').read().strip().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4)), np.zeros(0, int)
    a = np.array(rows, float)
    xy = np.stack([a[:, 1] - a[:, 3] / 2, a[:, 2] - a[:, 4] / 2, a[:, 1] + a[:, 3] / 2, a[:, 2] + a[:, 4] / 2], 1)
    return np.clip(xy, 0, 1), a[:, 0].astype(int)


# ---------------- symbolic / hybrid ----------------
def symbolic(det, B):
    b, c, s = det
    k = min(len(b), max(0, (B - OVERHEAD) * 8 // SYM_BITS))
    o = np.argsort(-s)[:k]
    q = np.round(b[o] * 255) / 255
    conf_bin = np.minimum((s[o] - 0.25) / 0.75 * 4, 3).astype(int) / 4 * 0.75 + 0.25
    return (q, c[o], conf_bin), -(-k * SYM_BITS // 8) + OVERHEAD


def merge(d1, d2):
    import torch, torchvision
    b = np.concatenate([d1[0], d2[0]]); c = np.concatenate([d1[1], d2[1]]); s = np.concatenate([d1[2], d2[2]])
    if len(b) == 0:
        return d1
    keep = torchvision.ops.batched_nms(torch.tensor(b, dtype=torch.float32), torch.tensor(s, dtype=torch.float32),
                                       torch.tensor(c), 0.5).numpy()
    return b[keep], c[keep], s[keep]


# ---------------- pipeline ----------------
def run_codec(pool, G, paths, codec, L, B, Lfallback=True):
    """returns decoded arrays (None for lost), bytes list; falls back to smaller L if infeasible."""
    idx = LS.index(L)
    decs = [None] * len(paths); nb = [0] * len(paths)
    todo = list(range(len(paths)))
    for ll in LS[:idx + 1][::-1]:
        if not todo:
            break
        res = list(pool.map(rate_control, [(paths[i], codec, ll, B) for i in todo], chunksize=4))
        nxt = []
        for i, r in zip(todo, res):
            if r is None:
                nxt.append(i)
            else:
                nb[i], decs[i] = r
        todo = nxt if Lfallback else []
    return decs, nb


def detect_decoded(G, decs):
    empty = (np.zeros((0, 4)), np.zeros(0, int), np.zeros(0))
    ok = [i for i, d in enumerate(decs) if d is not None]
    res = G([decs[i] for i in ok])
    out = [empty] * len(decs)
    for i, r in zip(ok, res):
        out[i] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ground', required=True); ap.add_argument('--onboard', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--ncal', type=int, default=80)
    ap.add_argument('--ntest', type=int, default=0); ap.add_argument('--budgets', type=int, nargs='*', default=BUDGETS)
    ap.add_argument('--workers', type=int, default=12); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    allp = sorted(glob.glob(f'{VIS}/images/val/*.jpg'))
    random.Random(a.seed).shuffle(allp)
    cal, test = allp[:a.ncal], allp[a.ncal:]
    if a.ntest:
        test = test[:a.ntest]
    G, O = Det(a.ground), Det(a.onboard)
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    gt_cal = [load_gt(stem(p)) for p in cal]; gt_test = [load_gt(stem(p)) for p in test]
    orig = lambda ps: [np.asarray(Image.open(p).convert('RGB')) for p in ps]
    R = {'config': vars(a) | dict(LS=LS, QS=QS, OVERHEAD=OVERHEAD, SYM_BITS=SYM_BITS, ncal=len(cal), ntest=len(test))}
    t0 = time.time()
    o_cal, o_test = O(orig(cal)), O(orig(test))
    R['ref_G_original'] = evaluate(G(orig(test)), gt_test)
    R['ref_O_original'] = evaluate(o_test, gt_test)
    print('refs', R['ref_G_original'], R['ref_O_original'], flush=True)

    pool = ProcessPoolExecutor(a.workers)
    Bs = sorted(set(a.budgets) | {int(B * (1 - r)) for B in a.budgets for r in RHOS})
    calt = {}
    for codec in CODECS:
        for B in Bs:
            if codec != 'avif' and B not in a.budgets:
                continue
            for L in LS:
                decs, nb = run_codec(pool, G, cal, codec, L, B, Lfallback=False)
                feas = sum(d is not None for d in decs) / len(decs)
                f1 = evaluate(detect_decoded(G, decs), gt_cal)['f1'] if feas >= 0.9 else -1
                calt[f'{codec}|{B}|{L}'] = dict(f1=f1, feas=feas)
            print('cal', codec, B, round(time.time() - t0), flush=True)
    R['cal'] = calt
    best = lambda codec, B: max(LS, key=lambda L: calt[f'{codec}|{B}|{L}']['f1'])
    R['test'] = {}
    for B in a.budgets:
        row = {}
        for codec in CODECS:
            L = best(codec, B)
            decs, nb = run_codec(pool, G, test, codec, L, B)
            m = evaluate(detect_decoded(G, decs), gt_test)
            m.update(L=L, mean_bytes=float(np.mean(nb)), lost=sum(d is None for d in decs) / len(decs))
            row[codec] = m
        sd = [symbolic(d, B) for d in o_test]
        m = evaluate([x[0] for x in sd], gt_test); m.update(mean_bytes=float(np.mean([x[1] for x in sd])))
        row['symbolic'] = m
        bh = None
        for rho in RHOS:
            Bs_, Bi = int(B * rho), int(B * (1 - rho))
            if Bs_ < OVERHEAD + 5 or f'avif|{Bi}|{LS[0]}' not in calt:
                continue
            L = best('avif', Bi)
            dcal, _ = run_codec(pool, G, cal, 'avif', L, Bi)
            gcal = detect_decoded(G, dcal)
            f1c = evaluate([merge(g, symbolic(o, Bs_)[0]) for g, o in zip(gcal, o_cal)], gt_cal)['f1']
            if bh is None or f1c > bh[0]:
                bh = (f1c, rho, Bs_, Bi, L)
        if bh:
            _, rho, Bs_, Bi, L = bh
            decs, nb = run_codec(pool, G, test, 'avif', L, Bi)
            gd = detect_decoded(G, decs)
            sy = [symbolic(o, Bs_) for o in o_test]
            m = evaluate([merge(g, s[0]) for g, s in zip(gd, sy)], gt_test)
            m.update(rho=rho, L=L, mean_bytes=float(np.mean(nb) + np.mean([s[1] for s in sy])))
            row['hybrid'] = m
        R['test'][B] = row
        print('B', B, {k: round(v['f1'], 3) for k, v in row.items()}, round(time.time() - t0), flush=True)
        json.dump(R, open(a.out, 'w'), indent=1, default=float)
    json.dump(R, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
