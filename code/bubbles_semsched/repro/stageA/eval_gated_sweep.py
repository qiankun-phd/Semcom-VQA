#!/usr/bin/env python3
"""v3 check (training-free): change-gated token transmission on top of the frozen SwinJSCC C96 (SNR-10) model.

UAV   : registers the reference to the current view (ORB homography + per-channel gain/offset; keeps the raw reference
        if that matches better), splits the 640x384 image into 16x16 blocks (= SwinJSCC latent tokens, 40x24 = 960),
        marks blocks whose blurred absolute difference to the reference exceeds tau, and transmits the 96-d latent of
        the marked tokens only (48 complex channel uses per token; AWGN; power-normalised over the sent tokens).
ground: applies the same registration (side information), encodes the registered reference with the same frozen
        encoder, fills the unsent tokens with the reference latent, decodes, and composites: marked blocks from the
        decoder output, the rest copied from the registered reference.
Side information counted separately: entropy-coded mask (960 * H2(p) + 16 bits), power scalar and flag (5 B),
registration parameters (56 B when used). It is sent digitally; the analysis converts it to channel uses with an SE model.
mask 'oracle' = blocks touching ground-truth changed boxes (upper bound of the gating). Same pairs/seeds as ref_gain.py."""
import os, sys, json, glob, random, time, zlib, argparse
import numpy as np
import cv2
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_gain as RG
import ref_gain_reg as RR
import jscc_lib as J
E = RG.E
WD = RG.WD
SNRS = [1, 7, 13]
TAUS = [6, 10, 16, 24]
BLK = 16


def pad128(a):
    h, w = a.shape[:2]
    H, W = -(-h // 128) * 128, -(-w // 128) * 128
    return np.pad(a, ((0, H - h), (0, W - w), (0, 0)), mode='edge')


def to_t(a):
    return torch.from_numpy(a).permute(2, 0, 1)[None].float().div(255.).cuda()


def block_diff(a, b):
    d = np.abs(cv2.GaussianBlur(a, (5, 5), 1.5).astype(np.float32) - cv2.GaussianBlur(b, (5, 5), 1.5).astype(np.float32)).mean(2)
    H, W = d.shape
    return d.reshape(H // BLK, BLK, W // BLK, BLK).mean((1, 3))


ONBOARD = os.path.expanduser('~/phd_research/BUBBLES_equalbyte_20260929/weights/onboard_v8n_visdrone.pt')


def det_mask(O, cur, ref_img, Hb, Wb, only_new, pad=4):
    """semantic gate from the ONBOARD detector (conf >= 0.1): blocks touching detections in the current image;
    only_new keeps detections with no counterpart (IoU >= 0.3, class-agnostic) among the detections in the reference."""
    h, w = cur.shape[:2]
    (bc, _, _), (br, _, _) = O([cur, ref_img])
    if only_new and len(bc) and len(br):
        bc = bc[E._iou(bc, br).max(1) < 0.3]
    m = np.zeros((Hb, Wb), bool)
    for x1, y1, x2, y2 in bc:
        m[max(0, int(y1 * h - pad) // BLK):min(Hb, int(np.ceil(y2 * h + pad) - 1) // BLK + 1),
          max(0, int(x1 * w - pad) // BLK):min(Wb, int(np.ceil(x2 * w + pad) - 1) // BLK + 1)] = True
    return m


def h2(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ground', default=os.path.expanduser('~/phd_research/BUBBLES_equalbyte_20260929/runs/v8m_visdrone/weights/best.pt'))
    ap.add_argument('--out', default=f'{WD}/results/eval_gated.json'); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--masks', nargs='*', default=None); ap.add_argument('--C', type=int, default=96)
    ap.add_argument('--snrs', type=int, nargs='*', default=SNRS); ap.add_argument('--conds', nargs='*', default=list(RG.CONDS))
    ap.add_argument('--changes', type=float, nargs='*', default=RG.CHANGE)
    a = ap.parse_args()
    allp = sorted(glob.glob(f'{E.VIS}/images/val/*.jpg'))
    random.Random(0).shuffle(allp)
    test = allp[80:]
    if a.limit:
        test = test[:a.limit]
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    gts = [E.load_gt(stem(p)) for p in test]
    G = E.Det(a.ground)
    net = J.build(f'{WD}/weights/SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C{a.C}.model', model='SwinJSCC_w/o_SAandRA', C=str(a.C), snrs='10')
    masks = a.masks or [f'tau{t}' for t in TAUS] + ['oracle']
    O = E.Det(ONBOARD, conf=0.1) if any(m.startswith('det') for m in masks) else None
    keys = [f'{m}|{c}@{rc}|{s}' for m in masks for c in a.conds for rc in a.changes for s in a.snrs]
    dets = {k: [] for k in keys}; ktok = {k: [] for k in keys}; side = {k: [] for k in keys}; nsel = {k: [] for k in keys}
    chg_all, used_reg = [], []
    t0 = time.time()

    @torch.no_grad()
    def tokens(img):
        x = to_t(img)
        J._set_res(net, x.shape[2], x.shape[3])
        return net.encoder(x, 10, a.C, net.model)[0]          # L x 96

    @torch.no_grad()
    def decode(tok, H, W):
        J._set_res(net, H, W)
        return net.decoder(tok[None], 10, net.model)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    for i, (p, g) in enumerate(zip(test, gts)):
        cur = RG.load_cur(p); h, w = cur.shape[:2]
        curp = pad128(cur); H, W = curp.shape[:2]; Hb, Wb = H // BLK, W // BLK
        zc = tokens(curp)
        seed = zlib.crc32(os.path.basename(p).encode())
        cs = RG.changed_sets(len(g[0]), seed)
        chg_all.append({rc: cs[rc].tolist() for rc in RG.CHANGE})
        for c in a.conds:
            for rc in a.changes:
                ref = RG.make_ref(cur, g[0], c, seed, cs[rc])[0]
                rreg, _ = RR.register(ref, cur)
                d_raw, d_reg = block_diff(curp, pad128(ref)), block_diff(curp, pad128(rreg))
                use_reg = d_reg.mean() < d_raw.mean()
                refp, d = (pad128(rreg), d_reg) if use_reg else (pad128(ref), d_raw)
                used_reg.append(bool(use_reg))
                zr = tokens(refp)
                om = np.zeros((Hb, Wb), bool)
                for x1, y1, x2, y2 in g[0][cs[rc]]:
                    om[max(0, int(y1 * h) // BLK):min(Hb, int(np.ceil(y2 * h) - 1) // BLK + 1),
                       max(0, int(x1 * w) // BLK):min(Wb, int(np.ceil(x2 * w) - 1) // BLK + 1)] = True
                outs, meta = [], []
                for m in masks:
                    sel = (om if m == 'oracle' else det_mask(O, cur, refp[:h, :w], Hb, Wb, m == 'det') if m.startswith('det')
                           else d > float(m[3:]))
                    n = int(sel.sum())
                    idx = torch.from_numpy(sel.reshape(-1)).cuda()
                    pix = np.repeat(np.repeat(sel, BLK, 0), BLK, 1)[..., None]
                    sb = (Hb * Wb * h2(n / (Hb * Wb)) + 16) / 8 + 5 + (RR.SIDE if use_reg else 0)
                    for s in a.snrs:
                        key = f'{m}|{c}@{rc}|{s}'
                        if n == 0:
                            img = refp
                        else:
                            xs = zc[idx]
                            torch.manual_seed((seed + s * 7 + len(outs)) % 2 ** 31)
                            pwr = xs.pow(2).mean() * 2
                            z = zr.clone()
                            z[idx] = xs + torch.sqrt(pwr) * np.sqrt(1.0 / (2 * 10 ** (s / 10))) * torch.randn_like(xs)
                            dec = (decode(z, H, W) * 255 + 0.5).astype(np.uint8)
                            img = np.where(pix, dec, refp)
                        outs.append(img[:h, :w]); meta.append((key, (a.C // 2) * n, sb, n))
                res = G(outs)
                for (key, kt, sb, n), dd in zip(meta, res):
                    dets[key].append(dd); ktok[key].append(kt); side[key].append(sb); nsel[key].append(n)
        if (i + 1) % 25 == 0:
            print(f'{i + 1}/{len(test)} {time.time() - t0:.0f}s', flush=True)

    def rec_new(dl, rc):
        tp = n = 0
        for (db, dc, ds), (gb, gc), ch in zip(dl, gts, chg_all):
            idx = np.array(ch[rc], int)
            if len(idx):
                t, _ = E._match(db, dc, ds, gb[idx], gc[idx], True); tp += t; n += len(idx)
        return tp / max(n, 1)
    out = dict(meta=dict(n=len(test), snrs=a.snrs, taus=TAUS, reg_used=float(np.mean(used_reg))), results={})
    for k in keys:
        rc = float(k.split('@')[1].split('|')[0])
        m = E.evaluate(dets[k], gts)
        out['results'][k] = dict(k_tok=float(np.mean(ktok[k])), side_bytes=float(np.mean(side[k])), n_sel=float(np.mean(nsel[k])),
                                 f1=m['f1'], rec_new=rec_new(dets[k], rc), n_sel_list=[int(x) for x in nsel[k]])
    json.dump(out, open(a.out, 'w'), indent=1)
    for k in keys:
        if '@0.1' in k:
            r = out['results'][k]
            print(f'{k:>24}: tokens {r["n_sel"]:6.1f}  k_tok {r["k_tok"]/1e3:5.2f}k  side {r["side_bytes"]:5.0f} B  rec_new {r["rec_new"]:.3f}  F1 {r["f1"]:.3f}')
    print('saved', a.out, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
