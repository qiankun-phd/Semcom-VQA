#!/usr/bin/env python3
"""A3 (TCOM plan): does the recall of a semantic level depend on how many tokens the evidence needs?

The scheduler model charges an evidence item in proportion to its token count and credits every item with the same recall per
level. If the recall gain of the tokens were the same for token-poor and token-rich images, a policy could cherry-pick the
cheap ones. This script stores, PER IMAGE of the stage-A test set (468 VisDrone images, synthetic revisit, 10 % of the objects
changed, 'gnss' registration, onboard-detector gate): the number of changed objects, the number of tokens sent, and the number
of changed objects found by the ground detector
  - in the gated reconstruction at each SNR and for C = 96 / 192 (the token levels),
  - from the symbolic layer alone (onboard detections, conf 0.25),
  - in the uncompressed image (upper end).
Same code path as eval_gated_sweep.py (imported), same seeds; only the bookkeeping is per image.

Usage (rented server, env bub): python a3_per_image.py [--limit N] [--out a3_per_image.json]"""
import os, sys, json, glob, random, time, zlib, argparse
import numpy as np
import torch
HR = os.path.expanduser('~/phd_research/BUBBLES_hoverreport_20260930')
sys.path.insert(0, f'{HR}/scripts')
import eval_gated_sweep as EG
RG, RR, J, E, WD, BLK = EG.RG, EG.RR, EG.J, EG.E, EG.WD, EG.BLK
SNRS, CS = [1, 4, 7, 10, 13], [96, 192]


def tp_new(d, g, idx):
    db, dc, ds = d; gb, gc = g
    return int(E._match(db, dc, ds, gb[idx], gc[idx], True)[0]) if len(idx) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ground', default=os.path.expanduser('~/phd_research/BUBBLES_equalbyte_20260929/runs/v8m_visdrone/weights/best.pt'))
    ap.add_argument('--out', default='a3_per_image.json'); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    allp = sorted(glob.glob(f'{E.VIS}/images/val/*.jpg')); random.Random(0).shuffle(allp)
    test = allp[80:][:a.limit] if a.limit else allp[80:]
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    gts = [E.load_gt(stem(p)) for p in test]
    G = E.Det(a.ground); Og = E.Det(EG.ONBOARD, conf=0.1); Os = E.Det(EG.ONBOARD, conf=0.25)
    nets = {C: J.build(f'{WD}/weights/SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C{C}.model', model='SwinJSCC_w/o_SAandRA', C=str(C), snrs='10') for C in CS}
    rows = []; t0 = time.time()
    for i, (p, g) in enumerate(zip(test, gts)):
        cur = RG.load_cur(p); h, w = cur.shape[:2]
        curp = EG.pad128(cur); H, W = curp.shape[:2]; Hb, Wb = H // BLK, W // BLK
        seed = zlib.crc32(os.path.basename(p).encode())
        cs = RG.changed_sets(len(g[0]), seed)[0.1]
        idx = np.array(cs, int)
        ref = RG.make_ref(cur, g[0], 'gnss', seed, cs)[0]
        rreg, _ = RR.register(ref, cur)
        d_raw, d_reg = EG.block_diff(curp, EG.pad128(ref)), EG.block_diff(curp, EG.pad128(rreg))
        refp = EG.pad128(rreg) if d_reg.mean() < d_raw.mean() else EG.pad128(ref)
        sel = EG.det_mask(Og, cur, refp[:h, :w], Hb, Wb, True)
        n = int(sel.sum()); ix = torch.from_numpy(sel.reshape(-1)).cuda()
        pix = np.repeat(np.repeat(sel, BLK, 0), BLK, 1)[..., None]
        outs, keys = [], []
        for C in CS:
            net = nets[C]
            with torch.no_grad():
                def tokens(img):
                    x = EG.to_t(img); J._set_res(net, x.shape[2], x.shape[3])
                    return net.encoder(x, 10, C, net.model)[0]
                zc, zr = tokens(curp), tokens(refp)
                for s in SNRS:
                    if n == 0:
                        img = refp
                    else:
                        xs = zc[ix]
                        torch.manual_seed((seed + s * 7 + C) % 2 ** 31)
                        z = zr.clone()
                        z[ix] = xs + torch.sqrt(xs.pow(2).mean() * 2) * np.sqrt(1.0 / (2 * 10 ** (s / 10))) * torch.randn_like(xs)
                        J._set_res(net, H, W)
                        dec = (net.decoder(z[None], 10, net.model)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
                        img = np.where(pix, dec, refp)
                    outs.append(img[:h, :w]); keys.append(f'C{C}@{s}')
        res = G(outs + [cur, refp[:h, :w]])
        row = dict(n_gt=int(len(g[0])), n_new=int(len(idx)), n_tok=n, tp={k: tp_new(d, g, idx) for k, d in zip(keys, res)},
                   tp_full=tp_new(res[-2], g, idx), tp_ref_only=tp_new(res[-1], g, idx), tp_sym=tp_new(Os([cur])[0], g, idx))
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f'{i + 1}/{len(test)} {time.time() - t0:.0f}s', flush=True)
    json.dump(dict(snrs=SNRS, Cs=CS, rows=rows), open(a.out, 'w'))
    tot = sum(r['n_new'] for r in rows)
    print(f"{len(rows)} images, {tot} changed objects; pooled recall: symbolic {sum(r['tp_sym'] for r in rows) / tot:.3f}, uncompressed {sum(r['tp_full'] for r in rows) / tot:.3f}, "
          + ', '.join(f"{k} {sum(r['tp'][k] for r in rows) / tot:.3f}" for k in rows[0]['tp']), flush=True)
    print('saved', a.out, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
