#!/usr/bin/env python3
"""Symbolic baseline for the change task: the ONBOARD detector O's own boxes (conf >= 0.25) sent as symbols
(38 bits per box as in E1a), evaluated on new-object recall / all-object F1, on (a) VisDrone-MOT pairs, (b) the
synthetic revisit set. Also G on the uncompressed current image for reference."""
import os, sys, json, glob, random, zlib
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_gain as RG
import ref_gain_mot as RM
E = RG.E
WD = RG.WD
O = E.Det(os.path.expanduser('~/phd_research/BUBBLES_equalbyte_20260929/weights/onboard_v8n_visdrone.pt'), conf=0.25)
G = E.Det(os.path.expanduser('~/phd_research/BUBBLES_equalbyte_20260929/runs/v8m_visdrone/weights/best.pt'))
out = {}


def rec_sub(dl, gts, idxs):
    tp = n = 0
    for (db, dc, ds), (gb, gc), idx in zip(dl, gts, idxs):
        idx = np.array(idx, int)
        if len(idx):
            t, _ = E._match(db, dc, ds, gb[idx], gc[idx], True); tp += t; n += len(idx)
    return tp / max(n, 1)


# (a) MOT pairs
pairs = RM.make_pairs()
anns = {s: RM.load_ann(s) for s in {p[0] for p in pairs}}
sizes = {s: Image.open(f'{RM.MOT}/sequences/{s}/0000001.jpg').size for s in anns}
for D in RM.DELTAS:
    gts, new, do, dg = [], [], [], []
    for seq, f0, f1, d in pairs:
        if d != D:
            continue
        W, H = sizes[seq]
        b1, c1, id1 = RM.frame_boxes(anns[seq], f1, W, H); _, _, id0 = RM.frame_boxes(anns[seq], f0, W, H)
        cur = RG.load_cur(f'{RM.MOT}/sequences/{seq}/{f1:07d}.jpg')
        gts.append((b1, c1)); new.append(np.where(~np.isin(id1, id0))[0]); do.append(O([cur])[0]); dg.append(G([cur])[0])
    nb = np.mean([len(d[0]) for d in do])
    out[f'mot_D{D}'] = dict(O_rec_new=rec_sub(do, gts, new), O_f1=E.evaluate(do, gts)['f1'], O_prec=E.evaluate(do, gts)['prec'],
                            G_rec_new=rec_sub(dg, gts, new), G_f1=E.evaluate(dg, gts)['f1'], mean_boxes=float(nb), sym_bytes=float(nb * 38 / 8 + 2))
    print(f'MOT D={D}:', {k: round(v, 3) for k, v in out[f'mot_D{D}'].items()}, flush=True)

# (b) synthetic revisit set
allp = sorted(glob.glob(f'{E.VIS}/images/val/*.jpg')); random.Random(0).shuffle(allp); test = allp[80:]
stem = lambda p: os.path.splitext(os.path.basename(p))[0]
gts = [E.load_gt(stem(p)) for p in test]
do, dg, chg = [], [], []
for p, g in zip(test, gts):
    cur = RG.load_cur(p)
    cs = RG.changed_sets(len(g[0]), zlib.crc32(os.path.basename(p).encode()))
    chg.append(cs); do.append(O([cur])[0]); dg.append(G([cur])[0])
nb = np.mean([len(d[0]) for d in do])
out['synthetic'] = dict(O_f1=E.evaluate(do, gts)['f1'], O_prec=E.evaluate(do, gts)['prec'], G_f1=E.evaluate(dg, gts)['f1'],
                        mean_boxes=float(nb), sym_bytes=float(nb * 38 / 8 + 2),
                        **{f'O_rec_new@{rc}': rec_sub(do, gts, [c[rc] for c in chg]) for rc in RG.CHANGE},
                        **{f'G_rec_new@{rc}': rec_sub(dg, gts, [c[rc] for c in chg]) for rc in RG.CHANGE})
print('synthetic:', {k: round(v, 3) for k, v in out['synthetic'].items()}, flush=True)
json.dump(out, open(f'{WD}/results/sym_baseline.json', 'w'), indent=1)
