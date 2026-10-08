#!/usr/bin/env python3
"""Stage A (Colab rebuild): the SYNTHETIC-revisit part of sym_baseline.py only.
The code below is part (b) of sym_baseline.py, copied unchanged; part (a) needs the VisDrone-MOT data, which is not
on the Colab VM yet, and sym_baseline.py cannot be imported without it. Output: results/sym_baseline_synth.json with
the same 'synthetic' entry sym_baseline.py writes."""
import os, sys, json, glob, random, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_gain as RG
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
json.dump(out, open(f'{WD}/results/sym_baseline_synth.json', 'w'), indent=1)
