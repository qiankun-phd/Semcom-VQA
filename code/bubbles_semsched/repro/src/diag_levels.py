#!/usr/bin/env python3
"""When does a scheduler send tokens? Evaluation runs (1.5 h, greedy policy, test traffic) with every decision recorded: the SNR of the UAV's
best channel in the slot, the time left before a hold (grace time minus the age of the oldest undelivered evidence), and the level chosen
(0 = symbolic layer only, >= 1 = tokens). Output: counts per SNR bin x level, per slack bin x level and per (SNR bin, slack bin) x {symbolic,
tokens}, so that the share of token decisions against the channel and against the urgency can be drawn.

Usage: python diag_levels.py <sim_inputs.json> <out.json> <unused> <M> <unused>
       env: DG_KIND = rule | sppo, DG_POL (rule: lyap<V> / lyapp<p>v<V>), DG_CKPT (sppo), DG_LAM (60), DG_SEEDS (12), DG_SEED0 (501), DG_PROCS (4)"""
import sys, os, json
import numpy as np
from multiprocessing import Pool
KIND, POL, CK = os.environ.get('DG_KIND', 'rule'), os.environ.get('DG_POL', 'lyapp3v0.1'), os.environ.get('DG_CKPT', '')
LAM, NS, S0 = float(os.environ.get('DG_LAM', 60)), int(os.environ.get('DG_SEEDS', 12)), int(os.environ.get('DG_SEED0', 501))
argv = sys.argv; M = int(argv[4]); sys.argv = argv[:4] + [str(M), '0', '1', '1']
W, mod = None, None
if KIND == 'sppo':
    import torch
    import sppo_hold as mod
    net = mod.SNet(); net.load_state_dict(torch.load(CK, map_location='cpu', weights_only=True)); W = mod.weights(net)
import sim_holdq as S
H, C = S.H, S.C
sys.argv = argv
SB = np.arange(-6.0, 30.01, 2.0)                               # SNR bins (dB)
TB = np.array([-1e9, 0.0, 10.0, 20.0, 30.0, 45.0, 1e9])        # time left before a hold (s): holding already, 0-10, 10-20, 20-30, 30-45, > 45


def one(seed):
    F = H.make_flights(LAM, M, seed); sim = S.Sim(F, M); rng = np.random.default_rng(0)
    a = np.zeros((len(SB) + 1, S.L)); b = np.zeros((len(TB) - 1, S.L)); c = np.zeros((len(SB) + 1, len(TB) - 1, 2))
    while not sim.finished():
        nh = sim.get()
        if not sim.cand:
            sim.step(None); continue
        snr = sim.s1 + np.array([g.max() for g in sim.gdb]); sl = sim.slack() * C.DT
        if KIND == 'sppo':
            o = mod.act(sim, nh, W, rng, False); lev, w = o[1], o[-1]
        elif POL.startswith('lyapp'):
            pp, _, vv = POL[5:].partition('v'); lev, w = S.lyap_action(sim, float(vv), p=float(pp))
        else:
            lev, w = S.lyap_action(sim, float(POL[4:]))
        i = np.searchsorted(SB, snr); j = np.clip(np.searchsorted(TB, sl, side='right') - 1, 0, len(TB) - 2)
        for ii, jj, ll in zip(i, j, np.asarray(lev)):
            a[ii, ll] += 1; b[jj, ll] += 1; c[ii, jj, int(ll > 0)] += 1
        sim.step(w, lev)
    return a, b, c


if __name__ == '__main__':
    with Pool(int(os.environ.get('DG_PROCS', 4))) as pool:
        rs = pool.map(one, range(S0, S0 + NS))
    a, b, c = (sum(r[k] for r in rs) for k in range(3))
    out = dict(kind=KIND, pol=POL if KIND == 'rule' else CK, par=(W['par'] if W else None), M=M, lam=LAM, seeds=[S0, NS], levels=S.QL.tolist(),
               snr_edges=SB.tolist(), slack_edges=[0.0, 10.0, 20.0, 30.0, 45.0], by_snr=a.tolist(), by_slack=b.tolist(), by_snr_slack=c.tolist())
    json.dump(out, open(argv[2], 'w'))
    tot = a.sum(0); print(KIND, out['pol'], 'decisions', int(tot.sum()), 'share per level', np.round(tot / tot.sum(), 3).tolist(), flush=True)
    print('token share by slack bin (holding, 0-10, 10-20, 20-30, 30-45, >45 s):', np.round(1 - b[:, 0] / np.maximum(b.sum(1), 1), 3).tolist(), flush=True)
