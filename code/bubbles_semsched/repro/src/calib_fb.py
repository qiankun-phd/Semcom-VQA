#!/usr/bin/env python3
"""Calibration of the adaptive constraint price of sppo_hold.py (SPPO_DUAL): the share of flights landing below the recall requirement in
TRAINING episodes (900 s warm-up from an empty system + 1200 s window, stochastic policy) is not the steady-state share of the evaluation
(1.5 h runs, greedy policy). For given checkpoints this prints the training-window share at every training rate, so that the target
SPPO_FB can be set to the value that corresponds to the wanted steady-state share.

Usage: python calib_fb.py <sim_inputs.json> <out.json> <unused> <M> <unused> <procs> <episodes per rate> name=ckpt.pt ..."""
import sys, json
import numpy as np
import torch
from multiprocessing import Pool
ARGS = sys.argv[8:]; NEP = int(sys.argv[7]); sys.argv = sys.argv[:7] + ['1']
import sppo_hold as SP
HP = SP.HP

if __name__ == '__main__':
    out = {}
    with Pool(HP.PROCS) as pool:
        for item in ARGS:
            name, path = item.split('=', 1)
            net = SP.SNet(); net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)); w = SP.weights(net)
            out[name] = dict(par=w['par'])
            for lam in HP.LAMS_TRAIN:
                tr = [t for t in pool.map(SP.rollout, [(w, lam, 7 * 10 ** 6 + k, 'train') for k in range(NEP)]) if t]
                nb = sum(t['fair'][0] for t in tr); nl = sum(t['fair'][1] for t in tr)
                out[name][str(lam)] = dict(below=int(nb), landed=int(nl), share=float(nb) / max(int(nl), 1))
            print(name, f"p {w['par']['p']:.2f} V {w['par']['V']:.3f}", ' | '.join(f"{la}/h {out[name][str(la)]['share'] * 100:.2f}% of {out[name][str(la)]['landed']}" for la in HP.LAMS_TRAIN), flush=True)
    json.dump(out, open(sys.argv[2], 'w'))
