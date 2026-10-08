#!/usr/bin/env python3
"""The hand-made schemes on the VALIDATION traffic (seeds 301-304, 1.5 h, no strategic deconfliction) - exactly what the learning curves
of hppo_hold / sppo_hold / offpol_hold are evaluated on: reference levels for the convergence figure.

Usage: python val_refs.py <sim_inputs.json> <out.json> <unused> [M=4] [procs=4]     env: VAL_POLS (comma-separated policies) and the switches of sim_holdq.py"""
import sys, json
from multiprocessing import Pool
import sim_holdq as S
M = int(sys.argv[4]) if len(sys.argv) > 4 else 4
import os
VAL, LAMS = [301, 302, 303, 304], [60.0, 65.0]
POLS = os.environ.get('VAL_POLS', 'edf,fixed2,lyap0.3,lyapp3v0.03').split(',')


def one(a):
    pol, lam, s = a
    return S.run(lam, M, s, pol)


if __name__ == '__main__':
    with Pool(int(sys.argv[5]) if len(sys.argv) > 5 else 4) as pool:
        out = {pol: {str(la): S.summary(pool.map(one, [(pol, la, s) for s in VAL]), la) for la in LAMS} for pol in POLS}
    json.dump(dict(M=M, seeds=VAL, res=out), open(sys.argv[2], 'w'))
    for pol in POLS:
        print(pol, ' | '.join(f"{la}/h hold {m['mean_hold']:.2f} s lost {m['lost']:.4f} recall {m['recall']:.4f} below {m['flights_below']:.3f}" for la, m in out[pol].items()), flush=True)
