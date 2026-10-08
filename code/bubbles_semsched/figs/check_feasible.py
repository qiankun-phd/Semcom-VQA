#!/usr/bin/env python3
"""Is the network-average recall condition of cap_strat.cell ever the ONLY requirement a cell misses? The paper states feasibility without
it (rejected <= 5 %, conformance >= 95 %, airborne <= N_ref, lost <= 1 %, flights below the recall requirement <= 5 %), so every cell whose
stored 'feasible' is False although those five hold is listed here. Reads the aggregated JSONs under res/ only.
usage: python check_feasible.py [folder ...]   (default: final final_182 qsweep extra pass1 main)"""
import sys, os, json, glob
R = os.environ.get('BUB_RES') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'res')
N_REF = 16.57


def flips(path):
    """(cells, [network average decides alone], [aggregated without the per-flight condition], [flips if lost evidence counted as recall 0])"""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    if not isinstance(d, dict) or 'cells' not in d or 'qbar' not in d:
        return None
    alone, noflight, denom = [], [], []
    for k, c in d['cells'].items():
        if not all(x in c for x in ('rejected', 'conformance', 'n_air', 'lost', 'recall', 'feasible')):
            return None
        four = c['rejected'] <= 0.05 and c['conformance'] >= 0.95 and c['n_air'] <= N_REF and c['lost'] <= 0.01
        five = four and c.get('flights_below', 0.0) <= 0.05
        row = (os.path.basename(path), k, c['recall'], d['qbar'], c.get('flights_below'), c['lost'])
        if five and not c['feasible']:
            alone.append(row)
        if c['feasible'] and not five:
            noflight.append(row)
        if c['feasible'] and c['recall'] * (1 - c['lost']) < d['qbar'] - 5e-4:      # mean recall with delivered + lost evidence as denominator
            denom.append(row)
    return len(d['cells']), alone, noflight, denom


def report(folders, verbose=True):
    tot = [0, 0, 0, 0]
    for folder in folders:
        n = 0; bad = [[], [], []]
        for f in sorted(glob.glob(os.path.join(R, folder, '*.json'))):
            if 'INVALID' in f:
                continue
            r = flips(f)
            if r:
                n += r[0]
                for i in range(3):
                    bad[i] += r[i + 1]
        tot = [tot[0] + n] + [tot[i + 1] + len(bad[i]) for i in range(3)]
        print(f'{folder}: {n} cells | network-average recall decides alone: {len(bad[0])} | stored feasible without the per-flight condition '
              f'(network-average runs): {len(bad[1])} | would flip with lost evidence counted as recall 0: {len(bad[2])}')
        if verbose:
            for lab, rows in (('alone', bad[0]), ('lost-as-0', bad[2])):
                for fn, k, rec, q, fb, lost in rows[:30]:
                    print(f'   [{lab}] {fn}  {k}  mean recall {rec:.4f} (requirement {q})  flights below {fb:.4f}  lost {lost:.4f}')
    return tot


def margin(folders):
    """over the feasible cells: smallest (network-average recall - requirement), with the delivered evidence as denominator (as stored) and
    with delivered + lost evidence as denominator; and how many feasible cells there are"""
    n, m1, m2, w1, w2 = 0, 9.0, 9.0, None, None
    for folder in folders:
        for f in sorted(glob.glob(os.path.join(R, folder, '*.json'))):
            if 'INVALID' in f or flips(f) is None:
                continue
            d = json.load(open(f))
            for k, c in d['cells'].items():
                if c['feasible']:
                    n += 1; a_ = c['recall'] - d['qbar']; b_ = c['recall'] * (1 - c['lost']) - d['qbar']
                    if a_ < m1:
                        m1, w1 = a_, (os.path.basename(f), k)
                    if b_ < m2:
                        m2, w2 = b_, (os.path.basename(f), k)
    return n, m1, w1, m2, w2


if __name__ == '__main__':
    t = report(sys.argv[1:] or ['final', 'final_182', 'qsweep', 'extra', 'pass1', 'main'])
    print(f'total {t[0]} cells: alone {t[1]}, without per-flight condition {t[2]}, lost-as-0 {t[3]}')
