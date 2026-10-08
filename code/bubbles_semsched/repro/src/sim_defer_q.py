#!/usr/bin/env python3
"""Headroom test, second dimension: the SEMANTIC LEVEL as a decision (before building a hybrid-action learner).

So far the quality target was fixed (0.44) and the scheduler only ordered the uploads. Here every piece of evidence can be
sent at one of several levels (target new-object recall 0.42 / 0.44 / 0.46 / 0.48; a higher level needs a wider latent or
more repetitions, i.e. more channel uses, from the same measured curves). Under load a scheduler can then trade quality
for timeliness instead of dropping evidence, and at light load it can upgrade.

Metric: mean recall GAIN over the symbolic floor per evidence item; images delivered before the deadline count at the
level they were sent with, undelivered images count zero (fractional credit, the same for all policies and the bound).
Reported normalised by the gain of level 0.44, so 1.0 = 'everything delivered at 0.44'.
Constraint (added after the first run showed that pure quality maximisation simply serves fewer items at the top level):
COVERAGE = mean delivered fraction of the evidence (at any level) must be >= 0.95. A policy setting that misses it is
infeasible; the bound is the LP with the same constraint.

Policies:
  fixed L   one level for everything, index = efficiency / time-to-deadline, hopeless items skipped (best hand-made rule
            of the fixed-level study); the best L per case is reported ('best fixed' = tuned to the load by an oracle);
  adaptive  per item the highest level it could finish within kappa x its remaining time with the whole band, index =
            gain x efficiency / time-to-deadline; kappa on a small grid, best kept;
  lagr      a stronger hand-made rule added after the first grid (the gap to the bound looked large, so the baseline had
            to be made harder): Lagrangian index. Per item the level maximising (gain + mu) x efficiency, i.e. value per unit
            of band with mu = price of the coverage constraint; served in descending (gain + mu) x efficiency / slack^alpha.
            mu and alpha on a grid, best feasible kept. This is the classical primal-dual greedy.
  bound     clairvoyant LP over (item, level, slot).
Usage: python sim_defer_q.py <sim_inputs.json> <out.json> <unused> [procs=4]"""
import sys, json, time, os
import numpy as np
from multiprocessing import Pool
import sim_defer as SD

LEVELS = [float(x) for x in os.environ.get('BUB_LEVELS', '0.42,0.44,0.46,0.48').split(',')]   # recall targets of the token levels (ablation: fewer levels)
COVER = 0.95
GAIN = np.array(LEVELS) - SD.Q_SYM
JCL = [np.array([min([r * C / 2 for C in (96, 192) for r in SD.REPS if SD.q_gated(C, s + 10 * np.log10(r)) >= q] or [np.inf]) for s in SD.GRID])
       for q in LEVELS]


def make_levels(lam, D, B, seed):
    Es = []
    for jc in JCL:
        SD.JC = jc                                            # same traffic and SNR for every level (same seed)
        E, _, rel, dl, cnt = SD.make(lam, 0, D, 'defer', B, seed)
        Es.append(E)
    return np.stack(Es), rel, dl, cnt                          # levels x items x lifetime slots


def run(Es, rel, dl, mode, par):
    L, n, _ = Es.shape
    rem = np.ones(n); gain = np.zeros(n)
    order = np.argsort(rel, kind='stable'); ptr = 0
    live = np.zeros(0, int)
    for s in range(int(rel.min()), int(dl.max())):
        a = ptr
        while ptr < n and rel[order[ptr]] <= s:
            ptr += 1
        if ptr > a:
            live = np.concatenate([live, order[a:ptr]])
        if len(live) == 0:
            continue
        live = live[(dl[live] > s) & (rem[live] > 1e-9)]
        if len(live) == 0:
            continue
        e = Es[:, live, s - rel[live]].astype(float)          # levels x live
        slack = (dl[live] - s).astype(float)
        need = np.where(e > 0, rem[live] / np.maximum(e, 1e-30), np.inf)
        if mode == 'fixed':
            lev = np.full(len(live), par)
            ok = need[par] <= slack
        elif mode == 'lagr':
            mu, alpha = par
            dens = np.where(need <= slack, (GAIN[:, None] + mu) * e, -1.0)
            lev = np.argmax(dens, 0)
            ok = dens[lev, np.arange(len(live))] > 0
        else:                                                 # adaptive: highest level that fits in kappa x slack, else lowest that fits at all
            fit = need <= par * slack
            lev = np.where(fit.any(0), L - 1 - np.argmax(fit[::-1], 0), 0)
            ok = need[lev, np.arange(len(live))] <= slack
        if not ok.any():
            continue
        cand, lv, sl = live[ok], lev[ok], slack[ok]
        ec = e[lv, np.nonzero(ok)[0]]
        pr = (GAIN[lv] + par[0]) * ec / sl ** par[1] if mode == 'lagr' else (GAIN[lv] if mode != 'fixed' else 1.0) * ec / sl
        o = np.argsort(-pr, kind='stable')
        cand, lv, ec = cand[o], lv[o], ec[o]
        d = SD.serve(rem[cand], ec)
        gain[cand] += d * GAIN[lv]; rem[cand] -= d
        rem[cand[rem[cand] <= 1e-9]] = 0.0
    return gain, 1.0 - rem


def lp_bound(Es, rel, dl, cover=None):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, vstack
    L, n, _ = Es.shape
    ll, ii, kk = np.nonzero(Es > 0)
    keep = kk < (dl - rel)[ii]
    ll, ii, kk = ll[keep], ii[keep], kk[keep]
    e = Es[ll, ii, kk].astype(float); ss = rel[ii] + kk
    slots, sidx = np.unique(ss, return_inverse=True)
    nv = len(ii); cols = np.arange(nv)
    A = vstack([coo_matrix((e, (ii, cols)), shape=(n, nv)), coo_matrix((np.ones(nv), (sidx, cols)), shape=(len(slots), nv))]).tocsr()
    b = np.ones(A.shape[0])
    if cover is not None:                                     # mean delivered fraction over all items >= cover
        A = vstack([A, coo_matrix((-e, (np.zeros(nv, int), cols)), shape=(1, nv))]).tocsr(); b = np.append(b, -cover * n)
    res = linprog(-GAIN[ll] * e, A_ub=A, b_ub=b, bounds=(0, None), method='highs')
    if res.status == 2:
        return None, None, nv
    assert res.status == 0, res.message
    return np.bincount(ii, weights=GAIN[ll] * e * res.x, minlength=n), np.bincount(ii, weights=e * res.x, minlength=n), nv


def job(args):
    lam, D, B = args
    t0 = time.time()
    Es, rel, dl, cnt = make_levels(lam, D, B, 7)
    norm = 0.44 - SD.Q_SYM
    out = dict(lam=lam, D=D, B_MHz=B / 1e6, items=int(cnt.sum()))
    gc = lambda r: [float(r[0][cnt].mean() / norm), float(r[1][cnt].mean())]          # [normalised gain, coverage]
    out['fixed'] = {str(LEVELS[l]): gc(run(Es, rel, dl, 'fixed', l)) for l in range(len(LEVELS))}
    out['adaptive'] = {str(k): gc(run(Es, rel, dl, 'adaptive', k)) for k in (0.01, 0.02, 0.05, 0.1, 0.2, 0.4)}
    out['lagr'] = {f'{mu},{al}': gc(run(Es, rel, dl, 'lagr', (mu, al))) for mu in (0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.4) for al in (0.0, 0.5, 1.0)}
    z, c, nv = lp_bound(Es, rel, dl)
    out['bound_free'] = gc((z, c))
    z, c, nv = lp_bound(Es, rel, dl, COVER)
    out['bound'] = gc((z, c)) if z is not None else None
    out['lp_vars'] = int(nv); out['secs'] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True)
    return out


def main():
    procs = int(float(sys.argv[4])) if len(sys.argv) > 4 else 4
    jobs = [(la, D, B) for B in (2e6, 1e6) for D in (60, 120) for la in (20, 40, 60)]
    if len(sys.argv) > 5 and sys.argv[5] == 'smoke':
        jobs = [(40, 60, 1e6)]
    print(f'levels {LEVELS}, gains over the symbolic floor {np.round(GAIN, 4).tolist()} (q_sym {SD.Q_SYM:.4f}); JSCC cost per token at 1/7/13 dB: '
          + ' | '.join('/'.join(f'{jc[int((s + 14) / 0.25)]:.0f}' for s in (1, 7, 13)) for jc in JCL), flush=True)
    with Pool(procs) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(dict(levels=LEVELS, q_sym=SD.Q_SYM, runs=res), open(SD.OUT, 'w'), indent=1)
    print(f'\n=== mean recall gain, normalised (1.0 = everything delivered at 0.44), subject to coverage >= {COVER} ===')
    print('gain/coverage per setting; * = meets the coverage constraint')
    for r in res:
        feas = {**{f'fixed {k}': v for k, v in r['fixed'].items()}, **{f'adaptive k={k}': v for k, v in r['adaptive'].items()},
                **{f'lagr mu,alpha={k}': v for k, v in r['lagr'].items()}}
        lo = {k: v for k, v in r['lagr'].items() if v[1] >= COVER}
        lb = max(lo, key=lambda k: lo[k][0]) if lo else None
        okp = {k: v for k, v in feas.items() if v[1] >= COVER}
        best = max(okp, key=lambda k: okp[k][0]) if okp else None
        print(f"--- {r['B_MHz']:g} MHz, D={r['D']}s, {r['lam']}/h")
        print('    fixed:    ' + '  '.join(f"{k}: {v[0]:.2f}/{v[1]:.3f}{'*' if v[1] >= COVER else ' '}" for k, v in r['fixed'].items()))
        print('    adaptive: ' + '  '.join(f"k={k}: {v[0]:.2f}/{v[1]:.3f}{'*' if v[1] >= COVER else ' '}" for k, v in r['adaptive'].items()))
        print('    lagr:     best feasible ' + (f'mu,alpha={lb}: {lo[lb][0]:.3f}/{lo[lb][1]:.3f}' if lb else 'none') + ' | highest gain regardless of coverage: ' + max((f'{v[0]:.2f}/{v[1]:.3f}' for v in r['lagr'].values()), key=lambda t: float(t.split('/')[0])))
        bd = f"{r['bound'][0]:.3f} (coverage {r['bound'][1]:.3f})" if r['bound'] else 'INFEASIBLE even for the clairvoyant'
        print(f"    best feasible online: {best + ' -> ' + format(okp[best][0], '.3f') if best else 'NONE meets the constraint'} | constrained bound: {bd} | unconstrained bound {r['bound_free'][0]:.3f}/{r['bound_free'][1]:.3f}")

if __name__ == '__main__':
    main()
