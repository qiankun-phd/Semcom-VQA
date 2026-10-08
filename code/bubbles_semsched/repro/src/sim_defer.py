#!/usr/bin/env python3
"""Headroom test for the DEFERRED-UPLOAD reformulation (before building any RL).

Question: if evidence captured at a stop may be uploaded later (other stops, cruise legs) as long as it arrives within a
deadline D and before landing, how much better is the clairvoyant optimum than simple online rules? That gap is the room
an RL scheduler could exploit. If simple rules already sit on the bound, RL has nothing to do here either.

Model (same frozen scenario and channel statistics as sim_sched_tmpl.py; additions are marked NEW):
  flight   : transit 150 s, K = 20 stops, each stop = capture 10 s + reserved hover window W + cruise 20 s, transit 150 s;
             Poisson arrivals (lam per hour); T_f = 300 + 20 (30 + W);  N = lam T_f / 3600 <= N_ref = 16.57.
  channel  : planned mean SNR per stop ~ N(5.7, 6) dB, known from the plan; NEW: linear in dB along a cruise/transit leg
             between its end points (transit end points drawn from the same law); fading N(0, 4) dB per segment (each hover
             and each leg has its own draw), revealed when the segment starts.
  evidence : one item per stop = M = 336 images, released at the end of the capture; scheme = reference-gated JSCC tokens
             + symbolic layer + side information (inputs: measured recall curves, tokens per image ~ lognormal, cv 0.8).
             Per image at SNR s: 8 (sym + side) / SE(s)  +  tokens * min{ r C/2 : recall_C(s + 10 log10 r) >= q* }  channel uses,
             SE = 3GPP TR 38.803 NR-UL (alpha 0.4, zero below -10 dB). The configuration is re-chosen at the SNR of the moment.
  uplink   : one shared band of B complex channel uses per second, time-shared (fluid: an item can be served in pieces).
  deadline : mode 'window' : item must finish inside its own hover window (the original setting, no deferral);
             mode 'defer'  : NEW: item must finish within D of its capture and before the UAV lands.
  an item not finished by its deadline is late (only the symbolic floor q_sym counts for it).

Policies (all online ones see the current SNR; 'defer' rules also use the PLANNED mean SNR of the rest of the flight):
  edf      earliest deadline first
  maxeff   cheapest item first (largest fraction completed per unit of band now), deadlines ignored
  slack    largest efficiency / time-to-deadline first
  defer    hold an item while its current efficiency is below theta x the best planned future efficiency in its lifetime,
           unless it is within tau seconds of its deadline; EDF among the released ones. theta, tau on a small grid, best kept.
  bound    clairvoyant LP (knows every future SNR and arrival): maximise the total completed fraction. Upper bound.

Usage: python sim_defer.py <sim_inputs.json> <out.json> [q_target=0.44] [procs=3]"""
import json, sys, time, itertools, os
import numpy as np
from multiprocessing import Pool

INP = json.load(open(sys.argv[1]))
INP = INP.get('core', INP)
OUT = sys.argv[2]
QT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.44
try:
    PROCS = int(float(sys.argv[4])) if len(sys.argv) > 4 else 3
except ValueError:                                           # other scripts import this module with their own argv[4]
    PROCS = 3
M, K = 336, 20
T_CAP, T_CRUISE, T_TR = 10.0, 20.0, 150.0
N_REF = 16.57
MU, SD_PLAN, SD_FADE = 5.7, 6.0, 4.0
DT, HOURS, WARM = 1.0, 1.5, 0.5
Q_SYM, SYM_B = INP['q_sym'], INP['sym_bytes']
SIDE_B, TOK = INP['gated_C96']['side_bytes'], INP['gated_C96']['tokens']
REPS = [1, 2, 4, 8, 16, 32, 64]
# Band model. 0 = one fluid band of B channel uses per second (first studies). M > 0 = M orthogonal channels of W_CH =
# 180 kHz (one resource block, the usual choice in semantic resource-allocation papers); B must then be M * W_CH; in a
# slot a channel serves ONE evidence item, an item may hold several channels (user decision 2026-10-04).
W_CH = 180e3
M_CH = int(os.environ.get('BUB_CHANNELS', 0))


def q_gated(C, x):
    p, y = INP['snr_pts'], INP[f'gated_C{C}']['rec']
    if x <= p[0]:
        return max(0.0, y[0] + (x - p[0]) * (y[1] - y[0]) / (p[1] - p[0]))
    return float(np.interp(min(x, p[-1]), p, y))


# per-image cost tables on an SNR grid
GRID = np.arange(-14.0, 26.01, 0.25)
SE = np.where(GRID < -10, 0.0, 0.4 * np.log2(1 + np.minimum(10 ** (GRID / 10), 10 ** 2.2)))
JC = np.array([min([r * C / 2 for C in (96, 192) for r in REPS if q_gated(C, s + 10 * np.log10(r)) >= QT] or [np.inf]) for s in GRID])
DIG = np.where(SE > 0, 8 * (SYM_B + SIDE_B) / np.maximum(SE, 1e-12), np.inf)     # channel uses per image, digital part


def eff(snr, ntok, B):
    """fraction of an item completed per slot with the whole band, for arrays of SNR (dB)"""
    i = np.clip(np.rint((snr - GRID[0]) / 0.25).astype(int), 0, len(GRID) - 1)
    uses = M * (DIG[i] + JC[i] * ntok)
    return np.where(np.isfinite(uses), B * DT / uses, 0.0)


def make(lam, W, D, mode, B, seed):
    """returns E (items x lifetime slots, efficiency with the realised SNR), EP (same with the planned SNR), release slot,
    deadline slot (exclusive), counted flag"""
    rng = np.random.default_rng(seed)
    T = (HOURS + WARM) * 3600
    t0s = np.cumsum(rng.exponential(3600 / lam, int(lam * (HOURS + WARM) * 1.5) + 50))
    t0s = t0s[t0s < T]
    cv = 0.8; sig = np.sqrt(np.log(1 + cv ** 2)); mu_ln = np.log(TOK) - sig ** 2 / 2
    rel, dl, cnt, rows, rowsp = [], [], [], [], []
    for fi, t0 in enumerate(t0s):
        frng = np.random.default_rng(seed * 100003 + fi)          # same per-flight draws for every W / D / B / policy
        mu = frng.normal(MU, SD_PLAN, K); fade = frng.normal(0, SD_FADE, K); ntok = np.exp(frng.normal(mu_ln, sig, K))
        mu_io = frng.normal(MU, SD_PLAN, 2); fade_leg = frng.normal(0, SD_FADE, K + 1)
        # segments of the flight: (t_start, t_end, planned SNR at start, at end, fading)
        seg, t = [(t0, t0 + T_TR, mu_io[0], mu[0], fade_leg[0])], t0 + T_TR
        caps = []
        for k in range(K):
            seg.append((t, t + T_CAP + W, mu[k], mu[k], fade[k])); caps.append(t + T_CAP); t += T_CAP + W
            nxt = mu[k + 1] if k + 1 < K else mu_io[1]
            seg.append((t, t + (T_CRUISE if k + 1 < K else T_TR), mu[k], nxt, fade_leg[k + 1]))
            t += T_CRUISE if k + 1 < K else T_TR
        t_land = t
        s0, s1 = int(np.ceil(t0 / DT)), int(np.floor(t_land / DT))
        ts = (np.arange(s0, s1) + 0.5) * DT
        plan = np.empty(len(ts)); real = np.empty(len(ts))
        for a, b, pa, pb, fd in seg:
            m = (ts >= a) & (ts < b)
            plan[m] = pa + (pb - pa) * (ts[m] - a) / max(b - a, 1e-9); real[m] = plan[m] + fd
        for k in range(K):
            r = int(np.ceil(caps[k] / DT))
            d = int(np.floor((caps[k] + W if mode == 'window' else min(caps[k] + D, t_land)) / DT))
            d = min(max(d, r), s1)
            rel.append(r); dl.append(d); cnt.append(caps[k] >= WARM * 3600)
            rows.append(eff(real[r - s0:d - s0], ntok[k], B)); rowsp.append(eff(plan[r - s0:d - s0], ntok[k], B))
    L = max((len(x) for x in rows), default=1)
    E = np.zeros((len(rows), max(L, 1)), np.float32); EP = np.zeros_like(E)
    for i, (x, y) in enumerate(zip(rows, rowsp)):
        E[i, :len(x)] = x; EP[i, :len(y)] = y
    return E, EP, np.array(rel), np.array(dl), np.array(cnt, bool)


def serve(remc, ec):
    """delivered fraction of each candidate this slot; candidates are already in priority order.
    remc = remaining fraction, ec = fraction completed per slot with the WHOLE band."""
    d = np.zeros(len(remc))
    if M_CH == 0:                                             # fluid: the band is shared, the last served item gets what is left
        cum = np.cumsum(remc / ec)
        j = int(np.searchsorted(cum, 1.0 + 1e-12, side='right'))
        d[:j] = remc[:j]
        if j < len(remc):
            d[j] = (1.0 - (cum[j - 1] if j else 0.0)) * ec[j]
    else:                                                     # channelised: whole channels, in priority order
        e1 = ec / M_CH
        cum = np.cumsum(np.ceil(remc / e1 - 1e-9))
        j = int(np.searchsorted(cum, M_CH + 1e-9, side='right'))
        d[:j] = remc[:j]
        if j < len(remc):
            d[j] = min(remc[j], (M_CH - (cum[j - 1] if j else 0.0)) * e1[j])
    return d


def best_future(EP):
    """best planned efficiency over the strictly later slots of each item's lifetime"""
    BF = np.zeros_like(EP)
    BF[:, :-1] = np.maximum.accumulate(EP[:, ::-1], axis=1)[:, ::-1][:, 1:]
    return BF


def run_policy(E, BF, rel, dl, policy, theta=0.0, tau=0.0):
    """slot-by-slot fluid simulation; returns the finished-before-deadline flag per item"""
    n = len(rel)
    rem = np.ones(n); done = np.zeros(n, bool)
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
        live = live[(dl[live] > s) & ~done[live]]             # expired items are late; finished ones leave
        if len(live) == 0:
            continue
        k = s - rel[live]
        e = E[live, k].astype(float)
        ok = e > 0
        if policy == 'defer':
            ok &= (e >= theta * BF[live, k]) | ((dl[live] - s) * DT <= tau)
        cand, ec = live[ok], e[ok]
        if len(cand) == 0:
            continue
        if policy in ('edf', 'defer'):
            o = np.lexsort((-ec, dl[cand]))
        elif policy == 'maxeff':
            o = np.argsort(-ec, kind='stable')
        else:                                                 # slack
            o = np.argsort(-ec / np.maximum(dl[cand] - s, 1), kind='stable')
        cand, ec = cand[o], ec[o]
        rem[cand] -= serve(rem[cand], ec)
        fin = cand[rem[cand] <= 1e-9]
        rem[fin] = 0.0; done[fin] = True
    return done


def lp_bound(E, rel, dl):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, vstack
    n = len(rel)
    ii, kk = np.nonzero(E > 0)
    keep = kk < (dl - rel)[ii]
    ii, kk = ii[keep], kk[keep]
    if len(ii) == 0:
        return np.zeros(n), 0
    e = E[ii, kk].astype(float); ss = rel[ii] + kk
    slots, sidx = np.unique(ss, return_inverse=True)
    nv = len(ii); cols = np.arange(nv)
    A = vstack([coo_matrix((e, (ii, cols)), shape=(n, nv)), coo_matrix((np.ones(nv), (sidx, cols)), shape=(len(slots), nv))]).tocsr()
    res = linprog(-e, A_ub=A, b_ub=np.ones(A.shape[0]), bounds=(0, None), method='highs')
    assert res.status == 0, res.message
    z = np.bincount(ii, weights=e * res.x, minlength=n)
    return np.minimum(z, 1.0), nv


def job(args):
    lam, W, D, mode, B = args
    t0 = time.time()
    E, EP, rel, dl, cnt = make(lam, W, D, mode, B, 7)
    out = dict(lam=lam, W=W, D=D, mode=mode, B_MHz=B / 1e6, items=int(cnt.sum()), T_f=2 * T_TR + K * (T_CAP + W + T_CRUISE),
               never=float(np.mean((E.max(1) <= 0)[cnt])), pol={})
    out['N'] = lam * out['T_f'] / 3600
    for p in ('edf', 'maxeff', 'slack'):
        out['pol'][p] = float(run_policy(E, None, rel, dl, p)[cnt].mean())
    if mode == 'defer':
        best = (-1, None); BF = best_future(EP)
        for th, tau in itertools.product((0.3, 0.5, 0.7, 0.9), (10, 30, 90)):
            v = float(run_policy(E, BF, rel, dl, 'defer', th, tau)[cnt].mean())
            best = max(best, (v, (th, tau)))
        out['pol']['defer'] = best[0]; out['defer_par'] = best[1]
        z, nv = lp_bound(E, rel, dl)
        out['pol']['bound'] = float(z[cnt].mean()); out['lp_vars'] = int(nv)
    out['secs'] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True)
    return out


def main():
    Bs = [20e6, 5e6, 2e6, 1e6]
    lams = [30, 60]
    jobs = [(la, W, 0, 'window', B) for B in Bs for la in lams for W in (2, 5, 12, 30, 60)] + \
           [(la, W, D, 'defer', B) for B in Bs for la in lams for W in (0,) for D in (60, 300, 900)]
    if len(sys.argv) > 5 and sys.argv[5] == 'smoke':
        jobs = [(30, 5, 0, 'window', 5e6), (30, 0, 300, 'defer', 5e6)]
    if len(sys.argv) > 5 and sys.argv[5] == 'cap':        # capacity scan in the scarce-band regime: largest arrival rate with >= 95 % on time
        Bs, lams = [2e6, 1e6, 0.5e6], [10, 20, 30, 40, 50, 60, 66]
        jobs = [(la, 0, D, 'defer', B) for B in Bs for D in (30, 60, 120, 300) for la in lams]
    print(f'{len(jobs)} runs; q_target {QT}, q_sym {Q_SYM:.3f}; per-image channel uses at -5/1/7/13 dB with {TOK:.1f} tokens: '
          + ' '.join(f'{(DIG[i] + JC[i] * TOK) / 1e3:.1f}k' for i in [int((s + 14) / 0.25) for s in (-5, 1, 7, 13)]), flush=True)
    with Pool(PROCS) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(dict(q_target=QT, inputs=INP, runs=res), open(OUT, 'w'), indent=1)
    print('\n=== on-time share of evidence items (need >= 0.95); capacity limit N <= 16.57 ===')
    for B in Bs:
        for la in lams:
            print(f'--- band {B / 1e6:g} MHz, {la} flights/h')
            for r in res:
                if r['B_MHz'] == B / 1e6 and r['lam'] == la:
                    tag = f"window W={r['W']:>2}s" if r['mode'] == 'window' else f"defer D={r['D']:>3}s W={r['W']}s"
                    print(f"   {tag:<20} N={r['N']:5.1f}{'*' if r['N'] > N_REF else ' '} never-sendable {r['never']:.3f} | "
                          + ' '.join(f'{k} {v:.3f}' for k, v in r['pol'].items()) + (f"  (theta,tau)={r.get('defer_par')}" if r['mode'] == 'defer' else ''))
    print('(* = more aircraft in the air than N_ref allows at this arrival rate)')


if __name__ == '__main__':
    main()
