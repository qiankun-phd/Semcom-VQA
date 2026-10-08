#!/usr/bin/env python3
"""BUBBLES-facing environment (user decision 2026-10-05): evidence MUST be delivered; a UAV whose evidence is still
undelivered D seconds after its capture HOLDS (hovers where it is) until that evidence is through. The hold delays the
rest of its 4D plan, so the quality of the uplink scheduling shows up in the U-space metrics, not only in a service metric.

Same network, channel model, evidence model and matching assignment as sim_ch.py. What changes:
  timeline    a flight has a plan clock that stops while it holds; position, large-scale gain, captures follow the plan clock;
              small-scale fading is drawn every real second. Deferred upload = the D seconds of grace before a hold.
  hold        triggered per evidence item at capture + D; ends when the item is delivered. After H_MAX seconds of hold on one
              item the item is dropped and counted as LOST (so a UAV parked in a spot without usable channel is not stuck
              for ever). A UAV that reaches the end of its plan with undelivered evidence also waits (counted as hold).
  metrics     per flight: total hold time. 4D temporal buffer = 95th percentile of the total hold per flight (the buffer that
              keeps 95 % of the flights inside their 4D volume). Airborne count with that buffer reserved:
              N = lam x (900 + buffer) / 3600, to be compared with N_ref = 16.57 (BUBBLES D2.1 App. F) -> capacity.
              Also: mean hold per flight, share of flights with any hold, share of evidence lost.
  objective   minimise the holding (reward of the learner = - number of UAVs holding in the slot).
Baselines (matching assignment, weight per UAV): thr = 1; urg = 1 / time left before a hold; urg+ = urg, overdue first
(x 100 once a hold has started); edf = sequential, earliest hold first.

Usage: python sim_hold.py <sim_inputs.json> <out.json> <q_target> table [procs]     env: BUB_IOT_DB, BUB_D, BUB_HMAX, BUB_MS, BUB_LAMS"""
import sys, json, os, time
import numpy as np
from multiprocessing import Pool
import sim_ch as C
SD = C.SD

D = float(os.environ.get('BUB_D', 60.0)); H_MAX = float(os.environ.get('BUB_HMAX', 120.0))
N_REF, T_PLAN = 16.57, 2 * C.T_TR + C.K * (C.T_CAP + C.T_CRUISE)
NTOK_CV = float(os.environ.get('BUB_NTOK_CV', 0.8))            # dispersion of the token count between evidence items (0 = all items alike)


def alloc(s1, gdb, ntok, rem, w, M):
    """weighted marginal-gain matching for candidates j = 0..: s1[j] = full-power single-channel SNR without fading (dB),
    gdb[j] = per-channel small-scale gains (dB). Returns the delivered fraction of each candidate's evidence this slot."""
    m = len(s1)
    held = [[] for _ in range(m)]; cur = np.zeros(m)
    free = np.ones(M, bool)
    while free.any():
        fidx = np.nonzero(free)[0]
        bv, bj, bc, br = 0.0, -1, -1, 0.0
        for j in range(m):
            if cur[j] >= rem[j] or len(held[j]) >= C.NMAX:
                continue
            c = fidx[np.argmax(gdb[j][fidx])]
            ch = held[j] + [c]
            r = float(SD.eff(s1[j] - 10 * np.log10(len(ch)) + gdb[j][ch], ntok[j], C.W).sum())
            dv = w[j] * (min(rem[j], r) - min(rem[j], cur[j]))
            if dv > bv:
                bv, bj, bc, br = dv, j, c, r
        if bj < 0:
            break
        held[bj].append(bc); cur[bj] = br; free[bc] = False
    return np.minimum(rem, cur)


def requests(lam, seed, hours=1.5, warm=0.5):
    """requested departure times (Poisson)"""
    rng = np.random.default_rng(seed)
    T = (hours + warm) * 3600
    t0s = np.cumsum(rng.exponential(3600 / lam, int(lam * (hours + warm) * 1.5) + 50))
    return t0s[t0s < T]


def make_flights(lam, M, seed, hours=1.5, warm=0.5, dep=None):
    """dep = None: every request departs when requested. dep = array (one entry per request, nan = rejected): departure times
    granted by the strategic deconfliction (sim_strat.schedule); flight fi keeps its own random stream, i.e. its layer and stops."""
    t0s = requests(lam, seed, hours, warm)
    cv = NTOK_CV; sig = np.sqrt(np.log(1 + cv ** 2)); mu_ln = np.log(SD.TOK) - sig ** 2 / 2
    out = []
    for fi, treq in enumerate(t0s):
        t0 = treq if dep is None else dep[fi]
        if not np.isfinite(t0):
            continue                                                           # no conflict-free 4D volume in time: request rejected
        frng = np.random.default_rng(seed * 100003 + fi)
        f = C.flight(t0, frng, 0.0, 1)                                         # plan-indexed large-scale gain; fading is drawn at run time
        f['fi'] = fi; f['req'] = float(treq); f['gdelay'] = float(t0 - treq)
        f['ntok'] = np.exp(frng.normal(mu_ln, sig, C.K))
        f['cap_idx'] = [int(np.ceil(c / C.DT)) - f['s0'] for c in f['caps']]   # plan slot of each capture
        f['n'] = len(f['G']); f['fade_seed'] = seed * 7919 + fi; f['counted'] = treq >= warm * 3600
        del f['g2']
        out.append(f)
    return out


class Sim:
    """slot-by-slot simulation; step(weights) advances one second. get() returns the candidates of the current slot."""
    def __init__(self, flights, M):
        self.F, self.M, self.nz = flights, M, C.noise_dbm()
        self.s = min(f['s0'] for f in flights)
        self.next = 0; self.order = sorted(range(len(flights)), key=lambda i: flights[i]['s0'])
        self.act = []                                                          # airborne flight indices
        for f in flights:
            f.update(tau=0, items=[], hold=0, lost=0, done_items=0, landed=None, rng=np.random.default_rng(f['fade_seed']), delay_at_stop=[], hs=[])
        self.holding = 0

    def finished(self):
        return self.next >= len(self.F) and not self.act

    def _advance(self):
        """move the plan clocks, start / continue holds, release captures; returns the number of UAVs holding"""
        s = self.s
        while self.next < len(self.F) and self.F[self.order[self.next]]['s0'] <= s:
            self.act.append(self.order[self.next]); self.next += 1
        nh = 0; keep = []
        for fi in self.act:
            f = self.F[fi]
            over = [it for it in f['items'] if s - it['cap'] >= D / C.DT]
            if over or (f['tau'] >= f['n'] and f['items']):
                nh += 1; f['hold'] += 1; f['hs'].append(s)                      # hs: the slots in which the flight holds (realised trajectory)
                for it in over:
                    it['held'] += 1
                    if it['held'] >= H_MAX / C.DT:                              # give up on this item
                        f['items'].remove(it); f['lost'] += 1
            else:
                if f['tau'] >= f['n']:
                    f['landed'] = s; continue
                f['tau'] += 1
                while len(f['delay_at_stop']) < C.K and f['cap_idx'][len(f['delay_at_stop'])] <= f['tau']:
                    k = len(f['delay_at_stop'])
                    f['items'].append(dict(cap=s, rem=1.0, ntok=f['ntok'][k], held=0))
                    f['delay_at_stop'].append(f['hold'])                        # delay accumulated when stop k is reached
            keep.append(fi)
        self.act = keep; self.holding = nh
        return nh

    def get(self):
        """candidates of this slot: one per airborne UAV with undelivered evidence (its oldest item)"""
        nh = self._advance()
        for fi in self.act:                                                    # fading is drawn for every airborne UAV every second, so that
            f = self.F[fi]; r = f['rng']                                       # all policies see the same channel realisations
            ray = r.exponential(1.0, self.M)
            x = np.sqrt(C.K_RICE / (C.K_RICE + 1)) + r.normal(0, 1, self.M) * np.sqrt(0.5 / (C.K_RICE + 1))
            y = r.normal(0, 1, self.M) * np.sqrt(0.5 / (C.K_RICE + 1))
            f['gdb'] = 10 * np.log10(np.maximum(x ** 2 + y ** 2 if f['LOS'][min(f['tau'], f['n'] - 1)] else ray, 1e-6))
        cand = [fi for fi in self.act if self.F[fi]['items']]
        s1 = np.empty(len(cand)); gdb = []; info = []
        for j, fi in enumerate(cand):
            f = self.F[fi]; t = min(f['tau'], f['n'] - 1)
            gdb.append(f['gdb'])
            s1[j] = C.P_MAX + f['G'][t] - self.nz
            info.append((fi, t, f['items'][0]))
        self.cand, self.s1, self.gdb, self.info = cand, s1, gdb, info
        return nh

    def step(self, w):
        if self.cand:
            rem = np.array([it['rem'] for _, _, it in self.info]); ntok = np.array([it['ntok'] for _, _, it in self.info])
            d = alloc(self.s1, self.gdb, ntok, rem, w, self.M)
            for (fi, t, it), dj in zip(self.info, d):
                it['rem'] -= dj
                if it['rem'] <= 1e-9:
                    self.F[fi]['items'].remove(it); self.F[fi]['done_items'] += 1
        self.s += 1


def weights(name, sim):
    slack = np.array([D / C.DT - (sim.s - it['cap']) for _, _, it in sim.info])   # slots left before a hold (<= 0: holding)
    if name == 'thr':
        return np.ones(len(slack))
    w = 1.0 / np.maximum(slack, 1.0)
    return w * np.where(slack <= 0, 100.0, 1.0) if name == 'urg+' else w


def run(lam, M, seed, policy, raw=False):
    """raw = True: return (hold per counted flight in s, evidence items lost, evidence items) instead of the metrics"""
    F = make_flights(lam, M, seed)
    sim = Sim(F, M)
    while not sim.finished():
        sim.get()
        if policy == 'edf' and sim.cand:                                       # sequential: earliest hold first takes what it wants
            order = np.argsort([it['cap'] for _, _, it in sim.info], kind='stable')
            w = np.zeros(len(order)); w[order] = 10.0 ** (-3.0 * np.arange(len(order)))
        else:
            w = weights(policy, sim) if sim.cand else None
        sim.step(w)
    if raw:
        fl = [f for f in F if f['counted']]
        return np.array([f['hold'] for f in fl], float) * C.DT, sum(f['lost'] for f in fl), sum(f['done_items'] + f['lost'] for f in fl)
    return metrics(F, lam)


def metrics(F, lam):
    fl = [f for f in F if f['counted']]
    hold = np.array([f['hold'] for f in fl], float) * C.DT
    items = sum(f['done_items'] + f['lost'] for f in fl)
    buf = float(np.percentile(hold, 95))
    return dict(flights=len(fl), mean_hold=float(hold.mean()), buffer95=buf, any_hold=float((hold > 0).mean()),
                lost=float(sum(f['lost'] for f in fl) / max(items, 1)), N_mean=lam * (T_PLAN + float(hold.mean())) / 3600,
                N_buf=lam * (T_PLAN + buf) / 3600)


POLS = os.environ.get('BUB_POLICIES', 'edf,thr,urg,urg+').split(',')


def job(args):
    lam, M, seed = args
    t0 = time.time()
    out = dict(lam=lam, M=M, seed=seed, pol={p: run(lam, M, seed, p) for p in POLS})
    out['secs'] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True)
    return out


def main():
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 9
    Ms = [int(x) for x in os.environ.get('BUB_MS', '3,4,6,10').split(',')]
    lams = [float(x) for x in os.environ.get('BUB_LAMS', '20,40,60').split(',')]
    jobs = [(la, M, sd) for M in Ms for la in lams for sd in (201, 202, 203)]
    print(f'{len(jobs)} runs; D {D:.0f} s, H_MAX {H_MAX:.0f} s, interference margin {C.IOT} dB, scheme {C.SCHEME}, q_target {SD.QT}', flush=True)
    with Pool(procs) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(dict(D=D, H_MAX=H_MAX, iot_db=C.IOT, scheme=C.SCHEME, runs=res), open(SD.OUT, 'w'), indent=1)
    print(f'\n=== mean of 3 test seeds: mean hold per flight (s) | 4D buffer = 95th pct of hold (s) | N with buffer (limit {N_REF}) | lost evidence ===')
    for M in Ms:
        for la in lams:
            rr = [r for r in res if r['M'] == M and r['lam'] == la]
            print(f'M={M:>3} {la:>4.0f}/h | ' + ' | '.join(
                f"{p}: {np.mean([r['pol'][p]['mean_hold'] for r in rr]):6.1f} {np.mean([r['pol'][p]['buffer95'] for r in rr]):6.1f} "
                f"N {np.mean([r['pol'][p]['N_buf'] for r in rr]):5.1f}{'*' if np.mean([r['pol'][p]['N_buf'] for r in rr]) > N_REF else ' '} "
                f"lost {np.mean([r['pol'][p]['lost'] for r in rr]):.3f}" for p in POLS))
    print('(* = airborne count with the buffer exceeds N_ref)')


if __name__ == '__main__':
    main()
