#!/usr/bin/env python3
"""Stage 2 of the BUBBLES-facing environment: sim_hold.py + the SEMANTIC LEVEL of what is sent as a decision.

Every UAV with undelivered evidence decides, each slot, the semantic level of the images it sends and its matching weight;
the base stations assign the channels by weighted marginal-gain matching (as in sim_hold.py). Evidence must still be
delivered, an overdue item still makes its UAV hold.
  levels      0 = symbolic layer only (recall q_sym = 0.397, no tokens); 1..4 = gated tokens reaching recall 0.42 / 0.44 /
              0.46 / 0.48 (cost per token from the measured curves, sim_defer_q.JCL). An item is a batch of images, the level
              may change from slot to slot: the recall of an item is the delivered-fraction-weighted mean of the levels used.
  objective   minimise the holding (-> 4D buffer, airborne count against N_ref), as in sim_hold.py
  constraint  mean recall over all evidence >= QBAR (default 0.44 = the fixed level of stage 1; lost evidence counts 0)
  virtual queue (Lyapunov)  Z <- max(Z + sum over the slot of delivered fraction x (QBAR - recall of its level), 0)
              BUB_ZMODE = net (default): one queue for the network (constraint on the network mean);
              BUB_ZMODE = uav: one queue per flight (every flight has to meet QBAR with its own evidence - the fair version).
Baselines (no learning; matching weight = urgency 1 / time left before a hold)
  fixed l     one level for everything (fixed 2 = stage 1)
  lyap V      drift-plus-penalty greedy, the usual Lyapunov baseline: per UAV the level maximising
              (V x urgency + Z x (recall_l - QBAR)) x best-channel efficiency_l ; matching weight = that bracket.
  snr th[z k] two-level channel-threshold rule (snr_action), static or driven by the constraint queue; urgency weight.
  lagr k      min-airtime Lagrangian rule (lagr_action): level minimising airtime - k Z (recall_l - QBAR), urgency weight.
  lyapfb V    the same as lyap with a deadline-aware fallback (lyapfb_action): a level that cannot be finished before the hold is
              replaced by the highest level that can, else by the symbolic layer.

Usage: python sim_holdq.py <sim_inputs.json> <out.json> <unused> table [procs]
       env: BUB_QBAR, BUB_QMARGIN, BUB_ZMODE, BUB_VS, BUB_NOFIXED, BUB_MS, BUB_LAMS, BUB_IOT_DB, BUB_D, BUB_HMAX, CAP_SEEDS"""
import sys, json, os, time
import numpy as np
from multiprocessing import Pool
import sim_hold as H
import sim_defer_q as Q
C, SD = H.C, H.SD

QL = np.array([SD.Q_SYM] + list(Q.LEVELS))                     # recall of each level
JCL = [np.zeros_like(Q.JCL[0])] + list(Q.JCL)                  # token cost per level and SNR grid point
L = len(QL)
QBAR = float(os.environ.get('BUB_QBAR', 0.44))
QINT = QBAR + float(os.environ.get('BUB_QMARGIN', 0.0))        # target used inside the constraint queue: a small margin absorbs the residual queue
ZMODE = os.environ.get('BUB_ZMODE', 'net')
DIG0 = np.where(SD.SE > 0, 8 * SD.SYM_B / np.maximum(SD.SE, 1e-12), np.inf)    # symbolic layer alone: no side information of the tokens
if C.SCHEME != 'gated':                                        # digital reference schemes (BUB_SCHEME): one operating point, the bit stream that reaches
    JCL = [np.zeros_like(Q.JCL[0])] * L                        # the recall target of sim_defer (sim_ch has put its cost in SD.DIG); no tokens, no levels.
    DIG0 = SD.DIG; QL = np.full(L, SD.QT)                      # Before this fix the token cost of the gated scheme was added on top (10-05: HEVC cells invalid).


def eff_l(snr, ntok, l):
    """fraction of an item delivered per slot on one channel at level l, for an array of SNR (dB)"""
    i = np.clip(np.rint((snr - SD.GRID[0]) / 0.25).astype(int), 0, len(SD.GRID) - 1)
    uses = SD.M * (DIG0[i] if l == 0 else SD.DIG[i] + JCL[l][i] * ntok)
    return np.where(np.isfinite(uses), C.W * C.DT / uses, 0.0)


def alloc(s1, gdb, ntok, rem, w, lev, M):
    """sim_hold.alloc with a semantic level per candidate"""
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
            r = float(eff_l(s1[j] - 10 * np.log10(len(ch)) + gdb[j][ch], ntok[j], lev[j]).sum())
            dv = w[j] * (min(rem[j], r) - min(rem[j], cur[j]))
            if dv > bv:
                bv, bj, bc, br = dv, j, c, r
        if bj < 0:
            break
        held[bj].append(bc); cur[bj] = br; free[bc] = False
    return np.minimum(rem, cur)


class Sim(H.Sim):
    def __init__(self, flights, M):
        super().__init__(flights, M)
        for f in flights:
            f['qsum'] = 0.0; f['Z'] = 0.0; f['lev'] = np.zeros(L)              # recall delivered, own constraint queue, evidence sent per level
        self.Z = 0.0

    def zvec(self):
        """constraint queue seen by each candidate"""
        return np.array([self.F[fi]['Z'] for fi, _, _ in self.info]) if ZMODE == 'uav' else np.full(len(self.info), self.Z)

    def e1(self):
        """levels x candidates: full-power efficiency on the best channel of the slot"""
        return np.array([[float(eff_l(np.array([self.s1[j] + self.gdb[j].max()]), it['ntok'], l)[0]) for j, (_, _, it) in enumerate(self.info)]
                         for l in range(L)])

    def slack(self):
        return np.array([H.D / C.DT - (self.s - it['cap']) for _, _, it in self.info])

    def step(self, w, lev=None):
        """returns (evidence delivered in the slot, recall delivered in the slot)"""
        dsum = qsum = 0.0
        if self.cand:
            rem = np.array([it['rem'] for _, _, it in self.info]); ntok = np.array([it['ntok'] for _, _, it in self.info])
            d = alloc(self.s1, self.gdb, ntok, rem, w, lev, self.M)
            for (fi, t, it), dj, l in zip(self.info, d, lev):
                f = self.F[fi]
                it['rem'] -= dj; f['qsum'] += dj * QL[l]; f['lev'][l] += dj; dsum += dj; qsum += dj * QL[l]
                f['Z'] = max(f['Z'] + dj * (QINT - QL[l]), 0.0)
                if it['rem'] <= 1e-9:
                    self.F[fi]['items'].remove(it); self.F[fi]['done_items'] += 1
            self.Z = max(self.Z + QINT * dsum - qsum, 0.0)
        self.s += 1
        return dsum, qsum


def lyap_action(sim, V, e=None, p=1.0):
    """drift-plus-penalty greedy level and matching weight of every candidate. p > 1 makes the urgency steeper:
    urgency = (1 / time left) x (D / time left)^(p - 1), equal to the plain 1 / time left right after capture and growing faster
    towards the hold (the learned policy turned out to do this: x 2 at 20-30 s left, x 4-8 below 20 s)."""
    e = sim.e1() if e is None else e
    sl = np.maximum(sim.slack(), 1.0)
    val = V * (1.0 / sl * (H.D / C.DT / sl) ** (p - 1.0))[None, :] + sim.zvec()[None, :] * (QL[:, None] - QINT)
    lev = np.argmax(np.where(e > 0, val * e, -np.inf), 0)
    return lev, np.maximum(val[lev, np.arange(e.shape[1])], 1e-6)


def lyapfb_action(sim, V, e=None):
    """drift-plus-penalty greedy with a deadline-aware fallback: if the greedy level cannot be finished before the hold (remaining
    fraction / best-channel efficiency > time left), take the highest level that still fits, else the symbolic layer; such a
    UAV keeps at least its urgency weight (the quality debt must not keep it from being served)"""
    e = sim.e1() if e is None else e
    lev, w = lyap_action(sim, V, e)
    slack = np.maximum(sim.slack(), 1.0); rem = np.array([it['rem'] for _, _, it in sim.info]); z = sim.zvec()
    need = rem[None, :] / np.maximum(e, 1e-12)
    for j in np.nonzero(need[lev, np.arange(len(lev))] > slack)[0]:
        ok = np.nonzero((e[:, j] > 0) & (need[:, j] <= slack[j]))[0]
        lev[j] = ok.max() if len(ok) else 0
        w[j] = max(V / slack[j] + z[j] * (QL[lev[j]] - QINT), V / slack[j])
    return lev, w


def lagr_action(sim, k, e=None):
    """min-airtime Lagrangian rule: per UAV the level minimising  (slots per item on its best channel) - k x Z x (recall_l - QBAR);
    the constraint queue is the multiplier of the recall constraint. Matching weight = urgency only, so that a UAV sending the
    symbolic layer is never held back by the quality debt (the weakness of the drift-plus-penalty weight)."""
    e = sim.e1() if e is None else e
    cost = np.where(e > 0, 1.0 / np.maximum(e, 1e-12), np.inf) - k * sim.zvec()[None, :] * (QL[:, None] - QINT)
    return np.argmin(cost, 0), 1.0 / np.maximum(sim.slack(), 1.0)


def snr_action(sim, theta, kz=0.0):
    """two-level channel-threshold rule (the structure the learned policy turned out to have): top level if the best-channel SNR of
    the slot is at least theta - kz x Z (dB), else the symbolic layer; matching weight = urgency. kz = 0: static threshold (the
    recall then follows from theta); kz > 0: the constraint queue lowers the threshold until the recall requirement is met."""
    snr = sim.s1 + np.array([g.max() for g in sim.gdb])
    top = np.array([float(eff_l(np.array([x]), it['ntok'], L - 1)[0]) > 0 for x, (_, _, it) in zip(snr, sim.info)])
    return np.where((snr >= theta - kz * sim.zvec()) & top, L - 1, 0), 1.0 / np.maximum(sim.slack(), 1.0)


def run(lam, M, seed, policy):
    """policy = 'fixed<l>', 'lyap<V>', 'lyapfb<V>', 'lagr<k>', 'snr<theta>', 'snr<theta>z<kz>', 'lyapp<p>v<V>' or 'edf'; returns raw(F)"""
    return raw(run_flights(lam, M, seed, policy))


def run_flights(lam, M, seed, policy, dep=None):
    """as run, returns the flights; dep = departure times from the strategic deconfliction (see sim_hold.make_flights)"""
    F = H.make_flights(lam, M, seed, dep=dep)
    if not F:
        return F
    sim = Sim(F, M)
    while not sim.finished():
        sim.get()
        if not sim.cand:
            sim.step(None)
        elif policy == 'edf':                                                  # fixed 0.44 level, sequential: earliest hold first takes what it wants
            order = np.argsort([it['cap'] for _, _, it in sim.info], kind='stable')
            w = np.zeros(len(order)); w[order] = 10.0 ** (-3.0 * np.arange(len(order)))
            sim.step(w, np.full(len(sim.cand), 2))
        elif policy.startswith('fixed'):
            sim.step(1.0 / np.maximum(sim.slack(), 1.0), np.full(len(sim.cand), int(policy[5:])))
        elif policy.startswith('snr'):                                         # 'snr<theta>' or 'snr<theta>z<kz>'
            th, _, kz = policy[3:].partition('z')
            lev, w = snr_action(sim, float(th), float(kz or 0))
            sim.step(w, lev)
        elif policy.startswith('lagr'):
            lev, w = lagr_action(sim, float(policy[4:]))
            sim.step(w, lev)
        elif policy.startswith('lyapp'):                                       # 'lyapp<p>v<V>': steeper urgency
            pp, _, vv = policy[5:].partition('v')
            lev, w = lyap_action(sim, float(vv), p=float(pp))
            sim.step(w, lev)
        elif policy.startswith('lyapfb'):
            lev, w = lyapfb_action(sim, float(policy[6:]))
            sim.step(w, lev)
        else:
            lev, w = lyap_action(sim, float(policy[4:]))
            sim.step(w, lev)
    return F


def raw(F):
    """per run: hold per counted flight (s), lost items, items, recall sum, evidence sent per level, mean recall per flight"""
    fl = [f for f in F if f['counted']]
    return (np.array([f['hold'] for f in fl], float) * C.DT, sum(f['lost'] for f in fl), sum(f['done_items'] + f['lost'] for f in fl),
            float(sum(f['qsum'] for f in fl)), np.sum([f['lev'] for f in fl], 0) if fl else np.zeros(L),
            np.array([f['qsum'] / max(f['done_items'] + f['lost'], 1) for f in fl]))


def summary(rs, lam):
    """pooled metrics of several runs (outputs of raw) at one arrival rate"""
    hold = np.concatenate([r[0] for r in rs]); items = max(sum(r[2] for r in rs), 1)
    b = float(np.percentile(hold, 95)); lev = np.sum([r[4] for r in rs], 0); fr = np.concatenate([r[5] for r in rs])
    return dict(flights=len(hold), mean_hold=float(hold.mean()), buffer95=b, N_buf=lam * (H.T_PLAN + b) / 3600, any_hold=float((hold > 0).mean()),
                lost=sum(r[1] for r in rs) / items, recall=sum(r[3] for r in rs) / items, level_share=(lev / max(lev.sum(), 1e-9)).tolist(),
                flight_recall_p05=float(np.percentile(fr, 5)), flight_recall_min=float(fr.min()), flights_below=float((fr < QBAR - 5e-3).mean()))


def one(args):
    p, lam, M, seed = args
    return run(lam, M, seed, p)


def main():
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    t0 = time.time()
    Ms = [int(x) for x in os.environ.get('BUB_MS', '4,5').split(',')]
    lams = [float(x) for x in os.environ.get('BUB_LAMS', '40,60').split(',')]
    pols = ([] if os.environ.get('BUB_NOFIXED') == '1' else [f'fixed{l}' for l in range(L)]) + [f'lyap{v}' for v in os.environ.get('BUB_VS', '0.3,1,3,10,30').split(',')]
    pols += [f'lyapfb{v}' for v in os.environ.get('BUB_VS_FB', '').split(',') if v] + [f'lagr{v}' for v in os.environ.get('BUB_KS', '').split(',') if v]
    pols += [f'snr{v}' for v in os.environ.get('BUB_THS', '').split(',') if v] + [x for x in os.environ.get('BUB_EXTRA', '').split(',') if x]
    seeds = list(range(401, 401 + int(os.environ.get('CAP_SEEDS', 6))))
    jobs = [(p, la, M, s) for M in Ms for la in lams for p in pols for s in seeds]
    print(f'{len(jobs)} runs; levels (recall) {np.round(QL, 3).tolist()}; required mean recall {QBAR} ({ZMODE} queue, target inside the queue {QINT:.4f}); D {H.D:.0f} s, H_MAX {H.H_MAX:.0f} s, '
          f'interference margin {C.IOT} dB; {len(seeds)} traffic seeds pooled', flush=True)
    with Pool(procs) as pool:
        res = dict(zip(jobs, pool.map(one, jobs, chunksize=1)))
    out = dict(levels=QL.tolist(), qbar=QBAR, qint=QINT, zmode=ZMODE, D=H.D, H_max=H.H_MAX, iot_db=C.IOT, seeds=seeds, cells={})
    for M in Ms:
        for la in lams:
            print(f'--- M={M}, {la:.0f} flights/h: mean hold (s) | 4D buffer (s) | N with buffer (limit {H.N_REF}) | lost | mean recall (* = meets {QBAR}) | '
                  f'share of the evidence sent at each level | per-flight recall: 5th pct, min, share of flights below {QBAR - 5e-3:.3f}')
            for p in pols:
                m = summary([res[(p, la, M, s)] for s in seeds], la); out['cells'][f'{M}|{la:.0f}|{p}'] = m
                print(f"    {p:<9} {m['mean_hold']:7.1f} | {m['buffer95']:7.1f} | {m['N_buf']:5.1f}{'!' if m['N_buf'] > H.N_REF else ' '} | {m['lost']:.4f} | "
                      f"{m['recall']:.4f}{'*' if m['recall'] >= QBAR - 5e-4 else ' '} | " + ' '.join(f'{x:.2f}' for x in m['level_share'])
                      + f" | {m['flight_recall_p05']:.3f} {m['flight_recall_min']:.3f} {m['flights_below']:.3f}", flush=True)
    out['secs'] = round(time.time() - t0, 1)
    json.dump(out, open(SD.OUT, 'w'), indent=1)
    print(f'DONE ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
