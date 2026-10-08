#!/usr/bin/env python3
"""U-space capacity with the strategic deconfliction in the loop (sim_strat.py + sim_holdq.py).

Chain:  requests -> 4D volumes with temporal buffer b granted by the service provider (ground delay or rejection)
        -> flights, uplink of the evidence, holds (sim_holdq)  -> conformance = share of flights whose total hold stays within b
        -> conflicts of the flown trajectories (BUBBLES criterion).
For every (uplink policy, channels M, requested rate, buffer b), flights of several traffic seeds pooled:
  carried rate, rejected share, ground delay; conformance; mean hold; evidence lost; recall of the delivered evidence;
  mean airborne count (against N_ref); conflict events per flight hour.
A cell is FEASIBLE if  rejected <= 5 %, conformance >= 95 %, airborne count <= N_ref, lost <= 1 %, recall >= QBAR.
Capacity of (policy, M) = largest requested rate that is feasible for some buffer; the buffer is the design variable that
trades conformance (needs a large buffer when the uplink makes flights hold) against what the deconfliction can fit.
b = -1: no strategic deconfliction (every request departs when requested; reference).

Usage: python cap_strat.py <sim_inputs.json> <out.json> <unused> cap [procs=12]
       env: BUB_MS, BUB_LAMS, STRAT_BUFS, BUB_POLS, CAP_LEARNED (name=ckpt.pt,...), CAP_KIND, CAP_SEEDS, CAP_SEED0, CAP_RAW, CAP_SCHED_CACHE (directory), BUB_QBAR, BUB_QMARGIN, BUB_NTOK_CV, BUB_ALTS, BUB_IOT_DB, STRAT_GMAX, STRAT_RPORT"""
import sys, json, os, time
import numpy as np
from multiprocessing import Pool
import sim_holdq as S
import sim_strat as ST
H, C, SD = S.H, S.C, S.SD

MS = [int(x) for x in os.environ.get('BUB_MS', '3,4,5').split(',')]
LAMS = [float(x) for x in os.environ.get('BUB_LAMS', '40,50,60,65').split(',')]
BUFS = [float(x) for x in os.environ.get('STRAT_BUFS', '-1,0,10,30,60,120').split(',')]
POLS = [x for x in os.environ.get('BUB_POLS', 'edf,fixed2,lyap30,lyapp3v30').split(',') if x]
SEED0 = int(os.environ.get('CAP_SEED0', 401))                  # first traffic seed (401 = the set used so far; another value = traffic never looked at)
SEEDS = list(range(SEED0, SEED0 + int(os.environ.get('CAP_SEEDS', 6))))
RAW = os.environ.get('CAP_RAW', '') == '1'                      # also write <out>_raw.json: per-flight recall, hold and ground delay of every cell
HOURS, WARM = 1.5, 0.5
SP, LW = None, {}                                              # the learner's module (imported only when needed) and the learned policies: name -> weights
# CAP_KIND: which learner the CAP_LEARNED checkpoints belong to - sppo (structured-actor PPO, default), hppo (hppo_hold.py), d3qn / td3
# (offpol_hold.py). One kind per run, with the SAME environment switches as in its training (PPO_PRIOR, PPO_BASE, PPO_REWARD, BUB_V, ...).
KIND = os.environ.get('CAP_KIND', 'sppo')


def run_learned(lam, M, seed, w, dep):
    """flights flown under a trained policy, greedy actions"""
    F = H.make_flights(lam, M, seed, dep=dep)
    if not F:
        return F
    sim = S.Sim(F, M); rng = np.random.default_rng(0)
    while not sim.finished():
        nh = sim.get()
        if not sim.cand:
            sim.step(None)
        elif KIND in ('sppo', 'hppo'):
            out = SP.act(sim, nh, w, rng, False)
            sim.step(out[-1], out[1])
        else:                                                  # as in offpol_hold.rollout, mode 'eval'
            e = sim.e1(); X, slack = SP.HP.observe(sim, nh, e)
            lev, a, _ = SP.decide(w, X, e.T > 0, rng, 0.0)
            sim.step(10.0 ** a / np.maximum(slack, 1.0), lev)
    return F


CACHE = os.environ.get('CAP_SCHED_CACHE', '')                  # directory for the granted departure times (they depend only on the traffic seed,
                                                               # the rate, the buffer and the geometry, not on the scheduler under test)


def sched_job(args):
    lam, seed, b = args
    if b < 0:
        return None
    if not CACHE:
        return ST.schedule(lam, seed, b)
    fn = os.path.join(CACHE, f"sched_a{'-'.join(f'{a:g}' for a in C.ALTS)}_p{C.NPORT}_g{ST.G_MAX:g}_r{ST.RPORT:g}_l{lam:g}_s{seed}_b{b:g}.npy")
    if os.path.exists(fn):
        return np.load(fn)
    dep = np.asarray(ST.schedule(lam, seed, b), dtype=float)
    tmp = fn + f'.{os.getpid()}.tmp.npy'; np.save(tmp, dep); os.replace(tmp, fn)
    return dep


def sim_job(args):
    pol, M, lam, seed, b, dep = args
    req = H.requests(lam, seed, HOURS, WARM); cnt_req = req >= WARM * 3600
    F = run_learned(lam, M, seed, LW[pol], dep) if pol in LW else S.run_flights(lam, M, seed, pol, dep=dep)
    fl = [f for f in F if f['counted']]
    s_end = max([f['landed'] or 0 for f in F] + [f['s0'] + f['n'] + f['hold'] for f in F]) if F else 0
    sec, ev, air = ST.tactical(F, seed, s_end) if F else (0, 0, 0)
    return dict(n_req=int(cnt_req.sum()), n_acc=len(fl), hold=[f['hold'] * C.DT for f in fl], gdelay=[f['gdelay'] for f in fl],
                lost=sum(f['lost'] for f in fl), done=sum(f['done_items'] for f in fl), qsum=float(sum(f['qsum'] for f in fl)),
                frec=[f['qsum'] / max(f['done_items'] + f['lost'], 1) for f in fl],
                air=float(sum(((f['landed'] if f['landed'] is not None else s_end) - f['s0']) for f in fl) * C.DT),
                conf_sec=sec, conf_ev=ev, air_all=air * C.DT)


def cell(rs, lam, b):
    hold = np.concatenate([np.array(r['hold']) for r in rs]); gd = np.concatenate([np.array(r['gdelay']) for r in rs])
    n_req = sum(r['n_req'] for r in rs); n_acc = sum(r['n_acc'] for r in rs)
    done = max(sum(r['done'] for r in rs), 1); lost = sum(r['lost'] for r in rs)
    hrs = len(rs) * HOURS
    m = dict(carried=n_acc / hrs, rejected=1 - n_acc / max(n_req, 1), gdelay_mean=float(gd.mean()) if len(gd) else 0.0,
             gdelay_p95=float(np.percentile(gd, 95)) if len(gd) else 0.0,
             conformance=float((hold <= max(b, 0.0)).mean()) if len(hold) else 0.0, mean_hold=float(hold.mean()) if len(hold) else 0.0,
             hold_p95=float(np.percentile(hold, 95)) if len(hold) else 0.0, lost=lost / (done + lost), recall=sum(r['qsum'] for r in rs) / done,
             n_air=sum(r['air'] for r in rs) / (hrs * 3600), conf_per_fh=sum(r['conf_ev'] for r in rs) / max(sum(r['air_all'] for r in rs) / 3600, 1e-9),
             conf_sec_per_fh=sum(r['conf_sec'] for r in rs) / max(sum(r['air_all'] for r in rs) / 3600, 1e-9))
    fr = np.concatenate([np.array(r['frec']) for r in rs]) if n_acc else np.zeros(0)
    m['flights_below'] = float((fr < S.QBAR - 5e-3).mean()) if len(fr) else 1.0          # flights whose own mean recall misses the requirement by > 0.005
    m['flight_recall_p05'] = float(np.percentile(fr, 5)) if len(fr) else 0.0
    m['feasible'] = bool(m['rejected'] <= 0.05 and m['conformance'] >= 0.95 and m['n_air'] <= H.N_REF and m['lost'] <= 0.01 and m['recall'] >= S.QBAR - 5e-4
                         and (S.ZMODE != 'uav' or m['flights_below'] <= 0.05))
    return m


def main():
    global SP
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    t0 = time.time()
    if os.environ.get('CAP_LEARNED'):                                          # name=checkpoint.pt,... (structured-actor PPO); loaded before the pool forks
        argv = sys.argv; sys.argv = argv[:4] + ['4', '0', '1', '1']             # hppo_hold reads M / procs / iterations from argv at import; unused here
        import torch
        if KIND == 'sppo':
            import sppo_hold as mod; mk, wf = mod.SNet, mod.weights
        elif KIND == 'hppo':
            import hppo_hold as mod; mk, wf = mod.Net, mod.weights
        else:
            os.environ['OFF_ALGO'] = KIND
            import offpol_hold as mod; mk, wf = (mod.QNet if KIND == 'd3qn' else mod.Actor), mod.npw
        SP = mod; sys.argv = argv
        for item in os.environ['CAP_LEARNED'].split(','):
            name, path = item.split('=', 1)
            net = mk(); net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)); LW[name] = wf(net)
            POLS.append(name)
            print(f"learned policy {name} ({KIND}): {path}" + (f"  p {LW[name]['par']['p']:.2f}  V {LW[name]['par']['V']:.3f}  kappa {LW[name]['par']['kappa']:.1f}"
                                                              if KIND == 'sppo' else ''), flush=True)
    print(f'policies {POLS}; channels {MS}; requested rates {LAMS}; buffers {BUFS} (-1 = no strategic deconfliction); {len(SEEDS)} traffic seeds; layers {C.ALTS} m; '
          f'take-off / landing sites {C.NPORT}; constraint queue {S.ZMODE}; token dispersion {H.NTOK_CV}; recall requirement {S.QBAR} (inside the queue {S.QINT}); ground delay offered up to {ST.G_MAX:.0f} s; port zone {ST.RPORT:.0f} m', flush=True)
    with Pool(procs) as pool:
        sj = [(la, s, b) for la in LAMS for s in SEEDS for b in BUFS]
        deps = dict(zip(sj, pool.map(sched_job, sj, chunksize=1)))
        print(f'schedules done ({time.time() - t0:.0f}s)', flush=True)
        jobs = [(p, M, la, s, b, deps[(la, s, b)]) for M in MS[::-1] for la in LAMS for b in BUFS for p in POLS for s in SEEDS]
        res = pool.map(sim_job, jobs, chunksize=1)
    R = {}
    for j, r in zip(jobs, res):
        R.setdefault((j[0], j[1], j[2], j[4]), []).append(r)
    out = dict(pols=POLS, Ms=MS, lams=LAMS, bufs=BUFS, seeds=SEEDS, alts=C.ALTS, ports=C.NPORT, ntok_cv=H.NTOK_CV, qbar=S.QBAR, qint=S.QINT, gmax=ST.G_MAX, rport=ST.RPORT,
               cells={}, capacity={})
    for p in POLS:
        for M in MS:
            print(f'\n=== {p}, M = {M} ({M * C.W / 1e6:.2f} MHz): carried /h | rejected | ground delay mean (s) | conformance | mean hold (s) | lost | recall | flights below {S.QBAR - 5e-3:.3f} | airborne | conflicts per flight hour | F = feasible')
            cap = (0.0, None)
            for la in LAMS:
                for b in BUFS:
                    m = cell(R[(p, M, la, b)], la, b); out['cells'][f'{p}|{M}|{la:.0f}|{b:.0f}'] = m
                    if m['feasible'] and b >= 0 and la > cap[0]:
                        cap = (la, b)
                    print(f"  {la:>3.0f}/h buffer {('none' if b < 0 else '%.0f s' % b):>5}: {m['carried']:5.1f} | {m['rejected']:.3f} | {m['gdelay_mean']:6.1f} | {m['conformance']:.3f} | "
                          f"{m['mean_hold']:7.1f} | {m['lost']:.4f} | {m['recall']:.4f} | {m['flights_below']:.3f} | {m['n_air']:5.2f} | {m['conf_per_fh']:6.3f} | {'F' if m['feasible'] else '-'}", flush=True)
            out['capacity'][f'{p}|{M}'] = dict(rate=cap[0], buffer=cap[1])
            print(f"  -> capacity: {cap[0]:.0f} flights/h" + (f' with buffer {cap[1]:.0f} s' if cap[1] is not None else ' (no feasible cell on the grid)'))
    out['secs'] = round(time.time() - t0, 1)
    json.dump(out, open(SD.OUT, 'w'), indent=1)
    if RAW:
        cat = lambda rs, k, nd: np.round(np.concatenate([np.array(r[k], dtype=float) for r in rs]), nd).tolist()
        json.dump({f'{k[0]}|{k[1]}|{k[2]:.0f}|{k[3]:.0f}': dict(n=[len(r['frec']) for r in rs], frec=cat(rs, 'frec', 4), hold=cat(rs, 'hold', 1), gdelay=cat(rs, 'gdelay', 1)) for k, rs in R.items()},   # n = flights per traffic seed, in seed order
                  open(SD.OUT.replace('.json', '_raw.json'), 'w'))
    print(f'DONE ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
