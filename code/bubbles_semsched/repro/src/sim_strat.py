#!/usr/bin/env python3
"""BUBBLES side of the model: volume-based strategic deconfliction with temporal buffers, and the conflicts of the flown
trajectories (user decision 2026-10-05: merge this into the simulator as the U-space metric).

  strategic   A flight that may be up to b seconds late occupies every point of its planned path from the planned time until b
              seconds later (D2.1: 4D volumes = trajectory + temporal buffer, linked to a conformance ratio). Two flights of the
              same layer are in conflict if two such occupancies overlap in time at points closer than 2 x d_SL = 271.22 m (the
              conflict-declaration criterion, D2.1 Table G-4; layers >= 30.32 m apart are vertically separated). The service
              provider grants each request, first come first served, the earliest departure not before the requested time that is
              conflict-free against the flights already accepted; a request that cannot depart within G_MAX is rejected.
  tactical    On the flown trajectories (a hold stops the plan clock): seconds and events in which two flights of the same layer
              are closer than 2 x d_SL. With all holds within the buffer there are none by construction; a flight whose total hold
              exceeds its buffer has left its 4D volume (non-conformant) and may run into the next one.
  port zone   points within R_PORT of the single port are left out of both tests (take-off and landing are the port's business);
              the single central port and the shared snake route are OUR geometry, not BUBBLES'.
Used by cap_strat.py. Geometry and timing are those of sim_ch.flight."""
import os
import numpy as np
import sim_hold as H
C = H.C

DSEP = 271.22
RPORT = float(os.environ.get('STRAT_RPORT', 300.0))
G_MAX = float(os.environ.get('STRAT_GMAX', 600.0))             # longest ground delay offered before a request is rejected (s)
STEP = 1.0                                                     # same sample points as the flown trajectories (slot centres): the two tests are then consistent
T_F = 2 * C.T_TR + C.K * C.T_CAP + (C.K - 1) * C.T_CRUISE      # 880 s


def port_dist(p):
    """distance of every point of p (n x 2) to the nearest take-off / landing site"""
    return np.min(np.linalg.norm(p[:, None, :] - C.PORTS[None, :, :], axis=2), axis=1)


def plan(seed, fi, step=STEP):
    """layer index and planned positions every `step` seconds of request fi (same random draws as sim_ch.flight)"""
    frng = np.random.default_rng(seed * 100003 + fi)
    li = int(frng.integers(len(C.ALTS))); start = frng.integers(len(C.POI))
    stops = C.POI[(start + np.arange(C.K)) % len(C.POI)]
    seg, t = [(0.0, C.T_TR, C.nearest_port(stops[0]), stops[0])], C.T_TR
    for k in range(C.K):
        seg.append((t, t + C.T_CAP, stops[k], stops[k])); t += C.T_CAP
        nxt = stops[k + 1] if k + 1 < C.K else C.nearest_port(stops[-1])
        dur = C.T_CRUISE if k + 1 < C.K else C.T_TR
        seg.append((t, t + dur, stops[k], nxt)); t += dur
    ts = np.arange(0, T_F, step) + step / 2
    pos = np.empty((len(ts), 2))
    for a, b, pa, pb in seg:
        m = (ts >= a) & (ts < b)
        pos[m] = pa + (pb - pa) * ((ts[m] - a) / max(b - a, 1e-9))[:, None]
    return li, pos


def offsets(pa, pb, skip_port=True):
    """time offsets (s) t_b - t_a (plan times) at which the two paths are closer than DSEP, as merged intervals [lo, hi]"""
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2) < DSEP
    if skip_port:
        d &= (port_dist(pa) > RPORT)[:, None] & (port_dist(pb) > RPORT)[None, :]
    i, j = np.nonzero(d)
    if len(i) == 0:
        return np.zeros((0, 2))
    off = np.unique((j - i) * STEP)
    cut = np.nonzero(np.diff(off) > STEP)[0]
    return np.stack([off[np.r_[0, cut + 1]], off[np.r_[cut, len(off) - 1]]], 1)


def schedule(lam, seed, b, skip_port=True, hours=1.5, warm=0.5):
    """departure time granted to every request of (lam, seed) with temporal buffer b; nan = rejected"""
    req = H.requests(lam, seed, hours, warm)
    paths = [plan(seed, i) for i in range(len(req))]
    dep = np.full(len(req), np.nan); acc = []
    for a in range(len(req)):
        la, pa = paths[a]
        forb = []
        for bi in acc:
            lb, pb = paths[bi]; tb = dep[bi]
            if lb != la or tb + T_F + b < req[a] - 1 or tb - T_F - b > req[a] + G_MAX + 1:
                continue
            for lo, hi in offsets(pa, pb, skip_port):                          # conflict iff ta within [tb + off - b, tb + off + b]
                forb.append((tb + lo - b - STEP, tb + hi + b + STEP))
        t = float(np.ceil(req[a]))                                             # departures on whole seconds: plan and flown trajectory share the sample points
        for lo, hi in sorted(forb):
            if lo <= t <= hi:
                t = float(np.floor(hi) + 1)
        if t - req[a] <= G_MAX:
            dep[a] = t; acc.append(a)
    return dep


def tactical(F, seed, s_end, skip_port=True):
    """conflicts of the flown trajectories: (seconds in conflict, conflict events, airborne seconds), summed over the flight pairs
    of the same layer; s_end = last simulated slot (for flights that have not landed)"""
    tr = []
    for f in F:
        li, P = plan(seed, f['fi'], 1.0)
        e = f['landed'] if f['landed'] is not None else s_end
        s = np.arange(f['s0'], max(e, f['s0']))
        tau = np.minimum(s - f['s0'] - np.searchsorted(np.array(f['hs']), s), len(P) - 1)
        tr.append((li, f['s0'], f['s0'] + len(s), P[tau]))
    sec = ev = 0
    for a in range(len(tr)):
        la, a0, a1, pa = tr[a]
        for b in range(a + 1, len(tr)):
            lb, b0, b1, pb = tr[b]
            lo, hi = max(a0, b0), min(a1, b1)
            if lb != la or hi <= lo:
                continue
            xa, xb = pa[lo - a0:hi - a0], pb[lo - b0:hi - b0]
            c = np.linalg.norm(xa - xb, axis=1) < DSEP
            if skip_port:
                c &= (port_dist(xa) > RPORT) & (port_dist(xb) > RPORT)
            if c.any():
                sec += int(c.sum()); ev += int(c[0]) + int(np.sum(c[1:] & ~c[:-1]))
    return sec, ev, int(sum(t[2] - t[1] for t in tr))
