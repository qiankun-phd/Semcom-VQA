#!/usr/bin/env python3
"""Uplink simulator with a STANDARD channel model and explicit channel / power decisions (replaces the empirical-SNR
model of sim_defer.py; user decision 2026-10-04: follow the common practice of resource-allocation papers).

Network   5 x 5 km airspace; 4 base stations (2 x 2 grid, height 25 m) share ONE pool of M orthogonal uplink channels of
          180 kHz (no reuse); a UAV is served by the base station with the best large-scale gain.
Flights   Poisson arrivals; altitude layer 40 / 70 / 100 m; 20 stops on a 282 m grid of inspection points (snake order),
          14 m/s (20 s between stops), 10 s capture at a stop, hover window W (0 with deferred upload), 150 s transit in/out.
Channel   h_{u,m}(t) = sqrt(beta_u(t)) g_{u,m}(t)
          large scale beta: 3GPP TR 36.777 Annex B, UMa-AV: LoS probability, LoS / NLoS path loss, log-normal shadowing
            (sigma_LoS = 4.64 exp(-0.0066 h), sigma_NLoS = 6 dB). LoS state and shadowing are drawn per flight segment
            (a hover or a leg) and per base station; f_c = 3.5 GHz (formulas take f_c in GHz as in TR 38.901 - to re-check).
          small scale g: Rician (K = 10 dB) if LoS, Rayleigh if NLoS; block fading, independent per slot and per channel.
          SINR_{u,m} = p_{u,m} beta_u |g_{u,m}|^2 / (N0 W F I), N0 = -174 dBm/Hz, noise figure F = 5 dB,
            I = uplink interference margin (BUB_IOT_DB), calibrated so that the wideband SINR at 40 m matches the measured
            NR n78 statistics (median 5.7 dB); see mode 'calib'.
          Perfect CSI in the current slot; the mean large-scale gain along the planned path is known in advance ('planned').
Power     a UAV has P_max = 23 dBm in total; on n channels it uses P_max / n on each (equal split; an optimised split is
          the continuous decision left for the learner).
Evidence  as in sim_defer.py: one item per stop (336 images), semantic cost per image from the measured recall-vs-SNR curves;
          deadline D after capture and before landing; a UAV transmits one item per slot.
Decision  per slot: which item gets which channels. Baselines here hand out channels in priority order; an item takes its
          best free channels, as many as maximise what it can deliver this slot (more channels = less power on each).
            edf / maxeff / slack / slack+drop  as before, with efficiency = best-channel, full-power efficiency
Bound     LP relaxation: every channel-slot given to an item yields its best-channel full-power efficiency, an item cannot
          get more in a slot than it could with all channels to itself; fractional sharing. Upper bound (optimistic).

Usage: python sim_ch.py <sim_inputs.json> <out.json> <q_target> calib|sweep|one [args]"""
import json, sys, os, time
import numpy as np
from multiprocessing import Pool
import sim_defer as SD            # cost tables (per-image channel uses vs SNR) from the measured curves; argv[1..3]

AREA, NG, SP = 5000.0, 18, 282.0
BS = np.array([[1250., 1250.], [1250., 3750.], [3750., 1250.], [3750., 3750.]]); H_BS = 25.0
FC, P_MAX, N0, NF, W = 3.5, 23.0, -174.0, 5.0, 180e3
K_RICE = 10 ** (float(os.environ.get('BUB_KRICE_DB', 10.0)) / 10)       # Rician K factor of the small-scale fading (dB); 10 dB = main setting
IOT = float(os.environ.get('BUB_IOT_DB', 0.0))
ALTS = [float(x) for x in os.environ.get('BUB_ALTS', '40,70,100').split(',')]   # flight layers (m); 39,69.5,100 keeps them 30.5 m apart (BUBBLES: >= 30.32 m)
V, T_CAP, T_CRUISE, T_TR, K, DT = 14.0, 10.0, 20.0, 150.0, 20, 1.0
NMAX = 16                                                      # a UAV never spreads its power over more channels than this
# power split of a UAV over the channels it holds: share_m proportional to |g_m|^(2 alpha).  alpha = 0: equal split (default);
# alpha > 0: more power on the stronger channels; alpha < 0: towards channel inversion. Used to test whether the power
# split is worth making a (continuous) decision of the learner.
PALPHA = float(os.environ.get('BUB_PALPHA', 0.0))


def split_db(gdb):
    """per-channel power share in dB (sums to 0 dB in linear terms) for small-scale gains gdb (dB)"""
    if PALPHA == 0.0:
        return np.full(len(gdb), -10 * np.log10(len(gdb)))
    w = 10 ** (PALPHA * gdb / 10)
    return 10 * np.log10(w / w.sum())
# Transmission scheme. 'gated' = reference-gated JSCC tokens (default). The digital schemes send every image as a bit stream
# over the same channel (bits / SE(SINR), 3GPP attenuated Shannon) with the byte size that reaches the same recall target:
# 'hevc_inter' (reference-aided HEVC, gnss registration), 'hevc_intra', 'raw' (uncompressed-resolution original, 148.9 kB).
SCHEME = os.environ.get('BUB_SCHEME', 'gated')
if SCHEME != 'gated':
    _pts = {'hevc_intra': SD.INP['hevc_intra'], 'hevc_inter': SD.INP['hevc_inter_gnss'], 'raw': [[148882.0, SD.INP['raw_rec']]]}[SCHEME]
    _pts = sorted(_pts); _b = np.array([p[0] for p in _pts]); _r = np.maximum.accumulate(np.array([p[1] for p in _pts]))
    assert _r[-1] >= SD.QT, f'{SCHEME} cannot reach recall {SD.QT}'
    BYTES = float(_b[0] if _r[0] >= SD.QT else np.exp(np.interp(SD.QT, _r, np.log(_b))))
    SD.JC = np.zeros_like(SD.JC)
    SD.DIG = np.where(SD.SE > 0, 8 * (BYTES + SD.SYM_B) / np.maximum(SD.SE, 1e-12), np.inf)
POI = np.array([[SP / 2 + SP * (i if j % 2 == 0 else NG - 1 - i), SP / 2 + SP * j] for j in range(NG) for i in range(NG)])   # snake order
POI += (AREA - SP * NG) / 2


def plos(d2, h):
    p1 = 4300 * np.log10(h) - 3800; d1 = max(460 * np.log10(h) - 700, 18.0)
    d2 = np.maximum(d2, 1.0)
    return np.where(d2 <= d1, 1.0, d1 / d2 + np.exp(-d2 / p1) * (1 - d1 / d2))


def pl_los(d3):
    return 28.0 + 22 * np.log10(d3) + 20 * np.log10(FC)


def pl_nlos(d3, h):
    return np.maximum(pl_los(d3), -17.5 + (46 - 7 * np.log10(h)) * np.log10(d3) + 20 * np.log10(40 * np.pi * FC / 3))


def noise_dbm():
    return N0 + 10 * np.log10(W) + NF + IOT


# Take-off / landing sites. 1 (default): one port in the centre for everything (the transit of 150 s then implies up to ~23 m/s).
# 4: ports at the four base-station sites; a flight takes off from the port nearest to its first stop and lands at the port
# nearest to its last stop (BUBBLES D2.1 App. H: a mission has its own take-off point and landing point); transits <= 1.8 km.
NPORT = int(os.environ.get('BUB_PORTS', 1))
PORTS = BS.copy() if NPORT == 4 else np.array([[AREA / 2, AREA / 2]])


def nearest_port(p):
    return PORTS[np.argmin(np.linalg.norm(PORTS - p, axis=1))]


def flight(t0, frng, hover, M):
    """one flight: per airborne slot the realised large-scale gain (dB), the planned mean gain (dB), LoS flag, small-scale
    power gains per channel; capture times"""
    h = ALTS[frng.integers(len(ALTS))]
    start = frng.integers(len(POI))
    stops = POI[(start + np.arange(K)) % len(POI)]
    port, port_l = nearest_port(stops[0]), nearest_port(stops[-1])
    seg, t, caps = [(t0, t0 + T_TR, port, stops[0])], t0 + T_TR, []
    for k in range(K):
        seg.append((t, t + T_CAP + hover, stops[k], stops[k])); caps.append(t + T_CAP); t += T_CAP + hover
        nxt = stops[k + 1] if k + 1 < K else port_l
        dur = T_CRUISE if k + 1 < K else T_TR
        seg.append((t, t + dur, stops[k], nxt)); t += dur
    t_land = t
    s0, s1 = int(np.ceil(t0 / DT)), int(np.floor(t_land / DT))
    ts = (np.arange(s0, s1) + 0.5) * DT
    G = np.empty(len(ts)); GP = np.empty(len(ts)); LOS = np.zeros(len(ts), bool)
    for a, b, pa, pb in seg:
        m = (ts >= a) & (ts < b)
        if not m.any():
            continue
        pos = pa + (pb - pa) * ((ts[m] - a) / max(b - a, 1e-9))[:, None]
        d2 = np.linalg.norm(pos[:, None, :] - BS[None, :, :], axis=2)                    # slots x BS
        d3 = np.sqrt(d2 ** 2 + (h - H_BS) ** 2)
        p = plos(d2, h)
        los = frng.random(len(BS)) < p[len(p) // 2]                                      # one state per segment and BS
        sig = np.where(los, 4.64 * np.exp(-0.0066 * h), 6.0)
        sh = frng.normal(0, 1, len(BS)) * sig
        gain = -np.where(los[None, :], pl_los(d3), pl_nlos(d3, h)) + sh[None, :]
        b_best = np.argmax(gain, 1)
        G[m] = gain[np.arange(len(pos)), b_best]; LOS[m] = los[b_best]
        GP[m] = np.max(-(p * pl_los(d3) + (1 - p) * pl_nlos(d3, h)), 1)
    n = len(ts)
    ray = frng.exponential(1.0, (n, M))
    x = np.sqrt(K_RICE / (K_RICE + 1)) + frng.normal(0, 1, (n, M)) * np.sqrt(0.5 / (K_RICE + 1))
    y = frng.normal(0, 1, (n, M)) * np.sqrt(0.5 / (K_RICE + 1))
    g2 = np.where(LOS[:, None], x ** 2 + y ** 2, ray).astype(np.float32)
    return dict(s0=s0, s1=s1, h=h, G=G, GP=GP, LOS=LOS, g2=g2, caps=caps, t_land=t_land)


def make(lam, hover, D, mode, M, seed, hours=1.5, warm=0.5):
    rng = np.random.default_rng(seed)
    T = (hours + warm) * 3600
    t0s = np.cumsum(rng.exponential(3600 / lam, int(lam * (hours + warm) * 1.5) + 50))
    t0s = t0s[t0s < T]
    cv = 0.8; sig = np.sqrt(np.log(1 + cv ** 2)); mu_ln = np.log(SD.TOK) - sig ** 2 / 2
    F, items = [], []
    for fi, t0 in enumerate(t0s):
        frng = np.random.default_rng(seed * 100003 + fi)
        f = flight(t0, frng, hover, M)
        ntok = np.exp(frng.normal(mu_ln, sig, K))
        for k in range(K):
            r = int(np.ceil(f['caps'][k] / DT))
            d = int(np.floor((f['caps'][k] + hover if mode == 'window' else min(f['caps'][k] + D, f['t_land'])) / DT))
            d = min(max(d, r), f['s1'])
            items.append((fi, r, d, ntok[k], f['caps'][k] >= warm * 3600))
        F.append(f)
    it = np.array([(a, b, c) for a, b, c, _, _ in items], int)
    return F, it[:, 0], it[:, 1], it[:, 2], np.array([x[3] for x in items]), np.array([x[4] for x in items], bool)


def rate_n(snr1_db, g2_free, ntok):
    """delivered fraction of an item in this slot when it uses its n best free channels with P_max / n on each, n = 1..;
    snr1_db = P_max + large-scale gain - noise (dB), g2_free = small-scale power gains of the free channels"""
    g = np.sort(g2_free)[::-1][:NMAX]
    gdb = 10 * np.log10(np.maximum(g, 1e-6))
    out = np.empty(len(g))
    for n in range(1, len(g) + 1):
        out[n - 1] = SD.eff(snr1_db + split_db(gdb[:n]) + gdb[:n], ntok, W).sum()
    return out


def alloc_mg(F, fl, ntok, rem, s, cand, w, M, nz):
    """Explicit channel assignment by weighted marginal gain (greedy matching): repeatedly give one free channel to the
    (item, channel) pair with the largest  weight x extra delivered fraction, where an item takes its best free channel and
    re-splits its power over all channels it then holds. One item per UAV (its heaviest). Returns {item: delivered}."""
    top = {}
    for i, wi in zip(cand, w):
        if fl[i] not in top or wi > top[fl[i]][1]:
            top[fl[i]] = (i, wi)
    items = [v[0] for v in top.values()]; ws = [v[1] for v in top.values()]
    g2 = {i: 10 * np.log10(np.maximum(F[fl[i]]['g2'][s - F[fl[i]]['s0']], 1e-6)) for i in items}
    s1 = {i: P_MAX + F[fl[i]]['G'][s - F[fl[i]]['s0']] - nz for i in items}
    held = {i: [] for i in items}; cur = {i: 0.0 for i in items}
    free = np.ones(M, bool)
    while free.any():
        fidx = np.nonzero(free)[0]
        bv, bi, bc, br = 0.0, None, None, 0.0
        for i, wi in zip(items, ws):
            if cur[i] >= rem[i] or len(held[i]) >= NMAX:
                continue
            c = fidx[np.argmax(g2[i][fidx])]
            ch = held[i] + [c]
            r = float(SD.eff(s1[i] + split_db(g2[i][ch]) + g2[i][ch], ntok[i], W).sum())
            dv = wi * (min(rem[i], r) - min(rem[i], cur[i]))
            if dv > bv:
                bv, bi, bc, br = dv, i, c, r
        if bi is None:
            break
        held[bi].append(bc); cur[bi] = br; free[bc] = False
    return {i: min(rem[i], cur[i]) for i in items if held[i]}


def precompute(F, fl, rel, dl, ntok, M):
    """per item and lifetime slot: E1 = best-channel full-power efficiency, CAP = most it can deliver alone with all channels,
    EP1 = the same efficiency from the planned mean gain (no small-scale term)"""
    L = int((dl - rel).max()) if len(rel) else 1
    E1 = np.zeros((len(rel), max(L, 1)), np.float32); CAP = np.zeros_like(E1); EP1 = np.zeros_like(E1)
    nz = noise_dbm()
    for i in range(len(rel)):
        f = F[fl[i]]
        a, b = rel[i] - f['s0'], dl[i] - f['s0']
        if b <= a:
            continue
        s1 = P_MAX + f['G'][a:b] - nz
        g2 = f['g2'][a:b]
        E1[i, :b - a] = SD.eff(s1 + 10 * np.log10(np.maximum(g2.max(1), 1e-6)), ntok[i], W)
        EP1[i, :b - a] = SD.eff(P_MAX + f['GP'][a:b] - nz, ntok[i], W)
        CAP[i, :b - a] = [rate_n(s1[j], g2[j], ntok[i]).max() for j in range(b - a)]
    return E1, CAP, EP1


def run_policy(F, fl, rel, dl, ntok, E1, CAP, M, policy):
    n = len(rel)
    rem = np.ones(n); done = np.zeros(n, bool)
    order = np.argsort(rel, kind='stable'); ptr = 0
    live = np.zeros(0, int)
    nz = noise_dbm(); used = 0.0; slots = 0
    for s in range(int(rel.min()), int(dl.max())):
        a = ptr
        while ptr < n and rel[order[ptr]] <= s:
            ptr += 1
        if ptr > a:
            live = np.concatenate([live, order[a:ptr]])
        if len(live) == 0:
            continue
        live = live[(dl[live] > s) & ~done[live]]
        if len(live) == 0:
            continue
        slots += 1
        k = s - rel[live]
        e1 = E1[live, k].astype(float); cap = CAP[live, k].astype(float)
        ok = e1 > 0
        slack = (dl[live] - s).astype(float)
        if policy.endswith('+drop'):
            ok &= rem[live] / np.maximum(cap, 1e-30) <= slack
        cand, e1, slack = live[ok], e1[ok], slack[ok]
        if len(cand) == 0:
            continue
        base = policy.split('+')[0]
        if base.startswith('mg'):                                  # marginal-gain channel assignment; weight = 1 (throughput) or 1 / slack
            w = np.ones(len(cand)) if base == 'mg-thr' else 1.0 / slack
            got = alloc_mg(F, fl, ntok, rem, s, cand, w, M, nz)
            for i, d in got.items():
                rem[i] -= d
                if rem[i] <= 1e-9:
                    rem[i] = 0.0; done[i] = True
            continue
        key = dl[cand] if base == 'edf' else -e1 if base == 'maxeff' else -e1 / slack
        cand = cand[np.argsort(key, kind='stable')]
        free = np.ones(M, bool); busy = set()
        for i in cand:
            if not free.any():
                break
            if fl[i] in busy:                                      # one item per UAV per slot
                continue
            f = F[fl[i]]; j = s - f['s0']
            idx = np.nonzero(free)[0]
            g2 = f['g2'][j, idx]
            r = rate_n(P_MAX + f['G'][j] - nz, g2, ntok[i])
            fin = np.nonzero(r >= rem[i])[0]
            nn = int(fin[0]) + 1 if len(fin) else int(np.argmax(r)) + 1
            if r[nn - 1] <= 0:
                continue
            free[idx[np.argsort(-g2)[:nn]]] = False; busy.add(fl[i]); used += nn
            rem[i] -= min(rem[i], r[nn - 1])
            if rem[i] <= 1e-9:
                rem[i] = 0.0; done[i] = True
    return done, used / max(slots * M, 1)


def lp_bound(E1, CAP, rel, dl, M):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, vstack
    n = len(rel)
    ii, kk = np.nonzero(E1 > 0)
    keep = kk < (dl - rel)[ii]
    ii, kk = ii[keep], kk[keep]
    if len(ii) == 0:
        return np.zeros(n)
    e = E1[ii, kk].astype(float); ub = CAP[ii, kk].astype(float) / e
    ss = rel[ii] + kk
    slots, sidx = np.unique(ss, return_inverse=True)
    nv = len(ii); cols = np.arange(nv)
    A = vstack([coo_matrix((e, (ii, cols)), shape=(n, nv)), coo_matrix((np.ones(nv), (sidx, cols)), shape=(len(slots), nv))]).tocsr()
    b = np.concatenate([np.ones(n), np.full(len(slots), float(M))])
    res = linprog(-e, A_ub=A, b_ub=b, bounds=np.stack([np.zeros(nv), ub], 1), method='highs')
    assert res.status == 0, res.message
    return np.minimum(np.bincount(ii, weights=e * res.x, minlength=n), 1.0)


def job(args):
    lam, hover, D, mode, M, seed = args
    t0 = time.time()
    F, fl, rel, dl, ntok, cnt = make(lam, hover, D, mode, M, seed)
    E1, CAP, EP1 = precompute(F, fl, rel, dl, ntok, M)
    out = dict(lam=lam, hover=hover, D=D, mode=mode, M=M, MHz=M * W / 1e6, seed=seed, items=int(cnt.sum()),
               T_f=2 * T_TR + K * (T_CAP + hover + T_CRUISE), never=float(np.mean((E1.max(1) <= 0)[cnt])), pol={}, util={})
    out['N'] = lam * out['T_f'] / 3600
    for p in (os.environ.get('BUB_POLICIES', 'edf,maxeff,slack,slack+drop,mg-thr,mg-thr+drop,mg-slack,mg-slack+drop').split(',')):
        d, u = run_policy(F, fl, rel, dl, ntok, E1, CAP, M, p)
        out['pol'][p] = float(d[cnt].mean()); out['util'][p] = float(u)
    if mode == 'defer' and os.environ.get('BUB_NOBOUND') != '1':
        out['pol']['bound'] = float(lp_bound(E1, CAP, rel, dl, M)[cnt].mean())
    out['palpha'] = PALPHA; out['scheme'] = SCHEME; out['secs'] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True)
    return out


def calib():
    """SINR statistics at the stops with the reference configuration (P_max spread over 6 channels), per altitude layer;
    and the interference margin that puts the 40 m median at 5.7 dB"""
    res = {}
    for hi, h in enumerate(ALTS):
        v, los = [], []
        for fi in range(3000):
            frng = np.random.default_rng(999000 + fi)
            f = flight(0.0, frng, 0.0, 1)
            if f['h'] != h:
                continue
            for c in f['caps']:
                j = int(c / DT) - f['s0']
                v.append(P_MAX - 10 * np.log10(6) + f['G'][j] - (N0 + 10 * np.log10(W) + NF)); los.append(f['LOS'][j])
        v = np.array(v)
        res[h] = dict(n=len(v), los_share=float(np.mean(los)), snr_no_interference=dict(p5=float(np.percentile(v, 5)), median=float(np.median(v)), p95=float(np.percentile(v, 95))))
        print(f'altitude {h:.0f} m: stops {len(v)}, LoS share {np.mean(los):.3f}, per-channel SNR without interference (dB): '
              f'5 % {np.percentile(v, 5):.1f}  median {np.median(v):.1f}  95 % {np.percentile(v, 95):.1f}', flush=True)
    iot = res[40.0]['snr_no_interference']['median'] - 5.7
    p5 = res[40.0]['snr_no_interference']['p5'] - iot
    print(f'interference margin that puts the 40 m median at 5.7 dB: {iot:.1f} dB  ->  5 % point {p5:.1f} dB (measured: -7.6 dB)', flush=True)
    res['iot_db'] = iot; res['p5_after'] = p5
    json.dump({str(k): v for k, v in res.items()}, open(SD.OUT, 'w'), indent=1)


def main():
    mode = sys.argv[4]
    if mode == 'calib':
        return calib()
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    if mode == 'schemes':                                          # scheme comparison over bandwidth and arrival rate, one seed
        jobs = [(la, 0, 60, 'defer', Mx, 201) for Mx in (3, 6, 10, 15, 28, 56, 111) for la in (20, 40, 60)]
    elif mode == 'one':
        jobs = [(la, 0, 60, 'defer', Mx, sd) for Mx in [int(x) for x in os.environ.get('BUB_MS', '3,4,6').split(',')] for la in (40, 60) for sd in (201, 202, 203)]
    else:
        Ms = [3, 6, 10, 15, 28]
        jobs = [(la, 0, 60, 'defer', M, 7) for M in Ms for la in (40, 60)] + [(la, hov, 0, 'window', M, 7) for M in (6, 28) for la in (60,) for hov in (5, 12, 30)]
    print(f'{len(jobs)} runs; scheme {SCHEME}' + (f' ({BYTES:.0f} B per image)' if SCHEME != 'gated' else '') + f'; interference margin {IOT:.1f} dB; noise per channel {noise_dbm():.1f} dBm; q_target {SD.QT}', flush=True)
    with Pool(procs) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(dict(iot_db=IOT, q_target=SD.QT, runs=res), open(SD.OUT, 'w'), indent=1)
    print('\n=== on-time share (deadline 60 s unless window mode) ===')
    for r in res:
        tag = f"defer D={r['D']}s" if r['mode'] == 'defer' else f"window W={r['hover']}s"
        print(f"M={r['M']:>3} ({r['MHz']:.2f} MHz) {r['lam']}/h {tag:<12} N={r['N']:.1f} never {r['never']:.3f} | " + ' '.join(f'{k} {v:.3f}' for k, v in r['pol'].items())
              + f" | palpha {r.get('palpha')}")


if __name__ == '__main__':
    main()
