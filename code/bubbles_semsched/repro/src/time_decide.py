#!/usr/bin/env python3
"""Computation per decision and size of a scheduler. One evaluation run per traffic seed (1.5 h, deterministic policy); the CPU time of the
process (not wall time, so that other jobs on the machine do not matter; one thread) is accumulated separately for
  policy    the scheduler's own computation in a slot with candidates (levels and weights of all candidates)
  matching  the channel-matching layer + bookkeeping of the slot (sim.step), which every scheme shares
Parameters = trainable parameters used at decision time (actor only).

Usage: python time_decide.py <sim_inputs.json> <out.json> <unused> <M> <unused>
       env: TD_KIND = rule | sppo | hppo | d3qn | td3;  TD_POL (rule: lyap<V>, lyapp<p>v<V>, fixed<l>);  TD_CKPT (learners);  TD_LAM (60);
            TD_SEEDS (3);  for the learners the environment switches of their training (PPO_PRIOR, PPO_BASE, PPO_REWARD, BUB_V, ...)"""
import sys, os, json, time
import numpy as np
KIND, POL, CK = os.environ.get('TD_KIND', 'rule'), os.environ.get('TD_POL', 'lyapp3v0.1'), os.environ.get('TD_CKPT', '')
LAM, NS = float(os.environ.get('TD_LAM', 60)), int(os.environ.get('TD_SEEDS', 3))
argv = sys.argv; M = int(argv[4]); sys.argv = argv[:4] + [str(M), '0', '1', '1']
os.environ.setdefault('OMP_NUM_THREADS', '1')
import torch
torch.set_num_threads(1)
W, npar, mod = None, 0, None
if KIND == 'sppo':
    import sppo_hold as mod
    net = mod.SNet(); net.load_state_dict(torch.load(CK, map_location='cpu', weights_only=True)); W = mod.weights(net)
    sd = net.state_dict(); resid = any(float(sd[k].abs().max()) > 0 for k in ('ac.weight', 'ad.weight', 'ac.bias', 'ad.bias'))   # residual heads start at zero
    npar = 3 + (sum(v.numel() for k, v in sd.items() if k[0] == 'a') if resid else 0)
elif KIND == 'hppo':
    import hppo_hold as mod
    net = mod.Net(); net.load_state_dict(torch.load(CK, map_location='cpu', weights_only=True)); W = mod.weights(net)
    npar = sum(v.numel() for k, v in net.state_dict().items() if k[0] == 'a')
elif KIND in ('d3qn', 'td3'):
    os.environ['OFF_ALGO'] = KIND
    import offpol_hold as mod
    net = (mod.QNet if KIND == 'd3qn' else mod.Actor)(); net.load_state_dict(torch.load(CK, map_location='cpu', weights_only=True)); W = mod.npw(net)
    npar = sum(v.numel() for v in net.state_dict().values())
import sim_holdq as S
H, C = S.H, S.C
sys.argv = argv


def decide(sim, nh, rng):
    if KIND == 'rule':
        if POL.startswith('lyapp'):
            pp, _, vv = POL[5:].partition('v'); return S.lyap_action(sim, float(vv), p=float(pp))
        if POL.startswith('lyap'):
            return S.lyap_action(sim, float(POL[4:]))
        return np.full(len(sim.cand), int(POL[5:])), 1.0 / np.maximum(sim.slack(), 1.0)
    if KIND in ('sppo', 'hppo'):
        o = mod.act(sim, nh, W, rng, False); return o[1], o[-1]
    e = sim.e1(); X, slack = mod.HP.observe(sim, nh, e)
    lev, a, _ = mod.decide(W, X, e.T > 0, rng, 0.0)
    return lev, 10.0 ** a / np.maximum(slack, 1.0)


if __name__ == '__main__':
    tp = tm = 0.0; slots = cands = 0; rng = np.random.default_rng(0); pc = time.process_time
    for seed in range(401, 401 + NS):
        F = H.make_flights(LAM, M, seed); sim = S.Sim(F, M)
        while not sim.finished():
            nh = sim.get()
            if sim.cand:
                t0 = pc(); lev, w = decide(sim, nh, rng); t1 = pc(); sim.step(w, lev); t2 = pc()
                tp += t1 - t0; tm += t2 - t1; slots += 1; cands += len(sim.cand)
            else:
                sim.step(None)
    out = dict(kind=KIND, pol=POL if KIND == 'rule' else CK, M=M, lam=LAM, seeds=NS, parameters=int(npar), slots=slots, candidates=cands,
               policy_ms_per_slot=1e3 * tp / max(slots, 1), policy_us_per_candidate=1e6 * tp / max(cands, 1), matching_ms_per_slot=1e3 * tm / max(slots, 1),
               candidates_per_slot=cands / max(slots, 1))
    json.dump(out, open(argv[2], 'w'))
    print(json.dumps(out), flush=True)
