#!/usr/bin/env python3
"""Structured-actor hybrid PPO on the BUBBLES-facing environment (user decision 2026-10-06): the shape of the drift-plus-penalty
rule is part of the ACTOR and is LEARNED, instead of being fixed (hppo_hold.py) or distilled by hand afterwards.

Actor of a UAV with undelivered evidence (time left sl before a hold, own constraint queue z, best-channel efficiency e_l per level):
  urgency          u = (1 / sl) x (D / sl)^(p - 1)                        p   learnable, starts at 1 (plain drift-plus-penalty)
  value of a level val_l = V u + z (recall_l - QINT)                      V   learnable, starts at BUB_V
  level (discrete) logits_l = kappa x s_l / max_l |s_l| + f_theta(o)_l    s_l = val_l e_l (the drift-plus-penalty score); kappa learnable
  weight (contin.) log10 w ~ N( log10 max(val_chosen, 1e-6) + g_theta(o), sigma^2 )
With f = g = 0 the greedy action (most likely level, mean weight) IS the drift-plus-penalty rule with (p, V); training starts
there (zero-initialised heads, p = 1). p and V enter the log-probabilities, so the policy gradient moves them like any weight.
  SPPO_MODE = full    structured part + neural residual (default)
              struct  only p, V, kappa (and the critic) are trained: a learned index rule, no neural residual
              resid   p, V, kappa frozen: the residual-only learner (= the first-stage H-PPO, here with the soft level prior)
Everything else (observation, reward with the per-flight constraint queues, matching layer, centralised critic, PPO settings,
checkpoint selection including iteration 0, evaluation) is as in hppo_hold.py.

Usage: python sppo_hold.py <sim_inputs.json> <out.json> <unused> <M> <unused> [procs=8] [iters=250]
       env: SPPO_MODE, SPPO_P0 (1), SPPO_K0 (6), SPPO_LRS (3e-3, learning rate of p, V, kappa), SPPO_REG (0), PPO_SEED, PPO_LAMS, PPO_LAMS_TEST,
            EVAL_LAMS, BUB_V, BUB_BETA, PPO_FAIR and the environment switches of sim_holdq.py"""
import sys, json, time, os, copy
import numpy as np
import torch, torch.nn as nn
from multiprocessing import Pool
import hppo_hold as HP
S, H, C, SD = HP.S, HP.H, HP.C, HP.SD

MODE = os.environ.get('SPPO_MODE', 'full')
P0, K0, LRS = float(os.environ.get('SPPO_P0', 1.0)), float(os.environ.get('SPPO_K0', 6.0)), float(os.environ.get('SPPO_LRS', 3e-3))
REG = float(os.environ.get('SPPO_REG', 0.0))                    # L2 pull of the neural residual towards zero: what the structure can explain, it should
M, PROCS, ITERS, SEED, L, NF_, HID, DS = HP.M, HP.PROCS, HP.ITERS, HP.SEED, HP.L, HP.NF_, HP.HID, HP.DS
EVAL_LAMS = [float(x) for x in os.environ.get('EVAL_LAMS', '55,60,65').split(',')]; EVAL_SEEDS = list(range(401, 413))
DQ = S.QL - S.QINT                                              # recall of each level minus the target inside the queue
# Options added 2026-10-07 (all off by default = the earlier behaviour):
#   SPPO_DUAL     step of the adaptive constraint price (dual ascent on the per-flight criterion). After every iteration the price BETA of the
#                 reward is multiplied by exp(SPPO_DUAL x (fb - SPPO_FB)); fb = share of the flights that landed below the recall requirement
#                 in the training episodes at the MIDDLE training rate, as a ratio of exponentially weighted counts (factor 0.98 per iteration).
#                 SPPO_FB is the target of that training-window statistic (it is not the steady-state share of the evaluation; calibrate it).
#   SPPO_LRS_END  the learning rate of p, V, kappa decays linearly from SPPO_LRS to SPPO_LRS_END x SPPO_LRS over the run (1 = constant)
#   SPPO_AVG      the saved 'final' structure is the average of p, log V, log kappa over the last SPPO_AVG iterations (0 = last iterate)
DUAL, FB_T = float(os.environ.get('SPPO_DUAL', 0.0)), float(os.environ.get('SPPO_FB', 0.04))
LRS_END, AVG = float(os.environ.get('SPPO_LRS_END', 1.0)), int(os.environ.get('SPPO_AVG', 0))
#   SPPO_FREEZE   structure parameters that stay at their initial value (comma-separated: p, V, kappa) - ablation of what is learned
FREEZE = [x for x in os.environ.get('SPPO_FREEZE', '').split(',') if x]
BETA_LO, BETA_HI = float(os.environ.get('SPPO_BETA_LO', 100.0)), float(os.environ.get('SPPO_BETA_HI', 1000.0))   # range of the adaptive price


def struct_np(par, sl, z, e, mask):
    """val (candidates x levels) and the structured level logits; sl already >= 1"""
    u = (1.0 / sl) * (DS / sl) ** (par['p'] - 1.0)
    val = par['V'] * u[:, None] + z[:, None] * DQ[None, :]
    s = np.where(mask, val * e, 0.0)
    return val, par['kappa'] * s / (np.abs(s).max(1, keepdims=True) + 1e-12)


def act(sim, nh, w, rng, train):
    """one decision; w = None: the plain rule (p = 1, V = BUB_V)"""
    e = sim.e1()
    X, slack = HP.observe(sim, nh, e)
    mask = e.T > 0; sl = np.maximum(slack, 1.0); z = sim.zvec(); n = len(X)
    par = dict(p=1.0, V=HP.V, kappa=K0) if w is None else w['par']
    val, pri = struct_np(par, sl, z, e.T, mask)
    mu_res, logit_res = (np.zeros(n), np.zeros((n, L))) if w is None else HP.actor_np(w, X)
    logit = np.where(mask, pri + logit_res, -1e9)
    if train:
        pr = np.exp(logit - logit.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        lev = np.array([rng.choice(L, p=pp) if m_.any() else 0 for pp, m_ in zip(pr, mask)])
    else:
        lev = np.argmax(logit, 1)
    mu = np.log10(np.maximum(val[np.arange(n), lev], 1e-6)) + mu_res
    b = mu + float(np.exp(w['logstd'])) * rng.standard_normal(n) if train else mu
    return X, lev, b, mask, sl, z, e.T, 10.0 ** np.clip(b, -8, 6)


def rollout(args):
    w, lam, seed, mode = args
    train = mode == 'train'
    F = H.make_flights(lam, M, seed, *((HP.T_EP / 3600, HP.WARM_S / 3600) if train else (1.5, 0.5)))
    sim = S.Sim(F, M)
    rng = np.random.default_rng(seed + 31337)
    s0 = int(HP.WARM_S / C.DT); s1 = s0 + HP.T_EP
    T = dict(X=[], lev=[], b=[], mask=[], sl=[], z=[], e=[]); cnts, rews, last_X = [], [], None
    lost0 = 0
    while not sim.finished():
        nh = sim.get()
        if train and sim.s >= s1:
            if sim.cand:
                last_X = HP.observe(sim, nh, sim.e1())[0].astype(np.float32)
            break
        rec = train and sim.s >= s0
        lost = sum(f['lost'] for f in F)
        r = -(nh + HP.LOSS_PEN * (lost - lost0)); lost0 = lost
        if HP.FAIR > 0 and train:
            for f in F:
                if f['landed'] is not None and not f.get('judged'):
                    f['judged'] = True
                    if f['qsum'] / max(f['done_items'] + f['lost'], 1) < S.QBAR - 5e-3:
                        r -= HP.FAIR
        if sim.cand:
            X, lev, b, mask, sl, z, e, wts = act(sim, nh, w, rng, train)
            pen = HP.price(sim, lev)
            sim.step(wts, lev)
            r = (r - (HP.BETA if w is None else w.get('beta', HP.BETA)) * pen(sim)) * HP.RSCALE
            if rec:
                for k, v in zip(('X', 'lev', 'b', 'mask', 'sl', 'z', 'e'), (X, lev, b, mask, sl, z, e)):
                    T[k].append(v)
                cnts.append(len(X)); rews.append(r)
        else:
            sim.step(None)
            if rec and rews:
                rews[-1] += r * HP.RSCALE
    if not train:
        return S.raw(F)
    if not cnts:
        return None
    out = {k: np.concatenate(v).astype(np.float32 if k not in ('lev', 'mask') else (np.int64 if k == 'lev' else bool)) for k, v in T.items()}
    out.update(cnt=np.array(cnts), rew=np.array(rews, np.float32), last=last_X, ret=float(np.sum(rews)) / HP.RSCALE)
    land = [f for f in F if f['landed'] is not None and f['landed'] >= s0]     # flights that landed inside the recorded window
    out['fair'] = (sum(f['qsum'] / max(f['done_items'] + f['lost'], 1) < S.QBAR - 5e-3 for f in land), len(land)); out['lam'] = lam
    return out


class SNet(HP.Net):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(P0)); self.logV = nn.Parameter(torch.tensor(float(np.log(HP.V)))); self.logk = nn.Parameter(torch.tensor(float(np.log(K0))))

    def struct(self, sl, z, e, mask, dq):
        u = (1.0 / sl) * (DS / sl) ** (self.p - 1.0)
        val = self.logV.exp() * u[:, None] + z[:, None] * dq[None, :]
        s = torch.where(mask, val * e, torch.zeros_like(e))
        return val, self.logk.exp() * s / (s.abs().max(1, keepdim=True).values + 1e-12)


def weights(net):
    w = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items() if k[0] == 'a'}
    w['logstd'] = float(net.logstd.detach().cpu())
    w['par'] = dict(p=float(net.p.detach().cpu()), V=float(net.logV.exp().detach().cpu()), kappa=float(net.logk.exp().detach().cpu()))
    return w


def main():
    t0 = time.time()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    net = SNet().to(dev)
    if os.environ.get('SPPO_INIT'):                                            # continue from an earlier run (structure and critic; the optimiser restarts)
        net.load_state_dict(torch.load(os.environ['SPPO_INIT'], map_location=dev, weights_only=True))
    sp = [t for t, n in ((net.p, 'p'), (net.logV, 'V'), (net.logk, 'kappa')) if n not in FREEZE]; heads = list(net.a1.parameters()) + list(net.a2.parameters()) + list(net.ac.parameters()) + list(net.ad.parameters())
    crit = list(net.c1.parameters()) + list(net.c2.parameters()) + list(net.c3.parameters()) + [net.logstd]
    groups = [dict(params=crit, lr=HP.LR)]
    if MODE in ('full', 'resid'):
        groups.append(dict(params=heads, lr=HP.LR))
    if MODE in ('full', 'struct'):
        groups.append(dict(params=sp, lr=LRS))
    opt = torch.optim.Adam(groups)
    dq = torch.tensor(DQ, dtype=torch.float32, device=dev)
    beta, en, ed, hist = HP.BETA, 0.0, 0.0, []
    lam_mid = sorted(HP.LAMS_TRAIN)[len(HP.LAMS_TRAIN) // 2]
    out = dict(algo='sppo', mode=MODE, M=M, seed=SEED, V0=HP.V, p0=P0, k0=K0, lrs=LRS, reg=REG, beta=HP.BETA, dual=DUAL, fb_target=FB_T, lrs_end=LRS_END, avg=AVG,
               qbar=S.QBAR, qint=S.QINT, zmode=S.ZMODE, ports=C.NPORT,
               fair=HP.FAIR, episodes=ITERS * HP.EPI, lams_train=HP.LAMS_TRAIN, curve=[], evals=[])
    with Pool(PROCS) as pool:
        def ev(wts, lam, seeds):
            return S.summary(pool.map(rollout, [(wts, lam, s, 'eval') for s in seeds]), lam)
        fmt = lambda d: ' '.join(f"{la}/h hold {m['mean_hold']:.1f}s buffer {m['buffer95']:.1f}s N {m['N_buf']:.1f} lost {m['lost']:.3f} recall {m['recall']:.4f} "
                                 f"below {m['flights_below']:.3f}" for la, m in d.items())
        print(f'structured-actor PPO, mode {MODE}: p0 {P0}, V0 {HP.V}, kappa0 {K0}; M {M}, constraint queue {S.ZMODE}, ports {C.NPORT}, train rates {HP.LAMS_TRAIN}', flush=True)
        vm0 = {str(la): ev(weights(net), la, HP.VAL) for la in HP.LAMS_TEST}
        best = (float(np.max([HP.cost(m) for m in vm0.values()])), 0, copy.deepcopy(net.state_dict()))
        out['evals'].append(dict(it=0, episodes=0, val=vm0, cost=best[0], par=weights(net)['par']))
        print(f"it   0 ({'the plain rule' if not os.environ.get('SPPO_INIT') else 'the loaded policy'}; p {weights(net)['par']['p']:.2f} V {weights(net)['par']['V']:.3f}): validation " + fmt(vm0) + f' | cost {best[0]:.1f}', flush=True)
        for it in range(1, ITERS + 1):
            w = weights(net); w['beta'] = beta
            if LRS_END != 1.0 and MODE in ('full', 'struct'):
                opt.param_groups[-1]['lr'] = LRS * (1.0 - (1.0 - LRS_END) * (it - 1) / max(ITERS - 1, 1))
            trajs = [t for t in pool.map(rollout, [(w, float(rng.choice(HP.LAMS_TRAIN)), int(10 ** 6 * (SEED + 1) + it * 100 + q), 'train') for q in range(HP.EPI)]) if t]
            en = 0.98 * en + sum(t['fair'][0] for t in trajs if t['lam'] == lam_mid); ed = 0.98 * ed + sum(t['fair'][1] for t in trajs if t['lam'] == lam_mid)
            fb = en / ed if ed > 0 else float('nan')
            if DUAL > 0 and ed > 200:                          # wait until the statistic rests on a few hundred flights
                beta = float(np.clip(beta * np.exp(DUAL * (fb - FB_T)), BETA_LO, BETA_HI))
            cat = lambda k: torch.from_numpy(np.concatenate([t[k] for t in trajs])).to(dev)
            X, lev, b, mask, SL, Z, E = cat('X'), cat('lev'), cat('b'), cat('mask'), cat('sl'), cat('z'), cat('e')
            cnt = np.concatenate([t['cnt'] for t in trajs]); T = len(cnt)
            sid = torch.from_numpy(np.repeat(np.arange(T), cnt)).to(dev); cntf = torch.from_numpy(cnt.astype(np.float32)).to(dev)
            anyok = mask.any(1)

            reg = [0.0]

            def logps():
                val, pri = net.struct(SL, Z, E, mask, dq)
                mu_res, logit_res = net.actor(X)
                reg[0] = (mu_res ** 2).mean() + (logit_res ** 2).mean()
                lg = torch.where(mask, pri + logit_res, torch.full_like(pri, -1e9))
                ld = torch.log_softmax(lg, 1).gather(1, lev[:, None]).squeeze(1)
                mu = torch.log10(val.gather(1, lev[:, None]).squeeze(1).clamp_min(1e-6)) + mu_res
                lc = -0.5 * ((b - mu) / net.logstd.exp()) ** 2 - net.logstd
                return lc, torch.where(anyok, ld, torch.zeros_like(ld))
            with torch.no_grad():
                lc0, ld0 = logps()
                v = net.value(X, sid, T, cntf).cpu().numpy()
                vlast = []
                for t in trajs:
                    if t['last'] is None:
                        vlast.append(0.0)
                    else:
                        xl = torch.from_numpy(t['last']).to(dev); nl = len(xl)
                        vlast.append(float(net.value(xl, torch.zeros(nl, dtype=torch.long, device=dev), 1, torch.tensor([float(nl)], device=dev))))
            adv = np.zeros(T, np.float32); ret = np.zeros(T, np.float32); q = 0
            for t, vl in zip(trajs, vlast):
                r = t['rew']; Lr = len(r); vv = np.append(v[q:q + Lr], vl); g = 0.0
                for i in range(Lr - 1, -1, -1):
                    g = r[i] + HP.GAMMA * vv[i + 1] - vv[i] + HP.GAMMA * HP.LAM_GAE * g
                    adv[q + i] = g
                ret[q:q + Lr] = adv[q:q + Lr] + v[q:q + Lr]; q += Lr
            A = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8)).to(dev)[sid]; rett = torch.from_numpy(ret).to(dev)
            stop = False
            for ep in range(HP.EPOCHS):
                perm = torch.randperm(T, device=dev)
                for mb in range(HP.NMB):
                    steps = torch.zeros(T, dtype=torch.bool, device=dev); steps[perm[mb::HP.NMB]] = True
                    rows = steps[sid]
                    lc, ld = logps()
                    lpi = 0.0
                    for l_new, l_old in ((lc, lc0), (ld, ld0)):
                        ratio = (l_new - l_old).exp()
                        lpi = lpi - torch.min(ratio * A, ratio.clamp(1 - HP.CLIP, 1 + HP.CLIP) * A)[rows].mean()
                    lv = ((net.value(X, sid, T, cntf) - rett) ** 2)[steps].mean()
                    opt.zero_grad(); (lpi + HP.VCOEF * lv + REG * reg[0]).backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
                    with torch.no_grad():
                        net.logstd.clamp_(-3.0, 0.5); net.p.clamp_(0.5, 6.0); net.logV.clamp_(np.log(1e-4), np.log(100.0)); net.logk.clamp_(0.0, np.log(50.0))
                        kl = float((lc0 - lc).mean() + (ld0 - ld).mean())
                    if kl > 0.03:
                        stop = True; break
                if stop:
                    break
            Rm = float(np.mean([t['ret'] for t in trajs])); par = weights(net)['par']
            out['curve'].append(dict(it=it, episodes=it * HP.EPI, mean_return=Rm, std=float(net.logstd.exp()), kl=kl, beta=beta, fb=fb, **par))
            hist.append((float(net.p.detach().cpu()), float(net.logV.detach().cpu()), float(net.logk.detach().cpu())))
            if it % 25 == 0 or it == ITERS:
                vm = {str(la): ev(weights(net), la, HP.VAL) for la in HP.LAMS_TEST}
                sc = float(np.max([HP.cost(m) for m in vm.values()]))
                out['evals'].append(dict(it=it, episodes=it * HP.EPI, val=vm, cost=sc, par=par))
                if sc < best[0]:
                    best = (sc, it, copy.deepcopy(net.state_dict()))
                print(f"it {it:3d} ({it * HP.EPI} episodes): return {Rm:.1f}  p {par['p']:.2f}  V {par['V']:.3f}  kappa {par['kappa']:.1f}  std {net.logstd.exp().item():.3f} | validation "
                      + fmt(vm) + f' | cost {sc:.1f} (best {best[0]:.1f} at it {best[1]})  ({time.time() - t0:.0f}s)', flush=True)
            elif it % 5 == 0:
                print(f"it {it:3d} ({it * HP.EPI} episodes): return {Rm:.1f}  p {par['p']:.2f}  V {par['V']:.3f}  kappa {par['kappa']:.1f}  std {net.logstd.exp().item():.3f}  kl {kl:.4f}"
                      + (f'  beta {beta:.0f}  fb {fb:.3f}' if DUAL > 0 else ''), flush=True)
        if AVG > 0 and len(hist) >= AVG:                       # iterate averaging of the structure
            m = np.mean(hist[-AVG:], 0)
            with torch.no_grad():
                net.p.fill_(float(m[0])); net.logV.fill_(float(m[1])); net.logk.fill_(float(m[2]))
        final = copy.deepcopy(net.state_dict())
        torch.save(final, SD.OUT.replace('.json', '_final.pt')); torch.save(best[2], SD.OUT.replace('.json', '_best.pt'))
        out['best_val_iter'] = best[1]; out['careful'] = {}
        for name, sd in (('final', final), ('best', best[2])):
            net.load_state_dict(sd); w = weights(net)
            print(f"--- sppo {MODE} {name} (it {ITERS if name == 'final' else best[1]}): p {w['par']['p']:.2f}  V {w['par']['V']:.3f}  kappa {w['par']['kappa']:.1f}; "
                  f'{len(EVAL_SEEDS)} traffic seeds pooled', flush=True)
            for la in EVAL_LAMS:
                rs = pool.map(rollout, [(w, la, s, 'eval') for s in EVAL_SEEDS]); m = S.summary(rs, la)
                m['holds'] = [np.round(r[0], 1).tolist() for r in rs]; m['par'] = w['par']; out['careful'][f'{name}|{la:.0f}'] = m
                print(f"  {la:.0f}/h  mean hold {m['mean_hold']:6.2f} s  buffer95 {m['buffer95']:6.1f} s  N with buffer {m['N_buf']:.2f}{'!' if m['N_buf'] > H.N_REF else ' '}  "
                      f"lost {m['lost']:.4f}  recall {m['recall']:.4f}{'*' if m['recall'] >= S.QBAR - 5e-4 else ' '}  flights below {m['flights_below']:.3f}  "
                      f"flights with a hold {m['any_hold']:.3f}", flush=True)
    json.dump(out, open(SD.OUT, 'w'))
    print(f'DONE ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
