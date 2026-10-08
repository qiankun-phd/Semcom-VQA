#!/usr/bin/env python3
"""RL scheme, stage 2 on the BUBBLES-facing environment (sim_holdq.py): hybrid-action PPO (H-PPO form: parallel sub-actors on a
shared trunk, one centralised critic). Each UAV with undelivered evidence chooses the SEMANTIC LEVEL of what it sends (discrete)
and its MATCHING WEIGHT (continuous); the learner minimises the holding subject to the mean-recall constraint.

MDP (conventions of mappo_hold.py: UAV agents, parameter sharing, 1 s slots, episode = 1200 slots after a 15 min warm-up,
truncated with a value bootstrap)
  observation  18 numbers per UAV: large-scale SNR; best / mean small-scale gain; planned improvement of the large-scale gain within
               the time left; remaining fraction of its oldest evidence; time left before a hold (negative while holding); time
               needed over time left (at the 0.44 level); number and size of its waiting evidence; best-channel efficiency at each
               of the 5 levels; number of UAVs with evidence; number of UAVs holding; the constraint queue Z (broadcast).
  action       discrete: semantic level (levels that cannot be sent at the current SNR are masked);
               continuous: log10 of a multiplier on the drift-plus-penalty matching weight V / time-left + Z (recall_l - QBAR).
               The level logits start biased (+3) towards the drift-plus-penalty greedy level and the multiplier at 1, so that
               training starts close to the Lyapunov baseline 'lyap V' of sim_holdq.py and learns corrections.
  transition   weighted marginal-gain matching (sim_holdq.alloc); plan clocks, holds, drops as in sim_hold.Sim;
               Z <- max(Z + QBAR x delivered - recall delivered, 0).
  reward       shared, drift-plus-penalty form:  - (UAVs holding + 20 x items dropped) - BETA x Z x (QBAR x delivered - recall delivered)
               in the slot, scaled by 0.1.
PPO          the configuration confirmed in stage 1 (= Stable-Baselines3 defaults, entropy 0); 8 episodes x 250 iterations.
evaluation   deterministic policy (most likely level, mean multiplier), 1.5 h runs on held-out traffic: mean hold, 4D buffer, airborne
             count with the buffer, lost share, mean recall. Checkpoint chosen on validation traffic by
             mean hold + 1000 x lost share + 5000 x recall shortfall.

Usage: python hppo_hold.py <sim_inputs.json> <out.json> <unused> <M> <unused> [procs=8] [iters=250]
       env: PPO_SEED, PPO_LAMS, PPO_LAMS_TEST, BUB_V (default 10), BUB_BETA (default 50 / V), BUB_QBAR, BUB_ZMODE, BUB_D, BUB_IOT_DB"""
import sys, json, time, os, copy
import numpy as np
import torch, torch.nn as nn
from multiprocessing import Pool
import sim_holdq as S
H, C, SD = S.H, S.C, S.SD

M = int(sys.argv[4])
PROCS = int(sys.argv[6]) if len(sys.argv) > 6 else 8
ITERS = int(sys.argv[7]) if len(sys.argv) > 7 else 250
SEED = int(os.environ.get('PPO_SEED', 0))
V = float(os.environ.get('BUB_V', 10.0))
BETA = float(os.environ.get('BUB_BETA', 50.0 / V))             # the network queue scales with V: keep price x queue comparable to the holding term
EPI, T_EP, WARM_S = int(os.environ.get('PPO_EPI', 8)), 1200, 900
LR, GAMMA, LAM_GAE, CLIP, EPOCHS, NMB, VCOEF, RSCALE, STD_INIT, LOSS_PEN = 3e-4, 0.99, 0.95, 0.2, 10, 4, 0.5, 0.1, 0.3, 20.0
# Variants (for the comparison set customary in the literature: other learners / ablations of the guidance):
#   PPO_PRIOR    bias of the level logits towards the guiding rule (default 3; 0 = no level guidance)
#   BUB_PRIOR_P  urgency exponent of the guiding rule (1 = plain drift-plus-penalty; 3 = the rule distilled from the first learned policy)
#   PPO_BASE     dpp (default): the weight multiplies the guiding rule's weight; urg: it multiplies the plain urgency 1 / time left
#   PPO_REWARD   queue (default): drift-plus-penalty reward with the constraint queue; penalty: fixed price PPO_PEN on the recall
#                shortfall of the slot, no queue in the reward and none in the observation (the usual 'DRL with a penalty term')
#   PPO_LR       learning rate (default 3e-4);  PPO_REG  weight of an L2 pull of both heads towards the guiding rule (0 = off):
#                loss += PPO_REG x (mean multiplier^2 + mean level-logit^2). Zero outputs = exactly the guiding rule, so the pull keeps
#                the policy at the rule unless the advantages consistently say otherwise (fine-tuning must not degrade the rule).
PRIOR = float(os.environ.get('PPO_PRIOR', 3.0)); PRIOR_P = float(os.environ.get('BUB_PRIOR_P', 1.0))
LR = float(os.environ.get('PPO_LR', LR)); REG = float(os.environ.get('PPO_REG', 0.0))
#   PPO_FAIR     penalty (in holding-seconds) charged when a flight lands with its own mean recall more than 0.005 below the requirement:
#                the per-flight criterion itself, as a (sparse) reward term next to the per-flight constraint queues.
FAIR = float(os.environ.get('PPO_FAIR', 0.0))
BASE = os.environ.get('PPO_BASE', 'dpp'); REWARD = os.environ.get('PPO_REWARD', 'queue'); PEN = float(os.environ.get('PPO_PEN', 300.0))
LAMS_TRAIN = [float(x) for x in os.environ.get('PPO_LAMS', '40,50,60').split(',')]
LAMS_TEST = [float(x) for x in os.environ.get('PPO_LAMS_TEST', '40,60').split(',')]
VAL, TEST = [301, 302, 303, 304], [201, 202, 203]
L = S.L; NF_, HID = 13 + L, 64
ZS = V if S.ZMODE == 'net' else 0.2                            # scale of the constraint queue in the observation
DS = H.D / C.DT


def actor_np(w, X):
    h = np.tanh(X @ w['a1.weight'].T + w['a1.bias'])
    h = np.tanh(h @ w['a2.weight'].T + w['a2.bias'])
    return (h @ w['ac.weight'].T + w['ac.bias']).ravel(), h @ w['ad.weight'].T + w['ad.bias']


def observe(sim, nh, e):
    """e = sim.e1() (levels x candidates)"""
    X = np.empty((len(sim.info), NF_)); slack = sim.slack()
    for j, (fi, t, it) in enumerate(sim.info):
        f = sim.F[fi]; sl = slack[j]; gdb = sim.gdb[j]
        fut = f['GP'][t + 1:t + 1 + int(max(sl, 0))] if sl > 0 else []
        X[j, :9] = [sim.s1[j] / 20, gdb.max() / 10, gdb.mean() / 10,
                    np.clip(((np.max(fut) if len(fut) else f['G'][t]) - f['G'][t]) / 20, -2, 2),
                    it['rem'], np.clip(sl / DS, -2, 1), min(it['rem'] / max(e[min(2, L - 1), j], 1e-9) / max(sl, 1.0), 3.0),
                    len(f['items']) / 3, sum(x['rem'] for x in f['items']) / 3]
    X[:, 9:9 + L] = np.clip(np.log10(np.maximum(e.T, 1e-9)), -4, 1) / 2
    X[:, 9 + L] = len(sim.info) / 15; X[:, 10 + L] = nh / 15; X[:, 12 + L] = np.clip(slack, -120, 0) / 120
    X[:, 11 + L] = np.minimum(3.0, sim.zvec() / ZS) if REWARD == 'queue' else 0.0
    return X, slack


def act(sim, nh, w, rng, train):
    """returns the pieces of one decision: features, level, multiplier, mask, Lyapunov-greedy level, matching weights"""
    e = sim.e1()
    X, slack = observe(sim, nh, e)
    lyap, _ = S.lyap_action(sim, V, e, p=PRIOR_P)
    mask = e.T > 0
    if w is None:                                                              # the Lyapunov baseline itself (reference run)
        lev, a = lyap, np.zeros(len(X))
    else:
        mu, logit = actor_np(w, X)
        logit = np.where(mask, logit + PRIOR * (np.arange(L)[None, :] == lyap[:, None]), -1e9)
        if train:
            a = mu + float(np.exp(w['logstd'])) * rng.standard_normal(len(mu))
            p = np.exp(logit - logit.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            lev = np.array([rng.choice(L, p=pp) if m_.any() else 0 for pp, m_ in zip(p, mask)])
        else:
            a, lev = mu, np.argmax(logit, 1)
    sl = np.maximum(slack, 1.0)
    if BASE == 'dpp':
        val = V * (1.0 / sl * (DS / sl) ** (PRIOR_P - 1.0)) + sim.zvec() * (S.QL[lev] - S.QINT)
    else:
        val = 1.0 / sl
    return X, lev, a, mask, lyap, 10.0 ** np.clip(a, -6, 6) * np.maximum(val, 1e-6)


def price(sim, lev):
    """constraint price of a slot = sum over the candidates of (queue before the step) x (QBAR - recall of its level) x (fraction delivered);
    returns a function to call after the step"""
    z0 = sim.zvec() if REWARD == 'queue' else np.full(len(sim.info), PEN / max(BETA, 1e-9))
    items = [it for _, _, it in sim.info]; rem0 = np.array([it['rem'] for it in items])
    return lambda sim_: float(np.sum(z0 * (S.QINT - S.QL[lev]) * (rem0 - np.array([max(it['rem'], 0.0) for it in items]))))


def rollout(args):
    """mode 'train': stochastic policy, returns the recorded T_EP-slot trajectory; mode 'eval': deterministic policy, 1.5 h run,
    returns the raw outputs of sim_holdq.run (w = None runs the Lyapunov baseline 'lyap V')"""
    w, lam, seed, mode = args
    train = mode == 'train'
    F = H.make_flights(lam, M, seed, *((T_EP / 3600, WARM_S / 3600) if train else (1.5, 0.5)))
    sim = S.Sim(F, M)
    rng = np.random.default_rng(seed + 31337)
    s_rec0 = int(WARM_S / C.DT); s_rec1 = s_rec0 + T_EP
    Xs, cnts, acts, levs, masks, priors, rews, last_X = [], [], [], [], [], [], [], None
    lost0 = 0
    while not sim.finished():
        nh = sim.get()
        if train and sim.s >= s_rec1:
            if sim.cand:
                last_X = observe(sim, nh, sim.e1())[0].astype(np.float32)
            break
        rec = train and sim.s >= s_rec0
        lost = sum(f['lost'] for f in F)
        r = -(nh + LOSS_PEN * (lost - lost0)); lost0 = lost
        if FAIR > 0 and train:
            for f in F:
                if f['landed'] is not None and not f.get('judged'):
                    f['judged'] = True
                    if f['qsum'] / max(f['done_items'] + f['lost'], 1) < S.QBAR - 5e-3:
                        r -= FAIR
        if sim.cand:
            X, lev, a, mask, lyap, wts = act(sim, nh, w, rng, train)
            pen = price(sim, lev)                                              # needs the queues and the remaining fractions before the step
            sim.step(wts, lev)
            r = (r - BETA * pen(sim)) * RSCALE
            if rec:
                Xs.append(X.astype(np.float32)); cnts.append(len(X)); acts.append(a.astype(np.float32)); levs.append(lev.astype(np.int64))
                masks.append(mask); priors.append(lyap.astype(np.int64)); rews.append(r)
        else:
            sim.step(None)
            if rec and rews:
                rews[-1] += r * RSCALE                                         # holding cost of slots without a decision
    if not train:
        return S.raw(F)
    if not Xs:
        return None
    return dict(X=np.concatenate(Xs), cnt=np.array(cnts), act=np.concatenate(acts), lev=np.concatenate(levs), mask=np.concatenate(masks),
                prior=np.concatenate(priors), rew=np.array(rews, np.float32), last=last_X, ret=float(np.sum(rews)) / RSCALE)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = nn.Linear(NF_, HID); self.a2 = nn.Linear(HID, HID)                    # shared actor trunk
        self.ac = nn.Linear(HID, 1); self.ad = nn.Linear(HID, L)                        # continuous head (weight), discrete head (level)
        self.c1 = nn.Linear(NF_, HID); self.c2 = nn.Linear(2 * HID, HID); self.c3 = nn.Linear(HID, 1)   # centralised critic
        self.logstd = nn.Parameter(torch.tensor(float(np.log(STD_INIT))))
        for m in (self.ac, self.ad):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def actor(self, X):
        h = torch.tanh(self.a2(torch.tanh(self.a1(X))))
        return self.ac(h).squeeze(1), self.ad(h)

    def value(self, X, sid, T, cntf):
        h = torch.tanh(self.c1(X))
        mean = torch.zeros(T, HID, device=X.device).index_add_(0, sid, h) / cntf[:, None]
        mx = torch.full((T, HID), -1e9, device=X.device).scatter_reduce(0, sid[:, None].expand(-1, HID), h, reduce='amax', include_self=True)
        return self.c3(torch.tanh(self.c2(torch.cat([mean, mx], 1)))).squeeze(1)


def weights(net):
    w = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items() if k[0] == 'a'}
    w['logstd'] = float(net.logstd.detach().cpu())
    return w


def cost(m):
    """what the checkpoint selection minimises"""
    c = m['mean_hold'] + 1000 * m['lost'] + 5000 * max(0.0, S.QBAR - m['recall'])
    return c + (200 * max(0.0, m['flights_below'] - 0.05) if S.ZMODE == 'uav' else 0.0)   # per-flight requirement: at most 5 % of the flights short by > 0.005


def main():
    t0 = time.time()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), lr=LR)
    out = dict(M=M, D=H.D, H_max=H.H_MAX, iot_db=C.IOT, V=V, beta=BETA, qbar=S.QBAR, qint=S.QINT, zmode=S.ZMODE, prior=PRIOR, prior_p=PRIOR_P, base=BASE,
               reward=REWARD, pen=PEN, ntok_cv=H.NTOK_CV, lr=LR, reg=REG, fair=FAIR, levels=S.QL.tolist(), seed=SEED, episodes=ITERS * EPI, T_ep=T_EP,
               hyper=dict(lr=LR, gamma=GAMMA, gae=LAM_GAE, clip=CLIP, epochs=EPOCHS, minibatches=NMB, entropy=0.0, hidden=HID, episodes_per_iter=EPI,
                          lams_train=LAMS_TRAIN, std_init=STD_INIT, prior=PRIOR), curve=[], evals=[])
    with Pool(PROCS) as pool:
        def ev(wts, lam, seeds):
            return S.summary(pool.map(rollout, [(wts, lam, s, 'eval') for s in seeds]), lam)
        fmt = lambda d: ' '.join(f"{la}/h hold {m['mean_hold']:.1f}s buffer {m['buffer95']:.1f}s N {m['N_buf']:.1f} lost {m['lost']:.3f} recall {m['recall']:.4f} "
                                 f"below {m['flights_below']:.3f}" for la, m in d.items())
        out['lyap_baseline'] = {str(la): ev(None, la, TEST) for la in LAMS_TEST}
        out['start'] = {str(la): ev(weights(net), la, TEST) for la in LAMS_TEST}
        print(f'variant: level prior {PRIOR} towards the rule with urgency exponent {PRIOR_P}; weight base {BASE}; reward {REWARD}', flush=True)
        print(f'TEST  guiding rule (V {V}, exponent {PRIOR_P}): ' + fmt(out['lyap_baseline']) + '\n      untrained policy: ' + fmt(out['start']), flush=True)
        vm0 = {str(la): ev(weights(net), la, VAL) for la in LAMS_TEST}        # iteration 0 = the guiding rule: a candidate like any other,
        best = (float(np.max([cost(m) for m in vm0.values()])), 0, copy.deepcopy(net.state_dict()))   # so the selected policy is never worse on validation
        out['evals'].append(dict(it=0, episodes=0, val=vm0, cost=best[0]))
        print(f'it   0: validation ' + fmt(vm0) + f' | cost {best[0]:.1f}', flush=True)
        for it in range(1, ITERS + 1):
            w = weights(net)
            trajs = [t for t in pool.map(rollout, [(w, float(rng.choice(LAMS_TRAIN)), int(10 ** 6 * (SEED + 1) + it * 100 + p), 'train') for p in range(EPI)]) if t]
            X = torch.from_numpy(np.concatenate([t['X'] for t in trajs])).to(dev)
            act_ = torch.from_numpy(np.concatenate([t['act'] for t in trajs])).to(dev)
            lev = torch.from_numpy(np.concatenate([t['lev'] for t in trajs])).to(dev)
            mask = torch.from_numpy(np.concatenate([t['mask'] for t in trajs])).to(dev)
            pri = PRIOR * torch.nn.functional.one_hot(torch.from_numpy(np.concatenate([t['prior'] for t in trajs])).to(dev), L).float()
            cnt = np.concatenate([t['cnt'] for t in trajs]); T = len(cnt)
            sid = torch.from_numpy(np.repeat(np.arange(T), cnt)).to(dev)
            cntf = torch.from_numpy(cnt.astype(np.float32)).to(dev)
            anyok = mask.any(1)

            reg = [0.0]

            def logps():
                mu, logit = net.actor(X)
                reg[0] = (mu ** 2).mean() + (logit ** 2).mean()
                lc = -0.5 * ((act_ - mu) / net.logstd.exp()) ** 2 - net.logstd
                lg = torch.where(mask, logit + pri, torch.full_like(logit, -1e9))
                ld = torch.log_softmax(lg, 1).gather(1, lev[:, None]).squeeze(1)
                return lc, torch.where(anyok, ld, torch.zeros_like(ld))
            with torch.no_grad():
                lc0, ld0 = logps()
                v = net.value(X, sid, T, cntf).cpu().numpy()
                vlast = []
                for t in trajs:                                                # bootstrap value at the truncation state
                    if t['last'] is None:
                        vlast.append(0.0)
                    else:
                        xl = torch.from_numpy(t['last']).to(dev); nl = len(xl)
                        vlast.append(float(net.value(xl, torch.zeros(nl, dtype=torch.long, device=dev), 1, torch.tensor([float(nl)], device=dev))))
            adv = np.zeros(T, np.float32); ret = np.zeros(T, np.float32); p = 0
            for t, vl in zip(trajs, vlast):
                r = t['rew']; Lr = len(r); vv = np.append(v[p:p + Lr], vl); g = 0.0
                for i in range(Lr - 1, -1, -1):
                    g = r[i] + GAMMA * vv[i + 1] - vv[i] + GAMMA * LAM_GAE * g
                    adv[p + i] = g
                ret[p:p + Lr] = adv[p:p + Lr] + v[p:p + Lr]; p += Lr
            A = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8)).to(dev)[sid]; rett = torch.from_numpy(ret).to(dev)
            stop = False
            for ep in range(EPOCHS):
                perm = torch.randperm(T, device=dev)
                for mb in range(NMB):
                    steps = torch.zeros(T, dtype=torch.bool, device=dev); steps[perm[mb::NMB]] = True
                    rows = steps[sid]
                    lc, ld = logps()
                    lpi = 0.0
                    for l_new, l_old in ((lc, lc0), (ld, ld0)):                    # one clipped surrogate per sub-actor, shared advantage
                        ratio = (l_new - l_old).exp()
                        lpi = lpi - torch.min(ratio * A, ratio.clamp(1 - CLIP, 1 + CLIP) * A)[rows].mean()
                    lv = ((net.value(X, sid, T, cntf) - rett) ** 2)[steps].mean()
                    opt.zero_grad(); (lpi + VCOEF * lv + REG * reg[0]).backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
                    with torch.no_grad():
                        net.logstd.clamp_(-3.0, 0.5)
                        kl = float((lc0 - lc).mean() + (ld0 - ld).mean())
                    if kl > 0.03:
                        stop = True; break
                if stop:
                    break
            Rm = float(np.mean([t['ret'] for t in trajs]))
            out['curve'].append(dict(it=it, episodes=it * EPI, mean_return=Rm, std=float(net.logstd.exp()), kl=kl, steps=int(T)))
            if it % 25 == 0 or it == ITERS:
                vm = {str(la): ev(weights(net), la, VAL) for la in LAMS_TEST}
                sc = float(np.max([cost(m) for m in vm.values()]))             # worst arrival rate: every rate has to be feasible
                out['evals'].append(dict(it=it, episodes=it * EPI, val=vm, cost=sc))
                if sc < best[0]:
                    best = (sc, it, copy.deepcopy(net.state_dict()))
                print(f'it {it:3d} ({it * EPI} episodes): mean return {Rm:.1f}  std {net.logstd.exp().item():.3f}  kl {kl:.4f} | validation ' + fmt(vm)
                      + f' | cost {sc:.1f} (best {best[0]:.1f} at it {best[1]})  ({time.time() - t0:.0f}s)', flush=True)
                torch.save(net.state_dict(), SD.OUT.replace('.json', f'_it{it}.pt'))
            elif it % 5 == 0:
                print(f'it {it:3d} ({it * EPI} episodes): mean return {Rm:.1f}  std {net.logstd.exp().item():.3f}  kl {kl:.4f}', flush=True)
        out['final'] = {str(la): ev(weights(net), la, TEST) for la in LAMS_TEST}
        torch.save(net.state_dict(), SD.OUT.replace('.json', '_final.pt'))
        net.load_state_dict(best[2]); torch.save(best[2], SD.OUT.replace('.json', '_best.pt'))
        out['best_val_iter'] = best[1]; out['best_val'] = {str(la): ev(weights(net), la, TEST) for la in LAMS_TEST}
    print('TEST final policy: ' + fmt(out['final']) + f'\n     best-validation checkpoint (it {best[1]}): ' + fmt(out['best_val']) + f'  ({time.time() - t0:.0f}s)', flush=True)
    json.dump(out, open(SD.OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
