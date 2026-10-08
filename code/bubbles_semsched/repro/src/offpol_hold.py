#!/usr/bin/env python3
"""Other learners on the SAME environment, observation, reward and training budget as hppo_hold.py - the comparison set that
TCOM / TWC papers on DRL resource allocation customarily report (e.g. D3QN, DDPG, TD3 next to the proposed scheme).

  OFF_ALGO=d3qn  dueling double DQN on a discretised action: semantic level (L) x matching-weight multiplier (5 bins, 0.1 ... 10).
  OFF_ALGO=td3   TD3 on a continuous relaxation of the hybrid action (Hausknecht & Stone style): L level scores + 1 weight
                 multiplier in [-1, 1]; the level is the arg max of the (masked) scores.
UAV agents share one network (independent learners, team reward), as in hppo_hold.py. No guidance by the hand rule: the weight
multiplies the plain urgency 1 / time left; the constraint queue is in the observation and in the drift-plus-penalty reward.
A transition of an agent runs from one decision of that UAV to its next one (discounted team reward in between); a UAV that
does not decide again inside the recorded window ends there.

Usage: python offpol_hold.py <sim_inputs.json> <out.json> <unused> <M> <unused> [procs=8] [iters=250]
       env: OFF_ALGO, PPO_SEED, PPO_LAMS, PPO_LAMS_TEST, EVAL_LAMS, and the environment switches of hppo_hold.py / sim_holdq.py"""
import sys, json, time, os, copy
import numpy as np
import torch, torch.nn as nn
from multiprocessing import Pool
import hppo_hold as HP
S, H, C, SD = HP.S, HP.H, HP.C, HP.SD

ALGO = os.environ.get('OFF_ALGO', 'd3qn')
M, PROCS, ITERS, SEED, L, NF_, HID = HP.M, HP.PROCS, HP.ITERS, HP.SEED, HP.L, HP.NF_, 64
WB = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]); NW = len(WB); NA = L * NW          # log10 multipliers of the discretised action
GAMMA, LR, TAU, BATCH, GSTEPS, BUF, START = 0.99, 3e-4, 0.005, 512, 250, 500000, 20000
EPS0, EPS1, EPS_FRAC, SIG0, SIG1, POL_NOISE, NOISE_CLIP, POL_DELAY = 1.0, 0.05, 0.6, 0.3, 0.1, 0.2, 0.5, 2
EVAL_LAMS = [float(x) for x in os.environ.get('EVAL_LAMS', '55,60,65').split(',')]; EVAL_SEEDS = list(range(401, 413))
relu = lambda x: np.maximum(x, 0.0)


def q_np(w, X):
    h = relu(relu(X @ w['a1.weight'].T + w['a1.bias']) @ w['a2.weight'].T + w['a2.bias'])
    A = h @ w['adv.weight'].T + w['adv.bias']
    return h @ w['v.weight'].T + w['v.bias'] + A - A.mean(1, keepdims=True)


def pi_np(w, X):
    h = relu(relu(X @ w['a1.weight'].T + w['a1.bias']) @ w['a2.weight'].T + w['a2.bias'])
    return np.tanh(h @ w['a3.weight'].T + w['a3.bias'])


def decide(w, X, mask, rng, expl):
    """returns (level, log10 multiplier, stored action) for every candidate; expl = epsilon (d3qn) or noise std (td3)"""
    n = len(X); anyok = mask.any(1)
    if ALGO == 'd3qn':
        q = np.where(np.repeat(mask, NW, axis=1), q_np(w, X), -1e18)
        act = np.argmax(q, 1)
        if expl > 0:
            rnd = rng.random(n) < expl
            for j in np.nonzero(rnd & anyok)[0]:
                act[j] = rng.choice(np.nonzero(mask[j])[0]) * NW + rng.integers(NW)
        act = np.where(anyok, act, NW // 2)
        return act // NW, WB[act % NW], act.astype(np.int64)
    a = pi_np(w, X)
    if expl > 0:
        a = np.clip(a + expl * rng.standard_normal(a.shape), -1.0, 1.0)
    lev = np.where(anyok, np.argmax(np.where(mask, a[:, :L], -1e18), 1), 0)
    return lev, a[:, L], a.astype(np.float32)


def rollout(args):
    """mode 'train': exploring policy, returns the recorded window as arrays; mode 'eval': greedy policy, returns sim_holdq.raw"""
    w, lam, seed, mode, expl = args
    train = mode == 'train'
    F = H.make_flights(lam, M, seed, *((HP.T_EP / 3600, HP.WARM_S / 3600) if train else (1.5, 0.5)))
    sim = S.Sim(F, M)
    rng = np.random.default_rng(seed + 4242)
    s0 = int(HP.WARM_S / C.DT); s1 = s0 + HP.T_EP
    fis, Xs, acts, masks, cnts, rews = [], [], [], [], [], []
    lost0 = 0
    while not sim.finished():
        nh = sim.get()
        if train and sim.s >= s1:
            break
        rec = train and sim.s >= s0
        lost = sum(f['lost'] for f in F)
        r = -(nh + HP.LOSS_PEN * (lost - lost0)); lost0 = lost
        if sim.cand:
            e = sim.e1()
            X, slack = HP.observe(sim, nh, e); mask = e.T > 0
            lev, a, act = decide(w, X, mask, rng, expl if train else 0.0)
            pen = HP.price(sim, lev); ids = [fi for fi, _, _ in sim.info]
            sim.step(10.0 ** a / np.maximum(slack, 1.0), lev)
            if rec:
                fis.append(ids); Xs.append(X.astype(np.float32)); acts.append(act); masks.append(mask); cnts.append(len(ids))
                rews.append((r - HP.BETA * pen(sim)) * HP.RSCALE)
        else:
            sim.step(None)
            if rec and rews:
                rews[-1] += r * HP.RSCALE
    if not train:
        return S.raw(F)
    if not Xs:
        return None
    return dict(fi=np.concatenate([np.array(x) for x in fis]), X=np.concatenate(Xs), act=np.concatenate(acts), mask=np.concatenate(masks),
                cnt=np.array(cnts), rew=np.array(rews), ret=float(np.sum(rews)) / HP.RSCALE)


def transitions(tr):
    """per-agent transitions (obs, act, mask, discounted reward to its next decision, discount, next obs, next mask)"""
    cnt = tr['cnt']; K = len(cnt)
    slot = np.repeat(np.arange(K), cnt)
    G = np.concatenate([[0.0], np.cumsum(GAMMA ** np.arange(K) * tr['rew'])])
    nxt = np.full(len(slot), -1); last = {}
    for i in range(len(slot) - 1, -1, -1):
        f = tr['fi'][i]; nxt[i] = last.get(f, -1); last[f] = i
    has = nxt >= 0; j = np.maximum(nxt, 0)
    k2 = np.where(has, slot[j], K)
    R = (G[k2] - G[slot]) / GAMMA ** slot
    return tr['X'], tr['act'], tr['mask'], R.astype(np.float32), np.where(has, GAMMA ** (k2 - slot), 0.0).astype(np.float32), tr['X'][j], tr['mask'][j]


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = nn.Linear(NF_, HID); self.a2 = nn.Linear(HID, HID); self.v = nn.Linear(HID, 1); self.adv = nn.Linear(HID, NA)

    def forward(self, X):
        h = torch.relu(self.a2(torch.relu(self.a1(X)))); A = self.adv(h)
        return self.v(h) + A - A.mean(1, keepdim=True)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = nn.Linear(NF_, HID); self.a2 = nn.Linear(HID, HID); self.a3 = nn.Linear(HID, L + 1)

    def forward(self, X):
        return torch.tanh(self.a3(torch.relu(self.a2(torch.relu(self.a1(X))))))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.ModuleList([nn.Sequential(nn.Linear(NF_ + L + 1, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, 1)) for _ in range(2)])

    def forward(self, X, a):
        z = torch.cat([X, a], 1)
        return self.q[0](z).squeeze(1), self.q[1](z).squeeze(1)


def npw(net):
    return {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}


def main():
    t0 = time.time()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    if ALGO == 'd3qn':
        net = QNet().to(dev); tgt = copy.deepcopy(net); opt = torch.optim.Adam(net.parameters(), lr=LR); act_net = net
    else:
        actor = Actor().to(dev); atgt = copy.deepcopy(actor); crit = Critic().to(dev); ctgt = copy.deepcopy(crit)
        opt_a = torch.optim.Adam(actor.parameters(), lr=LR); opt_c = torch.optim.Adam(crit.parameters(), lr=LR); act_net = actor
    adim = () if ALGO == 'd3qn' else (L + 1,)
    B = dict(X=np.zeros((BUF, NF_), np.float32), a=np.zeros((BUF,) + adim, np.int64 if ALGO == 'd3qn' else np.float32), m=np.zeros((BUF, L), bool),
             R=np.zeros(BUF, np.float32), d=np.zeros(BUF, np.float32), X2=np.zeros((BUF, NF_), np.float32), m2=np.zeros((BUF, L), bool))
    ptr = size = 0; nupd = 0
    out = dict(algo=ALGO, M=M, seed=SEED, qbar=S.QBAR, qint=S.QINT, zmode=S.ZMODE, ports=C.NPORT, beta=HP.BETA, episodes=ITERS * HP.EPI,
               hyper=dict(gamma=GAMMA, lr=LR, tau=TAU, batch=BATCH, gsteps=GSTEPS, buffer=BUF, hidden=HID, lams_train=HP.LAMS_TRAIN), curve=[], evals=[])
    with Pool(PROCS) as pool:
        def ev(w, lam, seeds):
            return S.summary(pool.map(rollout, [(w, lam, s, 'eval', 0.0) for s in seeds]), lam)
        fmt = lambda d: ' '.join(f"{la}/h hold {m['mean_hold']:.1f}s buffer {m['buffer95']:.1f}s N {m['N_buf']:.1f} lost {m['lost']:.3f} recall {m['recall']:.4f} "
                                 f"below {m['flights_below']:.3f}" for la, m in d.items())
        print(f'{ALGO}: {NA if ALGO == "d3qn" else L + 1} action outputs, M {M}, constraint queue {S.ZMODE}, ports {C.NPORT}, train rates {HP.LAMS_TRAIN}', flush=True)
        best = (1e18, 0, copy.deepcopy(act_net.state_dict()))
        for it in range(1, ITERS + 1):
            frac = min(1.0, (it - 1) / max(EPS_FRAC * ITERS, 1))
            expl = (EPS0 + (EPS1 - EPS0) * frac) if ALGO == 'd3qn' else (SIG0 + (SIG1 - SIG0) * frac)
            if ALGO == 'td3' and size < START:
                expl = 1.0                                                     # (nearly) random actions until the buffer has enough
            w = npw(act_net)
            trajs = [t for t in pool.map(rollout, [(w, float(rng.choice(HP.LAMS_TRAIN)), int(10 ** 6 * (SEED + 1) + it * 100 + p), 'train', expl)
                                                    for p in range(HP.EPI)]) if t]
            for tr in trajs:
                X, a, m, R, d, X2, m2 = transitions(tr)
                n = len(R); idx = (ptr + np.arange(n)) % BUF
                B['X'][idx], B['a'][idx], B['m'][idx], B['R'][idx], B['d'][idx], B['X2'][idx], B['m2'][idx] = X, a, m, R, d, X2, m2
                ptr = (ptr + n) % BUF; size = min(size + n, BUF)
            loss_v = 0.0
            if size >= START:
                for g in range(GSTEPS):
                    i = rng.integers(0, size, BATCH)
                    X = torch.from_numpy(B['X'][i]).to(dev); X2 = torch.from_numpy(B['X2'][i]).to(dev)
                    R = torch.from_numpy(B['R'][i]).to(dev); d = torch.from_numpy(B['d'][i]).to(dev)
                    m2 = torch.from_numpy(B['m2'][i]).to(dev); a = torch.from_numpy(B['a'][i]).to(dev)
                    if ALGO == 'd3qn':
                        with torch.no_grad():
                            ma = m2.repeat_interleave(NW, dim=1)
                            a2 = torch.where(ma, net(X2), torch.full((BATCH, NA), -1e18, device=dev)).argmax(1)
                            y = R + d * tgt(X2).gather(1, a2[:, None]).squeeze(1)
                        loss = nn.functional.smooth_l1_loss(net(X).gather(1, a[:, None]).squeeze(1), y)
                        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
                        with torch.no_grad():
                            for p, pt in zip(net.parameters(), tgt.parameters()):
                                pt.mul_(1 - TAU).add_(TAU * p)
                    else:
                        with torch.no_grad():
                            a2 = (atgt(X2) + (POL_NOISE * torch.randn(BATCH, L + 1, device=dev)).clamp(-NOISE_CLIP, NOISE_CLIP)).clamp(-1, 1)
                            q1, q2 = ctgt(X2, a2); y = R + d * torch.min(q1, q2)
                        q1, q2 = crit(X, a)
                        loss = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)
                        opt_c.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(crit.parameters(), 10.0); opt_c.step()
                        nupd += 1
                        if nupd % POL_DELAY == 0:
                            la = -crit(X, actor(X))[0].mean()
                            opt_a.zero_grad(); la.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 10.0); opt_a.step()
                            with torch.no_grad():
                                for n_, t_ in ((actor, atgt), (crit, ctgt)):
                                    for p, pt in zip(n_.parameters(), t_.parameters()):
                                        pt.mul_(1 - TAU).add_(TAU * p)
                    loss_v = float(loss)
            Rm = float(np.mean([t['ret'] for t in trajs])) if trajs else 0.0
            out['curve'].append(dict(it=it, episodes=it * HP.EPI, mean_return=Rm, explore=float(expl), buffer=int(size), loss=loss_v))
            if it % 25 == 0 or it == ITERS:
                vm = {str(la): ev(npw(act_net), la, HP.VAL) for la in HP.LAMS_TEST}
                sc = float(np.max([HP.cost(m) for m in vm.values()]))
                out['evals'].append(dict(it=it, episodes=it * HP.EPI, val=vm, cost=sc))
                if sc < best[0]:
                    best = (sc, it, copy.deepcopy(act_net.state_dict()))
                print(f'it {it:3d} ({it * HP.EPI} episodes): mean return {Rm:.1f}  explore {expl:.2f}  loss {loss_v:.4f} | validation ' + fmt(vm)
                      + f' | cost {sc:.1f} (best {best[0]:.1f} at it {best[1]})  ({time.time() - t0:.0f}s)', flush=True)
            elif it % 5 == 0:
                print(f'it {it:3d} ({it * HP.EPI} episodes): mean return {Rm:.1f}  explore {expl:.2f}  loss {loss_v:.4f}  buffer {size}', flush=True)
        torch.save(act_net.state_dict(), SD.OUT.replace('.json', '_final.pt')); torch.save(best[2], SD.OUT.replace('.json', '_best.pt'))
        out['best_val_iter'] = best[1]; out['careful'] = {}
        for name, sd in (('final', act_net.state_dict()), ('best', best[2])):
            act_net.load_state_dict(sd); w = npw(act_net)
            print(f'--- {ALGO} {name} (it {ITERS if name == "final" else best[1]}): {len(EVAL_SEEDS)} traffic seeds pooled', flush=True)
            for la in EVAL_LAMS:
                rs = pool.map(rollout, [(w, la, s, 'eval', 0.0) for s in EVAL_SEEDS]); m = S.summary(rs, la)
                m['holds'] = [np.round(r[0], 1).tolist() for r in rs]; out['careful'][f'{name}|{la:.0f}'] = m
                print(f"  {la:.0f}/h  mean hold {m['mean_hold']:6.2f} s  buffer95 {m['buffer95']:6.1f} s  N with buffer {m['N_buf']:.2f}{'!' if m['N_buf'] > H.N_REF else ' '}  "
                      f"lost {m['lost']:.4f}  recall {m['recall']:.4f}{'*' if m['recall'] >= S.QBAR - 5e-4 else ' '}  flights below {m['flights_below']:.3f}", flush=True)
    json.dump(out, open(SD.OUT, 'w'))
    print(f'DONE ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
