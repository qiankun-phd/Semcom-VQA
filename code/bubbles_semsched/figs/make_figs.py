#!/usr/bin/env python3
"""Figures of the BUBBLES / structured-PPO results in IEEE Transactions style (see ieee_style.py). Reads the aggregated JSONs under
scratch_bubbles/res/ (never the simulators) and writes PDF + PNG to figs/out/.

ONE comparison set in every figure (user, 2026-10-07):
  Proposed (structured-actor PPO) | H-PPO (plain hybrid-action PPO, same reward) | D3QN | TD3            learners
  Drift-plus-penalty (Lyapunov rule) | Fixed level (no semantic adaptation, channel matching)             no learning
  Grid search (56 points over the two structure parameters p, V of our own policy family; called 'Exhaustive search' until
  2026-10-08, renamed by the user: it is tuned at one operating point on the development traffic, not a global optimum)   reference, in ink
  HEVC inter (conventional codec)                                                                         capacity / semantic figures, in ink
The penalty-reward PPO and fixed level + EDF are ablation / table material, not in these figures.

  python make_figs.py              all figures whose data are present
  python make_figs.py conv sys     only the named groups

  group    files                                   data
  conv     fig_convergence (a)(b)                  res/gate/sppoF_p{1,2}_s*.json, mrl5k_{B,d3qn,td3}_M4_s0.json (5000 episodes, first pass: one seed;
                                                   until they exist: mrl_*_s?.json, 2000 episodes, 3 seeds), res/pass1/val_refs.json (rules on the
                                                   validation traffic, seeds 301-304, 60 flights/h)
  struct   fig_structure (a)(b)                    res/gate/sppoF_p{1,2}_s*.json ['curve']
  sys      fig_hold_vs_rate, fig_nonconf_vs_rate,  res/main/main_M4.json (12 traffic seeds, strategic deconfliction in the loop) + res/pass1/
           fig_nonconf_vs_buffer                   sys5k_{B,d3qn,td3}.json (else sys2k_*: the 2000-episode checkpoints)
  cap      fig_capacity_vs_channels                res/main/main_M{3,4,5,6to10}.json + main_hevc12.json (else main_hevc.json: 40/50/60 only, 6 seeds)
  sem      fig_semantic_recall_vs_snr              res/stageA/sim_inputs.json (468 test images; the table the simulator reads)
  land     fig_landscape                           res/pass1/landscape_pv.json + the parameter trajectories of sppoF_p1
  trade    fig_tradeoff                            res/pass1/tradeoff_mg*.json + res/main/main_M4_48seeds.json (48 traffic seeds)
  tokens   fig_recall_vs_tokens (supplementary)    res/a3/a3_per_image.json"""
import sys, os, json, glob
import numpy as np
import ieee_style as S
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

R = os.environ.get('BUB_RES') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'res')       # BUB_RES: another result tree (e.g. a reproduction)
J = lambda *p: json.load(open(os.path.join(R, *p)))
has = lambda *p: os.path.exists(os.path.join(R, *p))
plain = FuncFormatter(lambda v, _: f'{v:g}')
ORDER = ['prop', 'ppo', 'd3qn', 'td3', 'lyap', 'fixm', 'exh', 'hevc']
EXH = 'lyapp3v0.1'          # best feasible point of the grid search over (p, V) at M = 4 (landscape_pv, confirmed with 48 traffic seeds)


def logy(ax, lo, hi):
    ax.set_yscale('log'); ax.set_ylim(lo, hi); ax.yaxis.set_major_formatter(plain); ax.yaxis.set_minor_formatter(NullFormatter())


# ------------------------------------------------------------------ learning curves
def val_runs(pats):
    for pat in pats:
        runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(R, 'gate', pat)))]
        runs = [r for r in runs if len(r.get('evals', [])) > 1]
        if runs:
            return runs
    return []


def track(runs, get, skip0=False):
    """validation metric against training episodes on the grid common to all training seeds: episodes, values (seeds x points)"""
    grid = sorted(set.intersection(*[set(e['episodes'] for e in r['evals']) for r in runs]))
    grid = [g for g in grid if g > 0 or not skip0]
    return np.array(grid), np.array([[get(next(e for e in r['evals'] if e['episodes'] == g)['val']['60.0']) for g in grid] for r in runs])


NEW = sorted(glob.glob(os.path.join(R, 'pass1', 'sppoX_fast5_s?.json')))       # the five runs of the final training protocol


def proposed(get):
    """validation curve of the proposed learner: the final protocol (structure only, 4000 episodes) if its five runs are there, else the
    earlier two-phase protocol (4000 episodes structure only, then 1000 with the residual networks)"""
    if len(NEW) == 5:
        return track([json.load(open(f)) for f in NEW], get)
    x1, v1 = track(val_runs(['sppoF_p1_s?.json']), get); x2, v2 = track(val_runs(['sppoF_p2_s?.json']), get, skip0=True)
    return np.concatenate([x1, x1[-1] + x2]), np.concatenate([v1, v2], 1)


def par_tracks(par):
    """learned structure parameter against the training episodes, one row per training seed (8 episodes per iteration)"""
    if len(NEW) == 5:
        y = np.array([[e[par] for e in json.load(open(f))['curve']] for f in NEW])
    else:
        y = np.array([[e[par] for e in J('gate', f'sppoF_p1_s{s}.json')['curve']] + [e[par] for e in J('gate', f'sppoF_p2_s{s}.json')['curve']] for s in range(5)])
    return np.arange(1, y.shape[1] + 1) * 8, y


LEARN = (('td3', ['mrl5k_td3_M4_s?.json', 'mrl_td3_M4_s?.json']), ('d3qn', ['mrl5k_d3qn_M4_s?.json', 'mrl_d3qn_M4_s?.json']),
         ('ppo', ['mrl5k_B_M4_s?.json', 'mrl_B_M4_s?.json']))   # 5000 episodes, five training seeds (seed 0: 3090 server, 1-4: server 182)


def curve(ax, x, v, key, floor):
    v = np.maximum(v, floor)
    if len(v) > 1:                                             # several training seeds: median and range
        S.band(ax, x, v.min(0), v.max(0), key)
    S.line(ax, x, np.median(v, 0), key, markevery=max(1, len(x) // 8))


def fig_conv():
    """(a) objective, (b) the per-flight recall constraint: validation at 60 flights/h against the training episodes"""
    f, axs = S.fig(ncols=2, width=S.COL2, height=2.75)
    refs = J('pass1', 'val_refs.json')['res'] if has('pass1', 'val_refs.json') else {
        'lyap0.3': J('gate', 'sppoF_p1_s0.json')['evals'][0]['val'], 'lyapp3v0.03': J('gate', 'mrl_A_M4_s0.json')['evals'][0]['val']}
    if has('pass1', 'val_refs_exh.json'):
        refs = dict(refs, **J('pass1', 'val_refs_exh.json')['res'])
    for ax, get, floor, lab in ((axs[0], lambda m: m['mean_hold'], 0.1, 'Mean hold per flight (s)'),
                                (axs[1], lambda m: 100 * m['flights_below'], 0.5, 'Flights below the recall requirement (%)')):
        for key, pats in LEARN:                                # episode 0 of these is the untrained network (symbolic layer only): left out
            x, v = track(val_runs(pats), get, skip0=True); curve(ax, x, v, key, floor)
        x, v = proposed(get); curve(ax, x, v, 'prop', floor)
        for key, pol in (('lyap', 'lyap0.3'), ('fixm', 'fixed2'), ('exh', EXH if EXH in refs else 'lyapp3v0.03')):
            if pol in refs:
                S.level(ax, max(get(refs[pol]['60.0']), floor), key)
        ax.set_xlabel('Training episodes'); ax.set_xlim(0, 5000); ax.set_ylabel(lab)
    logy(axs[0], 0.07, 3000); logy(axs[1], 0.8, 130)
    axs[1].axhspan(5, 130, color='#000000', alpha=0.06, linewidth=0, zorder=0)           # the region that violates the requirement
    axs[1].axhline(5, color=S.INK, linewidth=0.7)
    axs[1].annotate('requirement violated (> 5%)', xy=(0.985, 5), xycoords=('axes fraction', 'data'), xytext=(0, 3), textcoords='offset points',
                    ha='right', va='bottom', fontsize=7.5)
    S.legend_fig(f, axs[0], ORDER, ncol=7); S.panels(axs)
    return S.save(f, 'fig_convergence')


def fig_conv_violation():
    """variant of the learning curves for readers who take every curve as 'lower is better': panel (b) shows the VIOLATION of the per-flight
    recall constraint (percentage points of flights below the requirement in excess of the allowed 5 %), so a scheme that meets the
    requirement is at zero whatever slack it leaves. Same data as fig_conv; saved under its own name."""
    f, axs = S.fig(ncols=2, width=S.COL2, height=2.75)
    refs = dict(J('pass1', 'val_refs.json')['res'], **(J('pass1', 'val_refs_exh.json')['res'] if has('pass1', 'val_refs_exh.json') else {}))
    for ax, get, floor, lab in ((axs[0], lambda m: m['mean_hold'], 0.1, 'Mean hold per flight (s)'),
                                (axs[1], lambda m: max(100 * m['flights_below'] - 5.0, 0.0), 0.0, 'Recall-constraint violation (points)')):
        for key, pats in LEARN:
            x, v = track(val_runs(pats), get, skip0=True); curve(ax, x, v, key, floor)
        x, v = proposed(get); curve(ax, x, v, 'prop', floor)
        for key, pol in (('lyap', 'lyap0.3'), ('fixm', 'fixed2'), ('exh', EXH if EXH in refs else 'lyapp3v0.03')):
            if pol in refs:
                S.level(ax, max(get(refs[pol]['60.0']), floor), key)
        ax.set_xlabel('Training episodes'); ax.set_xlim(0, 5000); ax.set_ylabel(lab)
    logy(axs[0], 0.07, 3000); axs[1].set_ylim(-4, 66)
    axs[1].annotate('share of flights below the recall requirement\nin excess of the allowed 5%; 0 = requirement met', xy=(0.975, 0.955), xycoords='axes fraction',
                    ha='right', va='top', fontsize=7.5, bbox=dict(boxstyle='square,pad=0.25', facecolor='white', edgecolor='none'))
    S.legend_fig(f, axs[0], ORDER, ncol=7); S.panels(axs)
    return S.save(f, 'fig_convergence_violation')


def fig_struct():
    """what the structured actor learns: (a) urgency exponent p, (b) weight V, five training seeds"""
    f, axs = plt.subplots(2, 1, figsize=(S.COL1, 3.75), layout='constrained', sharex=True)
    f.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0.03)
    c = S.STYLE['prop'][1]
    for ax, par, lab, tag in ((axs[0], 'p', 'Urgency exponent $p$', '(a)'), (axs[1], 'V', 'Weight $V$', '(b)')):
        x, y = par_tracks(par)
        for k, row in enumerate(y):
            ax.plot(x, row, color=c, linewidth=0.55, alpha=0.6, label='Individual training seeds' if k == 0 else None)
        ax.plot(x, y.mean(0), color=c, linewidth=1.5, label='Mean of five seeds')
        if len(NEW) != 5:
            ax.axvline(4000, color=S.INK, linestyle=(0, (4, 2)), linewidth=0.7)
        ax.set_ylabel(lab); ax.set_xlim(0, x[-1]); lo, hi = y.min(), y.max(); ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.14 * (hi - lo))
        ax.annotate(tag, xy=(0.5, 0.97), xycoords='axes fraction', ha='center', va='top', fontsize=8)
    axs[1].set_xlabel('Training episodes')
    h, l = axs[0].get_legend_handles_labels()
    lg = f.legend(h, l, loc='outside upper center', ncol=2, handlelength=2.3); lg.get_frame().set_linewidth(0.5)
    return S.save(f, 'fig_structure')


# ------------------------------------------------------------------ system results, M = 4
RULES = (('fixm', 'fixed2'), ('lyap', 'lyap0.3'))
OTHER = (('td3', 'td3', 'td3'), ('d3qn', 'd3qn', 'd3qn'), ('ppo', 'B', 'ppo'))        # style key, file tag, policy name inside the file


FINAL = has('final', 'final48_M4_main.json')                  # test traffic (seeds 501-548, 48 seeds): the numbers of the paper


def system(ax, xs, get):
    """get(cells, policy name, x) -> value. Test traffic if it is there (final48_M4_*: rules, grid-search rule, the five policies of
    the final protocol, and seed 0 of the comparison learners over the whole grid); else the earlier development-traffic files."""
    if FINAL:
        d = J('final', 'final48_M4_main.json')['cells']
        for key, pol in RULES + (('exh', EXH),):
            S.line(ax, xs, [get(d, pol, x) for x in xs], key)
        for key, tag, pol in OTHER:
            if has('final', f'final48_M4_{tag}.json'):
                c = J('final', f'final48_M4_{tag}.json')['cells']; S.line(ax, xs, [get(c, pol, x) for x in xs], key)
        v = np.array([[get(d, f'f{s}', x) for x in xs] for s in range(5)])
    else:
        d = J('main', 'main_M4.json')['cells']
        for key, pol in RULES:
            S.line(ax, xs, [get(d, pol, x) for x in xs], key)
        if has('main', 'main_M4_exh.json'):
            S.line(ax, xs, [get(J('main', 'main_M4_exh.json')['cells'], EXH, x) for x in xs], 'exh')
        else:
            S.line(ax, xs, [get(d, 'lyapp3v0.03', x) for x in xs], 'exh')
        for key, tag, pol in OTHER:
            fn = next((n for n in (f'sys5k_{tag}.json', f'sys2k_{tag}.json') if has('pass1', n)), None)
            if fn:
                c = J('pass1', fn)['cells']; S.line(ax, xs, [get(c, pol, x) for x in xs], key)
        if has('pass1', 'sysF_M4_grid.json'):                 # the five policies of the final training protocol (sppoX_fast5_s*)
            pf = J('pass1', 'sysF_M4_grid.json')['cells']; v = np.array([[get(pf, f'f{s}', x) for x in xs] for s in range(5)])
        else:
            v = np.array([[get(d, f's{s}', x) for x in xs] for s in range(5)])
    S.band(ax, xs, v.min(0), v.max(0), 'prop'); S.line(ax, xs, v.mean(0), 'prop')
    ax.xaxis.set_major_locator(FixedLocator(xs))


def fig_sys():
    m4 = J('final', 'final48_M4_main.json') if FINAL else J('main', 'main_M4.json'); lams, bufs = m4['lams'], m4['bufs']
    non = lambda c, p, la, b: max(100 * (1 - c[f'{p}|4|{la:.0f}|{b:.0f}']['conformance']), 0.05)
    out = []
    f, ax = S.fig()
    system(ax, lams, lambda c, p, la: max(c[f'{p}|4|{la:.0f}|0']['mean_hold'], 0.1))
    logy(ax, 0.1, 3000); ax.set_xlim(38.5, 66.5); ax.set_xlabel('Requested arrival rate (flights/h)'); ax.set_ylabel('Mean hold per flight (s)')
    S.legend_top(ax, order=ORDER); out.append(S.save(f, 'fig_hold_vs_rate'))
    for name, xs, get, xlab, xlim, lx, lha in (
            ('fig_nonconf_vs_rate', lams, lambda c, p, la: non(c, p, la, 0), 'Requested arrival rate (flights/h)', (38.5, 66.5), 0.985, 'right'),
            ('fig_nonconf_vs_buffer', bufs, lambda c, p, b: non(c, p, 60, b), 'Temporal buffer $b$ (s)', (-3, 63), 0.09, 'left')):
        f, ax = S.fig()
        system(ax, xs, get)
        S.ref(ax, 5, 'allowed 5%', x=lx, ha=lha, va='top')
        logy(ax, 0.08, 150); ax.set_xlim(*xlim); ax.set_xlabel(xlab); ax.set_ylabel('Non-conforming flights (%)')
        S.legend_top(ax, order=ORDER); out.append(S.save(f, name))
    return out


def fig_cap():
    """capacity (highest requested rate on the 5-flights/h grid that meets every requirement, with the best temporal buffer) against the
    number of channels. 48 traffic seeds where that run exists (main_M*_48seeds, sysX_48seeds), 12 otherwise (fixed level at M = 3, 4, where
    it is infeasible by a wide margin, and HEVC). Grid search = the best member of the rule family that was run at that M;
    the proposed scheme = median and range over its five trained policies. HEVC needs a logarithmic channel axis."""
    if FINAL:
        return fig_cap_final()
    cap, top = {}, 0
    for n in ('main_M3.json', 'main_M4.json', 'main_M5.json', 'main_M6to10.json', 'main_M3_48seeds.json', 'main_M4_48seeds.json', 'main_M5_48seeds.json',
              'main_M6to10_48seeds.json'):                     # later files (48 seeds) overwrite the 12-seed values
        if has('main', n):
            d = J('main', n); top = max(top, max(d['lams']))
            for k, v in d['capacity'].items():
                p, M = k.split('|'); cap.setdefault(p, {})[int(M)] = v['rate']
    for n in ('sysX_48seeds.json', 'sysN_M3_48seeds.json', 'sysN_M5_48seeds.json', 'sysF_M3_48seeds.json', 'sysF_M4_48seeds.json', 'sysF_M5_48seeds.json'):
        if has('pass1', n):                                    # more members of the rule family and the five policies of the final protocol (f0..f4)
            for k, v in J('pass1', n)['capacity'].items():
                p, M = k.split('|')
                if p.startswith('lyapp') or (p[0] == 'f' and p[1:].isdigit() and n.startswith('sysF')):
                    cap.setdefault(p, {})[int(M)] = v['rate']
    PROP = [f'f{s}' for s in range(5)] if 'f0' in cap else [f's{s}' for s in range(5)]
    for tag, pol, key in (('B', 'ppo', 'ppo'), ('d3qn', 'd3qn', 'd3qn'), ('td3', 'td3', 'td3')):   # comparison learners, 48 traffic seeds: trained at
        for M in (3, 4, 5):                                                                          # M = 4 and used as is, or trained for that M
            for n in (f'base48_{tag}_M{M}.json', f'base48own_{tag}_M{M}.json'):                      # (base48own_*): the better of the two
                if has('pass1', n):
                    r = J('pass1', n)['capacity'][f'{pol}|{M}']['rate']
                    cap.setdefault(key, {})[M] = max(cap.get(key, {}).get(M, 0), r)
    hevc = next((n for n in ('main_hevc12.json', 'main_hevc.json') if has('main', n)), None)
    if hevc:                                                   # main_hevc12 = same rate grid as the other schemes, 12 traffic seeds
        cap['hevc'] = {int(k.split('|')[1]): v['rate'] for k, v in J('main', hevc)['capacity'].items()}
    f, ax = S.fig()
    fam = [p for p in cap if p.startswith('lyapp')]
    for key, pols in (('hevc', ['hevc']), ('td3', ['td3']), ('d3qn', ['d3qn']), ('ppo', ['ppo']), ('fixm', ['fixed2']), ('lyap', ['lyap0.3', 'lyap0.1']), ('exh', fam)):
        Ms = sorted(set(M for p in pols for M in cap.get(p, {})))
        if Ms:                                                 # several members of one scheme: the best one
            S.line(ax, Ms, [max(cap[p][M] for p in pols if M in cap.get(p, {})) for M in Ms], key)
    Ms = sorted(set(M for p in PROP for M in cap.get(p, {})))
    v = [np.array([cap[p][M] for p in PROP if M in cap.get(p, {})]) for M in Ms]
    S.band(ax, Ms, [x.min() for x in v], [x.max() for x in v], 'prop'); S.line(ax, Ms, [np.median(x) for x in v], 'prop')
    S.ref(ax, top, 'highest rate tested', x=0.985)
    if hevc:
        ax.set_xscale('log'); ax.set_xlim(2.6, 190); t = [3, 4, 5, 6, 8, 10, 20, 40, 60, 80, 120, 160]
        ax.xaxis.set_major_locator(FixedLocator(t)); ax.xaxis.set_major_formatter(plain); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_locator(FixedLocator([7, 9, 30, 50, 70, 90, 100, 110]))
    else:
        Ms = sorted(set(M for c in cap.values() for M in c)); ax.set_xlim(Ms[0] - 0.4, Ms[-1] + 0.4); ax.xaxis.set_major_locator(FixedLocator(Ms))
    ax.set_xlabel('Number of channels $M$'); ax.set_ylabel('Capacity (flights/h)'); ax.set_ylim(-4, 78)
    S.legend_top(ax, order=ORDER)
    return S.save(f, 'fig_capacity_vs_channels')


ONTOP = dict(zorder=4, markersize=8.5, markerfacecolor='none', markeredgewidth=0.8, linewidth=0.9)   # grid search over a coinciding curve
CAPB = (0, 30)                                                 # temporal buffers allowed when the capacity is read off (common to all M)


def cap_of(d, pol, M, bufs=CAPB):
    """highest tested rate at which the scheme meets every requirement with one of the buffers; 0 = not even at the lowest tested rate"""
    ok = [la for la in d['lams'] if any(d['cells'][f'{pol}|{M}|{la:.0f}|{b:.0f}']['feasible'] for b in bufs if b in d['bufs'])]
    return max(ok) if ok else 0.0


def cap_runs(bufs=CAPB):
    """feasibility on the test traffic (seeds 501-548), merged over every file that holds a run:
    {scheme key: {M: {run: {rate: meets every requirement with one of the buffers}}}}. A run is a rule (family member) or a training run."""
    ok = {}
    def put(key, M, run, d, pol):
        o = ok.setdefault(key, {}).setdefault(M, {}).setdefault(run, {})
        for la in d['lams']:
            o[la] = o.get(la, False) or any(d['cells'][f'{pol}|{M}|{la:.0f}|{b:.0f}']['feasible'] for b in bufs if b in d['bufs'])
    for M in (3, 4, 5):
        d = J('final', f'final48_M{M}_main.json')
        for p in d['pols']:
            key = 'fixm' if p == 'fixed2' else 'exh' if p.startswith('lyapp') else 'lyap' if p.startswith('lyap') else 'prop' if p[0] == 'f' else None
            if key:
                put(key, M, p, d, p)
        if has('final_182', f'final48_M{M}_prop182.json'):      # the same protocol and training seeds on a second machine (g0..g4)
            e = J('final_182', f'final48_M{M}_prop182.json')
            for s_ in range(5):
                put('prop', M, f'g{s_}', e, f'g{s_}')
        for key, tag, pol in OTHER:                             # training seed 0 = 'ppo', seeds 1-4 = 'ppo1'..'ppo4'; low48* = the low end of the grid
            for dr, fn in (('final', f'final48_M{M}_{tag}.json'), ('final_182', f'final48_M{M}_{tag}_s1to4.json'), ('final', f'low48a_M{M}_{tag}.json'),
                           ('final', f'low48_M{M}_{tag}.json')):
                if has(dr, fn):
                    e = J(dr, fn)
                    for p in e['pols']:
                        put(key, M, p[len(pol):] or '0', e, p)
        if has('final', f'low48_M{M}_rules.json'):
            e = J('final', f'low48_M{M}_rules.json'); put('fixm', M, 'fixed2', e, 'fixed2'); put('lyap', M, 'lyap0.3', e, 'lyap0.3')
        if M == 4 and has('qsweep', 'qsweep0.44_fixed.json'):   # the main setting again, fixed level at 30-60 flights/h
            put('fixm', M, 'fixed2', J('qsweep', 'qsweep0.44_fixed.json'), 'fixed2')
    if has('final', 'final48_M6to10_rules.json'):
        d = J('final', 'final48_M6to10_rules.json')
        for M in d['Ms']:
            put('fixm', M, 'fixed2', d, 'fixed2')
    if has('pass1', 'main_hevc48.json'):
        d = J('pass1', 'main_hevc48.json')
        for M in d['Ms']:
            put('hevc', M, 'fixed2', d, 'fixed2')
    return ok


def cap_table(bufs=CAPB, low=None):
    """capacities on the test traffic: {scheme key: {M: [one value per training run]}}; a rule family (drift-plus-penalty, grid search)
    counts with its best member. 0 = infeasible at the lowest tested rate, which `low` ({(key, M): rate}) receives."""
    cap = {}
    for key, byM in cap_runs(bufs).items():
        for M, runs in byM.items():
            v = [max([la for la, f_ in o.items() if f_], default=0.0) for _, o in sorted(runs.items())]
            cap.setdefault(key, {})[M] = [max(v)] if key in ('exh', 'lyap') else v
            if low is not None:
                low[(key, M)] = max(min(o) for o in runs.values())      # the grid of the run that starts highest
    return cap


def fig_cap_final():
    """capacity on the test traffic (seeds 501-548) with a temporal buffer of at most 30 s: final48_M{3,4,5}_* (every scheme; the learners
    trained at M = 4 and used as they are), final48_M6to10_rules (fixed level); HEVC from main_hevc48 (development traffic, 48 seeds). The
    learners: median over their training runs (proposed: ten = five seeds on each of two machines; comparison learners: five seeds), with
    the range as a band. Rates tested: 30-40 (M = 3), 40-65 (M = 4; 55-65 for training seeds 1-4 of the comparison learners), 55-65
    (M >= 5); 0 = infeasible at the lowest tested rate."""
    cap = cap_table()
    f, ax = S.fig()
    for key in ('hevc', 'td3', 'd3qn', 'ppo', 'fixm', 'lyap', 'prop', 'exh'):
        if key in cap:
            Ms = sorted(cap[key]); S.line(ax, Ms, [float(np.median(cap[key][M])) for M in Ms], key, **(ONTOP if key == 'exh' else {}))
    for key in ('td3', 'd3qn', 'ppo', 'prop'):
        if key in cap:
            Ms = sorted(cap[key]); S.band(ax, Ms, [min(cap[key][M]) for M in Ms], [max(cap[key][M]) for M in Ms], key)
    S.ref(ax, 65, 'highest rate tested', x=0.985)
    ax.set_xscale('log'); ax.set_xlim(2.6, 190); t = [3, 4, 5, 6, 8, 10, 20, 40, 60, 80, 120, 160]
    ax.xaxis.set_major_locator(FixedLocator(t)); ax.xaxis.set_major_formatter(plain); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(FixedLocator([7, 9, 30, 50, 70, 90, 100, 110]))
    ax.set_xlabel('Number of channels $M$'); ax.set_ylabel('Capacity (flights/h)'); ax.set_ylim(-4, 78)
    S.legend_top(ax, order=ORDER)
    return S.save(f, 'fig_capacity_vs_channels')


QS = ('0.42', '0.43', '0.44', '0.45', '0.46')


def qsweep_table(bufs=CAPB, low=None, retrained=False):
    """capacity at M = 4 on the test traffic for every recall requirement: {scheme key: {q: [one value per training run]}}. Rules count with
    their best member (fixed level: the best of the four token levels; drift-plus-penalty: V = 0.1, 0.3, 1; grid search: the (p, V)
    grid). The learners are trained once in the main setting (requirement 0.44) and used as they are - the requirement enters through the
    constraint queue - (qsweep<q>_propasis: f0..f4; qsweep<q>_{B,d3qn,td3}asis[_s1to4]: five training seeds); at 0.44 the runs of the
    capacity figure. retrained=True: the learners retrained for the requirement instead (qsweep<q>_prop: three seeds; qsweep<q>_{B,d3qn,td3}:
    one seed), hyper-parameters as at 0.44. `low` receives the lowest tested rate {(key, q): rate}."""
    cap = {}
    for q in QS:
        runs = {}
        def put(key, run, d, pol):
            o = runs.setdefault(key, {}).setdefault(run, {})
            for la in d['lams']:
                o[la] = o.get(la, False) or any(d['cells'][f'{pol}|4|{la:.0f}|{b:.0f}']['feasible'] for b in bufs if b in d['bufs'])
        for fn in (f'qsweep{q}_rules.json', f'qsweep{q}_fixed.json', f'qsweep{q}_fixedlow.json'):
            if has('qsweep', fn):
                d = J('qsweep', fn)
                for p_ in d['pols']:
                    put('fixm' if p_.startswith('fixed') else 'exh' if p_.startswith('lyapp') else 'lyap', p_, d, p_)
        if q == '0.44':
            for key, byM in cap_runs(bufs).items():
                for run, o in byM.get(4, {}).items():
                    if key != 'hevc':
                        r = runs.setdefault(key, {}).setdefault(run, {})
                        for la, v in o.items():
                            r[la] = r.get(la, False) or v
        else:
            fn = f'qsweep{q}_prop.json' if retrained else f'qsweep{q}_propasis.json'
            if has('qsweep', fn):
                d = J('qsweep', fn)
                for p_ in d['pols']:
                    put('prop', p_, d, p_)
            for key, tag, pol in OTHER:
                for fn in ([f'qsweep{q}_{tag}.json'] if retrained else [f'qsweep{q}_{tag}asis.json', f'qsweep{q}_{tag}asis_s1to4.json']):
                    if has('qsweep', fn):
                        d = J('qsweep', fn)
                        for p_ in d['pols']:
                            put(key, p_[len(pol):] or '0', d, p_)
        for key, rs in runs.items():
            v = [max([la for la, f_ in o.items() if f_], default=0.0) for _, o in sorted(rs.items())]
            cap.setdefault(key, {})[q] = [max(v)] if key in ('exh', 'lyap', 'fixm') else v
            if low is not None:
                low[(key, q)] = (min(min(o) for o in rs.values()) if key in ('exh', 'lyap', 'fixm') else max(min(o) for o in rs.values()))
    return cap


def fig_qsweep():
    """capacity against the recall requirement (M = 4, test traffic, temporal buffer of at most 30 s): how much traffic a stricter semantic
    requirement costs. Learners: trained once at 0.44 and used as they are; median over their training runs, range as a band."""
    if not has('qsweep', 'qsweep0.45_rules.json'):
        return 'no data yet'
    cap = qsweep_table()
    f, ax = S.fig()
    for key in ('td3', 'd3qn', 'ppo', 'fixm', 'lyap', 'prop', 'exh'):
        if key in cap:
            qs = sorted(cap[key]); x = [float(q) for q in qs]
            S.line(ax, x, [float(np.median(cap[key][q])) for q in qs], key, **(ONTOP if key == 'exh' else {}))
            if key in ('td3', 'd3qn', 'ppo', 'prop'):
                S.band(ax, x, [min(cap[key][q]) for q in qs], [max(cap[key][q]) for q in qs], key)
    S.ref(ax, 65, 'highest rate tested', x=0.985)
    ax.set_xlim(0.4165, 0.4635); ax.xaxis.set_major_locator(FixedLocator([float(q) for q in QS])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.2f}'))
    ax.set_xlabel('Recall requirement of a flight'); ax.set_ylabel('Capacity (flights/h)'); ax.set_ylim(-4, 78)
    S.legend_top(ax, order=ORDER)
    return S.save(f, 'fig_capacity_vs_recall')


def fig_cdf():
    """constraint statistics on the test traffic, M = 4, 60 flights/h, no buffer: (a) distribution of the recall of a flight (the requirement
    allows 5 % of the flights below 0.435), (b) share of flights holding longer than x. Proposed = the flights of all five policies."""
    if not has('final', 'final48_M4_main_raw.json'):
        return 'no per-flight data yet'
    raw = J('final', 'final48_M4_main_raw.json'); other = {key: J('final', f'final48_M4_{tag}_raw.json') for key, tag, pol in OTHER if has('final', f'final48_M4_{tag}_raw.json')}
    get = lambda d, pols, k: np.concatenate([np.array(d[f'{p}|4|60|0'][k]) for p in pols])
    series = [('fixm', raw, ['fixed2']), ('lyap', raw, ['lyap0.3'])] + [(key, other[key], [pol]) for key, tag, pol in OTHER if key in other] + \
             [('exh', raw, [EXH]), ('prop', raw, [f'f{s}' for s in range(5)])]
    f, axs = S.fig(ncols=2, width=S.COL2, height=2.75)
    for key, d, pols in series:
        x = np.sort(get(d, pols, 'frec')); k = S.kw(key); k.update(marker=None)
        axs[0].plot(x, np.arange(1, len(x) + 1) / len(x), **k)
        h = np.sort(get(d, pols, 'hold')); xs = np.array([0, 1, 2, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360])
        k2 = S.kw(key); axs[1].plot(xs + 0.5, np.maximum((h[None, :] > xs[:, None]).mean(1), 2e-4), **k2)
    axs[0].axvline(0.435, color=S.INK, linestyle=(0, (4, 2)), linewidth=0.7); axs[0].axhline(0.05, color=S.INK, linestyle=(0, (4, 2)), linewidth=0.7)
    axs[0].annotate('5%', xy=(0.985, 0.05), xycoords=('axes fraction', 'data'), xytext=(0, 2), textcoords='offset points', ha='right', va='bottom', fontsize=7.5)
    axs[0].annotate('requirement $-$ 0.005', xy=(0.435, 0.97), xycoords=('data', 'axes fraction'), xytext=(-3, 0), textcoords='offset points', ha='right', va='top', fontsize=7.5)
    axs[0].set_xlim(0.40, 0.48); axs[0].set_ylim(0.0, 1.0); axs[0].set_xlabel('Recall of the changed objects of a flight'); axs[0].set_ylabel('Share of flights (CDF)')
    axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_xlim(0.45, 400); axs[1].set_ylim(1.5e-4, 1.2)
    axs[1].xaxis.set_major_locator(FixedLocator([0.5, 1.5, 5.5, 10.5, 30.5, 60.5, 120.5, 360.5])); axs[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v - 0.5:g}'))
    axs[1].xaxis.set_minor_formatter(NullFormatter())
    axs[1].set_xlabel('Hold time $x$ (s)'); axs[1].set_ylabel('Share of flights holding longer than $x$')
    S.legend_fig(f, axs[1], ORDER, ncol=7); S.panels(axs)
    return S.save(f, 'fig_flight_distributions')


def fig_behav():
    """when are tokens sent? Share of the decisions (one per UAV with undelivered evidence and slot) that send tokens, (a) against the time left
    before the flight has to hold and (b) against the SNR of its best channel; M = 4, 60 flights/h, 12 test-traffic seeds (diag_levels.py)."""
    if not has('extra', 'diag_prop.json'):
        return 'no behaviour data yet'
    f, axs = S.fig(ncols=2, width=S.COL2, height=2.6)
    for key, fn in (('lyap', 'diag_dpp.json'), ('exh', 'diag_exh.json'), ('prop', 'diag_prop.json')):
        d = J('extra', fn); a = np.array(d['by_snr']); b = np.array(d['by_slack'])
        S.line(axs[0], range(len(b)), 100 * (1 - b[:, 0] / np.maximum(b.sum(1), 1)), key)
        e = np.array(d['snr_edges']); xc = np.concatenate([[e[0] - 1], (e[:-1] + e[1:]) / 2, [e[-1] + 1]]); n = a.sum(1); ok = n > 500
        S.line(axs[1], xc[ok], 100 * (1 - a[ok, 0] / n[ok]), key, markevery=2)
    axs[0].set_xticks(range(6)); axs[0].set_xticklabels(['holding', '0-10', '10-20', '20-30', '30-45', '> 45'])
    axs[0].set_xlabel('Time left before a hold (s)'); axs[0].set_ylabel('Decisions that send tokens (%)'); axs[0].set_ylim(-4, 104)
    axs[1].set_xlabel('SNR of the best channel (dB)'); axs[1].set_ylabel('Decisions that send tokens (%)'); axs[1].set_ylim(-4, 104); axs[1].set_xlim(-8, 32)
    S.legend_fig(f, axs[0], ORDER, ncol=3); S.panels(axs)
    return S.save(f, 'fig_token_decisions')


# ------------------------------------------------------------------ semantic side
def fig_sem():
    """recall of the changed objects against the SNR: reference-gated JSCC tokens (the two channel-symbol budgets the levels are built from)
    against HEVC inter with an ideal capacity-achieving code in the SAME number of channel uses, the symbolic layer alone and the
    uncompressed image. HEVC budget at SNR g: tokens x C/2 channel uses x log2(1 + g) / 8 bytes + the side information of the tokens."""
    d = J('stageA', 'sim_inputs.json'); snr = np.array(d['snr_pts'], float)
    pts = sorted(d['hevc_inter_gnss']); hb = np.log([p[0] for p in pts]); hr = np.maximum.accumulate([p[1] for p in pts])     # as sim_ch.py reads it
    f, ax = S.fig()
    for C, c, ls, mk in ((192, S.RAMP[3], '-', 'o'), (96, S.RAMP[1], '--', 's')):
        g = d[f'gated_C{C}']
        ax.plot(snr, g['rec'], color=c, linestyle=ls, marker=mk, markerfacecolor='white', markeredgecolor=c, label=f'Tokens, $C={C}$')
    for C, ls, mk in ((192, (0, (4, 1.5, 1, 1.5)), '+'), (96, (0, (1, 1.5)), 'x')):
        g = d[f'gated_C{C}']
        byt = g['tokens'] * C / 2 * np.log2(1 + 10 ** (snr / 10)) / 8 + g['side_bytes']
        ax.plot(snr, np.interp(np.log(byt), hb, hr), color=S.INK, linestyle=ls, linewidth=0.9, marker=mk, label=f'HEVC inter + ideal channel code, channel uses of $C={C}$')
    for y, t, va in ((d['q_sym'], 'symbolic layer only', 'top'), (d['raw_rec'], 'uncompressed image', 'bottom')):
        ax.axhline(y, color=S.INK, linestyle=(0, (4, 2)), linewidth=0.7, zorder=1.5)
        ax.annotate(t, xy=(0.02, y), xycoords=('axes fraction', 'data'), xytext=(0, 2 if va == 'bottom' else -2.5), textcoords='offset points',
                    ha='left', va=va, fontsize=7.5)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Recall of the changed objects'); ax.set_xlim(-2.6, 13.6); ax.set_ylim(0, 0.6)
    ax.xaxis.set_major_locator(FixedLocator(snr))
    S.legend_top(ax, ncol=1)
    return S.save(f, 'fig_semantic_recall_vs_snr')


def fig_tokens():
    """supplementary (modelling check A3): recall against the token count of the image, quartile groups, tokens at 13 dB"""
    rows = J('a3', 'a3_per_image.json')['rows']
    nt = np.array([r['n_tok'] for r in rows]); nn = np.array([r['n_new'] for r in rows])
    q = np.quantile(nt, [0, .25, .5, .75, 1.0]); q[-1] += 1
    groups = [(nt >= a) & (nt < b) for a, b in zip(q[:-1], q[1:])]
    x = [nt[m].mean() for m in groups]
    rec = lambda fn: [sum(fn(rows[k]) for k in np.nonzero(m)[0]) / nn[m].sum() for m in groups]
    f, ax = S.fig()
    ax.plot(x, rec(lambda r: r['tp_full']), color=S.INK, linestyle=(0, (4, 2)), marker='x', linewidth=0.9, label='Uncompressed image')
    for lab, fn, c, ls, mk in (('Tokens, $C=192$', lambda r: r['tp']['C192@13'], S.RAMP[3], '-', 'o'), ('Tokens, $C=96$', lambda r: r['tp']['C96@13'], S.RAMP[2], '--', 's'),
                               ('Symbolic layer only', lambda r: r['tp_sym'], S.RAMP[1], '-.', '^')):
        ax.plot(x, rec(fn), color=c, linestyle=ls, marker=mk, markerfacecolor='white', markeredgecolor=c, label=lab)
    ax.set_xlabel('Mean number of tokens per image (quartile groups)'); ax.set_ylabel('Recall of the changed objects')
    ax.set_xlim(10, 128); ax.set_ylim(0.30, 0.60)
    S.legend_top(ax)
    return S.save(f, 'fig_recall_vs_tokens')


# ------------------------------------------------------------------ why the learner is right: landscape and trade-off
def fig_land():
    """grid search over (p, V) at 60 flights/h, M = 4 (mean hold as filled contours of one hue, dark = short hold) with the parameter
    trajectories of the structure-only training phase on top: does the learner walk into the region the grid search finds?"""
    if not has('pass1', 'landscape_pv.json'):
        return 'no landscape data yet'
    d = J('pass1', 'landscape_pv.json')['cells']
    pv = {tuple(float(t) for t in k.split('|')[0][5:].split('v')): c for k, c in d.items()}
    ps = sorted({a for a, _ in pv}); vs = sorted({b for _, b in pv})
    Z = np.array([[pv[(p, v)]['mean_hold'] for p in ps] for v in vs])
    f, ax = S.fig(height=2.6)
    lv = [0.3, 0.5, 1, 2, 4, 8, 16, 32]
    cs = ax.contourf(ps, vs, np.clip(Z, lv[0] * 1.001, lv[-1] * 0.999), levels=lv, norm=LogNorm(lv[0], lv[-1]),
                     cmap=LinearSegmentedColormap.from_list('hold', ['#104281', '#256abf', '#86b6ef', '#eef4fd']))
    cb = f.colorbar(cs, ax=ax, pad=0.02); cb.set_label('Mean hold per flight (s)'); cb.ax.yaxis.set_major_formatter(plain); cb.outline.set_linewidth(0.5)
    bad = [(p, v) for (p, v), c in pv.items() if not c['feasible']]
    if bad:                                                    # grid points that miss a requirement
        ax.plot(*zip(*bad), marker='x', color=S.INK, markersize=3, markeredgewidth=0.6, linestyle='none', label='Grid point misses a requirement')
    c = S.STYLE['prop'][1]
    xp, yp = par_tracks('p'); xv, yv = par_tracks('V')
    for k in range(5):
        x, y = yp[k], yv[k]
        ax.plot(x, y, color=c, linewidth=0.8, label='Learning trajectories (five seeds)' if k == 0 else None)
        ax.plot(x[-1], y[-1], marker='o', color=c, markeredgecolor='white', markeredgewidth=0.5, markersize=4.5, linestyle='none')
    ax.plot(1.0, 0.3, marker='s', markerfacecolor='white', markeredgecolor=S.INK, markersize=5, linestyle='none', label='Start: drift-plus-penalty')
    ax.set_yscale('log'); ax.set_xlabel('Urgency exponent $p$'); ax.set_ylabel('Weight $V$'); ax.yaxis.set_major_locator(FixedLocator(vs))
    ax.yaxis.set_major_formatter(plain); ax.yaxis.set_minor_formatter(NullFormatter()); ax.grid(False)
    S.legend_top(ax, ncol=1)
    return S.save(f, 'fig_landscape')


def fig_trade():
    """mean hold against the share of flights below the recall requirement, traced by the margin inside the constraint queue (48 traffic seeds,
    60 flights/h, M = 4, no buffer): which scheme gives less hold at the same fairness. Grid search = the best grid point (EXH) from
    tradeoff2_mg* and landscape_pv_M4_48 (margin 0.008); the other schemes from tradeoff_mg* and main_M4_48seeds."""
    mgs = [m for m in ('0.004', '0.006', '0.010', '0.012', '0.014') if has('pass1', f'tradeoff_mg{m}.json')]
    if not mgs:
        return 'no trade-off data yet'
    runs = sorted([(float(m), J('pass1', f'tradeoff_mg{m}.json')['cells']) for m in mgs] + [(0.008, J('main', 'main_M4_48seeds.json')['cells'])],
                  key=lambda t: t[0])
    f, ax = S.fig()
    pt = lambda c, p: (100 * c[f'{p}|4|60|0']['flights_below'], max(c[f'{p}|4|60|0']['mean_hold'], 0.1))
    xy = np.array([pt(c, 'lyap0.3') for _, c in runs]); S.line(ax, xy[:, 0], xy[:, 1], 'lyap')
    m2 = [m for m in ('0.004', '0.006', '0.010', '0.012') if has('pass1', f'tradeoff2_mg{m}.json')]
    if m2 and has('pass1', 'landscape_pv_M4_48.json'):
        r2 = sorted([(float(m), J('pass1', f'tradeoff2_mg{m}.json')['cells']) for m in m2] + [(0.008, J('pass1', 'landscape_pv_M4_48.json')['cells'])],
                    key=lambda t: t[0])
        xy = np.array([pt(c, EXH) for _, c in r2]); S.line(ax, xy[:, 0], xy[:, 1], 'exh')
    else:
        xy = np.array([pt(c, 'lyapp3v0.03') for _, c in runs]); S.line(ax, xy[:, 0], xy[:, 1], 'exh')
    if has('pass1', 'sysF_M4_48seeds.json'):                  # the five policies of the final training protocol: one point each at the
        c8 = J('pass1', 'sysF_M4_48seeds.json')['cells']       # protocol margin 0.008, and their curves over the margin once tradeoff3_* exist
        m3 = [m for m in ('0.004', '0.006', '0.010', '0.012') if has('pass1', f'tradeoff3_mg{m}.json')]
        r3 = sorted([(float(m), J('pass1', f'tradeoff3_mg{m}.json')['cells']) for m in m3] + [(0.008, c8)], key=lambda t: t[0])
        xy = np.array([[pt(c, f'f{s}') for _, c in r3] for s in range(5)])
        if len(r3) > 1:
            for one in xy:
                ax.plot(one[:, 0], one[:, 1], color=S.STYLE['prop'][1], linewidth=0.5, alpha=0.6)
        k = [t[0] for t in r3].index(0.008)
        S.line(ax, xy[:, k, 0], xy[:, k, 1], 'prop', linestyle='none')
    else:
        xy = np.array([[pt(c, f's{s}') for _, c in runs] for s in range(5)])   # trained policies x margins x 2
        for one in xy:
            ax.plot(one[:, 0], one[:, 1], color=S.STYLE['prop'][1], linewidth=0.5, alpha=0.6)
        S.line(ax, xy[:, :, 0].mean(0), xy[:, :, 1].mean(0), 'prop')
    ax.axvline(5, color=S.INK, linestyle=(0, (4, 2)), linewidth=0.7)
    ax.annotate('allowed 5%', xy=(5, 0.97), xycoords=('data', 'axes fraction'), xytext=(3, 0), textcoords='offset points', ha='left', va='top', fontsize=7.5)
    ax.set_yscale('log'); ax.yaxis.set_major_formatter(plain); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Flights below the recall requirement (%)'); ax.set_ylabel('Mean hold per flight (s)')
    S.legend_top(ax, order=ORDER)
    return S.save(f, 'fig_tradeoff')


FIGS = dict(conv=fig_conv, conv2=fig_conv_violation, struct=fig_struct, sys=fig_sys, cap=fig_cap, qsweep=fig_qsweep, cdf=fig_cdf, behav=fig_behav, sem=fig_sem, land=fig_land, trade=fig_trade, tokens=fig_tokens)
if __name__ == '__main__':
    S.use()
    for n in (sys.argv[1:] or [k for k in FIGS if k not in ('tokens', 'conv2')]):
        print(n, '->', FIGS[n]())
