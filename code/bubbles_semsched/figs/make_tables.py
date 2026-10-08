#!/usr/bin/env python3
"""Tables of the BUBBLES / structured-PPO results as LaTeX (IEEEtran style: caption above, \\hline rules) -> figs/out/tables.tex, and the
same numbers as plain text on stdout. Every number is read from the aggregated JSONs under scratch_bubbles/res/ (see make_figs.py for
the files); nothing is typed in by hand.

  T1  main comparison on the test traffic (seeds 501-548, 48 seeds), no temporal buffer: mean hold / flights below / conformance
  T2  capacity on the test traffic, without buffer and with a buffer of at most 30 s
  T3  robustness (development traffic 401-448): interference margin -3 / +3 dB, single take-off and landing site
  T4  sensitivity to the assumptions that are ours (development traffic; the strategic-layer ones on the test traffic)
  T5  computation per decision and number of parameters
  T6  number of token levels (rules) and the recall requirement on the network average instead of every flight
  T7  ablation: which of the two structure parameters has to be learned (test traffic)
  T8  development traffic, test traffic and both pooled at the operating point (per-flight data)"""
import sys, os, json
import numpy as np
sys.argv = sys.argv[:1]
import make_figs as F

J, has = F.J, F.has
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'tables.tex')
NAME = dict(prop='Proposed', exh='Grid search', lyap='Drift-plus-penalty', ppo='H-PPO', d3qn='D3QN', td3='TD3', fixm='Fixed level', hevc='HEVC inter')
ROWS = ('prop', 'exh', 'lyap', 'ppo', 'd3qn', 'td3', 'fixm')
tex, txt = [], []


def cellv(src, M, la, b=0):
    """src = [(cells, [policy names]), ...]: mean over all those policies of (hold s, flights below %, conformance), how many are feasible, how many"""
    xs = [c[f'{p}|{M}|{la:.0f}|{b:.0f}'] for c, pols in src for p in pols]
    return np.mean([x['mean_hold'] for x in xs]), 100 * np.mean([x['flights_below'] for x in xs]), np.mean([x['conformance'] for x in xs]), sum(x['feasible'] for x in xs), len(xs)


def schemes(M):
    """{key: [(cells, [policy names]), ...]} on the test traffic for one channel count; learners = every training run that was evaluated
    (proposed: five seeds on each of two machines; comparison learners: seed 0 and, from server 182, seeds 1-4)"""
    main = J('final', f'final48_M{M}_main.json')['cells']
    out = dict(prop=[(main, [f'f{s}' for s in range(5)])], exh=[(main, [F.EXH])], lyap=[(main, ['lyap0.3'])], fixm=[(main, ['fixed2'])])
    if has('final_182', f'final48_M{M}_prop182.json'):
        out['prop'].append((J('final_182', f'final48_M{M}_prop182.json')['cells'], [f'g{s}' for s in range(5)]))
    for key, tag, pol in F.OTHER:
        out[key] = [(J('final', f'final48_M{M}_{tag}.json')['cells'], [pol])]
        if has('final_182', f'final48_M{M}_{tag}_s1to4.json'):
            out[key].append((J('final_182', f'final48_M{M}_{tag}_s1to4.json')['cells'], [f'{pol}{s}' for s in range(1, 5)]))
    return out


def table(caption, label, head, rows, note=None):
    tex.append('\\begin{table}[t]\n\\caption{' + caption + '}\n\\label{' + label + '}\n\\centering\n\\begin{tabular}{' + 'l' + 'c' * (len(head) - 1) + '}\n\\hline')
    tex.append(' & '.join(head) + ' \\\\\n\\hline')
    for r in rows:
        tex.append(' & '.join(r) + ' \\\\')
    tex.append('\\hline\n\\end{tabular}' + ('\n\\\\[2pt]{\\footnotesize ' + note + '}' if note else '') + '\n\\end{table}\n')
    txt.append('\n== ' + caption); w = [max(len(str(x[i])) for x in [head] + rows) for i in range(len(head))]
    for r in [head] + rows:
        txt.append('  ' + '  '.join(str(x).ljust(w[i]) for i, x in enumerate(r)))


# ---------------------------------------------------------------- T1 main comparison, test traffic, no buffer
pts = ((3, 35), (3, 40), (4, 55), (4, 60), (5, 60), (5, 65))
data = {M: schemes(M) for M in (3, 4, 5)}
mark = lambda nf, n: '' if nf == n else '$^{\\dagger}$' if nf == 0 else '$^{\\ddagger}$'
rows = []
for key in ROWS:
    r = [NAME[key]]
    for M, la in pts:
        h, fb, cf, nf, n = cellv(data[M][key], M, la); r.append(f'{h:.2f} / {fb:.1f}' + mark(nf, n))
    rows.append(r)
nrun = {key: sorted({sum(len(p) for _, p in data[M][key]) for M in (3, 4, 5)}) for key in ('prop', 'ppo', 'd3qn', 'td3')}
table('Mean hold per flight (s) / flights below the recall requirement (\\%) on the test traffic, no temporal buffer', 'tab:main',
      ['Scheme'] + [f'$M={M}$, {la}/h' for M, la in pts], rows,
      '$^{\\dagger}$ misses a requirement; $^{\\ddagger}$ some of the training runs miss one. Mean over the training runs: proposed ' + '/'.join(map(str, nrun['prop'])) +
      ' (five seeds on each of two machines), comparison learners ' + '/'.join(map(str, sorted(set(sum((nrun[k] for k in ('ppo', 'd3qn', 'td3')), []))))) + ' seeds.')

# ---------------------------------------------------------------- T2 capacity
low0, low30 = {}, {}
c0, c30 = F.cap_table((0,), low0), F.cap_table((0, 30), low30)
def fmt(c, low, k, M):
    v = c[k][M]; lt = f'$<${low[(k, M)]:.0f}'; one = lambda x: lt if x == 0 else f'{x:.0f}'
    return one(v[0]) if len(set(v)) == 1 else f'{one(np.median(v))} ({one(min(v))}--{one(max(v))})'
rows = [[NAME[k]] + [fmt(c0, low0, k, M) + ' / ' + fmt(c30, low30, k, M) if M in c0.get(k, {}) else '--' for M in (3, 4, 5, 6)] for k in ROWS]
table('Capacity (flights/h) on the test traffic: without buffer / with a temporal buffer of at most 30 s', 'tab:capacity',
      ['Scheme', '$M=3$', '$M=4$', '$M=5$', '$M=6$'], rows,
      'Learners: median (range) over the training runs. $<x$: infeasible at $x$ flights/h, the lowest rate tested for that scheme; highest rate tested: 40 ($M=3$), 65 ($M\\geq 4$).')

# ---------------------------------------------------------------- T3 robustness (development traffic)
conds = (('Nominal', None), ('Interference $-3$ dB', 'iot17'), ('Interference $+3$ dB', 'iot23'), ('Single port', 'port1'))
rows = []
for key in ROWS:
    r = [NAME[key]]
    for lab, tag in conds:
        if tag is None:
            main = J('main', 'main_M4_48seeds.json')['cells']; exh = J('pass1', 'sysF_M4_48seeds.json')['cells']; base = J('pass1', 'base48_rules_M4.json')['cells']
            src = dict(prop=(exh, [f'f{s}' for s in range(5)]), exh=(exh, [F.EXH]), lyap=(main, ['lyap0.3']), fixm=(base, ['fixed2']),
                       ppo=(J('pass1', 'base48_B_M4.json')['cells'], ['ppo']), d3qn=(J('pass1', 'base48_d3qn_M4.json')['cells'], ['d3qn']), td3=(J('pass1', 'base48_td3_M4.json')['cells'], ['td3']))
        else:
            main = J('extra', f'robust_{tag}_main.json')['cells']
            src = dict(prop=(main, [f'f{s}' for s in range(5)]), exh=(main, [F.EXH]), lyap=(main, ['lyap0.3']), fixm=(main, ['fixed2']),
                       ppo=(J('extra', f'robust_{tag}_B.json')['cells'], ['ppo']), d3qn=(J('extra', f'robust_{tag}_d3qn.json')['cells'], ['d3qn']),
                       td3=(J('extra', f'robust_{tag}_td3.json')['cells'], ['td3']))
        h, fb, cf, nf, n = cellv([src[key]], 4, 55)
        r.append(f'{h:.2f} / {fb:.1f}' + ('' if nf == n else '$^{\\dagger}$' if nf == 0 else '$^{\\ddagger}$'))
    rows.append(r)
table('Robustness without retraining ($M=4$, 55 flights/h, no buffer, development traffic): mean hold (s) / flights below (\\%)', 'tab:robust',
      ['Scheme'] + [c[0] for c in conds], rows, '$^{\\dagger}$ misses a requirement; $^{\\ddagger}$ some of the trained policies miss one.')

# ---------------------------------------------------------------- T4 assumptions
sens = (('Main setting', 'pass1', 'sysF_M4_48seeds.json'), ('Grace time 30 s', 'extra', 'sensD30_main.json'), ('Grace time 120 s', 'extra', 'sensD120_main.json'),
        ('Longest hold 60 s', 'extra', 'sensH60_main.json'), ('Longest hold 240 s', 'extra', 'sensH240_main.json'), ('Rician $K$ 5 dB', 'extra', 'sensK5_main.json'),
        ('Rician $K$ 15 dB', 'extra', 'sensK15_main.json'), ('Ground delay $\\leq$ 300 s$^{*}$', 'extra', 'sensG300_main.json'),
        ('Ground delay $\\leq$ 1200 s$^{*}$', 'extra', 'sensG1200_main.json'), ('Port zone 150 m$^{*}$', 'extra', 'sensR150_main.json'), ('Port zone 600 m$^{*}$', 'extra', 'sensR600_main.json'))
rows = []
for lab, d_, fn in sens:
    if not has(d_, fn):
        continue
    d = J(d_, fn); c = d['cells']; r = [lab]
    for pols in ([f'f{s}' for s in range(5)], [F.EXH], ['lyap0.3']):
        if f'{pols[0]}|4|60|0' not in c:
            c2 = J('main', 'main_M4_48seeds.json')['cells']; h, fb, cf, nf, n = cellv([(c2, pols)], 4, 60)
            cp = [F.cap_of(J('main', 'main_M4_48seeds.json'), p, 4) for p in pols]
        else:
            h, fb, cf, nf, n = cellv([(c, pols)], 4, 60); cp = [F.cap_of(d, p, 4) for p in pols]
        r.append(f'{h:.2f} / {fb:.1f} / ' + (f'{cp[0]:.0f}' if len(set(cp)) == 1 else f'{min(cp):.0f}--{max(cp):.0f}'))
    rows.append(r)
table('Sensitivity to the assumptions ($M=4$): mean hold (s) / flights below (\\%) at 60 flights/h without buffer / capacity (flights/h, buffer $\\leq$ 30 s)', 'tab:sens',
      ['Assumption', 'Proposed', 'Grid search', 'Drift-plus-penalty'], rows, 'Development traffic; $^{*}$ test traffic. Rates tested: 55 and 60 flights/h (main setting: 55--65).')

# ---------------------------------------------------------------- T5 computation
rows = []
for key, fn in (('prop', 'prop'), ('exh', 'lyapp3v0.1'), ('lyap', 'lyap0.3'), ('ppo', 'B'), ('d3qn', 'd3qn'), ('td3', 'td3'), ('fixm', 'fixed2')):
    if has('extra', f'time_{fn}.json'):
        t = J('extra', f'time_{fn}.json'); rows.append([NAME[key], f"{t['parameters']}", f"{t['policy_ms_per_slot']:.2f}", f"{t['policy_us_per_candidate']:.0f}", f"{t['matching_ms_per_slot']:.2f}"])
table('Size of the scheduler and computation per 1-s slot (CPU time, one thread)', 'tab:time',
      ['Scheme', 'Parameters', 'Policy (ms/slot)', 'Policy ($\\mu$s/UAV)', 'Matching (ms/slot)'], rows)

# ---------------------------------------------------------------- T6 token levels, constraint type
rows = []
for lab, fn in (('One token level (0.48)', 'lev1_rules.json'), ('Two token levels (0.44, 0.48)', 'lev2_rules.json'), ('Four token levels', 'lev4_rules.json')):
    d = J('extra', fn); c = d['cells']; r = [lab]
    for p in ('lyapp3v0.1', 'lyapp3v0.05'):
        h, fb, cf, nf, n = cellv([(c, [p])], 4, 60); r.append(f'{h:.2f} / {fb:.1f} / {F.cap_of(d, p, 4):.0f}')
    rows.append(r)
table('Number of token levels of the semantic encoder ($M=4$, development traffic): mean hold (s) / flights below (\\%) at 60 flights/h without buffer / capacity', 'tab:levels',
      ['Levels', 'Rule $(p,V)=(3,0.1)$', 'Rule $(3,0.05)$'], rows)
# token levels on the TEST traffic: the rule of the grid search and the learner retrained for that set of levels (final checkpoints)
if has('final', 'final48_M4_lev1.json'):
    def rng(v, lo):
        one = lambda x: f'$<${lo:.0f}' if x == 0 else f'{x:.0f}'
        return one(v[0]) if len(set(v)) == 1 else f'{one(min(v))}--{one(max(v))}'
    sets = [('Four token levels', {M: [(J('final', f'final48_M{M}_main.json'), [f'f{s}' for s in range(5)])] + ([(J('final_182', f'final48_M{M}_prop182.json'), [f'g{s}' for s in range(5)])]
                                       if has('final_182', f'final48_M{M}_prop182.json') else []) for M in (3, 4, 5)}, {M: J('final', f'final48_M{M}_main.json') for M in (3, 4, 5)})]
    for lab, tag, names in (('Two token levels (0.44, 0.48)', 'lev2', ['t0', 't1', 't2']), ('One token level (0.48)', 'lev1', ['l0', 'l1', 'l2'])):
        Ms = [M for M in (3, 4, 5) if has('final', f'final48_M{M}_{tag}.json')]
        if Ms:
            sets.append((lab, {M: [(J('final', f'final48_M{M}_{tag}.json'), names)] for M in Ms}, {M: J('final', f'final48_M{M}_{tag}.json') for M in Ms}))
    BEST = {'Four token levels': ('final48_M4_lev4best.json', [f'fb{s}' for s in range(5)]), 'One token level (0.48)': ('final48_M4_lev1best.json', ['b0', 'b1', 'b2']),
            'Two token levels (0.44, 0.48)': ('final48_M4_lev2.json', ['tb0', 'tb1', 'tb2'])}
    rows = []
    for lab, lrn, rule in sets:
        r = [lab]
        for who in ('rule', 'learned'):
            src = [(rule[4], [F.EXH])] if who == 'rule' else lrn[4]
            h, fb, cf, nf, n = cellv([(d['cells'], pols) for d, pols in src], 4, 60); r.append(f'{h:.2f} / {fb:.1f}' + mark(nf, n))
            r.append(', '.join(rng([F.cap_of(d, p_, M, (0,)) for d, pols in ([(rule[M], [F.EXH])] if who == 'rule' else lrn[M]) for p_ in pols], min(rule[M]['lams']))
                               if M in rule else '--' for M in (3, 4, 5)))
        fn, names = BEST[lab]
        if has('final', fn):                                      # the same runs with the checkpoint that was best on the validation traffic
            d = J('final', fn); h, fb, cf, nf, n = cellv([(d['cells'], names)], 4, 60)
            r.append(f'{h:.2f} / {fb:.1f}' + mark(nf, n) + ' (' + rng([F.cap_of(d, p_, 4, (0,)) for p_ in names], min(d['lams'])) + ')')
        else:
            r.append('--')
        rows.append(r)
    table('Number of token levels on the test traffic: mean hold (s) / flights below (\\%) at $M=4$, 60 flights/h, and capacity without buffer at $M=3,4,5$', 'tab:levels-test',
          ['Levels', 'Rule $(3,0.1)$', 'Capacity', 'Learned, final', 'Capacity', 'Learned, best on validation ($M=4$ capacity)'], rows,
          '$^{\\ddagger}$ some of the training runs miss a requirement. Learned: retrained for that set of levels with the unchanged protocol (ten runs with four levels, three otherwise; best on validation: five and three); capacity as range over the runs; $<x$: infeasible at $x$, the lowest rate tested.')
if has('extra', 'net_rules.json'):
    d = J('extra', 'net_rules.json'); c = d['cells']; rows = []
    for lab, p in (('Network average, rule $(3,10)$', 'lyapp3v10'), ('Network average, rule $(3,30)$', 'lyapp3v30')):
        x = c[f'{p}|4|60|0']; rows.append([lab, f"{x['mean_hold']:.2f}", f"{x['recall']:.4f}", f"{100 * x['flights_below']:.1f}", f"{d['capacity'][p + '|4']['rate']:.0f}"])
    e = J('pass1', 'sysF_M4_48seeds.json'); x = e['cells'][f'{F.EXH}|4|60|0']
    rows.append(['Every flight, rule $(3,0.1)$', f"{x['mean_hold']:.2f}", f"{x['recall']:.4f}", f"{100 * x['flights_below']:.1f}", f"{F.cap_of(e, F.EXH, 4):.0f}"])
    table('Recall requirement on the network average or on every flight ($M=4$, 60 flights/h, no buffer, development traffic)', 'tab:constraint',
          ['Requirement and rule', 'Hold (s)', 'Mean recall', 'Flights below (\\%)', 'Capacity'], rows)

# ---------------------------------------------------------------- T7 which structure parameter has to be learned
if has('final', 'ablate48_M4.json'):
    ab = J('final', 'ablate48_M4.json'); m4 = J('final', 'final48_M4_main.json')
    rows = []
    for lab, src, d in (('Neither (drift-plus-penalty, $p=1$, $V=0.3$)', [(m4['cells'], ['lyap0.3'])], [(m4, 'lyap0.3')]),
                        ('$V$ only ($p=1$ fixed)', [(ab['cells'], [f'pfix{s}' for s in range(3)])], [(ab, f'pfix{s}') for s in range(3)]),
                        ('$p$ only ($V=0.3$ fixed)', [(ab['cells'], [f'vfix{s}' for s in range(3)])], [(ab, f'vfix{s}') for s in range(3)]),
                        ('$p$ and $V$ (proposed)', data[4]['prop'], [(m4, f'f{s}') for s in range(5)])):
        r = [lab]
        for la in (55, 60):
            h, fb, cf, nf, n = cellv(src, 4, la); r.append(f'{h:.2f} / {fb:.1f} / {100 * (1 - cf):.1f}' + mark(nf, n))
        cp = [[F.cap_of(dd, p, 4, bufs) for dd, p in d] for bufs in ((0,), (0, 30))]
        one = lambda x: f'$<${min(ab["lams"]):.0f}' if x == 0 else f'{x:.0f}'
        r.append(' / '.join(one(c[0]) if len(set(c)) == 1 else f'{one(min(c))}--{one(max(c))}' for c in cp)); rows.append(r)
    table('Which structure parameter has to be learned ($M=4$, test traffic, no buffer): mean hold (s) / flights below (\\%) / non-conforming flights (\\%)', 'tab:ablation',
          ['Learned', '55 flights/h', '60 flights/h', 'Capacity ($b=0$ / $b\\leq 30$ s)'], rows,
          '$^{\\dagger}$ misses a requirement; $^{\\ddagger}$ some of the training runs miss one. Three training seeds per ablation, ten training runs of the proposed method; rates tested 55--65 (the two rules: 40--65).')

# ---------------------------------------------------------------- T8 development traffic, test traffic and both pooled
# exact pooling: the aggregated cells (share of flights below, mean hold) weighted with the number of flights, which is the length of the
# per-flight record (the per-flight recall itself is stored rounded to 4 decimals, so it is not re-thresholded here)
if has('extra', 'raw48_main_raw.json') and has('final', 'final48_M4_main_raw.json'):
    def part(dr, fn, pols, la=60):
        c = J(dr, fn + '.json')['cells']; raw = J(dr, fn + '_raw.json'); k = [f'{p}|4|{la}|0' for p in pols]
        n = np.array([len(raw[x]['frec']) for x in k], float)
        return n, np.array([c[x]['flights_below'] for x in k]), np.array([c[x]['mean_hold'] for x in k])
    src = (('prop', 'main', [f'f{s}' for s in range(5)]), ('exh', 'main', [F.EXH]), ('lyap', 'main', ['lyap0.3']), ('ppo', 'B', ['ppo']), ('d3qn', 'd3qn', ['d3qn']),
           ('td3', 'td3', ['td3']), ('fixm', 'main', ['fixed2']))
    rows = []
    for key, tag, pols in src:
        dv = part('extra', f'raw48_{tag}', pols); te = part('final', f'final48_M4_{tag}', pols); both = tuple(np.concatenate([a_, b_]) for a_, b_ in zip(dv, te))
        rows.append([NAME[key]] + [f'{100 * (n * fb).sum() / n.sum():.1f} / {(n * ho).sum() / n.sum():.2f}' for n, fb, ho in (dv, te, both)])
    table('Development traffic, test traffic and both pooled ($M=4$, 60 flights/h, no buffer): flights below the recall requirement (\\%) / mean hold (s)', 'tab:devtest',
          ['Scheme', 'Development (48 seeds)', 'Test (48 seeds)', 'Pooled (96 seeds)'], rows,
          'All flights of the 48 (96) traffic seeds; proposed: the flights of the five training runs of one machine; comparison learners: training seed 0. At most 5\\% of the flights may be below.')

open(OUT, 'w').write('% generated by make_tables.py - do not edit by hand\n' + '\n'.join(tex))
print('\n'.join(txt)); print('\nwritten', OUT)
import check_feasible                                             # the paper states feasibility without the network-average recall condition:
print('\nfeasibility check on the files behind the tables')       # it must never decide alone in the test-traffic files
t = check_feasible.report(['final', 'final_182', 'qsweep'], verbose=True)
n_, m1, w1, m2, w2 = check_feasible.margin(['final', 'final_182', 'qsweep'])
print(f'feasible cells: {n_}; smallest network-average recall minus requirement: {m1:+.4f} {w1} (delivered evidence), {m2:+.4f} {w2} (delivered + lost)')
if t[1] or t[3]:
    print('*** WARNING: the network-average recall condition decides in', t[1], '+', t[3], 'cells - see above ***')
