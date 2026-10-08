#!/usr/bin/env python3
"""Builds the reproduction scripts in repro/exp/ and repro/MANIFEST.md from the record of what was actually run.

  .tmp/calls.jsonl   every command of every as-run chain script (as_run/run_*.sh), captured by replaying the scripts against a stand-in
                     interpreter (tools/replay_as_run.py): script, arguments, exported environment
  .tmp/opened.json   the result files the current figure and table scripts open (tools/trace_figs.py)
  ../res/*/*.json    the stored results: a recorded command is accepted for an output only if its settings agree with the metadata stored
                     in that output (policies, channel counts, rates, buffers, traffic seeds, recall requirement / training seed)
Nothing here runs a simulation."""
import json, os, re, glob, collections, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPRO = os.path.dirname(HERE); T = os.path.join(REPRO, '.tmp'); RES = os.path.join(REPRO, '..', 'res')
calls = [json.loads(l) for l in open(os.path.join(T, 'calls.jsonl'))]
for i, c in enumerate(calls):
    c['i'] = i
opened = json.load(open(os.path.join(T, 'opened.json')))
TRAIN = ('sppo_hold.py', 'hppo_hold.py', 'offpol_hold.py')
SERVER = lambda run: ('server 182' if run.startswith('run_182') else 'AutoDL' if run.startswith('run_new') or run in ('run_final.sh', 'run_extra.sh', 'run_robust.sh')
                      else 'rented 3090')


def out_of(c):
    a = c['argv']
    return a[2] if len(a) > 2 and a[2].endswith('.json') else None


by_out = collections.defaultdict(list)
for c in calls:
    if out_of(c):
        by_out[out_of(c)].append(c)
stored = {os.path.basename(f): f for f in glob.glob(os.path.join(RES, '*', '*.json'))}


def agrees(c, name):
    """does the recorded command agree with the metadata stored in the result file?"""
    if name not in stored:
        return None
    try:
        d = json.load(open(stored[name]))
    except Exception:
        return None
    e, a = c['env'], c['argv']
    fl = lambda s: [float(x) for x in s.split(',') if x != '']
    if a[0] == 'cap_strat.py' and 'cells' in d:
        pols = [p for p in e.get('BUB_POLS', '').split(',') if p] + [x.split('=')[0] for x in e.get('CAP_LEARNED', '').split(',') if '=' in x]
        ok = (d.get('pols') == pols if 'BUB_POLS' in e else set(pols) <= set(d.get('pols', []))) and [float(x) for x in d['Ms']] == fl(e['BUB_MS']) \
            and d['lams'] == fl(e['BUB_LAMS']) and d['bufs'] == fl(e['STRAT_BUFS']) and d['seeds'][0] == int(e.get('CAP_SEED0', 401)) \
            and len(d['seeds']) == int(e.get('CAP_SEEDS', 12)) and abs(d['qbar'] - float(e['BUB_QBAR'])) < 1e-12
        return bool(ok)
    if a[0] in TRAIN and 'seed' in d:
        return d['seed'] == int(e.get('PPO_SEED', 0)) and abs(d.get('qbar', 0.44) - float(e['BUB_QBAR'])) < 1e-12 and \
            ('beta' not in d or 'BUB_BETA' not in e or abs(d['beta'] - float(e['BUB_BETA'])) < 1e-9)
    return None


def pick(name):
    cs = by_out.get(name, [])
    good = [c for c in cs if agrees(c, name)]
    unk = [c for c in cs if agrees(c, name) is None]
    return (good[-1], 'checked') if good else (unk[-1], 'unchecked') if unk else (None, 'mismatch' if cs else 'no record')


def ckpts(c):
    v = [x.split('=')[1] for x in c['env'].get('CAP_LEARNED', '').split(',') if '=' in x]
    v += [c['env'][k] for k in ('TD_CKPT', 'DG_CKPT', 'SPPO_INIT', 'PPO_INIT') if c['env'].get(k, '').endswith('.pt')]
    return v


stem = lambda ck: re.sub(r'_(final|best|it\d+)\.pt$', '', ck)
# manual entries: outputs whose command was typed by hand in an earlier session; reconstructed from the configuration stored in the output
main_env = {'BUB_IOT_DB': '20.1', 'BUB_NTOK_CV': '0', 'BUB_QBAR': '0.44', 'BUB_ZMODE': 'uav', 'BUB_ALTS': '39,69.5,100', 'BUB_PORTS': '4', 'BUB_QMARGIN': '0.008',
            'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
manual = {}
ref = next((c for c in by_out.get('mrl5k_B_M4_s1.json', [])), None)
if ref:
    manual['mrl5k_B_M4_s0.json'] = dict(run='(by hand; rebuilt from the command of seeds 1-4, which states it is the same)', i=10 ** 6,
                                        argv=[x.replace('_s1.json', '_s0.json') for x in ref['argv']], env=dict(ref['env'], PPO_SEED='0'))
if 'main_M4_48seeds.json' in stored:
    d = json.load(open(stored['main_M4_48seeds.json']))
    manual['main_M4_48seeds.json'] = dict(run='(by hand; rebuilt from the metadata stored in the result)', i=10 ** 6 + 1,
        argv=['cap_strat.py', 'sim_inputs.json', 'main_M4_48seeds.json', '0.44', 'cap', '12'],
        env=dict(main_env, BUB_MS='4', BUB_LAMS=','.join(f'{x:g}' for x in d['lams']), STRAT_BUFS=','.join(f'{x:g}' for x in d['bufs']), BUB_POLS=','.join(d['pols']),
                 CAP_SEEDS=str(len(d['seeds'])), CAP_SCHED_CACHE='x'))
sel, status = {}, {}
todo = sorted({re.sub(r'_raw\.json$', '.json', os.path.basename(p)) for p in opened['figs'] + opened['tables']})
while todo:
    name = todo.pop()
    if name in sel or name in status:
        continue
    c, st = pick(name)
    if c is None and name in manual:
        c, st = manual[name], 'rebuilt'
    status[name] = st
    if c is None:
        continue
    sel[name] = c
    todo += [stem(k) + '.json' for k in ckpts(c)]
GROUPS = (('10_train_proposed', r'^sppoX_fast5?_s\d'), ('11_train_baselines', r'^mrl5k?_'), ('12_train_variants', r'^(sppo|mrlQ)'), ('20_validation_refs', r'^val_refs'),
          ('21_grid_search', r'^landscape_pv'), ('31_test_traffic', r'^(final48_M\w+_(main|B|d3qn|td3|C|rules|prop182)|low48|fresh48)'),
          ('32_ablation_levels', r'^(ablate48|final48_M\d_lev|lev\d_rules|net_rules)'), ('33_robustness_sensitivity', r'^(robust_|sens)'),
          ('34_recall_requirement', r'^(qbar|qsweep)'), ('40_behaviour_timing', r'^(diag_|time_)'), ('30_dev_traffic', r'.'))
group = lambda name: next(g for g, p in GROUPS if re.search(p, name))
PREFIX = ('BUB_', 'PPO_', 'SPPO_', 'OFF_', 'CAP_', 'STRAT_', 'EVAL_', 'VAL_', 'DG_', 'TD_', 'OMP_', 'MKL_')


def sig(c):
    """what is run, with the two things that do not change a result normalised: number of evaluation processes, cache directory"""
    a = list(c['argv']); e = {k: v for k, v in c['env'].items() if k.startswith(PREFIX) and k != 'CAP_SCHED_CACHE'}
    if a[0] in ('cap_strat.py', 'val_refs.py'):
        a[5] = '$PROCS'
    if 'DG_PROCS' in e:
        e['DG_PROCS'] = '$PROCS'
    return a, e


def q(v):
    return v if re.fullmatch(r'[A-Za-z0-9_.,=+\-/:$]*', v) and v != '' else "'" + v + "'"


by_group = collections.defaultdict(list)
for name, c in sel.items():
    by_group[group(name)].append((c['i'], name, c))
TITLE = {'10_train_proposed': 'the proposed learner, final protocol, two sets of five training seeds: sppoX_fast5_s* (2 rollout processes; run on the rented 3090) and sppoX_fast_s* (4 rollout processes; run on server 182) - same seeds and settings, different random streams',
         '11_train_baselines': 'the comparison learners (H-PPO = B, penalty PPO = C, D3QN, TD3): five training seeds at M = 4, seed 0 trained for M = 3 and 5',
         '12_train_variants': 'retrained variants of the learners: structure ablation, one / two token levels, other recall requirements, earlier protocols that a figure still reads',
         '20_validation_refs': 'the rules on the validation traffic (reference lines of the learning curves)',
         '21_grid_search': 'grid search over the two structure parameters (p, V) at M = 4, 60 flights/h',
         '30_dev_traffic': 'system level on the development traffic (seeds 401-448): grids, trade-off, per-M baselines, HEVC',
         '31_test_traffic': 'system level on the test traffic (seeds 501-548): the numbers of the paper',
         '32_ablation_levels': 'ablations: structure parameters, number of token levels, constraint on the network average',
         '33_robustness_sensitivity': 'robustness without retraining and sensitivity to the modelling assumptions',
         '34_recall_requirement': 'capacity against the recall requirement (trained once and used as is; retrained per requirement)',
         '40_behaviour_timing': 'behaviour diagnostics (when are tokens sent) and computation time per decision'}
for g in sorted(by_group):
    # deterministic order (the replay records commands that were started side by side in whatever order they finish): a command that
    # reads a checkpoint written in the same script comes after it, otherwise by output name
    names = {n for _, n, _ in by_group[g]}; cmd = {n: c for _, n, c in by_group[g]}
    def depth(n, seen=()):
        deps = [stem(k) + '.json' for k in ckpts(cmd[n]) if stem(k) + '.json' in names and stem(k) + '.json' not in seen + (n,)]
        return 1 + max(depth(d_, seen + (n,)) for d_ in deps) if deps else 0
    items = sorted(((depth(n), n, c) for _, n, c in by_group[g]), key=lambda t: (t[0], t[1])); sigs = [sig(c) for _, _, c in items]
    common = dict(set.intersection(*[set(e.items()) for _, e in sigs])) if len(sigs) > 1 else {}
    L = ['#!/bin/bash', f'# {TITLE[g]}', '# GENERATED by tools/build.py from the as-run chain scripts (as_run/) - edit the generator, not this file.',
         '# usage: bash exp/' + g + '.sh        (settings: env.sh; output directory: $OUT; SMOKE=1 for a quick execution test)',
         'source "$(dirname "$0")/../lib.sh"', '']
    if common:
        L += ['COMMON="' + ' '.join(f'{k}={v}' for k, v in sorted(common.items())) + '"', '']
    produced, pending, level = set(), False, 0
    for (dp, name, c), (a, e) in zip(items, sigs):
        train = a[0] in TRAIN
        if dp != level and pending:                             # the next level reads checkpoints of the one before
            L.append('waitall'); pending = False
        level = dp
        rest = ' '.join(f'{k}={q(v)}' for k, v in sorted(e.items()) if common.get(k) != v)
        L.append(f"{'runbg' if train else 'run  '} {name} {'$COMMON ' if common else ''}{rest} -- {' '.join(q(x) for x in a)}")
        produced.add(name); pending = pending or train
    if pending:
        L.append('waitall')
    L.append('report')
    p = os.path.join(REPRO, 'exp', g + '.sh'); open(p, 'w').write('\n'.join(L) + '\n'); os.chmod(p, 0o755)
# ---- manifest
use = collections.defaultdict(set)
for k in ('figs', 'tables'):
    for p in opened[k]:
        use[re.sub(r'_raw\.json$', '.json', os.path.basename(p))].add(k)
per_fig = json.load(open(os.path.join(T, 'per_fig.json'))) if os.path.exists(os.path.join(T, 'per_fig.json')) else {}
M = ['# 结果文件 → 生成命令 → 图表（由 tools/build.py 生成，勿手改）', '',
     '“核对”一栏：checked = 记录的命令与结果文件里存的元数据（策略、信道数、速率、缓冲、流量种子、召回要求 / 训练种子）一致；unchecked = 结果文件不含可比对的元数据；'
     'rebuilt = 当时是手敲的命令，按结果文件里的配置重建。', '',
     '| 结果文件 | res/ 下目录 | 复现脚本 | 当时运行的脚本 | 服务器 | 核对 | 用于 |', '|---|---|---|---|---|---|---|']
for name in sorted(sel, key=lambda n: (group(n), n)):
    c = sel[name]; folder = os.path.basename(os.path.dirname(stored[name])) if name in stored else '(未拉回本地)'
    figs = sorted(f for f, v in per_fig.items() if any(os.path.basename(x) in (name, name[:-5] + '_raw.json') for x in v))
    u = ', '.join(figs + (['表'] if 'tables' in use.get(name, ()) else [])) or '训练产物（检查点被评估用）'
    M.append(f"| `{name}` | {folder} | `exp/{group(name)}.sh` | `{c['run']}` | {SERVER(c['run']) if c['run'].startswith('run_') else '—'} | {status[name]} | {u} |")
miss = sorted(n for n, s in status.items() if n not in sel)
M += ['', '## 图表读取、但没有对应仿真命令的文件', ''] + [f'- `{n}`（{status[n]}）' for n in miss]
open(os.path.join(REPRO, 'MANIFEST.md'), 'w').write('\n'.join(M) + '\n')
json.dump({n: os.path.basename(os.path.dirname(stored[n])) for n in sel if n in stored}, open(os.path.join(REPRO, 'ref', 'folders.json'), 'w'), indent=0)
# ---- a full-fidelity check that needs minutes, not days: three cells of the main table with the shipped checkpoints, compared with the stored values
CHECK = (('final48_M4_main.json', 'check_M4_main.json', dict(BUB_POLS='lyap0.3,lyapp3v0.1', CAP_LEARNED='f0=sppoX_fast5_s0_final.pt'), ['lyap0.3', 'lyapp3v0.1', 'f0']),
         ('final48_M4_B.json', 'check_M4_B.json', {}, ['ppo']))
L = ['#!/bin/bash', '# Reproduction check at full fidelity (48 test-traffic seeds), a few minutes: four policies at M = 4, 60 flights/h, no buffer - a rule of each',
     '# family, a trained policy of the proposed learner and the H-PPO baseline, with the shipped checkpoints - compared with the stored values',
     '# (ref/expected.json, taken from the result files the tables are built from). GENERATED by tools/build.py.',
     'source "$(dirname "$0")/../lib.sh"', '']
exp_ref = {}
for src_name, out_name, over, pols in CHECK:
    if src_name not in sel:
        continue
    a_, e_ = sig(sel[src_name]); e_ = dict(e_, BUB_LAMS='60', STRAT_BUFS='0', **over); e_.pop('CAP_RAW', None)
    a_ = [out_name if x == src_name else x for x in a_]
    L.append(f"FORCE=1 run {out_name} {' '.join(f'{k}={q(v)}' for k, v in sorted(e_.items()))} -- {' '.join(q(x) for x in a_)}")
    d = json.load(open(stored[src_name]))['cells']
    exp_ref[out_name] = {f'{p_}|4|60|0': {k: d[f'{p_}|4|60|0'][k] for k in ('mean_hold', 'flights_below', 'conformance', 'recall', 'lost', 'n_air', 'feasible')} for p_ in pols}
L += ['$PY "$REPRO/tools/compare.py" "$OUT" "$REPRO/ref/expected.json"', 'report']
open(os.path.join(REPRO, 'exp', '90_check_reproduction.sh'), 'w').write('\n'.join(L) + '\n'); os.chmod(os.path.join(REPRO, 'exp', '90_check_reproduction.sh'), 0o755)
os.makedirs(os.path.join(REPRO, 'ref'), exist_ok=True); json.dump(exp_ref, open(os.path.join(REPRO, 'ref', 'expected.json'), 'w'), indent=1)
need = sorted({k for c in sel.values() for k in ckpts(c)})
json.dump(dict(ckpts=need, outputs=sorted(sel), status=status, folders={n: os.path.basename(os.path.dirname(stored[n])) for n in sel if n in stored}, groups={g: [n for _, n, _ in sorted(v)] for g, v in by_group.items()},
               sigs={n: dict(zip(('argv', 'env'), sig(c))) for n, c in sel.items()}), open(os.path.join(T, 'build.json'), 'w'))
print('outputs covered', len(sel), '| checkpoints needed', len(need)); print('status', dict(collections.Counter(status.values())))
print('not reproducible from a recorded command:', miss)
for g in sorted(by_group):
    print(f'  exp/{g}.sh  {len(by_group[g])} commands')
