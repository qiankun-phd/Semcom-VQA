#!/usr/bin/env python3
"""Do the generated scripts issue exactly the recorded commands? Replays exp/*.sh against tools/stub.py (nothing is simulated) and compares
script, arguments and simulator environment of every command with what tools/build.py selected from the as-run record (.tmp/build.json).
usage: python tools/check_equivalence.py"""
import os, json, subprocess, glob, shutil, sys
REPRO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); T = os.path.join(REPRO, '.tmp'); LOG = os.path.join(T, 'calls_exp.jsonl'); OUT = os.path.join(T, 'out_stub')
if os.path.exists(LOG):
    os.remove(LOG)
shutil.rmtree(OUT, ignore_errors=True); shutil.rmtree(OUT + '_cache', ignore_errors=True)
json.dump(dict(os.environ), open(os.path.join(T, 'base_env.json'), 'w'))
PFX = ('BUB_', 'PPO_', 'SPPO_', 'OFF_', 'CAP_', 'STRAT_', 'EVAL_', 'VAL_', 'DG_', 'TD_', 'OMP_', 'MKL_')
env = dict(os.environ, PY=f'python3 {REPRO}/tools/stub.py', PROCS='7777', JOBS='8', OUT=OUT, CACHE=OUT + '_cache', NODEPS='1', STUB_BASE=os.path.join(T, 'base_env.json'),
           STUB_LOG=LOG, BUB_LEAK='must-not-reach-a-command')            # BUB_LEAK: the clean-environment rule of lib.sh is tested too
for f in sorted(f for f in glob.glob(os.path.join(REPRO, 'exp', '[0-9]*.sh')) if not os.path.basename(f).startswith('9')):      # 9x = checks, not recorded runs
    r = subprocess.run(['bash', f], env=dict(env, AS_RUN=os.path.basename(f)), capture_output=True, text=True, timeout=600)
    if r.returncode:
        print(os.path.basename(f), 'exit', r.returncode, r.stdout[-300:], r.stderr[-300:])
want = json.load(open(os.path.join(T, 'build.json')))['sigs']
got = {}
for l in open(LOG):
    c = json.loads(l); a = ['$PROCS' if x == '7777' else x for x in c['argv']]
    e = {k: ('$PROCS' if v == '7777' else v) for k, v in c['env'].items() if k.startswith(PFX) and k != 'CAP_SCHED_CACHE'}
    got[a[2]] = dict(argv=a, env=e)
bad = [n for n in want if got.get(n) != want[n]]; extra = sorted(set(got) - set(want))
print('recorded commands', len(want), '| issued by exp/*.sh', len(got), '| identical', len(want) - len(bad), '| different or missing', len(bad), '| extra', len(extra))
for n in bad[:8]:
    w, g = want[n], got.get(n)
    print('  ', n, '->', 'missing' if g is None else {k: (w['env'].get(k), g['env'].get(k)) for k in set(w['env']) | set(g['env']) if w['env'].get(k) != g['env'].get(k)} or (w['argv'], g['argv']))
sys.exit(1 if bad or extra else 0)
