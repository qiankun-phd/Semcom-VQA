#!/usr/bin/env python3
"""Which result files do the figure and table scripts open? Runs ../figs/make_figs.py (every figure) and make_tables.py with open()
watched and writes .tmp/opened.json (figures / tables) and .tmp/per_fig.json (per figure). Reads aggregated JSONs only.
usage: python tools/trace_figs.py"""
import builtins, os, sys, json, runpy, io, contextlib
REPRO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); FIGS = os.path.join(REPRO, '..', 'figs'); R = os.path.realpath(os.path.join(REPRO, '..', 'res'))
os.chdir(FIGS); sys.path.insert(0, FIGS)
opened = set(); _open = builtins.open


def spy(f, *a, **k):
    try:
        p = os.path.realpath(f)
        if p.startswith(R) and p.endswith('.json') and (not a or 'r' in a[0]):
            opened.add(os.path.relpath(p, R))
    except Exception:
        pass
    return _open(f, *a, **k)


builtins.open = spy
per = {}
with contextlib.redirect_stdout(io.StringIO()):
    import make_figs as F, ieee_style as S, check_feasible
    S.use()
    for n in F.FIGS:
        before = set(opened)
        try:
            F.FIGS[n]()
        except Exception as e:
            print('figure', n, 'failed:', e, file=sys.stderr)
        per[n] = sorted(opened - before) if n != 'conv2' else sorted(per.get('conv', []))
    figs = set(opened)
    check_feasible.report = lambda *a, **k: [0, 0, 0, 0]; check_feasible.margin = lambda *a, **k: (0, 0, None, 0, None)
    sys.argv = ['make_tables.py']; runpy.run_path('make_tables.py', run_name='__main__')
builtins.open = _open
os.makedirs(os.path.join(REPRO, '.tmp'), exist_ok=True)
json.dump(dict(figs=sorted(figs), tables=sorted(opened - figs)), open(os.path.join(REPRO, '.tmp', 'opened.json'), 'w'), indent=0)
json.dump(per, open(os.path.join(REPRO, '.tmp', 'per_fig.json'), 'w'), indent=0)
print('figures open', len(figs), 'result files, the tables', len(opened - figs), 'more')
