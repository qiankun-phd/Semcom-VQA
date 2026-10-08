#!/usr/bin/env python3
"""Compares the cells of the check runs with the stored values. usage: python compare.py <output dir> <ref/expected.json>"""
import sys, os, json
out, ref = sys.argv[1], json.load(open(sys.argv[2])); worst = 0.0; n = 0
for name, cells in ref.items():
    p = os.path.join(out, name)
    if not os.path.exists(p):
        print('  missing', name); worst = float('inf'); continue
    got = json.load(open(p))['cells']
    for k, exp in cells.items():
        for m, v in exp.items():
            g = got[k][m]; d = abs(float(g) - float(v)); worst = max(worst, d); n += 1
        print(f"  {name:20s} {k:18s} hold {got[k]['mean_hold']:.6f} s (stored {exp['mean_hold']:.6f})  below {100 * got[k]['flights_below']:.4f} % (stored {100 * exp['flights_below']:.4f})")
print(f'{n} values compared, largest absolute difference {worst:.3g} ->', 'IDENTICAL' if worst == 0 else 'equal to 1e-9' if worst < 1e-9 else 'DIFFERENT')
sys.exit(0 if worst < 1e-9 else 1)
