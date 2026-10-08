#!/usr/bin/env python3
"""Puts reproduced outputs into the folder layout the figure scripts read: <dest>/<folder>/<file> (folders as in ../res, ref/folders.json;
the per-flight *_raw.json go next to their result). Files of stage A (sim_inputs.json, a3_per_image.json) are copied from <like> if given.
usage: python tools/collect.py <output dir> <dest> [like=../res]
then:  BUB_RES=<dest> python ../figs/make_figs.py ; BUB_RES=<dest> python ../figs/make_tables.py"""
import sys, os, json, shutil
REPRO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out, dest = sys.argv[1], sys.argv[2]; like = sys.argv[3] if len(sys.argv) > 3 else None
fold = json.load(open(os.path.join(REPRO, 'ref', 'folders.json'))); n = miss = 0
for name, f in sorted(fold.items()):
    for x in (name, name[:-5] + '_raw.json'):
        p = os.path.join(out, x)
        if os.path.exists(p):
            os.makedirs(os.path.join(dest, f), exist_ok=True); shutil.copy2(p, os.path.join(dest, f, x)); n += 1
        elif x == name:
            miss += 1
if like:
    for f, x in (('stageA', 'sim_inputs.json'), ('a3', 'a3_per_image.json')):
        if os.path.exists(os.path.join(like, f, x)):
            os.makedirs(os.path.join(dest, f), exist_ok=True); shutil.copy2(os.path.join(like, f, x), os.path.join(dest, f, x))
print(n, 'files placed under', dest, '|', miss, 'results of the manifest not reproduced yet')
