#!/usr/bin/env python3
"""Stage A (Colab rebuild): the HEVC arms the capacity model and the scheduling simulator read, and nothing else.

Runs ref_gain.py ('base') or ref_gain_reg.py ('reg') UNCHANGED, but on a subset of their grid:
    change fraction 0.1 only (instead of 1.0 / 0.3 / 0.1), registration conditions aligned / rtk / gnss (no 'poor'),
    registered-reference arms for rtk / gnss.
The subset is selected by overriding the module constants before calling their main(); the image pairs do not depend
on which other grid points are present (changed_sets draws one permutation per image and takes a prefix per fraction;
make_ref seeds its generator per image), so every arm produced here is the same arm the full grid would produce.
Per image: 24 + 12 encodes instead of 78 + 54 - the Colab VM has 2 CPU cores, the full grid is left for later.

Usage: stageA_hevc.py base|reg [workers] [limit]
Outputs: results/ref_gain_sub.json (base), results/ref_gain_sub_reg.json (reg; needs the base file) - same format as
ref_gain.json / ref_gain_reg.json. With a limit the names end in _smoke."""
import os, sys, multiprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_gain as RG
import ref_gain_reg as RR

assert multiprocessing.get_start_method() == 'fork', 'workers must inherit the overridden constants'
RG.CHANGE = [0.1]
RG.CONDS = {k: RG.CONDS[k] for k in ('aligned', 'rtk', 'gnss')}
RR.REG_CONDS = ['rtk', 'gnss']
mode = sys.argv[1]
workers = sys.argv[2] if len(sys.argv) > 2 else '2'
limit = ['--limit', sys.argv[3]] if len(sys.argv) > 3 else []
base = f'{RG.WD}/results/ref_gain_sub{"_smoke" if limit else ""}.json'
if mode == 'base':
    sys.argv = ['ref_gain.py', '--workers', workers, '--out', base] + limit
    RG.main()
elif mode == 'reg':
    assert os.path.exists(base), f'{base} missing: run the base mode first'
    sys.argv = ['ref_gain_reg.py', '--workers', workers, '--base', base] + limit
    RR.main()
else:
    raise SystemExit('mode must be base or reg')
print(f'STAGE A HEVC {mode} DONE', flush=True)
