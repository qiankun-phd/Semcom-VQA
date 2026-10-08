#!/usr/bin/env python3
"""Rebuild sim_inputs.json for the scheduling simulator from the Stage-A result files (local, JSON only).

Same definitions as the original builder (session transcript line 3502), with two stated differences:
  * source files: eval_gated_sweep_c{96,192}.json (det gate, rtk/gnss, change 0.1) instead of eval_gated_det_c*.json,
    ref_gain_sub_reg.json instead of ref_gain_reg.json, sym_baseline_synth.json instead of sym_baseline.json;
  * snr_pts: all measured points of the sweep by default (-2 ... 13 dB); pass --pts 1 7 13 for the original three.
Everything is read from the files; nothing is typed in by hand.
Usage: python3 build_sim_inputs.py [--res res] [--pts ...] [--out res/sim_inputs.json]"""
import json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--res', default='res'); ap.add_argument('--out', default=None)
ap.add_argument('--pts', type=int, nargs='*', default=None)
a = ap.parse_args()
load = lambda n: json.load(open(f'{a.res}/{n}'))
SW = {C: load(f'eval_gated_sweep_c{C}.json') for C in (96, 192)}
D = load('ref_gain_sub_reg.json'); S = load('sym_baseline_synth.json')['synthetic']
pts = a.pts or SW[96]['meta']['snrs']
assert SW[96]['meta']['snrs'] == SW[192]['meta']['snrs'] and all(p in SW[96]['meta']['snrs'] for p in pts)
inp = {'snr_pts': pts, 'raw_rec': D['raw640_rec_changed']['0.1'], 'q_sym': S['O_rec_new@0.1'], 'sym_bytes': S['sym_bytes']}
for C in (96, 192):
    R = SW[C]['results']
    rec = [float(np.mean([R[f'det|{c}@0.1|{s}']['rec_new'] for c in ('rtk', 'gnss')])) for s in pts]
    tok = float(np.mean([R[f'det|{c}@0.1|7']['n_sel'] for c in ('rtk', 'gnss')]))
    side = float(np.mean([R[f'det|{c}@0.1|7']['side_bytes'] for c in ('rtk', 'gnss')]))
    inp[f'gated_C{C}'] = dict(rec=rec, tokens=tok, side_bytes=side)


def curve(prefix):
    return [[D['arms'][f'{prefix}|{q}']['mean_bytes'], D['arms'][f'{prefix}|{q}']['rec_changed']['0.1']] for q in D['qps']]


inp['hevc_intra'] = curve('intra')
inp['hevc_inter_gnss'] = curve('gnss@0.1') + curve('gnss+reg@0.1')
inp['hevc_inter_rtk'] = curve('rtk@0.1') + curve('rtk+reg@0.1')
inp['provenance'] = dict(n_images=SW[96]['meta']['n'], sources=['eval_gated_sweep_c96.json', 'eval_gated_sweep_c192.json',
                                                                'ref_gain_sub_reg.json', 'sym_baseline_synth.json'])
out = a.out or f'{a.res}/sim_inputs.json'
json.dump(inp, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in inp.items() if not k.startswith('hevc')}, indent=1))
for k in ('hevc_intra', 'hevc_inter_gnss', 'hevc_inter_rtk'):
    print(k, [[round(b), round(r, 3)] for b, r in inp[k]])
print('saved', out)
