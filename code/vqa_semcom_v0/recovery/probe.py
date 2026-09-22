import json, glob, statistics

cfg = json.load(open('configs/v2_0_rician_cmp.json'))


def find(d, path=''):
    if isinstance(d, dict):
        for k, v in d.items():
            if 'fading' in str(k).lower() or k == 'link':
                print(path + '/' + k, json.dumps(v)[:300])
            find(v, path + '/' + k)


find(cfg)

try:
    from PIL import Image
    dims = [Image.open(p).size for p in glob.glob('data/raw/visdrone/DET/val/images/*.jpg')]
    print('images:', len(dims),
          'mean_w:', round(statistics.mean(w for w, h in dims), 1),
          'mean_h:', round(statistics.mean(h for w, h in dims), 1),
          'mean_source_reals:', round(statistics.mean(w * h * 3 for w, h in dims), 1))
except Exception as e:
    print('PIL probe failed:', e)

import sys
sys.path.insert(0, 'src')
from vqa_semcom.degradation.digital_link import LinkConfig, FadingConfig, ergodic_spectral_efficiency
lc = LinkConfig()
print('slot complex uses:', lc.bandwidth_hz * lc.tx_time_budget_s, 'code rate ~', 1 - lc.ldpc.d_v / lc.ldpc.d_c, 'mod', lc.modulation)
for kind in ('awgn', 'rayleigh', 'rician'):
    ses = {s: round(ergodic_spectral_efficiency(s, FadingConfig(kind=kind)), 3) for s in (-5, 0, 5, 10, 15, 20)}
    print(kind, ses)
