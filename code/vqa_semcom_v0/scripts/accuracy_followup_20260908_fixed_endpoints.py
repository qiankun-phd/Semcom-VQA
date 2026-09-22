#!/usr/bin/env python3
"""Exact fixed-branch companion table; does not select policies or alter predictions."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(exist_ok=False)
    source = args.repo / 'outputs/revision_20260907_independent/crossreceiver_v2'
    keys = json.loads((source / 'common_keys.json').read_text())
    result = {}
    lines = ['# Fixed-branch companion endpoints', '',
             'Same frozen train-only calibrated branch labels as the matched-input router study. '
             'These are separate routing-free endpoints, not newly inferred answers. '
             'Always-image uses 100% images and may lie outside the κ>=0 routing curves. '
             'No physical energy claim is added here: a routing-free always-image system need not compute a detector, '
             'whereas an image choice made by a detector-feature router already incurs detector computation.', '',
             '| Receiver | Split | Type | Decisions | Images | Detection accuracy (%) | Image accuracy (%) |',
             '|---|---|---|---:|---:|---:|---:|']
    for receiver in ('qwen2', 'qwen25', 'smol'):
        result[receiver] = {}
        with np.load(source / receiver / 'evaluation_inputs.npz') as z:
            for split in ('validation', 'test'):
                kk = [k for k in keys if k['split'] == split]
                y = z[f'{split}_y']
                assert len(kk) == len(y)
                result[receiver][split] = {}
                for qt in ('pooled', 'presence', 'counting', 'comparison', 'co_presence', 'threshold'):
                    mask = np.array([qt == 'pooled' or k['qt'] == qt for k in kk])
                    v = {'decisions': int(mask.sum()), 'images': len({k['image'] for k, m in zip(kk, mask) if m}),
                         'detection_accuracy': float(y[mask, 0].mean()), 'image_accuracy': float(y[mask, 1].mean()),
                         'detection_image_use': 0, 'image_image_use': 1}
                    result[receiver][split][qt] = v
                    lines.append(f'| {receiver} | {split} | {qt} | {v["decisions"]} | {v["images"]} | '
                                 f'{v["detection_accuracy"]*100:.3f} | {v["image_accuracy"]*100:.3f} |')
    (args.out / 'endpoints.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('Fixed branch endpoints reproduced; no model fits or policy selection.', flush=True)


if __name__ == '__main__':
    main()
