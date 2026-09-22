#!/usr/bin/env python3
"""Read-only, common-batch CPU scoring timing; no training or GPU use."""
import argparse
import os
from pathlib import Path
import platform
import time
import joblib
import numpy as np
import accuracy_followup_20260908_ab as a


def measure(fn, x):
    for _ in range(5):
        fn(x)
    seconds = []
    for _ in range(30):
        tick = time.perf_counter()
        fn(x)
        seconds.append(time.perf_counter() - tick)
    return {'batch_rows': len(x), 'warmup_batches': 5, 'measured_batches': 30,
            'batch_seconds': seconds, 'median_us_per_decision': float(np.median(seconds) * 1e6 / len(x)),
            'p95_us_per_decision': float(np.quantile(seconds, .95) * 1e6 / len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    b = a.b
    root = args.repo / 'outputs/accuracy_followup_20260908'
    prior = args.repo / 'outputs/accuracy_upgrade_20260908'
    source = args.repo / 'outputs/revision_20260907_independent/crossreceiver_v2'
    result = {}
    for receiver, methods in b.read(root / 'phase_ab/summary.json').items():
        with np.load(prior / f'phase_a/{receiver}_evaluation_inputs.npz') as z:
            x = z['validation_x'].copy()
        with np.load(source / receiver / 'evaluation_inputs.npz') as z:
            legacy_x, legacy_w = z['validation_x'].copy(), z['linear_weights'].copy()
        result[receiver] = {}
        for name in methods:
            xx = x
            if name.startswith('enhanced_network'):
                seed = int(name.rsplit('_', 1)[1])
                model = joblib.load(prior / f'phase_a/final/{receiver}/seed_{seed}/models.joblib')
                cfg = b.CONFIGS[a.NETWORKS[receiver]]
                fn = lambda features: b.route_score(model, cfg, features)
                models = model if isinstance(model, (list, tuple)) else [model]
                params = sum(sum(c.size for c in m.coefs_) + sum(c.size for c in m.intercepts_) for m in models)
            else:
                if name == 'legacy18_linear':
                    w, xx = legacy_w, legacy_x
                else:
                    with np.load(root / f'phase_ab/{receiver}/{name}/weights.npz') as z:
                        w = z['weights'].copy()
                fit = {'weights': w, 'objective': 'advantage_mse' if name == 'ridge_advantage' else 'dual_bce'}
                fn = lambda features: a.score(fit, features)
                params = w.size
            expected = np.load(root / f'phase_ab/{receiver}/{name}/validation_outcomes.npz')['score']
            np.testing.assert_array_equal(fn(xx), expected)
            result[receiver][name] = {**measure(fn, xx), 'parameters': params, 'feature_dimensions': xx.shape[1]}
    from threadpoolctl import threadpool_info
    b.dump(args.out / 'timings.json', result)
    b.dump(args.out / 'environment.json', {'platform': platform.platform(), 'cpu': platform.processor(),
                                          'threadpools': threadpool_info(), 'pid': os.getpid(),
                                          'scope': 'same 2646 validation rows, batch scoring only; no feature extraction/loading/training',
                                          'caveat': 'amortized CPU batch throughput, not per-query online latency or airborne energy'})
    lines = ['# Common-batch CPU scoring costs', '',
             'Five warmups and 30 measured batches per model, each with the same 2646 validation rows. '
             'Single BLAS thread and CUDA disabled. Model loading, input feature construction and standardization are excluded. '
             'This is amortized throughput, not batch-size-one latency or measured onboard energy. '
             'Models run sequentially while a separate GPU inference job is active; CPU contention can affect absolute times.', '',
             '| Receiver | Model | Parameters | Input dimensions | Median µs/decision |', '|---|---|---:|---:|---:|']
    for r, methods in result.items():
        for name, v in methods.items():
            lines.append(f'| {r} | {name} | {v["parameters"]} | {v["feature_dimensions"]} | {v["median_us_per_decision"]:.4f} |')
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('41 scoring controls benchmarked and prediction-verified; no model fitted.', flush=True)


if __name__ == '__main__':
    main()
