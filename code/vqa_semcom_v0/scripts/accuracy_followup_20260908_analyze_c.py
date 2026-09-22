#!/usr/bin/env python3
"""Verify full paired validation inference and report all outcomes; no new inference."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

QTYPES = ('presence', 'counting', 'comparison', 'co_presence', 'threshold')


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def contrast(difference, clusters, seed=20260908):
    """Pooled effect with image-cluster bootstrap and exploratory sign flips."""
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    sums = np.array([difference[clusters == k].sum() for k in unique])
    sizes = np.array([(clusters == k).sum() for k in unique])
    indices = rng.integers(0, len(unique), size=(10000, len(unique)))
    boots = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    signs = rng.choice([-1, 1], size=(10000, len(unique)))
    null = signs @ sums / sizes.sum()
    effect = float(difference.mean())
    return {'effect_pp': effect * 100, 'clusters': len(unique), 'n': len(difference),
            'bootstrap_95_percentile_pp': (np.quantile(boots, [.025, .975]) * 100).tolist(),
            'two_sided_sign_flip_p': float((1 + (np.abs(null) >= abs(effect) - 1e-15).sum()) / 10001),
            'replications': 10000, 'seed': seed}


def summarize(records):
    return {'n': len(records), 'images': len({r['image'] for r in records}),
            'correct': sum(r['correct'] for r in records),
            'accuracy': float(np.mean([r['correct'] for r in records])),
            'unknown': sum(r['normalized'] == 'unknown' for r in records)}


def timings(records):
    out = {}
    for label, rr in [('all', records), ('new_only', [r for r in records if not r['reused']]),
                      ('reused_pilot', [r for r in records if r['reused']])]:
        seconds = np.array([r['inference_seconds'] for r in rr])
        prep = np.array([r['prep_seconds'] for r in rr])
        out[label] = {'n': len(rr), 'mean_inference_seconds': float(seconds.mean()),
                      'median_inference_seconds': float(np.median(seconds)),
                      'p95_inference_seconds': float(np.quantile(seconds, .95)),
                      'sum_inference_seconds': float(seconds.sum()),
                      'mean_preprocessing_seconds': float(prep.mean()),
                      'mean_generated_tokens': float(np.mean([r['generated_tokens'] for r in rr])),
                      'peak_allocated_bytes': int(max(r['cuda_peak_allocated_bytes'] for r in rr))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root = args.repo / 'outputs/accuracy_followup_20260908'
    run = root / 'phase_c'
    assert read(run / 'status.json')['state'] == 'COMPLETE'
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo / 'src'))
    from vqa_semcom.vlm.answer import check_answer
    tasks = read(run / 'frozen_tasks.json')
    assert len(tasks) == 2646
    source = args.repo / 'outputs/revision_20260907_independent/crossreceiver_v2'
    keys = [k for k in read(source / 'common_keys.json') if k['split'] == 'validation']
    assert len(keys) == len(tasks)
    protocol = read(run / 'protocol.json')
    assert sha(args.repo / 'outputs/accuracy_followup_20260908_vlm.py') == protocol['script_sha256']
    for path, digest in read(run / 'source_sha256.json').items():
        assert sha(path) == digest, path
    records, reports, provenance, reuse_hashes = {}, {}, {}, {}
    runner_summary = read(run / 'summary.json')
    format_audit = {}
    for label in ('3b_nf4', '7b_nf4'):
        paths = sorted((run / label).glob('*.json'))
        assert len(paths) == 2646
        rr = [read(p) for p in paths]
        old_path = args.repo / f'outputs/accuracy_upgrade_20260908/phase_c/{label}_predictions.json'
        old_records = {(r['image'], r['question'], r['snr']): r for r in read(old_path)}
        assert len(old_records) == 120
        reuse_hashes[str(old_path)] = sha(old_path)
        for i, (r, t, k, p) in enumerate(zip(rr, tasks, keys, paths)):
            assert r['index'] == t['index'] == i and int(p.stem) == i
            assert (r['image'], r['question'], r['snr']) == (k['image'], k['question'], k['snr'])
            assert r['qt'] == t['question_type'] == k['qt']
            assert r['image_sha256'] == t['image_sha256'] and r['prompt_sha256'] == t['prompt_sha256']
            assert r['ground_truth'] == t['ground_truth_answer']
            if r['qt'] == 'counting':
                assert re.fullmatch(r'\d+', r['ground_truth']), 'Do not silently replace malformed count GT with zero'
            else:
                assert r['ground_truth'] in ('yes', 'no')
            answer = check_answer(r['qt'], r['prediction'], r['ground_truth'])
            assert answer.correct == r['correct'] and answer.normalized_prediction == r['normalized']
            assert np.isfinite(r['inference_seconds']) and r['inference_seconds'] > 0
            assert r['generated_tokens'] <= 24
            if r['reused']:
                old_record = old_records[(r['image'], r['question'], r['snr'])]
                assert r['origin_index'] == old_record['index']
                for field, value in old_record.items():
                    if field != 'index':
                        assert r[field] == value, (label, i, field)
            provenance[str(p)] = sha(p)
        assert sum(r['reused'] for r in rr) == 120
        records[label] = rr
        noncanonical = [r for r in rr if not re.fullmatch(r'\d+' if r['qt'] == 'counting' else r'(?i)(yes|no)', r['prediction'].strip())]
        format_audit[label] = {'noncanonical_raw_answers': len(noncanonical),
                               'by_type': dict(Counter(r['qt'] for r in noncanonical)),
                               'unknown': sum(r['normalized'] == 'unknown' for r in rr),
                               'examples_first_20_in_fixed_key_order': [
                                   {k: r[k] for k in ('index', 'qt', 'prediction', 'normalized', 'correct')}
                                   for r in noncanonical[:20]],
                               'scoring_unchanged': True}
        reports[label] = {'overall': summarize(rr), 'timing': timings(rr),
                          'by_type': {qt: summarize([r for r in rr if r['qt'] == qt]) for qt in QTYPES},
                          'by_snr': {str(s): summarize([r for r in rr if r['snr'] == s]) for s in [-5, 0, 5, 10, 15, 20]},
                          'type_snr': {f'{qt}_{s}': summarize([r for r in rr if r['qt'] == qt and r['snr'] == s]) for qt in QTYPES for s in [-5, 0, 5, 10, 15, 20]},
                          'prior_pilot': summarize([r for r in rr if r['reused']]),
                          'new_remaining_keys': summarize([r for r in rr if not r['reused']]),
                          'model': read(run / f'{label}_model.json')}
        assert reports[label]['overall']['accuracy'] == runner_summary[label]['accuracy']
        for qt in QTYPES:
            assert reports[label]['by_type'][qt]['accuracy'] == runner_summary[label]['per_type'][qt]['accuracy']
    for r3, r7 in zip(records['3b_nf4'], records['7b_nf4']):
        for field in ('image_sha256', 'prompt_sha256', 'image_grid_thw', 'ground_truth', 'reused'):
            assert r3[field] == r7[field], field
    y3 = np.array([r['correct'] for r in records['3b_nf4']], float)
    y7 = np.array([r['correct'] for r in records['7b_nf4']], float)
    images = np.array([k['image'] for k in keys])
    stats = contrast(y7 - y3, images)
    stats['discordance'] = {'only_7b_correct': int(((y7 == 1) & (y3 == 0)).sum()),
                            'only_3b_correct': int(((y7 == 0) & (y3 == 1)).sum()),
                            'both_correct': int(((y7 == 1) & (y3 == 1)).sum()),
                            'both_wrong': int(((y7 == 0) & (y3 == 0)).sum())}
    # Context only: this also changes quantization and the historical sampling flag.
    historical = {qt: {'n': sum(t['question_type'] == qt for t in tasks),
                       'accuracy': float(np.mean([t['correct'].lower() == 'true' for t in tasks if t['question_type'] == qt]))}
                  for qt in QTYPES}
    historical['overall'] = {'n': len(tasks), 'accuracy': float(np.mean([t['correct'].lower() == 'true' for t in tasks]))}
    dump(args.out / 'results.json', reports)
    dump(args.out / 'historical_3b_context_only.json', historical)
    dump(args.out / 'exploratory_paired_statistics.json', stats)
    dump(args.out / 'prediction_sha256.json', provenance)
    dump(args.out / 'reused_answer_source_sha256.json', reuse_hashes)
    dump(args.out / 'format_audit.json', format_audit)
    dump(args.out / 'prompt_scorer_source_sha256.json', {
        str(args.repo / 'src/vqa_semcom/evidence/builder.py'): sha(args.repo / 'src/vqa_semcom/evidence/builder.py'),
        str(args.repo / 'src/vqa_semcom/vlm/answer.py'): sha(args.repo / 'src/vqa_semcom/vlm/answer.py')})
    dump(args.out / 'qa.json', {'paired_keys': 2646, 'images': 101, 'predictions_checked': 5292,
                               'new': 5052, 'reused': 240, 'jpeg_prompt_grid_score_exact': True,
                               'full_validation_only': True, 'no_test_inference': True})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    figdir = args.out / 'figures'
    figdir.mkdir()
    def save(fig, name):
        fig.tight_layout()
        for ext in ('pdf', 'svg', 'png'):
            fig.savefig(figdir / f'{name}.{ext}', dpi=600, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    categories = ('overall',) + QTYPES
    for j, (label, color) in enumerate([('3b_nf4', '#0072B2'), ('7b_nf4', '#D55E00')]):
        values = [reports[label]['overall']['accuracy']] + [reports[label]['by_type'][qt]['accuracy'] for qt in QTYPES]
        ax.bar(np.arange(6) + (j - .5) * .36, np.array(values) * 100, width=.36,
               color=color, label=label.replace('_', ' ').upper())
    ax.set_xticks(np.arange(6), ['Pooled', 'Presence', 'Counting', 'Comparison', 'Co-presence', 'Threshold'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Validation answer accuracy (%)')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=.2)
    save(fig, 'full_validation_by_type')
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, color, marker in [('3b_nf4', '#0072B2', 'o'), ('7b_nf4', '#D55E00', 's')]:
        snrs = [-5, 0, 5, 10, 15, 20]
        ax.plot(snrs, [reports[label]['by_snr'][str(s)]['accuracy'] * 100 for s in snrs],
                color=color, marker=marker, label=label.replace('_', ' ').upper())
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Validation answer accuracy (%)')
    ax.set_xticks(snrs)
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    save(fig, 'full_validation_by_snr')
    lines = ['# Full controlled 3B/7B validation', '',
             'All 2646 common validation decisions across 101 images, six Rician SNRs. '
             'Same JPEG, prompt, official processor pixel limits and realized image grid, '
             'NF4 double quantization/BF16 compute, greedy 24-token decoding and scorer. '
             'This compares two quantized deployment models; it is not a BF16 model-size-only claim. '
             'No test inference, no prompt/model selection from these results.', '',
             '| Type | Decisions | Images | 3B NF4 (%) | 7B NF4 (%) | Difference (pp) |',
             '|---|---:|---:|---:|---:|---:|']
    for qt in categories:
        x = reports['3b_nf4']['overall'] if qt == 'overall' else reports['3b_nf4']['by_type'][qt]
        y = reports['7b_nf4']['overall'] if qt == 'overall' else reports['7b_nf4']['by_type'][qt]
        lines.append(f'| {qt} | {x["n"]} | {x["images"]} | {100*x["accuracy"]:.3f} | {100*y["accuracy"]:.3f} | {100*(y["accuracy"]-x["accuracy"]):+.3f} |')
    lines += ['', 'The prior 120-key pilot is a nested, differently weighted subset, not an independent replication. '
              'Its exact answers were reused (120 per model); the other 2526 per model were inferred once. '
              'Separate pilot/remainder/type/SNR results remain in results.json. Do not replace the full outcome with the pilot.', '',
              '## Runtime and scope', '', '| Model | New rows | Mean GPU inference (s) | Mean CPU preparation (s) | Peak allocation (GiB) |',
              '|---|---:|---:|---:|---:|']
    for label in ('3b_nf4', '7b_nf4'):
        t = reports[label]['timing']['new_only']
        lines.append(f'| {label} | {t["n"]} | {t["mean_inference_seconds"]:.4f} | {t["mean_preprocessing_seconds"]:.4f} | {t["peak_allocated_bytes"]/2**30:.3f} |')
    ci = stats['bootstrap_95_percentile_pp']
    lines += ['', 'Batch size 1, same fixed input keys and GPU. GPU inference timer is CUDA-synchronized around generate; '
              'it excludes CPU preparation, host-to-device transfer, answer decoding, checkpoint loading and JSON I/O. '
              'New-only rows provide a common current-run timing denominator; all-row and reused-pilot times are also retained. '
              'Models ran sequentially, so temporal hardware conditions are not randomized. '
              'NVML power.draw is unavailable, but Power Samples exists. Without continuous matched samples and a fresh idle baseline, '
              'this run makes no new Joule/airborne-power claim.', '',
              '## Exploratory paired uncertainty', '',
              f'Pooled 7B−3B effect {stats["effect_pp"]:+.3f} pp; 101-image-cluster percentile bootstrap 95% interval '
              f'[{ci[0]:+.3f}, {ci[1]:+.3f}] pp; two-sided cluster sign-flip p={stats["two_sided_sign_flip_p"]:.5f}. '
              '10,000 draws each, seed 20260908. This is one overall descriptive contrast; type/SNR tables do not carry unadjusted significance stars. '
              'Image clustering retains repeated questions/SNRs but nearby video frames may still correlate. '
              'Previously viewed validation/test benchmarks and subsequent development prevent pristine-holdout confirmation. '
              'No seed, type or unfavorable answer is removed.', '',
              '## Prompt/decoding follow-up checklist (not applied)', '',
              '- Historical BF16 cached 3B uses model-default sampling at temperature 1e-6; both NF4 runs explicitly set greedy decoding. '
              'Historical cached values are context only, not a clean quantization-only experiment.',
              '- Preserve unknown and malformed answers under the original scorer; do not retroactively repair model answers.',
              '- The current prompt explicitly specifies the output format only for presence and counting, although comparison, co-presence and threshold also need yes/no. '
              'This is a documentation/prompt-design issue for a separate controlled study, not a correction silently applied here.',
              '- The current count scorer extracts the first integer (or a zero-to-ten number word), clamps negative numeric answers to zero, '
              'and accepts error up to max(1, round(0.10 * ground-truth count)); it is not exact-match counting. '
              'All ground-truth formats are validated and the same unchanged scorer applies to both models. '
              'Raw noncanonical answers and unknown counts are retained in format_audit.json.',
              '- Compare counting, comparison and co-presence outcomes separately. More capacity need not improve every task.',
              '- Any shorter prompt, schema-constrained output or revised normalization requires its own frozen validation protocol; none was changed here.',
              '- No 7B routing model was trained, and no 7B labels were injected into the prior router/energy study.']
    (args.out / 'analysis-report.md').write_text('\n'.join(lines) + '\n')
    (args.out / 'stats-appendix.md').write_text(
        '# Exploratory full-validation paired statistics\n\n'
        f'7B NF4 minus 3B NF4 pooled effect: {stats["effect_pp"]:+.6f} percentage points. '
        f'Image-cluster percentile bootstrap 95% interval: [{ci[0]:+.6f}, {ci[1]:+.6f}] pp. '
        f'Two-sided image-cluster sign-flip Monte Carlo p: {stats["two_sided_sign_flip_p"]:.6f}.\n\n'
        'One prespecified overall contrast, 2646 paired decisions/101 images; 10000 bootstrap and sign-flip replicates, seed 20260908. '
        'All questions/SNRs for each image stay together. Resampled pooled effects retain unequal cluster sizes, '
        'rather than averaging per-image accuracies equally. No per-type/SNR significance claims or multiple favorable-subset tests. '
        'These are exploratory data-conditional checks, not model-training seed variation. Nearby video frames can remain correlated; '
        'the benchmark and nested 120-key pilot were previously viewed, so this is not pristine holdout confirmation. '
        'No null result is interpreted as proof of equivalence. Exact discordant/both-correct/both-wrong counts and full precision '
        'are in exploratory_paired_statistics.json.\n')
    (args.out / 'figure-catalog.md').write_text('# Figures\n\n'
        '`full_validation_by_type`: full paired validation, raw percentage accuracy; zero-based bars, actual uneven denominators in report. '
        'No trial/seed error bars because each deterministic deployment run supplies one answer per key. '
        'Assess pooled and adverse type effects together.\n\n'
        '`full_validation_by_snr`: same question/image composition at every SNR, complete six-point curves, no smoothing. '
        'Displays link-quality dependence without asserting monotonicity or unseen-SNR generalization.\n')
    print(json.dumps({'qa': read(args.out / 'qa.json'), 'overall': {k: v['overall'] for k, v in reports.items()}, 'stats': stats}), flush=True)


if __name__ == '__main__':
    main()
