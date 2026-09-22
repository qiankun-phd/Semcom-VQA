#!/usr/bin/env python3
"""Descriptive, single-training-seed analysis of completed baseline adaptations."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def style(run):
    if run['protocol']['kind'] == 'rsvqa':
        return ('#D55E00', 's', '--') if run['protocol'].get('training_image_augmentation') == 'd4' else ('#777777', 'x', ':')
    return {24: ('#0072B2', 'o', '-'), 48: ('#009E73', '^', '-.'),
            96: ('#CC79A7', 'D', ':')}[run['protocol']['complex_symbols']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for directory in sorted((args.root / 'runs').iterdir()):
        if '_smoke_' in directory.name or not (directory / 'status.json').exists():
            continue
        status = read(directory / 'status.json')
        if status['state'] != 'COMPLETE':
            continue
        protocol = read(directory / 'protocol.json')
        frozen = read(directory / 'selection_frozen.json')
        assert frozen['checkpoint_sha256'] == hashlib.sha256((directory / 'best.pt').read_bytes()).hexdigest()
        matched = read(directory / 'test_matched.json')
        clean = read(directory / 'test_noiseless_reference.json')
        records = matched['records']
        assert len({(r['index'], r['channel_seed']) for r in records}) == len(records)
        assert len(records) == 2808 * len(protocol['channel_validation_seeds'])
        assert abs(statistics.mean(r['correct'] for r in records) - matched['accuracy']) < 1e-12
        name = f"T-DeepSC adapted ({protocol['complex_symbols']} symbols)" if protocol['kind'] == 'tdeepsc' else ('RSVQA adapted (D4 training)' if protocol.get('training_image_augmentation') == 'd4' else 'RSVQA single-view pilot')
        runs.append({'name': name, 'directory': directory, 'protocol': protocol, 'status': status,
                     'frozen': frozen, 'matched': matched, 'clean': clean, 'history': read(directory / 'history.json')})
    assert runs, 'No completed runs; do not report partial training as test results.'
    dest = args.root / 'analysis'
    figures = dest / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(6.5, 3.3), layout='constrained')
    for run in runs:
        color, marker, line = style(run)
        pairs = sorted((float(k), v * 100) for k, v in run['matched']['per_snr'].items())
        ax.plot(*zip(*pairs), color=color, marker=marker, linestyle=line, label=run['name'], linewidth=1.7)
    ax.set(xlabel='SNR (dB)', ylabel='Task accuracy (%)', ylim=(0, 100), xticks=[-5, 0, 5, 10, 15, 20])
    ax.grid(alpha=.2)
    if len(runs) > 3:
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=2, fontsize=7.5, frameon=False)
    else:
        ax.legend(fontsize=8)
    fig.savefig(figures / 'accuracy_snr.pdf')
    fig.savefig(figures / 'accuracy_snr.png', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout='constrained')
    for run in runs:
        color, marker, line = style(run)
        history = run['history']
        axes[0].plot([r['epoch'] for r in history], [r['train_loss'] for r in history], color=color, linestyle=line, label=run['name'])
        axes[1].plot([r['epoch'] for r in history], [r['validation_accuracy'] * 100 for r in history], color=color, linestyle=line)
    axes[0].set(xlabel='Epoch', ylabel='Training cross entropy')
    axes[1].set(xlabel='Epoch', ylabel='Validation accuracy (%)', ylim=(0, 100))
    axes[0].legend(fontsize=6.5)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(figures / 'training_dynamics.pdf')
    fig.savefig(figures / 'training_dynamics.png', dpi=200)
    plt.close(fig)
    semantic_runs = sorted([r for r in runs if r['protocol']['kind'] == 'tdeepsc'], key=lambda r: r['protocol']['complex_symbols'])
    if len(semantic_runs) >= 2:
        fig, ax = plt.subplots(figsize=(4.5, 3.2), layout='constrained')
        ax.plot([r['protocol']['complex_symbols'] for r in semantic_runs],
                [100 * r['matched']['accuracy'] for r in semantic_runs], color='#0072B2', marker='o')
        ax.set(xlabel='Complex image symbols', ylabel='Mean task accuracy (%)', ylim=(0, 100), xticks=[24, 48, 96])
        ax.grid(alpha=.2)
        fig.savefig(figures / 'accuracy_symbols.pdf')
        fig.savefig(figures / 'accuracy_symbols.png', dpi=200)
        plt.close(fig)
    table = ['| Adapted method | Test accuracy | Clean/noiseless reference | Selected epoch | Trained epochs |',
             '|---|---:|---:|---:|---:|']
    detail = []
    for run in runs:
        table.append(f"| {run['name']} | {run['matched']['accuracy']*100:.3f}% | {run['clean']['accuracy']*100:.3f}% | {run['frozen'].get('selected_epoch', 'unknown')} | {run['status']['epochs']} |")
        detail.append({'name': run['name'], 'run': str(run['directory']), 'test_accuracy': run['matched']['accuracy'],
                       'per_type': run['matched']['per_type'], 'per_snr': run['matched']['per_snr'],
                       'train_seed': run['protocol']['seed'], 'channel_seeds': run['protocol']['channel_validation_seeds'],
                       'checkpoint_sha256': run['frozen']['checkpoint_sha256'],
                       'max_epochs_reached': run['status']['max_epochs_reached']})
    (dest / 'summary.json').write_text(json.dumps(detail, indent=2))
    (dest / 'analysis-report.md').write_text('# Single-seed literature-baseline analysis\n\n'
        'Question: how accurately do the two dedicated VisDrone adaptations answer the same468 held-out questions at six SNRs? '
        'This is not a matched-radio-resource superiority test. T-DeepSC sends learned complex symbols; RSVQA uses the existing received JPEG.\n\n'
        + '\n'.join(table) + '\n\n'
        'All completed runs are included. One training seed per configuration; no seed-level variance or significance claim. '
        'Clean/noiseless values use the same selected checkpoint and are not guaranteed upper bounds. '
        'Refer to PLAN.md for frontend, encoder-mode, augmentation, tokenization and optimizer deviations. '
        'Historical test exposure and image-not-video-disjoint splitting limit generalization claims. '
        'Inspect training_dynamics.pdf together with selected epochs; hitting the100-epoch cap does not establish convergence.\n')
    (dest / 'stats-appendix.md').write_text('# Statistical limits\n\n'
        'Primary metric: existing task-aware correct-answer fraction, higher is better. '
        '104 test images,468 unique questions,2808 image-question-SNR decisions. '
        'T-DeepSC averages three fixed channel realizations per decision; these are not independent training runs. '
        'Questions/SNRs on one image and nearby video frames are dependent. '
        'No independent-seed SD, confidence interval, p-value, or standardized effect size is inferred from one training seed. '
        'Accordingly no multiple-testing procedure is invoked and no significance stars appear. '
        'Descriptive per-type/per-SNR values and exact sample counts are retained in JSON. '
        'Any future paired inferential analysis must preserve image/sequence clustering and distinguish receiver accuracy from communication-cost matching.\n')
    (dest / 'figure-catalog.md').write_text('# Figure catalog\n\n'
        '## accuracy_snr.pdf\n\n'
        'Purpose: show descriptive channel sensitivity of completed adaptations. '
        'Caption must state one training seed,468 questions/SNR, three channel draws for T-DeepSC, and unmatched transmission representations/budgets. '
        'No error bars: seed uncertainty is unmeasured. Notice whether the slope is consistent across SNR, not only the highest point. '
        'A different slope motivates robustness analysis but does not identify a causal frontend advantage.\n\n'
        '## training_dynamics.pdf\n\n'
        'Purpose: assess optimization and validation saturation, without using test for checkpoint choice. '
        'Raw epochs are unsmoothed. Compare falling training loss with validation plateaus/declines. '
        'A continuing validation improvement at the cap prevents a convergence claim; validation decline with falling loss suggests overfitting and motivates a separately declared follow-up.\n\n'
        '## accuracy_symbols.pdf (when at least two budgets complete)\n\n'
        'Purpose: show the predeclared24/48/96-symbol tradeoff, averaged over six SNRs and three fixed channel draws. '
        'Each point is independently trained with one seed and validation-only epoch selection; no best-budget cherry-picking. '
        'A flat/non-monotonic curve does not prove added symbols are useless in general; optimization and finite target-data effects remain.\n\n'
        '## Observed completed-run patterns\n\n' + '\n'.join(
            f"- {r['name']}: selected epoch {r['frozen'].get('selected_epoch')}; training CE {r['history'][0]['train_loss']:.3f}→{r['history'][-1]['train_loss']:.3f}; "
            f"test SNR range {min(r['matched']['per_snr'].values())*100:.2f}–{max(r['matched']['per_snr'].values())*100:.2f}%. "
            'These are descriptive observations, not independent-seed uncertainty estimates.' for r in runs) + '\n')
    print('\n'.join(table))


if __name__ == '__main__':
    main()
