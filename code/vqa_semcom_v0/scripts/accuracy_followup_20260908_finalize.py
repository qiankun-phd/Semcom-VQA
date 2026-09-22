#!/usr/bin/env python3
"""Package completed and visually reviewed independent follow-up artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--visual-review-complete', action='store_true')
    args = ap.parse_args()
    assert args.visual_review_complete, 'Manual review of rendered final figures is required'
    root = args.repo / 'outputs/accuracy_followup_20260908'
    assert not (root / 'manifest.json').exists(), 'Do not overwrite a finalized bundle'
    assert read(root / 'phase_ab/status.json')['state'] == 'COMPLETE'
    assert read(root / 'phase_c/status.json')['state'] == 'COMPLETE'
    assert read(root / 'analysis_ab_v2/qa.json')['model_controls_verified'] == 41
    assert read(root / 'analysis_c/qa.json')['predictions_checked'] == 5292
    assert (root / 'cpu_cost/timings.json').exists()
    assert (root / 'fixed_endpoints/endpoints.json').exists()
    assert (root / '2026-09-08--accuracy-followup--r00--matched-baseline-vlm-validation.md').exists()
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(args.repo / 'outputs'),
                            '-p', 'test_accuracy_followup_20260908*.py'], capture_output=True, text=True, check=True)
    (root / 'unit_tests.log').write_text(tests.stdout + tests.stderr)
    shutil.copy2(args.repo / 'outputs/accuracy-followup-2026-09-08.md', root / 'protocol-plan.md')
    archive = root / 'code'
    archive.mkdir(exist_ok=False)
    for pattern in ('accuracy_followup_20260908*.py', 'accuracy_followup_20260908*.sh', 'test_accuracy_followup_20260908*.py'):
        for p in (args.repo / 'outputs').glob(pattern):
            shutil.copy2(p, archive / p.name)
    # Read-only copies retain auditability without modifying any previous task code.
    deps = root / 'code_dependencies'
    deps.mkdir(exist_ok=False)
    for name in ('accuracy_upgrade_20260908_vlm.py', 'accuracy_upgrade_20260908.py',
                 'router_network_revision_20260908.py', 'analyze_router_network_revision_20260908.py'):
        p = args.repo / 'outputs' / name
        if p.exists():
            shutil.copy2(p, deps / name)
    old = args.repo / 'outputs/revision_20260907_independent/revision_20260907.py'
    shutil.copy2(old, deps / old.name)
    shutil.copy2(args.repo / 'src/vqa_semcom/vlm/answer.py', deps / 'answer.py')
    shutil.copy2(args.repo / 'src/vqa_semcom/evidence/builder.py', deps / 'builder.py')
    ab = read(root / 'analysis_ab_v2/comparisons.json')
    vlm = read(root / 'analysis_c/results.json')
    training_cost = {}
    for receiver in ('qwen2', 'qwen25', 'smol'):
        rows = []
        for seed in range(10):
            path = args.repo / f'outputs/accuracy_upgrade_20260908/phase_a/final/{receiver}/seed_{seed}/record.json'
            value = read(path)
            rows.append({'seed': seed, 'seconds': value['training_seconds'], 'source': str(path), 'sha256': sha(path)})
        training_cost[receiver] = {'network_reused_training_records': rows,
                                  'network_mean_seconds': statistics.mean(r['seconds'] for r in rows),
                                  'new_linear_records': read(root / 'analysis_ab_v2/fit_convergence.json')[receiver]}
    dump(root / 'cpu_cost/training_cost_provenance.json', training_cost)
    lines = ['# Accuracy follow-up: completed evidence bundle', '',
             'No manuscript, old predictions, source models, data split or scoring rule was overwritten. '
             'A/B uses the existing development test benchmark; C uses validation only. '
             'This bundle is not pristine-holdout confirmation.', '',
             '## Stage A: matched-input linear controls', '',
             '| Receiver | Enhanced network mean ± SD (%) | Logistic GD400 (%) | Converged logistic (%) | Ridge advantage (%) |',
             '|---|---:|---:|---:|---:|']
    for r, methods in ab.items():
        v = methods['enhanced_network']['unpriced_accuracy']
        cells = [f'{100*v["mean"]:.3f} ± {100*v["sd"]:.3f}']
        for name in ('logistic_gd400', 'logistic_converged', 'ridge_advantage'):
            cells.append(f'{100*methods[name]["unpriced_accuracy"]["mean"]:.3f}' if name in methods else 'not applicable')
        lines.append(f'| {r} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Six dual-logistic fits and two fixed ridge fits; no nonlinear retraining. '
              'All 30 preceding enhanced-network seed models and three legacy18 linear models retained and prediction-verified. '
              'Qwen2 demonstrates that augmented input information, rather than nonlinear architecture alone, explains much of the gain. '
              'Every failed/non-winning control and convergence diagnostic remains in phase_ab and analysis_ab_v2.', '',
              'Historical enhanced-network training times are reused from the original 10 seed records, with hashes, '
              'in cpu_cost/training_cost_provenance.json. They include validation checks/checkpointing. '
              'Current linear solver times and historical network fit times were not collected in one randomized timing run; '
              'treat them as descriptive implementation costs, not a precision hardware-speedup benchmark.', '',
              '## Stage B: frozen validation resource policies', '',
              'All 26 dimensionless κ points and the common absolute validation target74% are retained. '
              'Network mean curves summarize ten individually evaluated seeds; they are not a deployed ensemble policy. '
              'Selected test accuracies need not meet74%, and actual test accuracy/image use must be read together. '
              'Qwen2.5 network selected points trade lower test accuracy for lower image use; do not claim universal dominance. '
              'The optional Qwen2 physical-λ curves pass5454 exact JPEG/payload rows and reuse only the old documented accounting model. '
              'Qwen2.5/Smol receive no borrowed energy values. Extra feature/router processing costs are not measured onboard. '
              'cpu_cost reports single-thread, same-validation-batch scoring throughput separately.', '',
              '## Stage C: complete controlled validation', '',
              '| NF4 deployment model | Decisions | Correct | Accuracy (%) | New inference mean (s) |',
              '|---|---:|---:|---:|---:|']
    for model in ('3b_nf4', '7b_nf4'):
        v, t = vlm[model]['overall'], vlm[model]['timing']['new_only']
        lines.append(f'| {model} | {v["n"]} | {v["correct"]} | {100*v["accuracy"]:.3f} | {t["mean_inference_seconds"]:.4f} |')
    lines += ['', 'The full2646-key set has101images and unequal per-type image coverage: presence/counting23images; '
              'comparison/co-presence/threshold101images.120exact prior pilot answers/model reused;2526missing answers/model generated. '
              'Both models use the same received JPEG/prompt/grid, NF4/BF16compute, explicit greedy24tokens and original scorer. '
              'Historical BF16 cached3B results differ in both precision and sampling flag; they are contextual only. '
              'No7B test inference/router training, no prompt fix or detector replay was performed. '
              'All types, including any negative outcomes, appear in analysis_c. No new model Joule estimate is claimed.', '',
              '## Artifact map', '',
              '- phase_ab: all model weights, fit/convergence records, validation selection, full validation/test scores and curves.',
              '- analysis_ab_v2: canonical tables, full curves, exploratory cluster statistics, verified publication-format figures. '
              'analysis_ab is an earlier retained report whose shared legend omitted ridge; v2 fixes the legend only.',
              '- cpu_cost: all41 parameter counts and common-batch CPU timing samples.',
              '- fixed_endpoints: always-detection/always-image validation and test endpoints with actual per-type coverage.',
              '- phase_c: frozen tasks, all5292 answer records, source/checkpoint hashes, exact reuse provenance, model metadata and status.',
              '- analysis_c: full paired/type/SNR results, negative cases, format audit, timings, figures and QA.',
              '- code/code_dependencies: owned runners/tests and read-only dependency copies; protocol/document/log provenance retained.',
              '- manifest.json and manifest.sha256: final content hashes; local synchronization must match every listed file.', '',
              'The daily-coding skill guided scoped code changes and tests; results-analysis guided complete controls, '
              'actual denominator reporting, uncertainty caveats and figure QA. The dated internal decision report follows '
              'the results-report main-file structure (its referenced template files were unavailable; no KB write-back). '
              'These are exploratory development results, '
              'not claims of unqualified statistical or operational superiority.']
    (root / 'README.md').write_text('\n'.join(lines) + '\n')
    dump(root / 'status.json', {'state': 'COMPLETE', 'completed_unix': time.time(),
                               'phase_ab': 'COMPLETE', 'phase_c': 'COMPLETE',
                               'new_linear_fits': 8, 'new_nonlinear_fits': 0, 'nonlinear_models_reused': 30,
                               'legacy_linear_reused': 3, 'new_vlm_answers': 5052, 'reused_vlm_answers': 240,
                               'visual_review_complete': True, 'manuscript_modified': False})
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.name not in ('manifest.json', 'manifest.sha256'))
    manifest = {'files': {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths},
                'external_inputs': 'phase_ab/source_sha256.json and phase_c/source_sha256.json/checkpoint_gate.json',
                'status': 'COMPLETE', 'file_count': len(paths)}
    dump(root / 'manifest.json', manifest)
    (root / 'manifest.sha256').write_text(sha(root / 'manifest.json') + '  manifest.json\n')
    print(json.dumps({'files': len(paths), 'manifest_sha256': sha(root / 'manifest.json')}), flush=True)


if __name__ == '__main__':
    main()
