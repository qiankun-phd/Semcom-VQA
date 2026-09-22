#!/usr/bin/env python3
"""Freeze matched benchmark inputs; never selects architecture from test labels."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.repo / 'outputs/revision_20260907_independent/crossreceiver_v2'
    keypath = source / 'common_keys.json'
    keys = json.loads(keypath.read_text())
    expected = {'train': 9150, 'validation': 2646, 'test': 2808}
    assert dict(Counter(k['split'] for k in keys)) == expected
    identities = {(k['image'], k['question'], k['snr']): k for k in keys}
    assert len(identities) == len(keys)
    source_hashes = {str(keypath): sha(keypath)}
    rows = {}
    for stem in ('main', 'cmp', 'extra'):
        path = args.repo / f'outputs/vlm/v25_rician_{stem}_predictions.csv'
        source_hashes[str(path)] = sha(path)
        with path.open() as handle:
            for row in csv.DictReader(handle):
                key = (row['image_id'], row['question'], int(float(row['sensed_snr_db'])))
                if row['service_level'] != '2' or key not in identities:
                    continue
                record = {**identities[key], 'answer': row['ground_truth_answer'],
                          'received_image': row['image_path'], 'channel_bin': row['channel_bin']}
                if key in rows:
                    assert rows[key] == record
                rows[key] = record
    assert set(rows) == set(identities)
    old_hashes = json.loads((source / 'receiver_image_sha256.json').read_text())
    image_hashes = {}
    for row in rows.values():
        path = Path(row['received_image'])
        if str(path) not in image_hashes:
            image_hashes[str(path)] = sha(path)
            assert image_hashes[str(path)] == old_hashes[str(path)]
        original = args.repo / 'data/raw/visdrone/DET/val/images' / (row['image'] + '.jpg')
        assert original.is_file(), original
        row['original_image'] = str(original)
        if str(original) not in image_hashes:
            image_hashes[str(original)] = sha(original)
    images = {split: {r['image'] for r in rows.values() if r['split'] == split} for split in expected}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not images[a] & images[b]
    train = [r for r in rows.values() if r['split'] == 'train']
    answer_vocab = sorted({r['answer'] for r in train})
    question_vocab = sorted({token for r in train for token in r['question'].lower().replace('?', '').split()})
    ordered = [rows[(k['image'], k['question'], k['snr'])] for k in keys]
    for split in expected:
        dump(args.out / f'{split}.json', [r for r in ordered if r['split'] == split])
    dump(args.out / 'vocab_train_only.json', {'answers': answer_vocab, 'question_tokens': question_vocab})
    dump(args.out / 'source_sha256.json', source_hashes)
    dump(args.out / 'image_sha256.json', image_hashes)
    dump(args.out / 'data_audit.json', {
        'rows': expected, 'images': {s: len(v) for s, v in images.items()},
        'unique_questions': {s: len({(r['image'], r['question']) for r in ordered if r['split'] == s}) for s in expected},
        'types': {s: dict(Counter(r['qt'] for r in ordered if r['split'] == s)) for s in expected},
        'snr_db': sorted({r['snr'] for r in ordered}), 'image_disjoint': True,
        'answer_vocabulary_source': 'training labels only; out-of-vocabulary evaluation labels remain unrepresentable',
        'answer_vocabulary_size': len(answer_vocab), 'prior_test_exposure': True,
        'selection': 'validation only; no claim of pristine holdout',
        'script_sha256': sha(Path(__file__)), 'received_jpeg_sha256_match': True})
    print((args.out / 'data_audit.json').read_text())


if __name__ == '__main__':
    main()
