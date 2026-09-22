#!/usr/bin/env python3
"""Predeclared validation-only checkpoint selection for two literature adaptations."""
from __future__ import annotations
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from literature_baselines_20260908_models import TDeepSCVisDrone, RSVQAHead


def dump(path: Path, value: object) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


@lru_cache(maxsize=4096)
def feature(path: str) -> torch.Tensor:
    return torch.load(path, map_location='cpu', weights_only=True)


def key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Inputs(Dataset):
    def __init__(self, root: Path, kind: str, split: str, original: bool = False, augmentation: str = 'none'):
        self.rows = json.loads((root / 'data' / f'{split}.json').read_text())
        self.root, self.kind, self.original = root, kind, original
        self.augmentation = augmentation if split == 'train' and kind == 'rsvqa' else 'none'
        self.image_hashes = json.loads((root / 'data/image_sha256.json').read_text())
        vocab = json.loads((root / 'data/vocab_train_only.json').read_text())
        self.answers = vocab['answers']
        self.tokenizer = BertTokenizer.from_pretrained(root / 'weights_bert/bert-small', local_files_only=True) if kind == 'tdeepsc' else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        path = row['original_image'] if self.kind == 'tdeepsc' or self.original else row['received_image']
        directory = 'features_regions' if self.kind == 'tdeepsc' else 'features_resnet'
        feature_id = key(path) if self.kind == 'tdeepsc' else self.image_hashes[path]
        feature_path = self.root / directory / (feature_id + '.pt')
        if self.augmentation == 'd4':
            view = int(torch.randint(8, ()).item())
            if view:
                feature_path = self.root / 'features_resnet_d4' / f'{feature_id}_view{view}.pt'
        result = {'image': feature(str(feature_path)),
                  'snr': float(row['snr']), 'index': index,
                  'target': self.answers.index(row['answer']) if row['answer'] in self.answers else -100}
        if self.kind == 'tdeepsc':
            result['tokens'] = {k: torch.tensor(v) for k, v in self.tokenizer(row['question'], padding='max_length', truncation=True, max_length=32).items()}
        else:
            result['question'] = feature(str(self.root / 'features_questions_normalized' / (key(row['question']) + '.pt')))
        return result


def forward(model: torch.nn.Module, batch: dict, kind: str, device: str, noiseless: bool = False) -> torch.Tensor:
    image = batch['image'].to(device).float()
    if kind == 'tdeepsc':
        return model(image, {k: v.to(device) for k, v in batch['tokens'].items()}, batch['snr'].to(device).float(), noiseless, batch.get('channel_seeds'))
    return model(image, batch['question'].to(device).float())


def evaluate(model: torch.nn.Module, data: Inputs, kind: str, device: str, batch_size: int, seeds: list[int], noiseless: bool = False) -> dict:
    from vqa_semcom.vlm.answer import check_answer
    model.eval()
    records, loss_sum, known, forward_seconds = [], 0., 0, 0.
    with torch.inference_mode():
        for seed in seeds:
            torch.manual_seed(seed)
            for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
                batch['channel_seeds'] = [int(key(f"{data.rows[i]['image']}|{data.rows[i]['snr']}|{seed}")[:8], 16)
                                          for i in batch['index'].tolist()]
                if device == 'cuda':
                    torch.cuda.synchronize()
                tick = time.perf_counter()
                logits = forward(model, batch, kind, device, noiseless)
                if device == 'cuda':
                    torch.cuda.synchronize()
                forward_seconds += time.perf_counter() - tick
                target = batch['target'].to(device)
                mask = target != -100
                if mask.any():
                    loss_sum += torch.nn.functional.cross_entropy(logits[mask], target[mask], reduction='sum').item()
                    known += int(mask.sum())
                predictions = logits.argmax(-1).cpu().tolist()
                for index, prediction in zip(batch['index'].tolist(), predictions):
                    row = data.rows[index]
                    answer = data.answers[prediction]
                    records.append({'index': index, 'channel_seed': seed, 'image': row['image'], 'qt': row['qt'],
                                    'snr': row['snr'], 'question': row['question'], 'answer': row['answer'],
                                    'prediction': answer, 'correct': bool(check_answer(row['qt'], answer, row['answer']).correct)})
    types, snrs = defaultdict(list), defaultdict(list)
    for row in records:
        types[row['qt']].append(row['correct'])
        snrs[row['snr']].append(row['correct'])
    return {'accuracy': sum(r['correct'] for r in records) / len(records),
            'loss_known_answers': loss_sum / known, 'known_answer_evaluations': known,
            'cached_feature_forward_seconds': forward_seconds,
            'cached_feature_batch_amortized_ms_per_query': forward_seconds * 1000 / len(records),
            'timing_scope': 'cached visual features through task model; excludes visual extraction, disk IO and real radio transmission; not end-to-end latency',
            'per_type': {k: sum(v) / len(v) for k, v in types.items()},
            'per_snr': {k: sum(v) / len(v) for k, v in snrs.items()}, 'records': records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--kind', choices=['tdeepsc', 'rsvqa'], required=True)
    parser.add_argument('--symbols', type=int, default=24)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--rsvqa-augmentation', default='none', choices=['none', 'd4'])
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / 'src'))
    torch.set_num_threads(2)
    if not args.smoke:
        required = ['features_regions'] if args.kind == 'tdeepsc' else ['features_resnet', 'features_questions_normalized']
        if args.kind == 'rsvqa' and args.rsvqa_augmentation == 'd4':
            required.append('features_resnet_d4')
        for directory in required:
            assert json.loads((args.root / directory / 'status.json').read_text())['state'] == 'COMPLETE', directory
    if args.device == 'cuda':
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if busy:
            raise RuntimeError(f'GPU occupied; no preemption: {busy}')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    name = f'{args.kind}_symbols{args.symbols}_seed{args.seed}' if args.kind == 'tdeepsc' else f'rsvqa_seed{args.seed}'
    if args.kind == 'rsvqa' and args.rsvqa_augmentation == 'd4':
        name = f'rsvqa_d4_seed{args.seed}'
    if args.smoke:
        name += f'_smoke_{int(time.time())}'
    dest = args.root / 'runs' / name
    dest.mkdir(parents=True, exist_ok=False)
    protocol = {'kind': args.kind, 'seed': args.seed, 'batch_size': 16, 'max_epochs': 100,
                'min_epochs': 30, 'patience': 15, 'optimizer': 'AdamW', 'learning_rate': .0001,
                'weight_decay': .0001, 'gradient_clip': 1., 'selection': 'highest validation task accuracy; tie lower known-answer CE',
                'channel_validation_seeds': [1701, 1702, 1703] if args.kind == 'tdeepsc' else [1701],
                'evaluation_channel_key': 'image ID + SNR + channel seed; identical realization across questions on one image and independent of batch order',
                'max_training_wall_seconds': 5400, 'complex_symbols': args.symbols if args.kind == 'tdeepsc' else None,
                'question_location': 'receiver, zero question uplink symbols', 'channel': 'Rician K6dB, normalized complex power1, perfectCSI',
                'adaptation_not_original_reproduction': True, 'historical_test_exposure': True,
                'pid': os.getpid(), 'smoke': args.smoke, 'device': args.device,
                'training_image_augmentation': args.rsvqa_augmentation if args.kind == 'rsvqa' else 'none',
                'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
                'dependencies': {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'transformers', 'timm', 'numpy']},
                'scripts_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('literature_baselines_20260908_*.py')}}
    dump(dest / 'protocol.json', protocol)
    train = Inputs(args.root, args.kind, 'train', augmentation=args.rsvqa_augmentation)
    validation = Inputs(args.root, args.kind, 'validation')
    model = (TDeepSCVisDrone(args.root, len(train.answers), args.symbols) if args.kind == 'tdeepsc' else RSVQAHead(len(train.answers))).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
    protocol['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dump(dest / 'protocol.json', protocol)
    if args.smoke:
        train.rows = train.rows[:32]
        validation.rows = validation.rows[:32]
    start, best, stale, history = time.time(), (-1., float('-inf')), 0, []
    best_epoch = 0
    dump(dest / 'status.json', {'state': 'RUNNING', 'pid': os.getpid(), 'epochs': 0, 'device': args.device})
    for epoch in range(1, 2 if args.smoke else 101):
        torch.manual_seed(args.seed + epoch * 100)
        model.train()
        loss_sum, count = 0., 0
        tick = time.time()
        for batch in DataLoader(train, batch_size=16, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            logits = forward(model, batch, args.kind, args.device)
            loss = torch.nn.functional.cross_entropy(logits, batch['target'].to(args.device))
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += loss.item() * len(batch['index'])
            count += len(batch['index'])
        result = evaluate(model, validation, args.kind, args.device, 16, protocol['channel_validation_seeds'])
        score = (result['accuracy'], -result['loss_known_answers'])
        if score > best:
            best, stale = score, 0
            best_epoch = epoch
            torch.save(model.state_dict(), dest / 'best.pt')
            dump(dest / 'best_validation.json', {k: v for k, v in result.items() if k != 'records'})
        else:
            stale += 1
        record = {'epoch': epoch, 'train_loss': loss_sum / count, 'validation_accuracy': result['accuracy'],
                  'validation_loss': result['loss_known_answers'], 'seconds': time.time() - tick}
        history.append(record)
        dump(dest / 'history.json', history)
        dump(dest / 'status.json', {'state': 'RUNNING', 'pid': os.getpid(), 'epochs': epoch,
                                    'best_epoch': best_epoch, 'stale_epochs': stale,
                                    'wall_seconds': time.time() - start, 'device': args.device})
        print(record, flush=True)
        if time.time() - start > 5400:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'best_validation_score': best, 'stale_epochs': stale,
                        'torch_rng': torch.get_rng_state()}, dest / 'budget_stop_state.pt')
            dump(dest / 'status.json', {'state': 'BUDGET_STOPPED_NOT_TESTED', 'epochs': epoch, 'convergence_not_established': True})
            return
        if epoch >= 30 and stale >= 15:
            break
    if args.smoke:
        dump(dest / 'status.json', {'state': 'SMOKE_COMPLETE', 'train_backward_and_validation_verified': True})
        return
    # Test is loaded only after model-selection loop and frozen checkpoint hash.
    dump(dest / 'selection_frozen.json', {'checkpoint_sha256': hashlib.sha256((dest / 'best.pt').read_bytes()).hexdigest(), 'epochs': epoch, 'selected_epoch': best_epoch})
    model.load_state_dict(torch.load(dest / 'best.pt', weights_only=True, map_location=args.device))
    test = Inputs(args.root, args.kind, 'test')
    for label, original in [('matched', False), ('noiseless_reference', True)]:
        test.original = original
        result = evaluate(model, test, args.kind, args.device, 16, [1701] if original else protocol['channel_validation_seeds'], noiseless=original)
        dump(dest / f'test_{label}.json', result)
    dump(dest / 'status.json', {'state': 'COMPLETE', 'epochs': epoch, 'wall_seconds': time.time() - start,
                                'max_epochs_reached': epoch == 100, 'no_measured_energy_j': True})


if __name__ == '__main__':
    main()
