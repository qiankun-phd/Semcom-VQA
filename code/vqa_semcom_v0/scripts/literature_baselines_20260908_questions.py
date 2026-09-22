#!/usr/bin/env python3
"""Cache original pretrained BayesianUniSkip in evaluation mode for frozen encoder."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch


def number_words(value: int) -> list[str]:
    units = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
             'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
    tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
    if value < 20:
        return [units[value]]
    if value < 100:
        return [tens[value // 10]] + (number_words(value % 10) if value % 10 else [])
    for scale, name in [(1000000, 'million'), (1000, 'thousand'), (100, 'hundred')]:
        if value >= scale:
            return number_words(value // scale) + [name] + (number_words(value % scale) if value % scale else [])
    raise ValueError(value)


def tokenize(text: str) -> list[str]:
    result = []
    for token in text.lower().replace('?', '').replace('-', ' ').split():
        result.extend(number_words(int(token)) if token.isdigit() else [token])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    expected = {'dictionary.txt': '26d8a3e6458500013723b380a4b4b55e',
                'utable.npy': '5871cc62fc01b79788c79c219b175617',
                'uni_skip.npz': '8eb7c6948001740c3111d71a2fa446c1'}
    for name, md5 in expected.items():
        h = hashlib.md5()
        with (args.weights / name).open('rb') as handle:
            for block in iter(lambda: handle.read(2**20), b''):
                h.update(block)
        assert h.hexdigest() == md5, name
    # Author-published weight hashes verified before the legacy object-array loader.
    sys.path.insert(0, str(args.root / 'upstream/skip-thoughts/pytorch'))
    from skipthoughts import BayesianUniSkip

    class ModernUniSkip(BayesianUniSkip):
        def _select_last(self, value, lengths):
            return value[torch.arange(value.shape[0]), torch.tensor(lengths) - 1]

    train = json.loads((args.root / 'data/train.json').read_text())
    numeric_lexicon = {t for n in range(100) for t in number_words(n)} | {'hundred', 'thousand', 'million'}
    words = ['UNK', *sorted({t for row in train for t in tokenize(row['question'])} | numeric_lexicon)]
    dictionary = set((args.weights / 'dictionary.txt').read_text().splitlines())
    unknown = [word for word in words if word not in dictionary]
    assert not unknown, unknown
    encoder = ModernUniSkip(str(args.weights), words, dropout=.25, fixed_emb=False).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    questions = sorted({r['question'] for split in ['train', 'validation', 'test'] for r in json.loads((args.root / 'data' / f'{split}.json').read_text())})
    out = args.root / 'features_questions_normalized'
    out.mkdir(exist_ok=True)
    (out / 'normalization_audit.json').write_text(json.dumps({'tokens': words, 'vocabulary_source': 'train questions + fixed English numeric lexicon; no heldout-derived tokens',
        'unknown_training_tokens': unknown, 'rule': 'Deterministic English number expansion and hyphen separation; same semantics, no answer information.',
        'reason': 'Literal tokenization collapsed17 training numerals and awning-tricycle to the same UNK embedding.',
        'previous_literal_cache_preserved_at': str(args.root / 'features_questions')}, indent=2))
    started = time.time()
    with torch.inference_mode():
        for offset in range(0, len(questions), 16):
            texts = questions[offset:offset + 16]
            ids = [[words.index(t) + 1 if t in words else 1 for t in tokenize(text)] for text in texts]
            lengths = [len(row) for row in ids]
            tensor = torch.zeros(len(ids), max(lengths), dtype=torch.long)
            for i, row in enumerate(ids):
                tensor[i, :len(row)] = torch.tensor(row)
            vectors = encoder(tensor, lengths=lengths)
            assert vectors.shape == (len(ids), 2400) and torch.isfinite(vectors).all()
            for text, vector in zip(texts, vectors):
                torch.save(vector.float(), out / (hashlib.sha256(text.encode()).hexdigest() + '.pt'))
            print(offset, len(questions), time.time() - started, flush=True)
    (out / 'status.json').write_text(json.dumps({'state': 'COMPLETE', 'questions': len(questions),
        'weights_md5_verified': expected, 'frozen_encoder_mode': 'eval', 'question_vocabulary': 'train questions + fixed numeric lexicon',
        'adaptation': 'Frozen encoder cached with recurrent dropout disabled; original head dropout retained. Legacy masked_select replaced by equivalent indexed gather.',
        'seconds': time.time() - started}, indent=2))


if __name__ == '__main__':
    main()
