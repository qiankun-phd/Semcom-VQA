#!/usr/bin/env python3
"""Acquire provenance-pinned public dependencies without loading pickle files."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
from huggingface_hub import HfApi, snapshot_download


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--group', choices=['all', 'vision', 'bert'], default='all')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items = {
        'dictionary.txt': ('https://mirror.nubenum.de/www.cs.toronto.edu/~rkiros/models/dictionary.txt', '26d8a3e6458500013723b380a4b4b55e'),
        'utable.npy': ('https://mirror.nubenum.de/www.cs.toronto.edu/~rkiros/models/utable.npy', '5871cc62fc01b79788c79c219b175617'),
        'uni_skip.npz': ('https://mirror.nubenum.de/www.cs.toronto.edu/~rkiros/models/uni_skip.npz', '8eb7c6948001740c3111d71a2fa446c1'),
        'resnet152-f82ba261.pth': ('https://download.pytorch.org/models/resnet152-f82ba261.pth', None),
        'fasterrcnn_resnet50_fpn_coco-258fb6c6.pth': ('https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth', None),
    }
    if args.group == 'vision':
        items = {k: v for k, v in items.items() if k.endswith('.pth')}
    if args.group == 'bert':
        items = {}
    status = {'started_unix': time.time(), 'state': 'DOWNLOADING', 'files': {},
              'skipthought_md5_source': 'https://github.com/Maluuba/nlg-eval#setup',
              'skipthought_mirror_source': 'https://github.com/Maluuba/nlg-eval/blob/master/bin/nlg-eval'}
    path = args.out / f'download_status_{args.group}.json'
    path.write_text(json.dumps(status, indent=2))

    def download(item: tuple) -> tuple:
        name, (url, md5) = item
        dest = args.out / name
        subprocess.run(['curl', '-L', '--fail', '--retry', '2', '--connect-timeout', '20', '--max-time', '3600', '-C', '-', url, '-o', str(dest)], check=True)
        if md5:
            assert digest(dest, 'md5') == md5, name
        sha = digest(dest, 'sha256')
        if name.endswith('.pth'):
            assert sha.startswith(name.rsplit('-', 1)[1].split('.')[0]), name
        return name, {'url': url, 'bytes': dest.stat().st_size, 'sha256': sha, 'verified_md5': md5}

    with ThreadPoolExecutor(max_workers=3) as executor:
        for name, record in executor.map(download, items.items()):
            status['files'][name] = record
            path.write_text(json.dumps(status, indent=2))
    if args.group in ('all', 'bert'):
        info = HfApi().model_info('prajjwal1/bert-small')
        checkpoint = snapshot_download('prajjwal1/bert-small', revision=info.sha,
                                       allow_patterns=['config.json', 'pytorch_model.bin', 'vocab.txt'], local_dir=args.out / 'bert-small')
        status['bert_small'] = {'repo': 'prajjwal1/bert-small', 'revision': info.sha, 'path': checkpoint,
                                'sha256': {p.name: digest(p, 'sha256') for p in Path(checkpoint).iterdir() if p.is_file()}}
    status['state'] = 'COMPLETE'
    path.write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
