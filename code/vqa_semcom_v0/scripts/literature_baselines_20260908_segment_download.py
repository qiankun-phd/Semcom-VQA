#!/usr/bin/env python3
"""Bounded resumable public-weight transport with published whole-file MD5 gate."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items = {'dictionary.txt': (7996547, '26d8a3e6458500013723b380a4b4b55e'),
             'uni_skip.npz': (663989216, '8eb7c6948001740c3111d71a2fa446c1'),
             'utable.npy': (2342138474, '5871cc62fc01b79788c79c219b175617')}
    tasks = []
    size = 16 * 2**20
    status = {'state': 'DOWNLOADING', 'started_unix': time.time(), 'completed_chunks': 0,
              'transport_workers': 8, 'files': {}, 'md5_source': 'https://github.com/Maluuba/nlg-eval#setup'}
    for name, (length, _) in items.items():
        for start in range(0, length, size):
            tasks.append((name, start, min(start + size, length) - 1))

    def chunk(task: tuple) -> None:
        name, start, end = task
        dest = args.out / f'{name}.part{start:012d}'
        if dest.exists() and dest.stat().st_size == end - start + 1:
            return
        # Keep partial bytes after a dropped connection; never restart a whole16MB
        # range merely because the transport ended early.
        for attempt in range(30):
            have = dest.stat().st_size if dest.exists() else 0
            if have == end - start + 1:
                break
            assert have < end - start + 1
            tail = dest.with_suffix(dest.suffix + '.tail')
            headers = dest.with_suffix(dest.suffix + '.headers')
            proc = subprocess.run(['curl', '-sSL', '--fail', '--connect-timeout', '20', '--max-time', '90',
                                   '--range', f'{start + have}-{end}', '-D', str(headers),
                                   f'https://mirror.nubenum.de/www.cs.toronto.edu/~rkiros/models/{name}', '-o', str(tail)])
            if tail.exists() and tail.stat().st_size:
                metadata = headers.read_text().lower()
                assert f'content-range: bytes {start + have}-{end}/' in metadata, metadata
                assert tail.stat().st_size <= end - start + 1 - have
                with dest.open('ab') as output, tail.open('rb') as handle:
                    shutil.copyfileobj(handle, output)
                tail.unlink()
            if proc.returncode not in (0, 18, 28, 22, 56, 35):
                raise RuntimeError(f'curl failed unexpectedly: {proc.returncode}')
        assert dest.stat().st_size == end - start + 1, (dest, dest.stat().st_size)

    failures = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for future in as_completed([executor.submit(chunk, t) for t in tasks]):
            try:
                future.result()
            except Exception as exc:
                failures.append(repr(exc))
            status['completed_chunks'] += 1
            status['failures'] = failures
            status['total_chunks'] = len(tasks)
            (args.out / 'status.json').write_text(json.dumps(status, indent=2))
            print(status['completed_chunks'], len(tasks), time.time() - status['started_unix'], flush=True)
    if failures:
        status['state'] = 'PARTIAL_RETRY_NEEDED'
        (args.out / 'status.json').write_text(json.dumps(status, indent=2))
        raise RuntimeError(failures)
    for name, (length, expected_md5) in items.items():
        md5, sha = hashlib.md5(), hashlib.sha256()
        with (args.out / name).open('wb') as output:
            for start in range(0, length, size):
                with (args.out / f'{name}.part{start:012d}').open('rb') as handle:
                    for block in iter(lambda: handle.read(2**20), b''):
                        output.write(block)
                        md5.update(block)
                        sha.update(block)
        assert md5.hexdigest() == expected_md5, name
        status['files'][name] = {'bytes': length, 'md5': md5.hexdigest(), 'sha256': sha.hexdigest()}
    status['state'] = 'COMPLETE'
    (args.out / 'status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
