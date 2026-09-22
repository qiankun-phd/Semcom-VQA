#!/usr/bin/env python3
"""Full post-repair D4 cache integrity and disk-space gate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
directory = args.root / 'features_resnet_d4'
assert json.loads((directory / 'status.json').read_text())['state'] == 'COMPLETE'
rows = json.loads((args.root / 'data/train.json').read_text())
image_hashes = json.loads((args.root / 'data/image_sha256.json').read_text())
training_hashes = {image_hashes[row['received_image']] for row in rows}
expected = {f'{digest}_view{view}.pt' for digest in training_hashes for view in range(1, 8)}
actual = {p.name for p in directory.glob('*.pt')}
assert actual == expected, {'missing': sorted(expected - actual), 'unexpected': sorted(actual - expected)}
started = time.time()
hashes, sizes = {}, []
for index, name in enumerate(sorted(expected)):
    path = directory / name
    value = torch.load(path, map_location='cpu', weights_only=True)
    assert value.shape == (2048, 16, 16) and value.dtype == torch.float16
    assert torch.isfinite(value).all()
    assert value.untyped_storage().nbytes() == value.numel() * value.element_size(), name
    size = path.stat().st_size
    assert size < 2 * value.numel() * value.element_size(), (name, size)
    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    sizes.append(size)
    if index % 2000 == 0:
        print(index, len(expected), flush=True)
free = shutil.disk_usage(directory).free
assert free > 10 * 1024**3, 'Keep at least10GiB free before continuing training'
result = {'state': 'PASSED', 'training_image_contents': len(training_hashes), 'files': len(expected),
          'bytes_total': sum(sizes), 'bytes_min': min(sizes), 'bytes_max': max(sizes),
          'disk_available_bytes': free, 'seconds': time.time() - started,
          'all_readable_finite_correct_shape_and_independent_storage': True,
          'source': 'train received-image hashes only; no heldout-derived augmentation choices'}
(args.root / 'd4_cache_sha256.json').write_text(json.dumps(hashes, indent=2))
(args.root / 'd4_cache_audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
