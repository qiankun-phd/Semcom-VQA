#!/usr/bin/env python3
"""Losslessly compact only this run's regenerable D4 tensor-cache storage."""
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import tempfile

root = Path(sys.argv[1]).resolve()
assert str(root) == '/home/qiankun/phd_research/vqa_semcom/outputs/literature_baselines_20260908'
# Disk-full bootstrap: Torch/dill imports need a writable temporary directory.
# Use a private memory-backed directory only for this repair process.
repair_temp = tempfile.TemporaryDirectory(prefix='literature-baseline-repair-', dir='/dev/shm')
tempfile.tempdir = repair_temp.name
import torch
directory = root / 'features_resnet_d4'
torch.set_num_threads(2)
removed, compacted, saved = [], 0, 0
started = time.time()
for index, path in enumerate(sorted(directory.glob('*.pt'))):
    assert re.fullmatch(r'[0-9a-f]{64}_view[1-7]\.pt', path.name), path
    before = path.stat().st_size
    try:
        value = torch.load(path, weights_only=True, map_location='cpu')
        assert value.shape == (2048, 16, 16) and value.dtype == torch.float16 and torch.isfinite(value).all()
    except Exception as error:
        # This is a derived training-view cache, not source data/checkpoints.
        removed.append({'file': path.name, 'reason': repr(error), 'bytes': before})
        path.unlink()
        continue
    if value.untyped_storage().nbytes() <= value.numel() * value.element_size():
        continue
    buffer = io.BytesIO()
    torch.save(value.clone(), buffer)
    content = buffer.getvalue()
    assert len(content) < 2 * value.numel() * value.element_size()
    if shutil.disk_usage(directory).free < len(content) * 2:
        # Bootstrap one owned, regenerable cache file after all tensor values
        # are safely loaded in RAM. Truncation releases its bloated storage.
        # Reuse already allocated blocks before truncating: freeing blocks
        # first can still fail under delayed-allocation disk-full conditions.
        with path.open('r+b') as handle:
            handle.seek(0)
            handle.write(content)
            handle.flush()
            handle.truncate(len(content))
            os.fsync(handle.fileno())
        os.sync()
        torch.testing.assert_close(torch.load(path, weights_only=True), value, rtol=0, atol=0)
    else:
        temporary = path.with_suffix('.compact-tmp')
        temporary.write_bytes(content)
        torch.testing.assert_close(torch.load(temporary, weights_only=True), value, rtol=0, atol=0)
        temporary.replace(path)
    compacted += 1
    saved += before - len(content)
    if index % 500 == 0:
        print(index, compacted, saved, flush=True)
audit = {'state': 'COMPACTED_REGENERATE_MISSING', 'compacted': compacted, 'saved_bytes': saved,
         'removed_corrupt_derived_caches': removed, 'seconds': time.time() - started,
         'repair': 'clone tensor storage, exact-value round-trip verification; source data and checkpoints untouched'}
(root / 'storage_repair.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit), flush=True)
