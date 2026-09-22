#!/usr/bin/env python3
"""Read-only source identity gate before a resumed VLM process may run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
for path,digest in json.loads((args.out/'source_sha256.json').read_text()).items():assert sha(path)==digest,path
gate=json.loads((args.out/'checkpoint_gate.json').read_text())
for name,manifest in gate['manifests'].items():
    for filename,meta in manifest['files'].items():assert sha(Path(manifest['snapshot'])/filename)==meta['sha256'],(name,filename)
runner=args.repo/'outputs/accuracy_followup_20260908_vlm.py'
assert sha(runner)==json.loads((args.out/'protocol.json').read_text())['script_sha256']
print('Resume source/checkpoint/runner SHA gates passed; no content changed.',flush=True)
if not args.check_only:os.execv(sys.executable,[sys.executable,'-u',str(runner),'--repo',str(args.repo),'--out',str(args.out),'--resume'])
