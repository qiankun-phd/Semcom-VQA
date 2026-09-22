#!/usr/bin/env python3
"""Rebuild or verify the artifact manifest after final reporting-only additions."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--verify',action='store_true');ap.add_argument('--checkpoint',type=Path);args=ap.parse_args()
if args.checkpoint:
    weights={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(args.checkpoint.iterdir()) if p.is_file()}
    (args.root/'3b_checkpoint_manifest.json').write_text(json.dumps({'snapshot':str(args.checkpoint),'files':weights},indent=2)+'\n')
files={str(p.relative_to(args.root)):sha(p) for p in sorted(args.root.rglob('*')) if p.is_file()
    and not any(part in ('quant_env','wheels') for part in p.relative_to(args.root).parts) and p.name!='bundle_manifest.json'}
digest=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
if args.verify:
    expected=json.loads((args.root/'bundle_manifest.json').read_text())
    assert files==expected['files'] and digest==expected['digest'],'Artifact checksum mismatch'
else:
    (args.root/'bundle_manifest.json').write_text(json.dumps({'files':files,'digest':digest,'file_count':len(files),'exclusions':['quant_env','wheels']},indent=2)+'\n')
print(json.dumps({'verified' if args.verify else 'written':True,'files':len(files),'digest':digest}))
