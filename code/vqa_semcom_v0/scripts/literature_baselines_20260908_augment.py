#!/usr/bin/env python3
"""Training-only pixel-space D4 augmentation; preserve frozen encoder convention."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import time

import torch
from torchvision import models, transforms
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU occupied'
    torch.set_num_threads(2)
    rows = json.loads((args.root / 'data/train.json').read_text())
    hashes = json.loads((args.root / 'data/image_sha256.json').read_text())
    paths = sorted({hashes[r['received_image']]: r['received_image'] for r in rows}.values())
    total = len(paths)
    if args.limit:
        paths = paths[:args.limit]
    net = models.resnet152(weights=None)
    net.load_state_dict(torch.load(args.root / 'weights_vision/resnet152-f82ba261.pth', weights_only=True))
    net = torch.nn.Sequential(*list(net.children())[:8]).cuda().eval()
    pre = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(),
                              transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    out = args.root / 'features_resnet_d4'
    out.mkdir(exist_ok=True)
    remaining = sum(not all((out / f'{hashes[path]}_view{view}.pt').exists() for view in range(1, 8)) for path in paths)
    expected_bytes = remaining * 7 * (2048 * 16 * 16 * 2 + 4096)
    assert shutil.disk_usage(out).free > expected_bytes + 1024**3, 'Insufficient cache space plus1GiB safety margin'
    if not args.limit and remaining == 0 and (out / 'status.json').exists() and json.loads((out / 'status.json').read_text()).get('state') == 'COMPLETE':
        print('Verified complete cache; no extraction needed.', flush=True)
        return
    started = time.time()
    completed = 0
    status = {'state': 'EXTRACTING', 'training_images': total, 'completed_new_images': 0,
              'train_only': True, 'identity_view_source': 'features_resnet'}
    for index, path in enumerate(paths):
        destinations = [out / f'{hashes[path]}_view{view}.pt' for view in range(1, 8)]
        if all(p.exists() for p in destinations):
            continue
        with Image.open(path) as image:
            base = pre(image.convert('RGB'))
        # D4 identity is the already-verified view0. Other seven are real pixel
        # transformations BEFORE ResNet, not approximate feature-map rotations.
        variants = [torch.rot90(base, k, (1, 2)) for k in range(1, 4)]
        variants += [torch.rot90(torch.flip(base, (2,)), k, (1, 2)) for k in range(4)]
        with torch.inference_mode():
            features = net(torch.stack(variants).cuda()).cpu().half()
        assert features.shape == (7, 2048, 16, 16) and torch.isfinite(features).all()
        for value, dest in zip(features, destinations):
            temporary = dest.with_suffix('.tmp')
            # A batch slice shares its backing storage: clone before saving,
            # otherwise torch.save serializes all seven maps into EVERY file.
            torch.save(value.clone(), temporary)
            assert temporary.stat().st_size < 2 * value.numel() * value.element_size()
            temporary.replace(dest)
        completed += 1
        status = {'state': 'EXTRACTING', 'training_images': total, 'completed_new_images': completed,
                  'last_index': index, 'seconds': time.time() - started,
                  'seconds_per_image_seven_views': (time.time() - started) / completed,
                  'train_only': True, 'identity_view_source': 'features_resnet',
                  'encoder_mode': 'frozen eval; pretrained BN; pixel-space D4 before encoder',
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (out / 'status.json').write_text(json.dumps(status, indent=2))
        print(index, total, status['seconds_per_image_seven_views'], flush=True)
    status['state'] = 'SMOKE_COMPLETE' if args.limit else 'COMPLETE'
    (out / 'status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
