#!/usr/bin/env python3
"""Frozen visual feature extraction with explicit upstream substitutions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import torch
from torchvision import models, transforms
from torchvision.ops import roi_align
from PIL import Image


def identity(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--kind', choices=['regions', 'resnet'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device == 'cuda':
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if active:
            raise RuntimeError(f'GPU occupied; no preemption: {active}')
    weights = args.root / 'weights_vision'
    manifest = json.loads((weights / 'download_status_vision.json').read_text())
    assert manifest['state'] == 'COMPLETE'
    rows = sum([json.loads((args.root / 'data' / f'{s}.json').read_text()) for s in ['train', 'validation', 'test']], [])
    paths = sorted({r['original_image'] for r in rows})
    hashes = json.loads((args.root / 'data/image_sha256.json').read_text())
    if args.kind == 'resnet':
        paths = sorted(set(paths) | {r['received_image'] for r in rows})
        # Identical JPEG bytes under question/cache-specific paths need one extraction.
        paths = sorted({hashes[path]: path for path in paths}.values())
    all_count = len(paths)
    if args.limit:
        paths = paths[:args.limit]
    out = args.root / f'features_{args.kind}'
    out.mkdir(exist_ok=True)
    captured = {}
    if args.kind == 'regions':
        net = models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        net.load_state_dict(torch.load(weights / 'fasterrcnn_resnet50_fpn_coco-258fb6c6.pth', weights_only=True, map_location='cpu'))
        net.backbone.body.layer4.register_forward_hook(lambda module, inputs, output: captured.update(c5=output))
        pre = transforms.ToTensor()
        description = 'COCO FasterRCNN ResNet50 FPN top100 RPN proposals; 2048-D C5 RoIAlign mean features, NOT original VisualGenome bottom-up extractor'
    else:
        net = models.resnet152(weights=None)
        net.load_state_dict(torch.load(weights / 'resnet152-f82ba261.pth', weights_only=True, map_location='cpu'))
        net = torch.nn.Sequential(*list(net.children())[:8])
        pre = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(),
                                  transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        description = 'RSVQA original frozen ImageNet1K V1 ResNet152 stage4 at512x512; 2048x16x16 maps'
    net.to(args.device).eval()
    started = time.time()
    times = []
    record = {'state': 'EXTRACTING', 'requested': len(paths), 'total_images': all_count,
              'feature_method': description, 'device': args.device, 'weights': manifest}
    for index, path in enumerate(paths):
        dest = out / ((hashes[path] if args.kind == 'resnet' else identity(path)) + '.pt')
        if dest.exists():
            continue
        tick = time.perf_counter()
        with Image.open(path) as im:
            tensor = pre(im.convert('RGB')).to(args.device)
        with torch.inference_mode():
            if args.kind == 'regions':
                images, _ = net.transform([tensor], None)
                features = net.backbone(images.tensors)
                proposals, _ = net.rpn(images, features, None)
                boxes = proposals[0][:100]
                value = roi_align(captured['c5'], [boxes], output_size=(1, 1), spatial_scale=1 / 32, aligned=False).flatten(1)
                assert value.shape == (100, 2048), value.shape
            else:
                value = net(tensor.unsqueeze(0))[0]
                assert value.shape == (2048, 16, 16), value.shape
            assert torch.isfinite(value).all()
            torch.save(value.cpu().half(), dest)
        seconds = time.perf_counter() - tick
        times.append(seconds)
        record = {'state': 'EXTRACTING', 'last_index': index, 'requested': len(paths), 'total_images': all_count,
                  'feature_method': description, 'device': args.device, 'seconds': time.time() - started,
                  'mean_seconds_new_image': sum(times) / len(times), 'new_images': len(times),
                  'input': path, 'weights': manifest, 'no_ground_truth_features': True}
        (out / 'status.json').write_text(json.dumps(record, indent=2))
        print(index, path, seconds, flush=True)
    record['state'] = 'SMOKE_COMPLETE' if args.limit else 'COMPLETE'
    (out / 'status.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
