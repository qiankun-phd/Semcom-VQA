"""EXP-014 actual codec/GPU grid with immutable per-image restart journals.

No answer file is read here. Train/validation and test artifacts are separate.
The test phase requires all controller dependencies to be frozen first.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import re
import shutil
import sys
import time
from types import ModuleType

sys.dont_write_bytecode = True


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.tmp')
    part.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    part.replace(path)


def verify(hashes: dict) -> None:
    for name, digest in hashes.items():
        if not Path(name).is_file() or sha(Path(name)) != digest:
            raise ValueError(f'Frozen dependency changed: {name}')


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def old_modules(code: Path):
    package = ModuleType('exp014_source')
    package.__path__ = [str(code)]
    sys.modules[package.__name__] = package
    codec = load('exp014_source.encode_data', code / 'encode_data.py')
    frame = load('route_frame', code / 'route_frame.py')
    infer = load('exp014_source.infer_supervision', code / 'infer_supervision.py')
    return codec, frame, infer


def freeze(path: Path, fingerprint: dict) -> None:
    if path.exists():
        if read(path) != fingerprint:
            raise ValueError(f'Changed run fingerprint: {path.name}')
    else:
        save(path, fingerprint)
    verify(fingerprint['sha256'])


def controller_guard(output: Path) -> str:
    path = output / 'controller_frozen.json'
    control = read(path)
    if control.get('state') != 'FROZEN_BEFORE_TEST' or not control.get('sha256'):
        raise ValueError('Test processing requires frozen trained controllers')
    verify(control['sha256'])
    verify(control.get('source_inputs_sha256', {}))
    # Test must use the same downstream codec/receiver dependencies as trainval.
    for name in ('codec_frozen.json', 'inference_frozen.json'):
        frozen_path = output / 'grid_trainval' / name
        expected = control.get('source_inputs_sha256', {}).get(str(frozen_path.resolve()))
        if expected != sha(frozen_path):
            raise ValueError('Trainval model dependencies were not pinned by controller')
        frozen = read(frozen_path)
        if frozen.get('phase') != 'trainval' or not frozen.get('sha256'):
            raise ValueError('Invalid trainval model fingerprint')
        verify(frozen['sha256'])
    return sha(path)


def dataset_rows(output: Path, protocol_path: Path, phase: str) -> tuple[list, dict]:
    if phase not in ('trainval', 'test'):
        raise ValueError('Unknown dataset phase')
    control_hash = controller_guard(output) if phase == 'test' else None
    protocol = read(protocol_path)
    frozen = read(output / ('frozen_data.json' if phase == 'trainval' else 'test_data_frozen.json'))
    if frozen['protocol_sha256'] != sha(protocol_path):
        raise ValueError('Data protocol fingerprint mismatch')
    verify(frozen['sha256'])  # Data contract contains no truth paths here.
    splits = ('train', 'validation') if phase == 'trainval' else ('test',)
    rows, identities, images = [], set(), set()
    forbidden = {'answer', 'answers', 'label', 'labels', 'correct', 'ground_truth'}
    hashes_path = output / ('image_hashes.json' if phase == 'trainval' else 'test_image_hashes.json')
    image_hashes = read(hashes_path)
    for split in splits:
        manifest = output / f'{split}_manifest.json'
        # Manifest identity must be pinned by the original selection freeze.
        selection = read(output / 'frozen_data.json')
        expected_manifest = selection.get('test_manifest_sha256') if split == 'test' else selection['sha256'].get(str(manifest.resolve()))
        if expected_manifest != sha(manifest):
            raise ValueError('Unfrozen manifest')
        selected = read(manifest)
        expected = Counter({task: protocol['data']['per_type'][split] for task in protocol['data']['tasks']})
        if Counter(row['question_type'] for row in selected) != expected:
            raise ValueError('Frozen per-type quotas disagree')
        for row in selected:
            if forbidden.intersection(row) or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', row['id']):
                raise ValueError('Invalid label-free manifest row')
            if row['id'] in identities or row['image_id'] in images:
                raise ValueError('Repeated image or question identity')
            if sha(Path(row['file'])) != image_hashes[row['id']]:
                raise ValueError('Source JPEG changed')
            identities.add(row['id'])
            images.add(row['image_id'])
        selected.sort(key=lambda r: hashlib.sha256(f"{protocol['seed']}|{r['id']}".encode()).hexdigest())
        groups = {task: [r for r in selected if r['question_type'] == task] for task in sorted(expected)}
        rows.extend({**groups[task][i], 'split': split} for i in range(protocol['data']['per_type'][split]) for task in groups)
    return rows, {'sha256': {str(output / ('frozen_data.json' if phase == 'trainval' else 'test_data_frozen.json')): sha(output / ('frozen_data.json' if phase == 'trainval' else 'test_data_frozen.json')),
                            str(hashes_path): sha(hashes_path)}, 'controller_sha256': control_hash}


def disk_guard(output: Path, protocol: dict) -> None:
    if shutil.disk_usage(output).free < protocol['resources']['minimum_free_disk_gb'] * 1024**3:
        raise RuntimeError('Registered disk free-space floor reached')


def encode(args, rows: list, data_fingerprint: dict, old) -> dict:
    import torch
    from PIL import Image
    protocol, output = read(args.protocol), args.output
    directory = output / f'grid_{args.phase}'
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(protocol['resources']['codec_threads'])
    stage1 = old.load_stage1(args.stage1_root)
    region, _ = stage1.context()  # Never stage1.verify(): old sealed-test inputs.
    codec = region.alignment.parent.train.load_codec('cpu')
    codec.eval()
    fingerprint = old.dependency_snapshot(args.stage1_root, args.protocol, output)
    fingerprint['sha256'].update(data_fingerprint['sha256'])
    fingerprint['sha256'][str(Path(__file__).resolve())] = sha(Path(__file__))
    fingerprint.update(phase=args.phase, controller_sha256=data_fingerprint['controller_sha256'])
    freeze(directory / 'codec_frozen.json', fingerprint)
    all_keys = {(r['id'], b) for r in rows for b in protocol['budgets']}
    records = {}
    for path in sorted((directory / 'encoding_records').glob('*.json')):
        record = read(path)
        key = record['id'], record['budget']
        if key not in all_keys or key in records:
            raise ValueError('Foreign or repeated codec journal record')
        old.validate_record(record)
        records[key] = record
    complete_path = directory / 'encoding_complete.json'
    if complete_path.exists():
        done = read(complete_path)
        if done['representations_sha256'] != sha(directory / 'representations.json') or len(records) != len(all_keys):
            raise ValueError('Completed codec manifest changed')
        return done
    selected = rows[:6] if args.smoke else rows
    started, newly_encoded = time.monotonic(), 0
    for row in selected:
        disk_guard(output, protocol)
        if all((row['id'], budget) in records for budget in protocol['budgets']):
            continue
        start = time.perf_counter()
        with Image.open(row['file']) as source:
            tensor, hw = region.alignment.parent.train.image_tensor(source.convert('RGB'), 320, device='cpu')
        with torch.inference_mode():
            latent = codec.g_a(tensor)
        source_seconds = time.perf_counter() - start
        padded_hw = tuple(tensor.shape[-2:])
        for budget in protocol['budgets']:
            key = row['id'], budget
            if key in records:
                continue
            start = time.perf_counter()
            with torch.inference_mode():
                indices = torch.full((5, 5), 8, dtype=torch.int64)
                chosen = region.alignment.old.select_global_gain(lambda gain: region.alignment.encode_latent(
                    codec, latent, hw, padded_hw, indices, gain, region.alignment.parent.wire, aligned=True), budget, iterations=12)
            entropy_seconds = time.perf_counter() - start
            payload = chosen.payload
            if payload is None or not 0 < len(payload) <= budget:
                raise ValueError('Codec could not meet registered byte budget')
            start = time.perf_counter()
            with torch.inference_mode():
                rgb, header = region.alignment.decode(codec, payload, region.alignment.parent.wire)
                image = region.alignment.parent.canonical_reconstruction_image(rgb, header['image_hw'])
            decode_seconds = time.perf_counter() - start
            with torch.inference_mode():
                rgb2, header2 = region.alignment.decode(codec, payload, region.alignment.parent.wire)
                repeat = region.alignment.parent.canonical_reconstruction_image(rgb2, header2['image_hw'])
            pixels_sha = region.alignment.parent.reconstruction_sha256(image)
            if pixels_sha != region.alignment.parent.reconstruction_sha256(repeat):
                raise ValueError('Packet-only RGB roundtrip mismatch')
            folder = directory / 'decoded' / row['id']
            packet, png = folder / f'{budget}.bin', folder / f'{budget}.png'
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            old.publish_bytes(packet, payload)
            old.publish_bytes(png, buffer.getvalue())
            record = {'id': row['id'], 'budget': budget, 'codec_image_bytes': len(payload), 'image_bytes': len(payload)+1,
                      'packet': str(packet), 'packet_sha256': sha(packet), 'decoded': str(png), 'decoded_sha256': sha(png),
                      'pixels_sha256': pixels_sha, 'roundtrip': True, 'encode_seconds': source_seconds+entropy_seconds,
                      'decode_seconds': decode_seconds, 'global_gain': chosen.gain, 'route_bytes': 1,
                      'timing_scope': 'cached CPU codec component; not live end-to-end latency'}
            old.validate_record(record)
            save(directory / 'encoding_records' / f"{row['id']}-{budget}.json", record)
            records[key] = record
            newly_encoded += 1
        elapsed = time.monotonic()-started
        save(output / 'encoding_status.json', {'state': 'RUNNING', 'phase': args.phase, 'smoke': args.smoke,
             'done': len(records), 'total': len(all_keys), 'elapsed_seconds': elapsed,
             'estimated_remaining_seconds': (len(all_keys)-len(records))*elapsed/newly_encoded if newly_encoded else None})
    exported = [records[(r['id'], b)] for r in rows for b in protocol['budgets'] if (r['id'], b) in records]
    save(directory / 'representations.json', exported)
    if not args.smoke and len(exported) != len(all_keys):
        raise ValueError('Incomplete codec grid')
    verify(fingerprint['sha256'])
    result = {'state': 'SMOKE_COMPLETE' if args.smoke else 'COMPLETE', 'phase': args.phase,
              'representations': len(exported), 'representations_sha256': sha(directory / 'representations.json'),
              'protocol_sha256': sha(args.protocol), 'controller_sha256': data_fingerprint['controller_sha256'],
              'elapsed_seconds': time.monotonic()-started}
    save(directory / ('encoding_smoke_complete.json' if args.smoke else 'encoding_complete.json'), result)
    save(output / 'encoding_status.json', result)
    return result


def infer(args, rows: list, data_fingerprint: dict, old, frame) -> dict:
    protocol, output = read(args.protocol), args.output
    directory = output / f'grid_{args.phase}'
    runtime_path = args.grid_code / 'runtime.py'
    runtime = load('exp014_grid_runtime', runtime_path)
    records_path = output / ('supervision_records.json' if args.phase == 'trainval' else 'test_records.json')
    selected = rows[:6] if args.smoke else rows
    if not args.smoke:
        done = read(directory / 'encoding_complete.json')
        if done['representations_sha256'] != sha(directory / 'representations.json'):
            raise ValueError('Incomplete/changed codec grid')
    representations = old.verify_packets(read(directory / 'representations.json'), selected, protocol['budgets'], sha)
    adapter_weight = args.adapter / 'adapter_model.safetensors'
    if sha(adapter_weight) != protocol['receiver_sha256']:
        raise ValueError('Receiver adapter differs from registered frozen model')
    stage1 = runtime.load_stage1(args.stage1_root / 'run_stage1.py')
    _, qwen = stage1.context()
    paths = {args.protocol, Path(__file__).resolve(), runtime_path, adapter_weight,
             args.adapter / 'adapter_config.json', args.stage1_root / 'run_stage1.py',
             args.source_code / 'infer_supervision.py', args.source_code / 'route_frame.py'}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.suffix == '.py' and args.stage1_root.parent in path.parents and not any(
                    part in ('venv', '.venv', 'site-packages') for part in path.parts):
                paths.add(path)
    fingerprint = {'sha256': {str(p): sha(p) for p in sorted(paths)}, 'phase': args.phase,
                   'controller_sha256': data_fingerprint['controller_sha256']}
    fingerprint['sha256'].update(data_fingerprint['sha256'])
    freeze(directory / 'inference_frozen.json', fingerprint)
    allowed = {(r['id'], b, t['name']) for r in selected for b in protocol['budgets'] for t in protocol['tiers']}
    full_completion = output / ('supervision_complete.json' if args.phase == 'trainval' else 'test_inference_complete.json')
    if full_completion.exists():
        done = read(full_completion)
        if done['records_sha256'] != sha(records_path) or done['records'] != len(rows)*9:
            raise ValueError('Completed receiver outputs changed')
        return done
    records, completed = [], set()
    selected_ids = {r['id'] for r in selected}
    for path in sorted((directory / 'inference_records').glob('*.json')):
        cached = read(path)
        if not cached or any(r['id'] != cached[0]['id'] for r in cached) or len(cached) != 9:
            raise ValueError('Corrupt per-image inference journal')
        if cached[0]['id'] not in selected_ids:
            if args.smoke:
                continue
            raise ValueError('Foreign inference journal image')
        old.verify_cached_records(cached, allowed, representations, protocol['receiver_sha256'], sha(args.protocol))
        records.extend(cached)
        completed.add(cached[0]['id'])
    pending = [r for r in selected if r['id'] not in completed]
    started, new_predictions = time.monotonic(), 0
    if pending:
        import torch
        from PIL import Image
        import importlib.metadata
        torch.set_num_threads(2)
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required for VLM; CPU fallback forbidden')
        save(output / 'inference_status.json', {'state': 'LOADING_RECEIVER', 'phase': args.phase,
                                              'done': len(records), 'total': len(allowed)})
        model, processor = stage1.load_receiver(args.adapter, False)
        model.eval()
        random.seed(protocol['seed'])
        torch.manual_seed(protocol['seed'])
        runtime.freeze_or_verify(directory / 'model_runtime.json', {
            'model_config': model.config.to_dict(), 'gpu': torch.cuda.get_device_name(),
            'versions': {name: importlib.metadata.version(name) for name in ('torch','transformers','peft','bitsandbytes','Pillow')}})
        token = getattr(model.config, 'image_token_id', None)
        if token is None:
            token = processor.tokenizer.convert_tokens_to_ids(processor.image_token)

        def predict(row, representation, tier):
            runtime.set_pixel_target(processor, tier['pixels'])
            torch.cuda.synchronize()
            start = time.perf_counter()
            with Image.open(representation['decoded']) as source:
                image = source.convert('RGB')
            messages = [{'role':'user','content':[{'type':'image'},{'type':'text','text':qwen.prompt(row['question'])}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors='pt', min_pixels=tier['pixels'], max_pixels=tier['pixels'])
            measures = runtime.visual_measurements(inputs, processor, int(token))
            inputs = inputs.to('cuda')
            torch.cuda.synchronize()
            prep = time.perf_counter()-start
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=16)
            torch.cuda.synchronize()
            duration = time.perf_counter()-start
            continuation = generated[:,measures['input_tokens']:]
            return {'prediction':processor.batch_decode(continuation,skip_special_tokens=True)[0].strip(), **measures,
                    'receiver_seconds':duration,'preprocessing_seconds':prep,'generated_tokens':int(continuation.shape[1]),
                    'energy_j':None}

        warmups = [predict(pending[0], representations[(pending[0]['id'],4000)], tier) for tier in protocol['tiers']]
        counts = [r['actual_visual_tokens'] for r in warmups]
        if counts != sorted(set(counts)):
            raise ValueError('Visual tiers ineffective')
        save(directory / 'warmup_latest.json', {'excluded_from_metrics':True,'actual_visual_tokens':counts})
        tiers = {t['name']:t for t in protocol['tiers']}
        for index,row in enumerate(selected):
            if row['id'] in completed:
                continue
            disk_guard(output,protocol)
            image_records = []
            for budget,tier in runtime.configuration_order(protocol['budgets'],protocol['tiers'],protocol['seed'],index):
                representation = representations[(row['id'],budget)]
                payload = Path(representation['packet']).read_bytes()
                wire = frame.pack(payload,tier)
                if frame.unpack(wire) != (tier,payload) or len(wire) != representation['image_bytes']:
                    raise ValueError('Frame/tier byte accounting mismatch')
                result = predict(row,representation,tiers[tier])
                image_records.append({'id':row['id'],'image_id':row['image_id'],'split':row['split'],
                    'question_type':row['question_type'],'budget':budget,'tier':tier,**result,
                    **{key:representation[key] for key in ('image_bytes','codec_image_bytes','packet_sha256','decoded_sha256','encode_seconds','decode_seconds')},
                    'receiver_sha256':protocol['receiver_sha256'],'protocol_sha256':sha(args.protocol),
                    'frame_sha256':hashlib.sha256(wire).hexdigest()})
            save(directory / 'inference_records' / f"{row['id']}.json", image_records)
            records.extend(image_records)
            completed.add(row['id'])
            new_predictions += 9
            elapsed = time.monotonic()-started
            save(output / 'inference_status.json', {'state':'RUNNING','phase':args.phase,'smoke':args.smoke,
                 'done':len(records),'total':len(allowed),'elapsed_seconds':elapsed,
                 'estimated_remaining_seconds':(len(allowed)-len(records))*elapsed/new_predictions})
    old.verify_cached_records(records,allowed,representations,protocol['receiver_sha256'],sha(args.protocol))
    if len(records) != len(allowed):
        raise ValueError('Incomplete VLM grid')
    runtime.verify_tier_separation(records,protocol['tiers'])
    verify(fingerprint['sha256'])
    save(records_path,records)
    result = {'state':'SMOKE_COMPLETE' if args.smoke else 'SUPERVISION_COMPLETE','phase':args.phase,
              'records':len(records),'records_sha256':sha(records_path),'protocol_sha256':sha(args.protocol),
              'receiver_sha256':protocol['receiver_sha256'],'controller_sha256':data_fingerprint['controller_sha256'],
              'labels_loaded':False,'old_test300_opened':False,'elapsed_seconds':time.monotonic()-started}
    name = 'inference_smoke_complete.json' if args.smoke else ('supervision_complete.json' if args.phase == 'trainval' else 'test_inference_complete.json')
    save(output/name,result)
    save(output/'inference_status.json',result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output','protocol','source-code','stage1-root','grid-code','adapter'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=('encode','infer'),required=True)
    parser.add_argument('--phase',choices=('trainval','test'),default='trainval')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    for name in ('output','protocol','source_code','stage1_root','grid_code','adapter'):
        setattr(args,name,getattr(args,name).resolve())
    if args.smoke and args.phase != 'trainval':
        raise ValueError('Smoke is training-only')
    rows,fingerprint = dataset_rows(args.output,args.protocol,args.phase)
    codec,frame,inference = old_modules(args.source_code)
    result = encode(args,rows,fingerprint,codec) if args.stage == 'encode' else infer(args,rows,fingerprint,inference,frame)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
