#!/usr/bin/env python3
"""Train a VisDrone detector on the Colab VM with per-epoch checkpoint sync to Google Drive and auto-resume.

Recipe = the original command used on server 182 for the ground detector G (found in the session transcript):
    yolo detect train model=yolov8m.pt data=.../visdrone.yaml epochs=30 imgsz=640 batch=16 seed=0   (other args default)
Only `workers` differs (8 on 182, 2 here: the Colab VM has 2 vCPUs). The onboard detector O (yolov8n.pt) is trained
with the same recipe because its original recipe is not known - it is NOT a reproduction of the original O.

Usage:  python3 colab_train_det.py --model yolov8m.pt --name v8m_visdrone [--epochs 30] [--fraction 1.0]
Local run dir: /content/runs/<name>.  Drive: MyDrive/BUBBLES_hoverreport/runs/<name>/{last.pt,best.pt,results.csv,args.yaml}
and, when finished, MyDrive/BUBBLES_hoverreport/weights/<name>_best.pt plus runs/<name>/DONE.json.

BUB_ROOT (environment variable) replaces the Drive folder by another directory with the same layout. Used to train
before Drive is mounted (store on the VM disk, /content/local_store/...); colab_run_all_cell.py later stops such a run
and merges its store into Drive, where the run then resumes with normal Drive sync."""
import os, sys, json, shutil, argparse, time, hashlib

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--name', required=True)
ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--fraction', type=float, default=1.0)
a = ap.parse_args()

W = os.environ.get('BUB_DATA', '/content')   # working directory of the job (Colab VM: /content; a server: e.g. ~/bub_work)
WORKERS = int(os.environ.get('BUB_WORKERS', 2))   # dataloader workers: 2 on the Colab VM (2 vCPUs), 8 = the original recipe on 182
DATA = f'{W}/visdrone_yolo/visdrone.yaml'
PROJECT = f'{W}/runs'
DRIVE = os.environ.get('BUB_ROOT', '/content/drive/MyDrive/BUBBLES_hoverreport')
STORE = 'Drive' if DRIVE.startswith('/content/drive/') else 'local store'
if STORE != 'Drive':
    os.makedirs(f'{DRIVE}/weights', exist_ok=True)
LOCAL = f'{PROJECT}/{a.name}'
DRUN = f'{DRIVE}/runs/{a.name}'
assert os.path.exists(DATA), 'dataset not staged: run colab_convert_visdrone.py first'
assert os.path.isdir(DRIVE), 'Drive working folder not mounted'
assert STORE != 'Drive' or os.path.ismount('/content/drive'), 'Drive is not mounted'
os.makedirs(DRUN, exist_ok=True)

import torch, ultralytics
from ultralytics import YOLO
print(f'[train] {a.name}: model={a.model} epochs={a.epochs} fraction={a.fraction} | ultralytics {ultralytics.__version__} '
      f'torch {torch.__version__} {torch.cuda.get_device_name(0)}', flush=True)


def to_drive(src, dst_name):
    """copy via a temp name then rename, so an interrupted copy never replaces a good checkpoint"""
    if not os.path.exists(src):
        return False
    tmp = os.path.join(DRUN, dst_name + '.tmp')
    shutil.copyfile(src, tmp)
    os.replace(tmp, os.path.join(DRUN, dst_name))
    return True


def on_model_save(trainer):
    sd = str(trainer.save_dir)
    if os.path.exists(f'{DRUN}/last.pt'):   # keep the previous checkpoint: Drive uploads are asynchronous, and a VM
        os.replace(f'{DRUN}/last.pt', f'{DRUN}/last_prev.pt')   # reclaimed mid-upload could leave last.pt unreadable
    ok = to_drive(f'{sd}/weights/last.pt', 'last.pt')
    if ok:   # manifest of the checkpoint just stored, so a copy pulled elsewhere (the Mac) can be verified
        h = hashlib.sha256()
        with open(f'{sd}/weights/last.pt', 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        json.dump(dict(name=a.name, epochs_done=trainer.epoch + 1, epochs=trainer.epochs, bytes=os.path.getsize(f'{sd}/weights/last.pt'),
                       sha256=h.hexdigest()), open(f'{DRUN}/last.json.tmp', 'w'))
        os.replace(f'{DRUN}/last.json.tmp', f'{DRUN}/last.json')
    to_drive(f'{sd}/weights/best.pt', 'best.pt'); to_drive(f'{sd}/results.csv', 'results.csv'); to_drive(f'{sd}/args.yaml', 'args.yaml')
    line = f'[sync] {a.name} epoch {trainer.epoch + 1}/{trainer.epochs} -> {STORE} ({"ok" if ok else "last.pt missing"}) {time.strftime("%H:%M:%S")}'
    print(line, flush=True)
    with open(f'{W}/hb.log', 'a') as f:
        f.write(line + '\n')


if os.path.exists(f'{DRUN}/DONE.json'):
    print('[train] already finished according to Drive DONE.json - nothing to do', flush=True); sys.exit(0)

if os.path.exists(f'{DRUN}/last.pt') or os.path.exists(f'{DRUN}/last_prev.pt'):
    os.makedirs(f'{LOCAL}/weights', exist_ok=True)
    for f in ('results.csv', 'args.yaml'):
        if os.path.exists(f'{DRUN}/{f}'):
            shutil.copyfile(f'{DRUN}/{f}', f'{LOCAL}/{f}')
    ck = None
    for cand in ('last.pt', 'last_prev.pt'):   # newest first; fall back if the newest did not finish uploading
        if not os.path.exists(f'{DRUN}/{cand}'):
            continue
        try:
            shutil.copyfile(f'{DRUN}/{cand}', f'{LOCAL}/weights/last.pt')
            ck = torch.load(f'{LOCAL}/weights/last.pt', map_location='cpu', weights_only=False)   # our own checkpoint
            print(f'[train] using Drive checkpoint {cand} ({os.path.getsize(f"{LOCAL}/weights/last.pt"):,} bytes)', flush=True)
            break
        except Exception as e:
            print(f'[train] Drive checkpoint {cand} is unreadable ({type(e).__name__}: {str(e)[:120]})', flush=True)
    assert ck is not None, 'no readable checkpoint on Drive - inspect the run folder before training from scratch'
    ep, total = ck.get('epoch'), (ck.get('train_args') or {}).get('epochs', a.epochs)
    del ck
    if ep is None or ep < 0 or ep + 1 >= total:
        # interrupted after the last epoch but before DONE.json: nothing left to train, only finalise
        print(f'[train] Drive checkpoint is already at the final epoch (epoch index {ep}, epochs {total}) - finalising only', flush=True)
        if os.path.exists(f'{DRUN}/best.pt'):
            shutil.copyfile(f'{DRUN}/best.pt', f'{LOCAL}/weights/best.pt')
    else:
        rc = f'{LOCAL}/results.csv'   # drop rows of epochs after the checkpoint (they will be trained again)
        if os.path.exists(rc):
            rows = [r for r in open(rc).read().splitlines() if r.strip()]
            keep = rows[:1] + [r for r in rows[1:] if int(float(r.split(',')[0])) <= ep + 1]
            if len(keep) != len(rows):
                open(rc, 'w').write('\n'.join(keep) + '\n')
                print(f'[train] results.csv: dropped {len(rows) - len(keep)} row(s) newer than the checkpoint', flush=True)
        print(f'[train] RESUMING from the Drive checkpoint: {ep + 1} of {total} epochs done', flush=True)
        model = YOLO(f'{LOCAL}/weights/last.pt')
        model.add_callback('on_model_save', on_model_save)
        model.train(resume=True)
else:
    print('[train] starting from the official base weights', flush=True)
    model = YOLO(a.model)
    model.add_callback('on_model_save', on_model_save)
    model.train(data=DATA, epochs=a.epochs, imgsz=640, batch=16, device=0, workers=WORKERS, project=PROJECT, name=a.name,
                exist_ok=True, seed=0, fraction=a.fraction)

best = f'{LOCAL}/weights/best.pt'
m = YOLO(best).val(data=DATA, imgsz=640, batch=16, device=0, workers=WORKERS, plots=False, verbose=False)
done = dict(name=a.name, base=a.model, epochs=a.epochs, fraction=a.fraction, workers=WORKERS, map50=float(m.box.map50), map5095=float(m.box.map),
            precision=float(m.box.mp), recall=float(m.box.mr), ultralytics=ultralytics.__version__, torch=torch.__version__,
            finished=time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()))
to_drive(best, 'best.pt'); to_drive(f'{LOCAL}/results.csv', 'results.csv')
shutil.copyfile(best, f'{DRIVE}/weights/{a.name}_best.pt')
json.dump(done, open(f'{DRUN}/DONE.json', 'w'), indent=1)
print('[train] DONE', json.dumps(done), flush=True)
with open(f'{W}/hb.log', 'a') as f:
    f.write(f'[train] {a.name} DONE ({STORE}) mAP50 {done["map50"]:.4f} {time.strftime("%H:%M:%S")}\n')
