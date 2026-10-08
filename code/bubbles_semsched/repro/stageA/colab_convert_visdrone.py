#!/usr/bin/env python3
"""Rebuild the YOLO-format VisDrone dataset on the Colab VM (the copy on server 182 is unreachable).

Rule (determined from the data, 2026-10-03): score == 0 occurs exactly for category 0 (ignored regions) and
category 11 (others); score == 1 exactly for categories 1..10. So: drop score == 0 rows, class = category - 1
(10 classes, 0..9), box -> normalised cx, cy, w, h. Zero-area boxes (3 in train) are dropped. No clipping.
This is the standard ultralytics VisDrone conversion; it was NOT checked against the label files on 182.

Output layout (same as on 182 so the experiment scripts run unmodified):
  /content/visdrone_yolo/{images,labels}/{train,val}/   and a symlink from
  /root/phd_research/uav-vqa-semantic-rl/data/processed/visdrone_yolo
Also writes visdrone.yaml, a conversion report, and the E1a split fingerprint; copies labels + report to Drive."""
import os, sys, json, glob, random, hashlib, zipfile, shutil, tarfile
from PIL import Image

BASE = os.environ.get('BUB_DATA', '/content')   # working directory of the job (Colab VM: /content; a server: e.g. ~/bub_work)
SRC = {'val': f'{BASE}/data/val_A.zip', 'train': f'{BASE}/data/train.zip'}
OUT = f'{BASE}/visdrone_yolo'
DRIVE = '/content/drive/MyDrive/BUBBLES_hoverreport'
NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
report = {}
for split, zp in SRC.items():
    tmp = f'{BASE}/_raw_{split}'
    if not os.path.isdir(tmp):
        zipfile.ZipFile(zp).extractall(tmp)
    root = glob.glob(f'{tmp}/VisDrone2019-DET-{split}')[0]
    os.makedirs(f'{OUT}/images/{split}', exist_ok=True); os.makedirs(f'{OUT}/labels/{split}', exist_ok=True)
    n_img = n_obj = n_drop_score = n_drop_area = n_empty = 0
    per_class = [0] * 10
    for ap in sorted(glob.glob(f'{root}/annotations/*.txt')):
        stem = os.path.splitext(os.path.basename(ap))[0]
        ip = f'{root}/images/{stem}.jpg'
        W, H = Image.open(ip).size
        rows = []
        for line in open(ap).read().strip().splitlines():
            r = line.strip().rstrip(',').split(',')
            x, y, w, h, score, cat = (int(v) for v in r[:6])
            if score == 0:
                n_drop_score += 1; continue
            if w <= 0 or h <= 0:
                n_drop_area += 1; continue
            assert 1 <= cat <= 10, (ap, line)
            rows.append(f'{cat - 1} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}')
            per_class[cat - 1] += 1
        open(f'{OUT}/labels/{split}/{stem}.txt', 'w').write('\n'.join(rows) + ('\n' if rows else ''))
        dst = f'{OUT}/images/{split}/{stem}.jpg'
        if not os.path.exists(dst):
            shutil.move(ip, dst)
        n_img += 1; n_obj += len(rows); n_empty += (len(rows) == 0)
    report[split] = dict(images=n_img, objects=n_obj, dropped_score0=n_drop_score, dropped_zero_area=n_drop_area,
                         images_without_objects=n_empty, per_class=dict(zip(NAMES, per_class)))
    print(split, report[split], flush=True)

open(f'{OUT}/visdrone.yaml', 'w').write(
    f'path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(NAMES)))

# E1a split fingerprint: sorted val images, random.Random(0).shuffle, first 80 = calibration, rest = test
allp = sorted(glob.glob(f'{OUT}/images/val/*.jpg'))
names = [os.path.basename(p) for p in allp]
random.Random(0).shuffle(names)
report['e1a_split'] = dict(n_cal=80, n_test=len(names) - 80, cal_first3=names[:3], test_first3=names[80:83],
                           sha256_of_ordered_names=hashlib.sha256('\n'.join(names).encode()).hexdigest())
print('split', report['e1a_split'], flush=True)

link = os.path.expanduser('~/phd_research/uav-vqa-semantic-rl/data/processed/visdrone_yolo')
os.makedirs(os.path.dirname(link), exist_ok=True)
if not os.path.lexists(link):
    os.symlink(OUT, link)
json.dump(report, open(f'{OUT}/conversion_report.json', 'w'), indent=1)
with tarfile.open(f'{BASE}/visdrone_yolo_labels.tar.gz', 'w:gz') as t:
    t.add(f'{OUT}/labels', arcname='labels'); t.add(f'{OUT}/visdrone.yaml', arcname='visdrone.yaml')
    t.add(f'{OUT}/conversion_report.json', arcname='conversion_report.json')
if os.path.isdir(DRIVE + '/data'):
    shutil.copy(f'{BASE}/visdrone_yolo_labels.tar.gz', DRIVE + '/data/visdrone_yolo_labels.tar.gz')
    shutil.copy(f'{OUT}/conversion_report.json', DRIVE + '/results/visdrone_conversion_report.json')
    shutil.copy(os.path.abspath(sys.argv[0]), DRIVE + '/scripts/colab_convert_visdrone.py')
    print('copied labels tarball, report and this script to Drive', flush=True)
print('CONVERT DONE', flush=True)
