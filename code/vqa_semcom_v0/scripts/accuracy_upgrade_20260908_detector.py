#!/usr/bin/env python3
"""Small validation-only detection diagnosis, not cached-channel VQA evaluation."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from ultralytics import YOLO

CLASSES=('pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','others')
VARIANTS={'original_640_025':(640,.25),'resolution_1280_025':(1280,.25),
          'threshold_640_015':(640,.15),'combined_1280_015':(1280,.15)}

def dump(path: Path, obj: object) -> None:
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def iou(a: list,b: list) -> float:
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    denom=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/denom if denom>0 else 0

def evaluate(pred: list, gt: list) -> dict:
    result={}
    for c in CLASSES[:10]:
        p=sorted([v for v in pred if v['category']==c],key=lambda v:-v['confidence'])
        g=[v for v in gt if v['category']==c]
        used=set(); tp=0
        for box in p:
            pairs=[(iou(box['xyxy'],other['xyxy']),j) for j,other in enumerate(g) if j not in used]
            if pairs and max(pairs)[0]>=.5:
                used.add(max(pairs)[1]);tp+=1
        overlap=sum(iou(a['xyxy'],b['xyxy'])>.7 for i,a in enumerate(p) for b in p[i+1:])
        result[c]={'predicted_count':len(p),'gt_count':len(g),'true_positive_iou05':tp,
            'unmatched_predicted':len(p)-tp,'missed_gt':len(g)-tp,'same_class_overlap_pairs_iou07':overlap,
            'count_error':len(p)-len(g)}
    return result

def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    weights=args.repo/'outputs/detector/visdrone_yolov8n/weights/best.pt'
    inventory=args.repo/'outputs/accuracy_upgrade_20260908/phase_a/validation_pilot_keys.json'
    images=json.loads(inventory.read_text())['images']
    dump(args.out/'protocol.json',{'variants':VARIANTS,'images':images,'split':'validation','pid':__import__('os').getpid(),
        'script_sha256':sha(Path(__file__)),'weights_sha256':sha(weights),'inventory_sha256':sha(inventory),
        'metric':'class-aware greedy confidence-order one-to-one IoU.5; count errors; not official AP',
        'limitations':'10 known task classes only; ignored-region suppression not applied; overlapping pairs are a duplicate proxy, not verified duplicates',
        'channel_replay':False,'VQA_gain_claim_permitted':False,'imgsz_conf_only_changes':True,'iou':.7,'max_det':300,
        'energy':'No energy claim; GPU power sensor unavailable in preflight; old energy not reused'})
    try:
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if active:raise RuntimeError(f'GPU occupied; will not preempt {active}')
        tick=time.perf_counter();model=YOLO(str(weights));load_seconds=time.perf_counter()-tick
        assert tuple(model.names[i] for i in range(11))==CLASSES,model.names
        records={};sourcehash={}
        for name,(size,conf) in VARIANTS.items():
            # Same first validation image for warmup; excluded from timed rows.
            model.predict(str(args.repo/f'data/raw/visdrone/DET/val/images/{images[0]}.jpg'),device=0,imgsz=size,conf=conf,iou=.7,max_det=300,verbose=False)
            records[name]=[]
            for iid in images:
                image=args.repo/f'data/raw/visdrone/DET/val/images/{iid}.jpg'
                annotation=args.repo/f'data/raw/visdrone/DET/val/annotations/{iid}.txt'
                sourcehash[str(image)]=sha(image);sourcehash[str(annotation)]=sha(annotation)
                gt=[];ignored=0
                for line in annotation.read_text().splitlines():
                    row=[int(float(x)) for x in line.strip().rstrip(',').split(',')]
                    if row[5] in range(1,11) and row[4]>0:
                        x,y,w,h=row[:4];gt.append({'category':CLASSES[row[5]-1],'xyxy':[x,y,x+w,y+h]})
                    elif row[5]==0:ignored+=1
                torch.cuda.synchronize();tick=time.perf_counter()
                prediction=model.predict(str(image),device=0,imgsz=size,conf=conf,iou=.7,max_det=300,verbose=False)[0]
                torch.cuda.synchronize();seconds=time.perf_counter()-tick
                pred=[{'category':model.names[int(c)],'confidence':float(score),'xyxy':[float(v) for v in box]}
                    for box,score,c in zip(prediction.boxes.xyxy.cpu().tolist(),prediction.boxes.conf.cpu().tolist(),prediction.boxes.cls.cpu().tolist())]
                record={'image':iid,'seconds':seconds,'ultralytics_speed_ms':prediction.speed,'predictions':pred,
                    'ground_truth':gt,'ignored_regions':ignored,'metrics':evaluate(pred,gt)}
                records[name].append(record)
                dump(args.out/f'{name}.json',records[name])
                print(f'[{name}/{iid}] {seconds:.3f}s boxes={len(pred)}',flush=True)
        summary={}
        for name,rows in records.items():
            metrics=[m for row in rows for m in row['metrics'].values()]
            tp=sum(m['true_positive_iou05'] for m in metrics);p=sum(m['predicted_count'] for m in metrics);g=sum(m['gt_count'] for m in metrics)
            summary[name]={'images':len(rows),'prediction_count':p,'gt_count':g,'true_positive_iou05':tp,
                'precision_iou05':tp/p if p else 0,'recall_iou05':tp/g if g else 0,
                'count_mae_per_image_class':float(np.mean([abs(m['count_error']) for m in metrics])),
                'count_bias_per_image_class':float(np.mean([m['count_error'] for m in metrics])),
                'overlap_pairs_iou07':sum(m['same_class_overlap_pairs_iou07'] for m in metrics),
                'mean_seconds_per_image':float(np.mean([r['seconds'] for r in rows])),
                'per_class':{c:{'gt':sum(r['metrics'][c]['gt_count'] for r in rows),
                    'predicted':sum(r['metrics'][c]['predicted_count'] for r in rows),
                    'mae':float(np.mean([abs(r['metrics'][c]['count_error']) for r in rows]))} for c in CLASSES[:10]}}
        dump(args.out/'summary.json',summary);dump(args.out/'source_sha256.json',sourcehash)
        dump(args.out/'status.json',{'state':'COMPLETE','scope':'validation detector-only diagnostic','seconds':time.time()-started,'model_load_seconds':load_seconds})
    except Exception as exc:
        dump(args.out/'status.json',{'state':'FAILED_OR_BLOCKED','error':repr(exc),'seconds':time.time()-started});raise

if __name__=='__main__':main()
