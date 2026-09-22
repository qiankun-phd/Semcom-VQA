#!/usr/bin/env python3
"""Queued, bounded same-budget n/s pilot. Run only after explicit GPU release."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import yaml
sys.path.append(str(Path(__file__).parent/'revision_20260907_independent'))
import revision_20260907 as base
import revision_20260907_crossreceiver as cross
from detector_decoder_controls_20260908_decoder import packet,predict,metrics

def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    dump=lambda n,o:base.dump(args.out/n,o)
    dataset=args.repo/'data/processed/visdrone_yolo'
    paths=sorted((dataset/'images/train').glob('*.jpg'))
    fit=[p for p in paths if int(hashlib.sha256(p.stem.encode()).hexdigest()[:8],16)%10!=0]
    val=[p for p in paths if p not in fit]
    common=json.loads((args.repo/'outputs/revision_20260907_independent/crossreceiver_v2/common_keys.json').read_text())
    assert not {k['image'] for k in common}&{p.stem for p in paths}
    for name,items in [('fit',fit),('internal_val',val)]:
        (args.out/f'{name}.txt').write_text('\n'.join(str(p) for p in items)+'\n')
    names=yaml.safe_load((dataset/'visdrone.yaml').read_text())['names']
    (args.out/'dataset.yaml').write_text(yaml.safe_dump({'path':str(dataset),'train':str(args.out/'fit.txt'),'val':str(args.out/'internal_val.txt'),'names':names}))
    protocol={'models':['yolov8n','yolov8s'],'epochs':20,'imgsz':640,'batch':4,'seed':0,'optimizer':'AdamW','lr0':.001,
        'fit_images':len(fit),'internal_val_images':len(val),'official_val_overlap':0,'budget_seconds':7200,
        'selection':'best internal detector-val fitness, no official-val checkpoint selection',
        'inference':{'conf':.25,'iou':.7,'max_det':300},'decoder':'unchanged train-only multiplier counting, deterministic other tasks',
        'limitations':'pilot, one seed, old50epoch not a same-budget comparator; historical ignored-region conversion retained',
        'channel':'unchanged per-record loss simulation; text payload bytes recomputed; fixed nominal TOKEN_PAYLOAD_BYTES in existing outage model retained and disclosed',
        'power':'no energy estimate when power sensor N/A; no reuse old detector J',
        'script_sha256':base.sha(Path(__file__)),'mapping':names}
    if (args.out/'protocol.json').exists():
        old=json.loads((args.out/'protocol.json').read_text())
        hashes={'script_sha256','prepared_script_sha256'}
        canonical=lambda p:json.loads(json.dumps({k:v for k,v in p.items() if k not in hashes}))
        assert canonical(old)==canonical(protocol),'Scientific protocol changed after preparation'
        protocol['prepared_script_sha256']=old.get('prepared_script_sha256',old['script_sha256'])
    dump('protocol.json',protocol)
    if args.prepare_only:
        label_paths=[dataset/'labels/train'/f'{p.stem}.txt' for p in paths]
        dump('data_provenance.json',{'fit_images':[p.stem for p in fit],'internal_val_images':[p.stem for p in val],
            'label_sha256':{str(p):base.sha(p) for p in label_paths},
            'official_task_images':sorted({k['image'] for k in common}),
            'initialization_source':'official Ultralytics assets release v8.4.0, COCO weights used ONLY as initialization before VisDrone training',
            'source_module_sha256':{str(p):base.sha(p) for p in [args.repo/'src/vqa_semcom/detector/visdrone_yolo.py',args.repo/'src/vqa_semcom/degradation/digital_link.py']}})
        dump('status.json',{'state':'QUEUED_FOR_EXPLICIT_GPU_RELEASE','pid':os.getpid()});return
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError(f'GPU occupied; not preempting {active}')
    import torch
    from ultralytics import YOLO
    from accuracy_upgrade_20260908_detector import evaluate, CLASSES
    sys.path.insert(0,str(args.repo/'src'))
    from vqa_semcom.detector.visdrone_yolo import DetectionRecord,build_detector_lightweight_evidence
    started=time.time();dump('status.json',{'state':'RUNNING','pid':os.getpid(),'started':started})
    try:
        groups,_=cross.load_receiver(args.repo,'qwen2')
        rows=[groups[(k['image'],k['question'],k['snr'])] for k in common]
        config=json.loads((args.repo/'configs/v2_0_rician_cmp.json').read_text())
        results={}
        for name in protocol['models']:
            modeldir=args.out/name
            done=modeldir/'qa_summary.json'
            if done.exists():results[name]=json.loads(done.read_text());continue
            checkpoint=modeldir/'weights/best.pt'
            if checkpoint.exists():
                with (modeldir/'results.csv').open() as stream:
                    completed=list(csv.DictReader(stream))
                if len(completed)<20:
                    raise RuntimeError(f'{name} partial checkpoint has only {len(completed)} epochs; do not silently compare it')
            if not checkpoint.exists():
                pretrained=args.out/f'{name}.pt'
                model=YOLO(str(pretrained))
                initialhash=base.sha(pretrained)
                def budget(trainer):
                    elapsed=time.time()-started
                    if elapsed>protocol['budget_seconds']:
                        raise RuntimeError('Two-hour GPU pilot budget exhausted; incomplete pair must not be compared')
                    dump(f'{name}_progress.json',{'epoch':trainer.epoch+1,'elapsed':elapsed,'initial_sha256':initialhash})
                model.add_callback('on_train_epoch_end',budget)
                model.train(data=str(args.out/'dataset.yaml'),epochs=20,imgsz=640,batch=4,device=0,
                    workers=4,seed=0,optimizer='AdamW',lr0=.001,deterministic=True,patience=100,
                    project=str(args.out),name=name,exist_ok=False,plots=False,save=True,verbose=False)
            model=YOLO(str(checkpoint));assert model.names==names
            boxes={};timings=[]
            ids=sorted({g['image'] for g in rows})
            for iid in ids:
                image=args.repo/f'data/raw/visdrone/DET/val/images/{iid}.jpg'
                torch.cuda.synchronize();t=time.perf_counter()
                p=model.predict(str(image),imgsz=640,conf=.25,iou=.7,max_det=300,device=0,verbose=False)[0]
                torch.cuda.synchronize();timings.append(time.perf_counter()-t)
                boxes[iid]=[DetectionRecord(model.names[int(c)],max(0,round(x1)),max(0,round(y1)),max(1,round(x2-x1)),max(1,round(y2-y1)),round(float(score),4))
                    for (x1,y1,x2,y2),score,c in zip(p.boxes.xyxy.cpu().tolist(),p.boxes.conf.cpu().tolist(),p.boxes.cls.cpu().tolist())]
            base.dump(modeldir/'boxes.json',{i:[vars(b) for b in v] for i,v in boxes.items()})
            changed=copy.deepcopy(rows);bytesizes=[]
            for g in changed:
                task={'image_id':g['image'],'question_type':g['qt'],'target_class':g['class']}
                evidence=build_detector_lightweight_evidence(task,boxes[g['image']],f'{g["snr"]}dB',config)
                g['1']['evidence']=evidence;g['1']['transmitted']=packet(evidence).get(g['class'],0)
                bytesizes.append(len(evidence.encode()))
            ratio=base.fit_calibration([g for g in changed if g['split']=='train'])
            base.dump(modeldir/'qa_calibration.json',ratio)
            report={};outs=[]
            for split in ['validation','test']:
                subset=[g for g in changed if g['split']==split]
                answers=[predict(g,packet(g['1']['evidence']),ratio,'baseline') for g in subset]
                report[split]=metrics(subset,answers)
                detection_metrics=[]
                for iid in sorted({g['image'] for g in subset}):
                    annotation=args.repo/f'data/raw/visdrone/DET/val/annotations/{iid}.txt'
                    ground=[]
                    for line in annotation.read_text().splitlines():
                        r=[int(float(v)) for v in line.strip().rstrip(',').split(',')]
                        if 1<=r[5]<=10:
                            x,y,w,h=r[:4];ground.append({'category':CLASSES[r[5]-1],'xyxy':[x,y,x+w,y+h]})
                    predicted=[{'category':b.category,'confidence':b.confidence,'xyxy':[b.bbox_x,b.bbox_y,b.bbox_x+b.bbox_w,b.bbox_y+b.bbox_h]} for b in boxes[iid]]
                    detection_metrics.extend(evaluate(predicted,ground).values())
                tp=sum(v['true_positive_iou05'] for v in detection_metrics)
                n_pred=sum(v['predicted_count'] for v in detection_metrics);n_gt=sum(v['gt_count'] for v in detection_metrics)
                report[split]['detector_diagnostic']={'precision_iou05':tp/n_pred if n_pred else 0.,'recall_iou05':tp/n_gt if n_gt else 0.,
                    'count_mae_per_image_class':float(np.mean([abs(v['count_error']) for v in detection_metrics])),
                    'scope':'10 task classes; count annotations like task generator; ignored-region suppression absent; not official AP'}
                outs.extend([{**{k:g[k] for k in ('image','question','qt','snr','split','gt','answer')},'prediction':p,'received_evidence':g['1']['evidence']} for g,p in zip(subset,answers)])
                if split=='validation':base.dump(modeldir/'validation_frozen.json',{'metrics':report[split],'configuration_changed':False})
            report.update(checkpoint_sha256=base.sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
                parameters=sum(p.numel() for p in model.model.parameters()),mean_inference_seconds=float(np.mean(timings[1:])),
                received_text_bytes_mean=float(np.mean(bytesizes)),received_text_bytes_min=min(bytesizes),received_text_bytes_max=max(bytesizes),
                power_w=None,energy_j=None)
            base.dump(modeldir/'qa_outcomes.json',outs);base.dump(done,report);results[name]=report
            del model;torch.cuda.empty_cache()
        dump('summary.json',results);dump('status.json',{'state':'COMPLETE','seconds':time.time()-started})
    except Exception as exc:
        dump('status.json',{'state':'FAILED_OR_BUDGET_BLOCKED','error':repr(exc),'seconds':time.time()-started});raise

if __name__=='__main__':main()
