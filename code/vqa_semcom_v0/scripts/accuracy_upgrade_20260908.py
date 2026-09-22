#!/usr/bin/env python3
"""Sender-only feature development; freezes validation choices before test use."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import re
import time
import joblib
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier, MLPRegressor
import router_network_revision_20260908 as base

CLASSES = ('pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor')
QUALITY = ['target_conf_mean','target_conf_max','target_conf_sd','target_conf_below_0.4',
           'target_area_mean','target_area_max','target_area_sum','all_count_per100','all_conf_mean','all_area_sum']
SEMANTICS = [f'second_class_{c}' for c in CLASSES] + ['has_second','has_threshold','explicit_negation',
    'primary_count_per60','second_count_per60','count_difference_per60','count_sum_per60',
    'count_min_per60','count_max_per60','primary_count_share','both_positive',
    'threshold_per60','threshold_margin_per60','threshold_met']
GROUPS = {'base18': [], 'quality': QUALITY, 'semantics': SEMANTICS, 'combined': QUALITY + SEMANTICS}
NETWORKS = {'qwen2':'wide_bce','qwen25':'advantage_mse','smol':'advantage_mse'}

def parse_question(text: str, qt: str, primary: str) -> tuple[str | None, int | None]:
    c = '(' + '|'.join(re.escape(c) for c in CLASSES) + ')'
    patterns = {'presence':rf'Are there {c} objects in this area\?',
        'counting':rf'How many {c} objects are in this area\?',
        'comparison':rf'Are there more {c} than {c} objects in this area\?',
        'co_presence':rf'Are there both {c} and {c} objects in this area\?',
        'threshold':rf'Are there at least (\d+) {c} objects in this area\?'}
    match = re.fullmatch(patterns[qt], text)
    if match is None:
        raise ValueError(f'Unsupported question template: {qt}: {text}')
    parts = match.groups()
    actual = parts[1] if qt == 'threshold' else parts[0]
    if actual != primary:
        raise ValueError(f'Primary class mismatch {actual} != {primary}')
    return (parts[1] if qt in ('comparison','co_presence') else None,
            int(parts[0]) if qt == 'threshold' else None)

def load_boxes(path: Path) -> dict:
    boxes = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row['category'] not in (*CLASSES, 'others'):
                raise ValueError('Unexpected detector category')
            boxes.setdefault(row['image_id'], []).append((row['category'], float(row['confidence']),
                float(row['bbox_w']) * float(row['bbox_h'])))
    return boxes

def enhanced_features(keys: list, boxes: dict, images: Path) -> np.ndarray:
    sizes, rows = {}, []
    for key in keys:
        iid, cls = key['image'], key['class']
        second, threshold = parse_question(key['question'], key['qt'], cls)
        if iid not in sizes:
            with Image.open(images / f'{iid}.jpg') as im:
                sizes[iid] = im.width * im.height
        records = boxes.get(iid, [])
        target = [r for r in records if r[0] == cls]
        conf = np.array([r[1] for r in target])
        areas = np.array([r[2] / sizes[iid] for r in target])
        quality = [float(conf.mean()) if len(conf) else 0, float(conf.max()) if len(conf) else 0,
            float(conf.std()) if len(conf) else 0, float((conf < .4).mean()) if len(conf) else 0,
            float(areas.mean()) if len(areas) else 0, float(areas.max()) if len(areas) else 0,
            float(areas.sum()), len(records)/100, float(np.mean([r[1] for r in records])) if records else 0,
            sum(r[2] for r in records)/sizes[iid]]
        a, b = len(target), sum(r[0] == second for r in records) if second else 0
        semantic = [int(second == c) for c in CLASSES] + [int(second is not None), int(threshold is not None), 0,
            a/60,b/60,(a-b)/60,(a+b)/60,min(a,b)/60,max(a,b)/60,a/(a+b+1),int(a>0 and b>0),
            threshold/60 if threshold is not None else 0,
            (a-threshold)/60 if threshold is not None else 0,
            int(a >= threshold) if threshold is not None else 0]
        rows.append(quality + semantic)
    result = np.asarray(rows, dtype=float)
    assert result.shape == (len(keys),34) and np.isfinite(result).all()
    return result

def scaler(train: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    mean, sd = train.mean(axis=0), train.std(axis=0)
    sd[sd == 0] = 1
    return mean, sd

def compose(x: np.ndarray, extra: np.ndarray, group: str, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    if group == 'base18':
        return x.copy()
    indices = [ (QUALITY + SEMANTICS).index(name) for name in GROUPS[group] ]
    result = np.concatenate([x, ((extra - mean) / sd)[:, indices]], axis=1)
    np.testing.assert_array_equal(result[:,:18], x)
    return result

def params(config: dict, dimension: int) -> int:
    widths = [dimension,*config['hidden'],1]
    return sum((a+1)*b for a,b in zip(widths,widths[1:])) * (2 if config['objective']=='dual_bce' else 1)

def fit(config: dict, data: dict, seed: int, dest: Path, epochs: int=300) -> tuple[list,dict]:
    dest.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    models, histories = [], []
    objective = config['objective']
    for head in range(2 if objective == 'dual_bce' else 1):
        tm, yt = base.targets(data['train_y'], objective, head)
        vm, yv = base.targets(data['validation_y'], objective, head)
        cls = MLPRegressor if objective == 'advantage_mse' else MLPClassifier
        model = cls(hidden_layer_sizes=tuple(config['hidden']), activation='relu', solver='adam',
            alpha=config['alpha'], batch_size=200, learning_rate_init=.001,
            early_stopping=False, random_state=seed, shuffle=True)
        best, best_loss, stale, history = None, float('inf'), 0, []
        for epoch in range(epochs):
            if objective == 'advantage_mse':
                model.partial_fit(data['train_x'][tm], yt)
                pred = model.predict(data['validation_x'][vm])
                loss = float(np.mean((pred-yv)**2))
            else:
                model.partial_fit(data['train_x'][tm], yt, classes=np.array([0,1]))
                pred = np.clip(base.probability(model,data['validation_x'][vm]),1e-12,1-1e-12)
                loss = float(-np.mean(yv*np.log(pred)+(1-yv)*np.log(1-pred)))
            assert np.isfinite(loss)
            history.append(loss)
            if loss < best_loss - 1e-4:
                best,best_loss,stale = copy.deepcopy(model),loss,0
                joblib.dump(best,dest/f'best_head_{head}.joblib')
            else:
                stale += 1
            if stale >= 10:
                break
        models.append(best)
        histories.append({'head':head,'epochs':len(history),'loss_history':history,'best_validation_loss':best_loss})
    seconds = time.perf_counter()-begin
    score = base.route_score(models,config,data['validation_x'])
    count = sum(a.size for m in models for a in [*m.coefs_,*m.intercepts_])
    assert count == params(config,data['train_x'].shape[1])
    record = {'seed':seed,'config':config,'dimension':data['train_x'].shape[1],'parameter_count':count,
        'training_seconds':seconds,'training':histories,'validation_accuracy':base.accuracy(data['validation_y'],score),
        'inference':base.benchmark(models,config,data['validation_x'])}
    joblib.dump(models,dest/'models.joblib')
    np.savez_compressed(dest/'validation_outcomes.npz',score=score,pick=(score>0).astype(np.int8))
    base.dump(dest/'record.json',record)
    return models,record

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    start=time.time()
    source=args.repo/'outputs/revision_20260907_independent/crossreceiver_v2'
    previous=args.repo/'outputs/router_network_revision_20260908'
    boxpath=args.repo/'outputs/detector/v2_0_snr_detections.csv'
    images=args.repo/'data/raw/visdrone/DET/val/images'
    protocol={'phase':'A','pid':os.getpid(),'started_unix':start,'smoke':args.smoke,
        'feature_groups':GROUPS,'base18_unchanged':True,'networks':NETWORKS,'configs':base.CONFIGS,
        'screen_seeds':[0,1,2],'final_seeds':list(range(10)),'max_epochs':300,'patience':10,'min_delta':1e-4,
        'selection':'mean full validation accuracy; exact ties fewer dimensions then name',
        'test_gate':'global selection written before loading test arrays; no retuning',
        'prior_test_exposure':True,'source':str(source),'script_sha256':base.sha(Path(__file__)),
        'base_script_sha256':base.sha(Path(base.__file__)), 'feature_source':str(boxpath),
        'semantics':'exact full-match question text only; primary class cross-check; unknown templates fail',
        'input_exclusions':['ground_truth','answers','received_count','correctness','annotation_count'],
        'scaling':'training fitted; appended columns only','threads':1,'lambda':0}
    base.dump(args.out/'protocol.json',protocol)
    base.dump(args.out/'status.json',{'state':'PREPARING'})
    try:
        data,hashes,keys=base.load_development(source)
        assert base.read(previous/'status.json')['state']=='COMPLETE'
        assert base.read(previous/'validation_selection.json')['selected']==NETWORKS
        hashes[str(boxpath)]=base.sha(boxpath)
        hashes[str(previous/'validation_selection.json')]=base.sha(previous/'validation_selection.json')
        base.dump(args.out/'source_sha256.json',hashes)
        boxes=load_boxes(boxpath)
        keysets={s:[k for k in keys if k['split']==s] for s in ('train','validation','test')}
        # Freeze validation pilot image/key inventory without labels or performance.
        valid_images=sorted({k['image'] for k in keysets['validation']})
        pilot_images=[iid for iid in valid_images if {k['qt'] for k in keysets['validation'] if k['image']==iid}==set(base.QTYPES)][:8]
        pilot=[]
        for iid in pilot_images:
            for qt in base.QTYPES:
                question=min(k['question'] for k in keysets['validation'] if k['image']==iid and k['qt']==qt)
                pilot.extend(k for k in keysets['validation'] if k['image']==iid and k['question']==question and k['snr'] in (-5,10,20))
        base.dump(args.out/'validation_pilot_keys.json',{'images':pilot_images,'keys':pilot,'scope':'validation only; first question per type; SNR -5,10,20'})
        tick=time.perf_counter()
        extra={s:enhanced_features(keysets[s],boxes,images) for s in ('train','validation')}
        feature_seconds=time.perf_counter()-tick
        mean,sd=scaler(extra['train'])
        base.dump(args.out/'scaler.json',{'columns':QUALITY+SEMANTICS,'mean':mean.tolist(),'sd':sd.tolist(),'fit_split':'train'})
        for split in ('train','validation'):
            x=data['qwen2'][f'{split}_x']
            counts=np.array([sum(r[0]==k['class'] for r in boxes.get(k['image'],[])) for k in keysets[split]])
            np.testing.assert_array_equal(x[:,16],np.minimum(counts,60)/60)
            np.testing.assert_array_equal(x[:,17],counts>0)
        base.dump(args.out/'feature_audit.json',{'safe_sender_count_assertion':'PASS train/validation all keys',
            'development_feature_seconds':feature_seconds,'rows':{s:len(v) for s,v in extra.items()},
            'dimensions':{g:18+len(cols) for g,cols in GROUPS.items()},'zero_sd_columns':[n for n,v in zip(QUALITY+SEMANTICS,extra['train'].std(axis=0)) if v==0]})
        screen={}
        completed=0
        for r in base.RECEIVERS:
            screen[r]={}
            cfg=base.CONFIGS[NETWORKS[r]]
            for group in GROUPS:
                screen[r][group]=[]
                augmented={**data[r],**{f'{s}_x':compose(data[r][f'{s}_x'],extra[s],group,mean,sd) for s in extra}}
                for seed in ([0] if args.smoke else [0,1,2]):
                    dest=args.out/'screen'/r/group/f'seed_{seed}'
                    _,record=fit(cfg,augmented,seed,dest,2 if args.smoke else 300)
                    if group=='base18' and not args.smoke:
                        with np.load(dest/'validation_outcomes.npz') as z, np.load(previous/'screen'/r/NETWORKS[r]/f'seed_{seed}'/'validation_outcomes.npz') as old:
                            np.testing.assert_array_equal(z['score'],old['score'])
                        record['prior_network_exact_reproduction']=True
                        base.dump(dest/'record.json',record)
                    screen[r][group].append(record)
                    completed+=1
                    base.dump(args.out/'status.json',{'state':'SCREENING','completed':completed,'expected':12 if args.smoke else 36})
                    print(f'[screen {r}/{group}/{seed}] seconds={record["training_seconds"]:.3f} validation={record["validation_accuracy"]:.6f}',flush=True)
        if args.smoke:
            base.dump(args.out/'status.json',{'state':'COMPLETE','scope':'2epoch validation-only smoke','seconds':time.time()-start})
            return
        selected={r:min(GROUPS,key=lambda g:(-np.mean([v['validation_accuracy'] for v in screen[r][g]]),len(GROUPS[g]),g)) for r in base.RECEIVERS}
        selection={'selected':selected,'frozen_unix':time.time(),'test_used':False,'screen_results':{
            r:{g:{'seeds':[v['validation_accuracy'] for v in records],'mean':float(np.mean([v['validation_accuracy'] for v in records])),
            'sd':float(np.std([v['validation_accuracy'] for v in records],ddof=1))} for g,records in methods.items()} for r,methods in screen.items()},
            'protocol_sha256':base.sha(args.out/'protocol.json')}
        base.dump(args.out/'validation_selection.json',selection)
        selection_hash=base.sha(args.out/'validation_selection.json')
        print(f'[selection frozen] {selected} sha={selection_hash}',flush=True)
        extra['test']=enhanced_features(keysets['test'],boxes,images)
        summary={}
        completed=0
        for r in base.RECEIVERS:
            cfg=base.CONFIGS[NETWORKS[r]]
            group=selected[r]
            augmented={**data[r],**{f'{s}_x':compose(data[r][f'{s}_x'],extra[s],group,mean,sd) for s in ('train','validation')}}
            with np.load(source/r/'evaluation_inputs.npz') as z:
                original_test=z['test_x'].copy()
                xt=compose(original_test,extra['test'],group,mean,sd)
                yt=z['test_y'].astype(np.int8)
            counts=np.array([sum(row[0]==k['class'] for row in boxes.get(k['image'],[])) for k in keysets['test']])
            np.testing.assert_array_equal(original_test[:,16],np.minimum(counts,60)/60)
            np.testing.assert_array_equal(original_test[:,17],counts>0)
            np.savez_compressed(args.out/f'{r}_evaluation_inputs.npz',**augmented,test_x=xt,test_y=yt)
            records=[]
            for seed in range(10):
                dest=args.out/'final'/r/f'seed_{seed}'
                models,record=fit(cfg,augmented,seed,dest)
                assert base.sha(args.out/'validation_selection.json')==selection_hash
                score=base.route_score(models,cfg,xt)
                record.update(test=base.test_summary(keysets['test'],yt,score),group=group,selection_sha256=selection_hash)
                base.dump(dest/'record.json',record)
                np.savez_compressed(dest/'test_outcomes.npz',score=score,pick=(score>0).astype(np.int8),y=yt)
                records.append(record)
                completed+=1
                base.dump(args.out/'status.json',{'state':'FINAL_EVALUATION','completed':completed,'expected':30})
                print(f'[final {r}/{seed}] seconds={record["training_seconds"]:.3f} test={record["test"]["accuracy"]:.6f}',flush=True)
            summary[r]={'group':group,'network':NETWORKS[r],'dimension':xt.shape[1],
                'parameter_count':records[0]['parameter_count'],'accuracy_mean':float(np.mean([v['test']['accuracy'] for v in records])),
                'accuracy_sd':float(np.std([v['test']['accuracy'] for v in records],ddof=1)),
                'all_seeds':records,'prior_network':base.read(previous/'summary.json')[r]}
            base.dump(args.out/'summary.json',summary)
        base.dump(args.out/'status.json',{'state':'COMPLETE','screen_fits':36,'final_fits':30,'seconds':time.time()-start,'selection_sha256':selection_hash})
    except Exception as exc:
        base.dump(args.out/'status.json',{'state':'FAILED','error':repr(exc),'seconds':time.time()-start})
        raise

if __name__=='__main__':
    main()
