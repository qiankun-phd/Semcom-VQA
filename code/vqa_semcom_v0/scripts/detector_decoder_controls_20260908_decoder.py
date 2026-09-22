#!/usr/bin/env python3
"""Receiver-only count calibration controls; no detector or VLM inference."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
sys.path.append(str(Path(__file__).parent/'revision_20260907_independent'))
import revision_20260907 as base
import revision_20260907_crossreceiver as cross
from accuracy_upgrade_20260908 import CLASSES, parse_question

def packet(text: str) -> dict[str, int]:
    match = re.search(r'^detector_counts_by_class: (.*)$', text, re.M)
    if match is None:
        raise ValueError('No received class counts: do not substitute sender counts')
    result = {} if match[1] == 'none' else {k: int(v) for k,v in (p.strip().rsplit(':',1) for p in match[1].split(','))}
    if any(v < 0 for v in result.values()):
        raise ValueError('Negative packet count')
    return result

def predict(g: dict, counts: dict, calibration: dict, method: str) -> str:
    second, threshold = parse_question(g['question'],g['qt'],g['class'])
    def count(c: str) -> int:
        x = counts.get(c,0)
        model = calibration.get(f'{c}|{g["snr"]}')
        if method == 'baseline':
            return base.apply_ratio(x,model if model is not None else 1.) if g['qt']=='counting' else x
        if x == 0 or model is None:
            return x  # preserve zero evidence; cannot hallucinate an absent category
        if method == 'linear':
            value = model['slope']*x + model['intercept']
        else:
            value = float(np.interp(x,model['x'],model['y']))
        return max(0,round(value))
    a = count(g['class'])
    qt = g['qt']
    if qt == 'counting': return str(a)
    yes = a>0 if qt=='presence' else a>count(second) if qt=='comparison' else a>0 and count(second)>0 if qt=='co_presence' else a>=threshold
    return 'yes' if yes else 'no'

def correct(g: dict, answer: str) -> bool:
    return base.count_correct(int(answer),g['gt']) if g['qt']=='counting' else answer==g['answer'].lower()

def metrics(rows: list[dict], predictions: list[str]) -> dict:
    outcomes=np.array([correct(g,p) for g,p in zip(rows,predictions)],dtype=np.int8)
    return {'n':len(rows),'images':len({g['image'] for g in rows}),'accuracy':float(outcomes.mean()),
        'per_type':{q:{'n':sum(g['qt']==q for g in rows),'accuracy':float(outcomes[[g['qt']==q for g in rows]].mean())} for q in base.QTYPES}}

def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    dump=lambda name,obj:base.dump(args.out/name,obj)
    protocol={'pid':os.getpid(),'candidates':['linear','isotonic'],'baseline':'original class-SNR train ratio, counting only',
        'fit':'one sample per train image/SNR/class, all 10 task classes; train GT count targets only',
        'selection':'validation overall QA; tie linear first; retain baseline if best candidate fails to improve',
        'input':'counts parsed ONLY from received evidence_repr, class and SNR; source boxes never inputs',
        'logic':'same deterministic comparisons; candidates calibrate both classes/all five tasks, zeros preserved',
        'split':'existing crossreceiver_v2; test previously inspected, exploratory development not pristine holdout',
        'score':'unchanged yes/no exact; counting abs error <= max(1, round(.1*GT))',
        'energy':'no Joule claim; CPU batch timing separately; image branch unchanged',
        'source_sha256':base.sha(Path(__file__))}
    dump('protocol.json',protocol);dump('status.json',{'state':'RUNNING','pid':os.getpid()})
    try:
        common_path=args.repo/'outputs/revision_20260907_independent/crossreceiver_v2/common_keys.json'
        keys=json.loads(common_path.read_text());loaded,audit=cross.load_receiver(args.repo,'qwen2')
        rows=[loaded[(k['image'],k['question'],k['snr'])] for k in keys]
        counts=[packet(g['1']['evidence']) for g in rows]
        parts={s:[g for g in rows if g['split']==s] for s in cross.SPLITS}
        ratio=base.fit_calibration(parts['train'])
        mismatches=[]
        for g,c in zip(rows,counts):
            expected = base.labels([g],ratio)[0,0]
            got = correct(g,predict(g,c,ratio,'baseline'))
            if got!=expected: mismatches.append({**{k:g[k] for k in ('image','qt','question','snr')},'got':bool(got),'expected':bool(expected)})
        dump('baseline_reproduction.json',{'decisions':len(rows),'mismatches':mismatches})
        if mismatches: raise ValueError('Baseline packet replay does not reproduce cached decoder')
        gt={};sourcehash={str(common_path):base.sha(common_path)}
        for g in parts['train']:
            iid=g['image']
            if iid not in gt:
                path=args.repo/f'data/raw/visdrone/DET/val/annotations/{iid}.txt'
                sourcehash[str(path)]=base.sha(path);gt[iid]=Counter()
                for line in path.read_text().splitlines():
                    r=[int(float(v)) for v in line.strip().rstrip(',').split(',')]
                    if 1<=r[5]<=10:gt[iid][CLASSES[r[5]-1]]+=1
            if gt[iid][g['class']]!=g['gt']:
                raise ValueError(f'GT annotation/task count mismatch {iid}')
        samples={}
        for g,c in zip(rows,counts):
            if g['split']!='train':continue
            key=(g['image'],g['snr'])
            if key in samples and samples[key]!=c:raise ValueError('Received packet counts vary by question')
            samples[key]=c
        models={'baseline':ratio,'linear':{},'isotonic':{}}
        for cls in CLASSES:
            for snr in base.SNRS:
                xy=[(c.get(cls,0),gt[iid][cls]) for (iid,s),c in samples.items() if s==snr and c.get(cls,0)>0]
                if not xy:continue
                x,y=np.asarray(xy,dtype=float).T
                linear=LinearRegression(positive=True).fit(x[:,None],y)
                models['linear'][f'{cls}|{snr}']={'slope':float(linear.coef_[0]),'intercept':float(linear.intercept_),'n':len(x)}
                iso=IsotonicRegression(y_min=0,out_of_bounds='clip').fit(x,y)
                models['isotonic'][f'{cls}|{snr}']={'x':iso.X_thresholds_.tolist(),'y':iso.y_thresholds_.tolist(),'n':len(x)}
        dump('models.json',models);dump('source_sha256.json',{**sourcehash,**audit['source_sha256']})
        validation={}
        for name,model in models.items():
            p=[predict(g,packet(g['1']['evidence']),model,name) for g in parts['validation']]
            validation[name]=metrics(parts['validation'],p)
        selected=max(models,key=lambda n:validation[n]['accuracy'])
        dump('validation_selection.json',{'selected':selected,'metrics':validation,'frozen_before_test':True})
        results={};outcomes=[]
        for name,model in models.items():
            test=parts['test'];pc=[packet(g['1']['evidence']) for g in test]
            start=time.perf_counter()
            for _ in range(20):p=[predict(g,c,model,name) for g,c in zip(test,pc)]
            seconds=(time.perf_counter()-start)/20/len(test)
            results[name]={**metrics(test,p),'mean_cpu_seconds_per_query':seconds,'serialized_model_bytes':len(json.dumps(model).encode())}
            outcomes.extend([{**{k:g[k] for k in ('image','question','qt','snr','class','gt','answer')},'method':name,'prediction':v,'correct':correct(g,v)} for g,v in zip(test,p)])
        dump('test_outcomes.json',outcomes);dump('summary.json',{'selected':selected,'test':results,'validation':validation})
        dump('status.json',{'state':'COMPLETE','pid':os.getpid(),'test_previously_seen':True})
    except Exception as exc:
        dump('status.json',{'state':'FAILED','error':repr(exc),'pid':os.getpid()});raise

if __name__=='__main__':main()
