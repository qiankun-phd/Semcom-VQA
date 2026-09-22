#!/usr/bin/env python3
"""Read-only train/validation cache and validation-pilot diagnosis."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from transformers import AutoProcessor
from huggingface_hub import snapshot_download
from qwen_vl_utils import process_vision_info
import accuracy_upgrade_20260908 as a

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False);b=a.base
    sys.path.insert(0,str(args.repo/'src'))
    from vqa_semcom.vlm.answer import check_answer
    root=args.repo/'outputs/accuracy_upgrade_20260908';source=args.repo/'outputs/revision_20260907_independent/crossreceiver_v2'
    keys=[k for k in b.read(source/'common_keys.json') if k['split']!='test']
    keymap={(k['image'],k['question'],k['snr']):k for k in keys}
    report={};cached={};hashes={}
    for r,stems in {'qwen2':['v3_0_rician'],'qwen25':['v25_rician_main','v25_rician_cmp','v25_rician_extra'],
                    'smol':['v26_rician_main','v26_rician_cmp','v26_rician_extra']}.items():
        rows={}
        for stem in stems:
            path=args.repo/f'outputs/vlm/{stem}_predictions.csv';hashes[str(path)]=b.sha(path)
            with path.open() as f:
                for row in csv.DictReader(f):
                    if row['service_level']!='2':continue
                    k=(row['image_id'],row['question'],int(float(row['sensed_snr_db'])))
                    if k not in keymap:continue
                    if k in rows:
                        for field in ('predicted_answer','normalized_prediction','correct','ground_truth_answer','image_path'):
                            assert rows[k][field]==row[field],(r,k,field)
                    else:rows[k]=row
        assert set(rows)==set(keymap),(r,len(rows),len(keymap))
        cached[r]=rows
        report[r]={}
        for split in ('train','validation'):
            report[r][split]={}
            for qt in b.QTYPES:
                chosen=[row for k,row in rows.items() if keymap[k]['split']==split and keymap[k]['qt']==qt]
                checks=[check_answer(qt,row['predicted_answer'],row['ground_truth_answer']) for row in chosen]
                disagreement=[row for row,check in zip(chosen,checks) if check.correct!=(row['correct'].lower()=='true') or check.normalized_prediction!=row['normalized_prediction']]
                report[r][split][qt]={'n':len(chosen),'accuracy':float(np.mean([c.correct for c in checks])),
                    'unknown':sum(c.normalized_prediction=='unknown' for c in checks),'rescoring_mismatches':len(disagreement)}
                assert not disagreement,(r,split,qt)
    b.dump(args.out/'parser_audit.json',report)
    inventory=b.read(root/'phase_a/validation_pilot_keys.json');pilots=inventory['images']
    # Processor-only diagnosis: no new VLM inference and no test samples.
    snapshot=snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct',local_files_only=True)
    processor=AutoProcessor.from_pretrained(snapshot,local_files_only=True,use_fast=False,min_pixels=200704,max_pixels=802816)
    sizes=[]
    for k in inventory['keys']:
        row=cached['qwen25'][(k['image'],k['question'],k['snr'])];path=Path(row['image_path'])
        sourceimg=args.repo/f'data/raw/visdrone/DET/val/images/{k["image"]}.jpg'
        with Image.open(sourceimg) as im:original=list(im.size)
        with Image.open(path) as im:received=list(im.size)
        vision,_=process_vision_info([{'role':'user','content':[{'type':'image','image':str(path)}]}])
        processed=processor.image_processor(images=vision,return_tensors='np')
        sizes.append({'image':k['image'],'qt':k['qt'],'snr':k['snr'],'original_size':original,'received_size':received,
            'qwen_utils_size':list(vision[0].size),'processor_grid_thw':processed['image_grid_thw'].tolist(),'received_sha256':b.sha(path)})
    b.dump(args.out/'image_size_audit.json',sizes)
    # B original640/.25 counts should match canonical cache, modulo documented inference rounding effects.
    canonical=a.load_boxes(args.repo/'outputs/detector/v2_0_snr_detections.csv')
    base_rows=b.read(root/'phase_b/original_640_025.json');countaudit=[]
    for row in base_rows:
        old=Counter(x[0] for x in canonical.get(row['image'],[]));new=Counter(x['category'] for x in row['predictions'])
        countaudit.append({'image':row['image'],'cached_counts':dict(old),'rerun_counts':dict(new),'counts_equal':old==new})
    b.dump(args.out/'detector_cache_reproduction.json',countaudit)
    # Before-link semantic answers from each detector configuration: diagnostic only, no channel/calibration claim.
    diagnostic={}
    for name in ('original_640_025','resolution_1280_025','threshold_640_015','combined_1280_015'):
        det={r['image']:Counter(p['category'] for p in r['predictions']) for r in b.read(root/f'phase_b/{name}.json')}
        outcomes=[]
        unique={(k['image'],k['question']):k for k in inventory['keys']}
        for (iid,question),k in unique.items():
            row=cached['qwen25'][(iid,question,k['snr'])];second,threshold=a.parse_question(question,k['qt'],k['class'])
            count=det[iid][k['class']];other=det[iid][second] if second else 0
            if k['qt']=='counting':answer=str(count)
            else:
                flag={'presence':count>0,'comparison':count>other,'co_presence':count>0 and other>0,
                      'threshold':count>=threshold if threshold is not None else False}[k['qt']]
                answer='yes' if flag else 'no'
            checked=check_answer(k['qt'],answer,row['ground_truth_answer'])
            outcomes.append({'image':iid,'question':question,'qt':k['qt'],'answer':answer,'correct':checked.correct,'gt':row['ground_truth_answer']})
        diagnostic[name]={'n_unique_questions':len(outcomes),'accuracy':float(np.mean([r['correct'] for r in outcomes])),
            'per_type':{qt:float(np.mean([r['correct'] for r in outcomes if r['qt']==qt])) for qt in b.QTYPES},'rows':outcomes,
            'scope':'sender-side uncalibrated/no-channel symbolic-answer diagnostic, NOT received branch or end-to-end VQA'}
    b.dump(args.out/'sender_no_channel_qa_diagnostic.json',diagnostic)
    b.dump(args.out/'source_sha256.json',hashes)
    b.dump(args.out/'status.json',{'state':'COMPLETE','scope':'train/validation only; no inference or cache edits','parser_mismatch':0})

if __name__=='__main__':main()
