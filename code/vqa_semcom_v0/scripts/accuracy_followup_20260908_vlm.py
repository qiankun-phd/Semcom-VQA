#!/usr/bin/env python3
"""Resumable full-validation NF4 comparison; reuses exact prior pilot answers."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from transformers import AutoProcessor,BitsAndBytesConfig,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import accuracy_upgrade_20260908_vlm as old

EXPECTED={'presence':552,'counting':276,'comparison':606,'co_presence':606,'threshold':606}
SETTINGS={'min_pixels':200704,'max_pixels':802816,'max_new_tokens':24,'do_sample':False,
          'quant_type':'nf4','double_quant':True,'compute_dtype':'bfloat16','attn':'sdpa','repetition_penalty':1.05}

def identity(row:dict) -> tuple:
    return row.get('image',row.get('image_id')),row['question'],int(float(row.get('snr',row.get('sensed_snr_db'))))

def gpu_idle():
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    # The driver may retain this process's CUDA context between model loads.
    other=[p.strip() for p in result.splitlines() if p.strip() and p.strip()!=str(os.getpid())]
    if other:raise RuntimeError(f'Other GPU work present; no preemption: {other}')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    sys.path.insert(0,str(args.repo/'src'))
    from vqa_semcom.evidence.builder import build_vlm_prompt
    from vqa_semcom.vlm.answer import check_answer
    prior=args.repo/'outputs/accuracy_upgrade_20260908';source=args.repo/'outputs/revision_20260907_independent/crossreceiver_v2'
    if args.resume:
        protocol=json.loads((args.out/'protocol.json').read_text())
        assert protocol['settings']==SETTINGS and protocol['script_sha256']==old.sha(Path(__file__))
    else:
        args.out.mkdir(parents=True,exist_ok=False)
        protocol={'settings':SETTINGS,'scope':'full common validation only','keys':2646,'images':101,
            'expected_type_rows':EXPECTED,'prior_exposure':'benchmark development; not pristine holdout',
            'script_sha256':old.sha(Path(__file__)),'max_gpu_wall_seconds':7200,'prior':str(prior),
            'old_decode_difference':'cached BF16 default sampling temperature1e-6; both new NF4 runs explicit greedy',
            'frozen_unix':time.time()}
        old.dump(args.out/'protocol.json',protocol)
    try:
        keys=[k for k in json.loads((source/'common_keys.json').read_text()) if k['split']=='validation']
        assert len(keys)==2646 and len({k['image'] for k in keys})==101
        assert dict(Counter(k['qt'] for k in keys))==EXPECTED
        allowed={(k['image'],k['question'],k['snr']) for k in keys};rows={};hashes={}
        for stem in ('main','cmp','extra'):
            path=args.repo/f'outputs/vlm/v25_rician_{stem}_predictions.csv';hashes[str(path)]=old.sha(path)
            with path.open() as f:
                for row in csv.DictReader(f):
                    if row['service_level']!='2' or old.key(row) not in allowed:continue
                    k=old.key(row)
                    if k in rows:
                        for field in ('ground_truth_answer','predicted_answer','correct','question_type','image_path','snr_bin','channel_bin'):
                            assert rows[k][field]==row[field],(k,field)
                    else:rows[k]=row
        assert set(rows)==allowed
        jpegmap=json.loads((source/'receiver_image_sha256.json').read_text());tasks=[];seen={}
        for index,k in enumerate(keys):
            row=rows[(k['image'],k['question'],k['snr'])].copy();path=Path(row['image_path'])
            if str(path) not in seen:
                seen[str(path)]=old.sha(path);assert seen[str(path)]==jpegmap[str(path)]
            prompt=f"service_level=2 snr_bin={row['snr_bin']} channel={row['channel_bin']} evidence_source=image\n"+build_vlm_prompt(row)
            row.update(index=index,image_sha256=seen[str(path)],prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
            tasks.append(row)
        old.dump(args.out/'coverage.json',{qt:{'n':sum(k['qt']==qt for k in keys),'images':len({k['image'] for k in keys if k['qt']==qt})} for qt in EXPECTED})
        hashes.update(seen);old.dump(args.out/'source_sha256.json',hashes)
        old.dump(args.out/'frozen_tasks.json',tasks)
        prior_protocol=json.loads((prior/'phase_c/protocol.json').read_text())
        for field in ('min_pixels','max_pixels','max_new_tokens','do_sample'):assert prior_protocol[field]==SETTINGS[field]
        assert json.loads((prior/'phase_c/status.json').read_text())['state']=='COMPLETE'
        manifests={name:json.loads((prior/f'{name}_checkpoint_manifest.json').read_text()) for name in ('7b','3b')}
        # Immutable checkpoint identity gate: hashes already verified in completed prior bundle;
        # validate every file again once when preparing this new run.
        if not args.resume:
            for name,manifest in manifests.items():
                for filename,meta in manifest['files'].items():
                    path=Path(manifest['snapshot'])/filename
                    assert path.stat().st_size==meta['bytes'] and old.sha(path)==meta['sha256'],(name,filename)
            old.dump(args.out/'checkpoint_gate.json',{'verified':True,'manifests':manifests})
        else:assert json.loads((args.out/'checkpoint_gate.json').read_text())['manifests']==manifests
        records={};reused={}
        taskmap={old.key(t):t for t in tasks}
        for label in ('7b_nf4','3b_nf4'):
            dest=args.out/label;dest.mkdir(exist_ok=True)
            previous=json.loads((prior/f'phase_c/{label}_predictions.json').read_text());reused[label]=len(previous)
            assert len(previous)==120
            for item in previous:
                k=identity(item);t=taskmap[k]
                assert item['image_sha256']==t['image_sha256'] and item['prompt_sha256']==t['prompt_sha256']
                assert item['ground_truth']==t['ground_truth_answer'] and item['qt']==t['question_type']
                check=check_answer(item['qt'],item['prediction'],item['ground_truth']);assert check.correct==item['correct']
                path=dest/f'{t["index"]:05d}.json'
                record={**item,'index':t['index'],'origin_index':item['index'],'reused':True,
                    'source_artifact':str(prior/f'phase_c/{label}_predictions.json')}
                if path.exists():assert json.loads(path.read_text())==record
                else:old.dump(path,record)
            records[label]={int(p.stem):json.loads(p.read_text()) for p in dest.glob('*.json')}
            for index,r in records[label].items():
                t=tasks[index];assert identity(r)==old.key(t) and r['image_sha256']==t['image_sha256'] and r['prompt_sha256']==t['prompt_sha256']
                assert check_answer(r['qt'],r['prediction'],r['ground_truth']).correct==r['correct']
        old.dump(args.out/'reuse_audit.json',{'counts':reused,'exact_task_prompt_jpeg_scoring':True,'checkpoint_files_verified':True})
        if args.prepare_only:
            old.dump(args.out/'status.json',{'state':'PREPARED','scope':'full2646keys verified,120answers/model reused; no GPU inference yet'});return
        started=time.time();prior_seconds=0
        if (args.out/'runtime.json').exists():prior_seconds=json.loads((args.out/'runtime.json').read_text()).get('gpu_wall_seconds',0)
        old.dump(args.out/'active_process.json',{'pid':os.getpid(),'started_unix':started,'prior_gpu_seconds':prior_seconds})
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        for label,name in (('7b_nf4','7b'),('3b_nf4','3b')):
            missing=[t for t in tasks if t['index'] not in records[label]]
            if not missing:continue
            gpu_idle();snapshot=manifests[name]['snapshot'];tick=time.perf_counter()
            torch.cuda.init();torch.cuda.reset_peak_memory_stats()
            model=Qwen2_5_VLForConditionalGeneration.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.bfloat16,
                quantization_config=quant,device_map={'':0},attn_implementation='sdpa').eval()
            processor=AutoProcessor.from_pretrained(snapshot,local_files_only=True,use_fast=False,min_pixels=200704,max_pixels=802816)
            old.dump(args.out/f'{label}_model.json',{'load_seconds':time.perf_counter()-tick,'model_footprint_bytes':model.get_memory_footprint(),
                'snapshot':snapshot,'generation_defaults':model.generation_config.to_dict(),'effective_overrides':{'do_sample':False,'max_new_tokens':24,'repetition_penalty':1.05},
                'processor':processor.to_dict()})
            for t in missing:
                elapsed=prior_seconds+time.time()-started
                if elapsed>=7200:raise TimeoutError('Frozen2h budget reached; retained partial full-set evaluation, no favorable subset chosen')
                tick=time.perf_counter();messages=[{'role':'user','content':[{'type':'image','image':t['image_path']},{'type':'text','text':t['prompt']}]}]
                text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                vision,_=process_vision_info(messages)
                inputs=processor(text=[text],images=vision,padding=True,return_tensors='pt');grid=inputs['image_grid_thw'].tolist()
                if t['index'] in records['7b_nf4']:assert grid==records['7b_nf4'][t['index']]['image_grid_thw']
                prep=time.perf_counter()-tick;inputs=inputs.to('cuda');torch.cuda.synchronize();tick=time.perf_counter()
                with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=24,do_sample=False,repetition_penalty=1.05)
                torch.cuda.synchronize();seconds=time.perf_counter()-tick
                ids=[o[len(i):] for i,o in zip(inputs.input_ids,generated)]
                prediction=processor.batch_decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                check=check_answer(t['question_type'],prediction,t['ground_truth_answer'])
                r={'index':t['index'],'image':t['image_id'],'question':t['question'],'qt':t['question_type'],'snr':int(float(t['sensed_snr_db'])),
                    'prediction':prediction,'normalized':check.normalized_prediction,'correct':check.correct,'ground_truth':t['ground_truth_answer'],
                    'cached_3b_prediction':t['predicted_answer'],'cached_3b_correct':t['correct'].lower()=='true','reused':False,
                    'image_sha256':t['image_sha256'],'prompt_sha256':t['prompt_sha256'],'image_grid_thw':grid,
                    'vision_size':list(vision[0].size),'prep_seconds':prep,'inference_seconds':seconds,'generated_tokens':len(ids[0]),
                    'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
                old.dump(args.out/label/f'{t["index"]:05d}.json',r);records[label][t['index']]=r
                runtime=prior_seconds+time.time()-started
                old.dump(args.out/'runtime.json',{'gpu_wall_seconds':runtime})
                old.dump(args.out/'status.json',{'state':'RUNNING','model':label,'completed':len(records[label]),'total':2646,
                    'reused':120,'new':len(records[label])-120,'gpu_wall_seconds':runtime})
                if len(records[label])%25==0:print(f'[{label}] completed={len(records[label])}/2646 new={len(records[label])-120} last={seconds:.3f}s wall={runtime:.1f}s',flush=True)
            del model,processor;gc.collect();torch.cuda.empty_cache()
        summary={}
        for label,recordmap in records.items():
            assert set(recordmap)==set(range(2646));rr=[recordmap[i] for i in range(2646)]
            summary[label]={'n':len(rr),'images':101,'reused':120,'new':2526,'accuracy':float(np.mean([r['correct'] for r in rr])),
                'new_inference_seconds_mean':float(np.mean([r['inference_seconds'] for r in rr if not r['reused']])),
                'all_inference_seconds_mean':float(np.mean([r['inference_seconds'] for r in rr])),
                'unknown':sum(r['normalized']=='unknown' for r in rr),
                'per_type':{qt:{'n':sum(r['qt']==qt for r in rr),'images':len({r['image'] for r in rr if r['qt']==qt}),
                    'accuracy':float(np.mean([r['correct'] for r in rr if r['qt']==qt]))} for qt in EXPECTED}}
        for i in range(2646):
            for field in ('image_sha256','prompt_sha256','image_grid_thw','ground_truth'):
                assert records['7b_nf4'][i][field]==records['3b_nf4'][i][field],(i,field)
        old.dump(args.out/'summary.json',summary)
        old.dump(args.out/'status.json',{'state':'COMPLETE','gpu_wall_seconds':prior_seconds+time.time()-started,
            'new_answers':5052,'reused_answers':240,'paired_keys':2646,'paired_hash_grid_scoring_gate':True})
    except Exception as exc:
        old.dump(args.out/'status.json',{'state':'BLOCKED_OR_INCOMPLETE','error':repr(exc),'pid':os.getpid()});raise

if __name__=='__main__':main()
