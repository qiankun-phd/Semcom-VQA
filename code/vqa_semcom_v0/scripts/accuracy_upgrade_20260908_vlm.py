#!/usr/bin/env python3
"""Equal-NF4 Qwen2.5 7B/3B controlled validation pilot, no cached-answer edits."""
from __future__ import annotations
import argparse
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
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from huggingface_hub import snapshot_download

def dump(path: Path,obj: object) -> None:
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def key(row: dict) -> tuple:
    return row['image_id'],row['question'],int(float(row['sensed_snr_db']))

def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.repo/'src'))
    from vqa_semcom.evidence.builder import build_vlm_prompt
    from vqa_semcom.vlm.answer import check_answer
    started=time.time()
    root=args.repo/'outputs/accuracy_upgrade_20260908'
    inventory=json.loads((root/'phase_a/validation_pilot_keys.json').read_text())
    allowed={(k['image'],k['question'],k['snr']) for k in inventory['keys']}
    rows={};hashes={}
    protocol={'pid':os.getpid(),'started_unix':started,'model_ids':['Qwen/Qwen2.5-VL-7B-Instruct','Qwen/Qwen2.5-VL-3B-Instruct'],
        'precision':'both NF4 double-quantized linear layers, BF16 compute and nonquantized modules',
        'min_pixels':200704,'max_pixels':802816,'max_new_tokens':24,'do_sample':False,
        'device_map':{'':0},'attn_implementation':'sdpa','max_images':8,'max_decisions_per_model':120,
        'budget':'after first7Bimage 15questions: n_images=min(8,floor(3000/(2*first_image_seconds))); require n>=2; hard3600s total',
        'scope':'validation-only, same JPEG/question/prompt/scoring; whole models under matched quantization, not official BF16 benchmark',
        'no_silent_resizing_on_OOM':True,'script_sha256':sha(Path(__file__)),'inventory_sha256':sha(root/'phase_a/validation_pilot_keys.json')}
    dump(args.out/'protocol.json',protocol)
    try:
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if active:raise RuntimeError(f'GPU occupied; no preemption {active}')
        for stem in ('main','cmp','extra'):
            path=args.repo/f'outputs/vlm/v25_rician_{stem}_predictions.csv';hashes[str(path)]=sha(path)
            with path.open() as f:
                for row in csv.DictReader(f):
                    if row['service_level']!='2' or key(row) not in allowed:continue
                    k=key(row)
                    if k in rows:
                        for field in ('ground_truth_answer','predicted_answer','correct','question_type','image_path','snr_bin','channel_bin'):
                            if row[field]!=rows[k][field]:raise ValueError(f'Conflicting duplicate {k} {field}')
                    else:rows[k]=row
        assert set(rows)==allowed and len(rows)==120
        audited=json.loads((args.repo/'outputs/revision_20260907_independent/crossreceiver_v2/receiver_image_sha256.json').read_text())
        ordered=[]
        for k in inventory['keys']:
            r=rows[(k['image'],k['question'],k['snr'])].copy()
            path=Path(r['image_path']);digest=sha(path)
            assert digest==audited[str(path)]
            hashes[str(path)]=digest
            cached=check_answer(r['question_type'],r['predicted_answer'],r['ground_truth_answer'])
            assert cached.correct==(r['correct'].lower()=='true')
            r['prompt']=f"service_level=2 snr_bin={r['snr_bin']} channel={r['channel_bin']} evidence_source=image\n"+build_vlm_prompt(r)
            ordered.append(r)
        for path in (args.repo/'src/vqa_semcom/evidence/builder.py',args.repo/'src/vqa_semcom/vlm/answer.py'):
            hashes[str(path)]=sha(path)
        dump(args.out/'source_sha256.json',hashes)
        dump(args.out/'frozen_tasks.json',ordered)
        loadstate=json.loads((root/'download_status.json').read_text())
        if loadstate['state']!='COMPLETE':raise RuntimeError('7B checkpoint acquisition incomplete')
        config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        model_paths=[loadstate['path'],snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct',local_files_only=True)]
        n_images=None;allresults={};modelmeta={};grids={}
        for label,path in zip(('7b_nf4','3b_nf4'),model_paths):
            begin=time.perf_counter()
            model=Qwen2_5_VLForConditionalGeneration.from_pretrained(path,local_files_only=True,
                torch_dtype=torch.bfloat16,quantization_config=config,device_map={'':0},attn_implementation='sdpa').eval()
            processor=AutoProcessor.from_pretrained(path,local_files_only=True,use_fast=False,min_pixels=200704,max_pixels=802816)
            modelmeta[label]={'load_seconds':time.perf_counter()-begin,'model_class':type(model).__name__,
                'memory_footprint_bytes':model.get_memory_footprint(),'processor':processor.to_dict(),
                'generation_config':model.generation_config.to_dict(),'device_map':{k:str(v) for k,v in model.hf_device_map.items()},
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
            dump(args.out/'model_metadata.json',modelmeta)
            results=[];image_begin=time.perf_counter()
            for index,row in enumerate(ordered):
                if n_images is not None and index>=n_images*15:break
                if time.time()-started>3600:raise TimeoutError('GPU wall-clock budget reached; preserve partial outputs')
                messages=[{'role':'user','content':[{'type':'image','image':row['image_path']},{'type':'text','text':row['prompt']}]}]
                tick=time.perf_counter()
                text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                vision,_=process_vision_info(messages)
                inputs=processor(text=[text],images=vision,padding=True,return_tensors='pt')
                grid=inputs['image_grid_thw'].tolist()
                if label=='7b_nf4':grids[index]=grid
                else:assert grid==grids[index],f'Processor image grid mismatch at {index}'
                with Image.open(row['image_path']) as im:received_size=list(im.size)
                prep_seconds=time.perf_counter()-tick
                inputs=inputs.to('cuda');torch.cuda.synchronize();tick=time.perf_counter()
                with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=24,do_sample=False)
                torch.cuda.synchronize();infer_seconds=time.perf_counter()-tick
                newids=[o[len(i):] for i,o in zip(inputs.input_ids,generated)]
                prediction=processor.batch_decode(newids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                check=check_answer(row['question_type'],prediction,row['ground_truth_answer'])
                result={'index':index,'image':row['image_id'],'question':row['question'],'qt':row['question_type'],'snr':int(float(row['sensed_snr_db'])),
                    'prediction':prediction,'normalized':check.normalized_prediction,'correct':check.correct,'ground_truth':row['ground_truth_answer'],
                    'cached_3b_prediction':row['predicted_answer'],'cached_3b_correct':row['correct'].lower()=='true',
                    'image_sha256':hashes[row['image_path']],'received_size':received_size,'vision_size':list(vision[0].size),'image_grid_thw':grid,
                    'prompt_sha256':hashlib.sha256(row['prompt'].encode()).hexdigest(),'prep_seconds':prep_seconds,'inference_seconds':infer_seconds,
                    'generated_tokens':len(newids[0]),'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
                results.append(result);dump(args.out/f'{label}_predictions.json',results)
                dump(args.out/'status.json',{'state':'RUNNING','model':label,'completed':len(results),'frozen_images':n_images,'seconds':time.time()-started})
                print(f'[{label}/{index+1}] inference={infer_seconds:.3f}s prediction={prediction!r}',flush=True)
                if label=='7b_nf4' and index==14:
                    elapsed=time.perf_counter()-image_begin;n_images=min(8,int(3000/(2*elapsed)))
                    if n_images<2:raise TimeoutError('First-image throughput cannot support2matchedimages under budget')
                    dump(args.out/'budget_selection.json',{'images':inventory['images'][:n_images],'decisions_per_model':n_images*15,
                        'first_7b_image_seconds':elapsed,'basis':'wall time only; no accuracy input','frozen_unix':time.time()})
            allresults[label]=results
            del model,processor;gc.collect();torch.cuda.empty_cache()
        assert len(allresults['7b_nf4'])==len(allresults['3b_nf4'])==n_images*15
        summary={label:{'n':len(records),'images':n_images,'accuracy':float(np.mean([r['correct'] for r in records])),
            'mean_inference_seconds':float(np.mean([r['inference_seconds'] for r in records])),
            'unknown_answers':sum(r['normalized']=='unknown' for r in records),
            'per_type':{qt:float(np.mean([r['correct'] for r in records if r['qt']==qt])) for qt in ('presence','counting','comparison','co_presence','threshold')}} for label,records in allresults.items()}
        dump(args.out/'summary.json',summary)
        dump(args.out/'status.json',{'state':'COMPLETE','seconds':time.time()-started,'images':n_images,'decisions_per_model':n_images*15})
    except Exception as exc:
        dump(args.out/'status.json',{'state':'BLOCKED_OR_INCOMPLETE','error':repr(exc),'seconds':time.time()-started})
        raise

if __name__=='__main__':main()
