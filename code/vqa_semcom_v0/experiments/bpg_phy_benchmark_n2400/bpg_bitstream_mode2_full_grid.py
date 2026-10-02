"""Isolated BPG real-bitstream CQEM benchmark; configurable Colab root, no proxy."""
import argparse, ast, csv, gc, hashlib, importlib.util, io, json, math, os, fcntl
import multiprocessing as mp
from pathlib import Path
import random, statistics, struct, sys, time, zlib

LEGACY_ROOT=Path('/home/qiankun/phd_research/vqa_semcom')
ROOT=Path(os.environ.get('VQA_SEMCOM_ROOT',str(LEGACY_ROOT))).resolve()
DATA=ROOT/'outputs/rgb_selector_large_20260922'
PHY=ROOT/'outputs/prerun_protocol_finalization_20260908/phy.py'
ADAPTER=ROOT/'outputs/tdiuc_qlora_pilot_20260916/standard/final_adapter'
EXPECTED_ADAPTER='a2c9476840f0aa56553ff12f7eaec5be4b7b4df32df3c2d7865aea4ebe3c6a21'

def asset_path(path):
    path=Path(path)
    if path.is_relative_to(LEGACY_ROOT):return ROOT/path.relative_to(LEGACY_ROOT)
    return path

def prior_hash(protocol,path):
    path=Path(path)
    candidates=[str(path)]
    if path.is_relative_to(ROOT):candidates.append(str(LEGACY_ROOT/path.relative_to(ROOT)))
    if protocol.get('repo_root') and path.is_relative_to(ROOT):
        candidates.append(str(Path(protocol['repo_root'])/path.relative_to(ROOT)))
    return next(protocol['hashes_before'][k] for k in candidates if k in protocol['hashes_before'])

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def csvsave(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def journal_rows(path,repair=False):
    if not path.exists():return []
    data=path.read_bytes();lines=data.splitlines(keepends=True);rows=[];offset=0
    for i,line in enumerate(lines):
        # A crash may leave a final unterminated write. Interior damage is fatal.
        if not line.endswith(b'\n'):
            assert i==len(lines)-1
            if repair:
                with path.open('r+b') as f:f.truncate(offset)
            break
        rows.append(json.loads(line));offset+=len(line)
    return rows
def init_worker():
    global digital, realization, numpy_channel, bp_decode, tx_cache
    sys.path.insert(0,str(PHY.parent))
    from phy import Digital, realization, numpy_channel, bp_decode
    digital=Digital();tx_cache={}
def transmission(task):
    key,path,packet_sha,image,seed,snr=task
    import numpy as np
    if packet_sha not in tx_cache:
        payload=Path(path).read_bytes();assert hashlib.sha256(payload).hexdigest()==packet_sha
        _,_,tx=digital.encode(payload)
        # Bound each worker's RAM while allowing repeated seed/SNR tasks.
        if len(tx_cache)>32:tx_cache.clear()
        tx_cache[packet_sha]=(payload,tx)
    payload,tx=tx_cache[packet_sha];ns=len(tx)
    eff=snr+10*math.log10(21420/ns)
    h,noise=realization(str(image),seed,ns,'rician');rx=numpy_channel(tx,eff,h,noise)
    obs=np.stack([rx.real,rx.imag],-1).reshape(-1,digital.n)*np.sqrt(2)
    llrs=2*obs/(10**(-eff/10)/abs(h)**2)
    # Actual coded bits are sent. Stop on first bad CRC, no analytic SNR cutoff.
    parts=[];iters=0;blocks_decoded=0;crc_ok=True
    for i,llr in enumerate(llrs):
        decoded,it=bp_decode(llr,digital.check_edges,digital.edge_vars);iters+=it;blocks_decoded+=1
        raw=np.packbits(decoded[:digital.k]).tobytes()
        index,count,size=struct.unpack('!HHH',raw[2:8]);end=8+size
        good=(raw[:2]==b'V2' and index==i and count==len(llrs) and size<=48 and end+4<=len(raw) and zlib.crc32(raw[:end])==int.from_bytes(raw[end:end+4],'big'))
        if not good:crc_ok=False;break
        parts.append(raw[8:end])
    recovered=b''.join(parts) if crc_ok else None
    ok=recovered==payload
    if crc_ok:assert ok,'CRC collision or payload mismatch'
    return {'task_key':key,'packet_sha256':packet_sha,'image_id':image,'channel_seed':seed,'nominal_snr_db':snr,'effective_snr_db':eff,
      'actual_symbols':ns,'payload_bytes':len(payload),'success':ok,'payload_identity':ok,'blocks_decoded':blocks_decoded,'total_blocks':len(llrs),'bp_iterations':iters,'n':digital.n,'k':digital.k,'fade_power':float(abs(h)**2),'delivery':'real_bitstream_phy'}

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,choices=[6,60,2400],required=True);p.add_argument('--workers',type=int,default=12)
    p.add_argument('--resume',action='store_true');p.add_argument('--reuse-phy',type=Path,required=True,help='Completed JPEG2400 real-PHY output; learned baseline only')
    p.add_argument('--reuse-bpg-smoke',type=Path);p.add_argument('--bpgenc',default='bpgenc');p.add_argument('--bpgdec',default='bpgdec')
    p.add_argument('--max-new-vlm',type=int,default=4800,help='Hard clean-inference count limit, excluding one warmup')
    a=p.parse_args();assert 1<=a.workers<=12 and a.max_new_vlm>=0
    from verify_rule_b import verify_scanner
    from budgeted_bpg import fingerprints
    scanner=ROOT/'code/vqa_semcom_v0/experiments/rgb_channel_snr_scan/run_snr_scan.py'
    freeze=verify_scanner(scanner)
    binary_fingerprints=fingerprints(a.bpgenc,a.bpgdec)
    if a.output.exists() and not a.resume:raise SystemExit('Existing output requires explicit --resume')
    t0=time.monotonic();out=a.output;out.mkdir(parents=True,exist_ok=True);(out/'streams').mkdir(exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def progress(phase,**counts):
        save(out/'progress.json',{'phase':phase,'pid':os.getpid(),'n':a.n,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'elapsed_seconds':time.monotonic()-t0,**counts})
    progress('LOADING_INPUTS')
    snrs=[0.,5.,10.,20.];seeds=[1701,1702,1703];budgets=[2000,4000]
    manifest=json.loads((DATA/'test_manifest.json').read_text());groups={}
    for row in manifest:row['file']=str(asset_path(row['file']))
    for r in manifest:groups.setdefault(r['question_type'],[]).append(r)
    for g in groups.values():g.sort(key=lambda r:hashlib.sha256(('B-smoke-v1|'+r['id']).encode()).hexdigest())
    rows=[groups[t][i] for i in range(a.n//6) for t in sorted(groups)]
    assert len(rows)==a.n and len({r['image_id'] for r in rows})==a.n
    keys={(r['id'],b) for r in rows for b in budgets}
    learned={(r['id'],r['budget']):r for r in json.loads((DATA/'test_records.json').read_text()) if r['tier']=='medium' and (r['id'],r['budget']) in keys}
    reps={(r['id'],r['budget']):r for r in json.loads((DATA/'grid_test/representations.json').read_text()) if (r['id'],r['budget']) in keys}
    assert set(learned)==keys and set(reps)==keys
    for rep in reps.values():
        for field in ['packet','decoded']:rep[field]=str(asset_path(rep[field]))
    bpg_path=Path(__file__).with_name('budgeted_bpg.py')
    runtime_path=ROOT/'outputs/rgb_rate_visual_budget_20260922/code/runtime.py'
    stage_path=ROOT/'outputs/qwen_compression_joint_stage1_20260921/run_stage1.py'
    prompt_path=ROOT/'outputs/tdiuc_qlora_pilot_20260916/run_training.py'
    scanner=ROOT/'code/vqa_semcom_v0/experiments/rgb_channel_snr_scan/run_snr_scan.py'
    frame_path=ROOT/'outputs/rgb_joint_selector_20260922/code/route_frame.py'
    frame_module=load('real_route_frame',frame_path)
    paths=[Path(__file__),PHY,bpg_path,Path(__file__).with_name('verify_rule_b.py'),runtime_path,stage_path,prompt_path,frame_path,scanner,ADAPTER/'adapter_model.safetensors',
      DATA/'test_manifest.json',DATA/'test_records.json',DATA/'grid_test/representations.json',DATA/'test_truth.sealed.json']
    paths += [Path(x['path']) for x in binary_fingerprints.values()]
    paths += [a.reuse_phy/'protocol.json',a.reuse_phy/'transmissions.jsonl',a.reuse_phy/'report.json']
    if a.reuse_bpg_smoke:paths += [a.reuse_bpg_smoke/'protocol.json',a.reuse_bpg_smoke/'transmissions.jsonl',a.reuse_bpg_smoke/'clean_predictions.json']
    paths += [Path(r['file']) for r in rows]
    paths += [Path(reps[k][field]) for k in sorted(keys) for field in ['packet','decoded']]
    before={}
    for i,x in enumerate(paths):
        before[str(x)]=sha(x)
        if i%250==0:progress('HASHING_INPUTS',hashed=i+1,total=len(paths))
    assert before[str(ADAPTER/'adapter_model.safetensors')]==EXPECTED_ADAPTER
    baseline_protocol=json.loads((a.reuse_phy/'protocol.json').read_text())
    baseline_report=json.loads((a.reuse_phy/'report.json').read_text())
    assert baseline_report['status']=='PASS' and baseline_report['historical_proxy_used'] is False
    for dep in [PHY,runtime_path,stage_path,prompt_path,frame_path,scanner,ADAPTER/'adapter_model.safetensors',DATA/'test_records.json']:
        assert prior_hash(baseline_protocol,dep)==before[str(dep)],'JPEG baseline dependency changed: '+str(dep)
    v2_ast=freeze['rule_b_ast']
    protocol={'ids':[r['id'] for r in rows],'selection':'label-free per-type SHA256 B-smoke-v1; first n/6 each type',
      'n':a.n,'snrs':snrs,'seeds':seeds,'budgets':budgets,'tier':'medium','pixels':100352,'codecs':['bpg','learned_frozen'],
      'delivery':'real_bitstream_phy','energy':'fixed RF E0 proxy, common N_ref=21420; gamma_eff uses actual packet symbols for BOTH codecs',
      'radio_rounding':'no rounded cache proxy SNR','failed_packet':'strict incorrect, no VLM on failed payload','early_stop':'first failed CRC; no analytical SNR cutoff',
      'v2_ast':v2_ast,'hashes_before':before,'bpg_binaries':binary_fingerprints,
      'bpg_encoder':{'version':'libbpg0.9.8','sides':[384,320,256,224,192,160,128,96,64],'qps':[20,25,30,35,40,45,50,51],'encoder':'x265','level':1,'bit_depth':8,'chroma':'420','color_space':'ycbcr','transport_prefix':'B'},
      'repo_root':str(ROOT),'baseline_reuse':'JPEG2400 learned_frozen exact frames only'}
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol,'Resume fingerprint changed'
    else:save(out/'protocol.json',protocol)
    from PIL import Image
    bpg=load('real_bpg',bpg_path);runtime=load('real_runtime',runtime_path)
    encoding_journal=out/'encoded.jsonl'
    previous_encoding={r['id']:r for r in journal_rows(encoding_journal,repair=True)}
    encoded=[];phy_records=[]
    for row_index,row in enumerate(rows):
        with Image.open(row['file']) as im:source=im.convert('RGB')
        saved_encoding=previous_encoding.get(row['id'])
        candidate_results=None if saved_encoding else bpg.encode_bpg_caps(source,[b-1 for b in budgets],bpgenc=a.bpgenc)
        image_encoded=[]
        for b in budgets:
            rep=reps[row['id'],b]
            assert sha(rep['packet'])==rep['packet_sha256'] and sha(rep['decoded'])==rep['decoded_sha256']
            assert Path(rep['packet']).stat().st_size==rep['codec_image_bytes'] and rep.get('roundtrip') is True
            ref=learned[row['id'],b]
            assert rep['packet_sha256']==ref['packet_sha256'] and rep['decoded_sha256']==ref['decoded_sha256']
            learned_frame=frame_module.pack(Path(rep['packet']).read_bytes(),'medium')
            assert frame_module.unpack(learned_frame)==('medium',Path(rep['packet']).read_bytes())
            assert len(learned_frame)==rep['image_bytes']==ref['image_bytes'] and len(learned_frame)<=b
            assert hashlib.sha256(learned_frame).hexdigest()==ref['frame_sha256']
            learned_folder=out/'streams'/row['id'];learned_folder.mkdir(exist_ok=True)
            learned_packet=learned_folder/f'{b}_learned_frame.bin';learned_packet.write_bytes(learned_frame)
            with Image.open(rep['decoded']) as im:geometry=im.size
            if saved_encoding:
                record=next(r for r in saved_encoding['records'] if r['budget']==b)
                assert record['geometry']==list(geometry) and record['codec']=='bpg'
                if not record['encode_failure']:
                    assert sha(record['packet'])==record['packet_sha256'] and sha(record['decoded'])==record['decoded_sha256']
                    assert len(Path(record['packet']).read_bytes())==record['image_bytes']<=b
                    assert bpg.unframe(Path(record['packet']).read_bytes()).startswith(bpg.NATIVE_MAGIC)
            else:
                result=candidate_results[b-1]
                record={'id':row['id'],'image_id':row['image_id'],'question_type':row['question_type'],'budget':b,'codec':'bpg','encode_failure':result.payload is None,'bpg_side':result.side,'bpg_qp':result.qp,'encode_attempts_to_first_fit':result.attempts,'encode_seconds_to_first_fit':result.seconds,'geometry':list(geometry)}
                if result.payload is not None:
                    frame=bpg.frame(result.payload);assert len(frame)<=b
                    folder=out/'streams'/row['id'];folder.mkdir(exist_ok=True)
                    packet=folder/f'{b}_bpg.bin';packet.write_bytes(frame)
                    decoded=bpg.decode_bpg(bpg.unframe(frame),bpgdec=a.bpgdec).resize(geometry,Image.Resampling.LANCZOS)
                    dest=folder/f'{b}_bpg.png';decoded.save(dest)
                    record.update(packet=str(packet),packet_sha256=sha(packet),decoded=str(dest),decoded_sha256=sha(dest),image_bytes=len(frame),actual_symbols=510*math.ceil(len(frame)/48))
            if not record['encode_failure']:phy_records.append(record)
            encoded.append(record);image_encoded.append(record)
            phy_records.append({'id':row['id'],'image_id':row['image_id'],'budget':b,'codec':'learned_frozen','packet':str(learned_packet),'packet_sha256':sha(learned_packet),'image_bytes':len(learned_frame),'actual_symbols':510*math.ceil(len(learned_frame)/48),'encode_failure':False})
        if not saved_encoding:
            with encoding_journal.open('a') as journal:
                journal.write(json.dumps({'id':row['id'],'records':image_encoded},allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno())
        if (row_index+1)%6==0:
            progress('ENCODING',images_encoded=row_index+1,total=a.n)
            print(f'ENCODE {row_index+1}/{a.n}',flush=True)
    save(out/'encoded.json',encoded)
    # Clean BPG predictions come only from identical BPG frames and receiver fingerprints.
    prior=[]
    if a.reuse_bpg_smoke:
        smoke_protocol=json.loads((a.reuse_bpg_smoke/'protocol.json').read_text())
        assert all(smoke_protocol['bpg_binaries'][name]['sha256']==binary_fingerprints[name]['sha256'] for name in binary_fingerprints)
        assert smoke_protocol['bpg_encoder']==protocol['bpg_encoder']
        for dep in [bpg_path,Path(__file__),runtime_path,stage_path,prompt_path,frame_path,ADAPTER/'adapter_model.safetensors']:
            assert prior_hash(smoke_protocol,dep)==before[str(dep)],'BPG smoke dependency changed'
        prior+=json.loads((a.reuse_bpg_smoke/'clean_predictions.json').read_text())
    prior += json.loads((out/'clean_predictions.json').read_text()) if (out/'clean_predictions.json').exists() else []
    clean_journal=out/'clean_predictions.jsonl'
    prior += journal_rows(clean_journal,repair=True)
    prior_map={(r['id'],r['budget'],r['packet_sha256']):r for r in prior if not r.get('encode_failure')}
    clean=[];pending=[];reused=0
    for e in encoded:
        if e['encode_failure']:clean.append(e);continue
        cached=prior_map.get((e['id'],e['budget'],e['packet_sha256']))
        if cached:
            assert cached['geometry']==e['geometry'] and cached['actual_visual_tokens']==learned[e['id'],e['budget']]['actual_visual_tokens']
            clean.append({**e,**{k:cached[k] for k in ['prediction','actual_visual_tokens','image_grid_thw','input_tokens']},'clean_reused':True});reused+=1
        else:pending.append(e)
    new_vlm=0;warmups=0;load_seconds=0
    progress('CLEAN_VLM',clean_completed=len(clean),clean_total=len(encoded),clean_reused=reused,new_clean_this_invocation=0)
    if len(pending)>a.max_new_vlm:raise RuntimeError(f'New clean VLM count {len(pending)} exceeds --max-new-vlm {a.max_new_vlm}; stop before loading receiver')
    if pending:
        import torch
        assert torch.cuda.is_available();torch.set_num_threads(2);torch.manual_seed(20260923);random.seed(20260923)
        stage=load('real_stage1',stage_path);qwen=load('real_prompt',prompt_path)
        started=time.monotonic();model,processor=stage.load_receiver(ADAPTER,False);load_seconds=time.monotonic()-started
        token=int(model.config.image_token_id);byid={r['id']:r for r in rows}
        def predict(rep):
            row=byid[rep['id']];runtime.set_pixel_target(processor,100352)
            with Image.open(rep['decoded']) as im:image=im.convert('RGB')
            msg=[{'role':'user','content':[{'type':'image'},{'type':'text','text':qwen.prompt(row['question'])}]}]
            text=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[text],images=[image],return_tensors='pt',min_pixels=100352,max_pixels=100352)
            measures=runtime.visual_measurements(inputs,processor,token);ref=learned[row['id'],rep['budget']]
            assert measures['actual_visual_tokens']==ref['actual_visual_tokens'] and measures['image_grid_thw']==ref['image_grid_thw']
            inputs=inputs.to('cuda')
            with torch.inference_mode():gen=model.generate(**inputs,do_sample=False,max_new_tokens=16)
            pred=processor.batch_decode(gen[:,measures['input_tokens']:],skip_special_tokens=True)[0].strip()
            return {**rep,**measures,'prediction':pred,'clean_reused':False}
        predict(pending[0]);warmups=1
        with clean_journal.open('a') as journal:
            for index,e in enumerate(pending):
                record=predict(e);clean.append(record);new_vlm+=1
                journal.write(json.dumps(record,allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno())
                if (index+1)%12==0:
                    progress('CLEAN_VLM',clean_completed=len(clean),clean_total=len(encoded),clean_reused=reused,new_clean_this_invocation=new_vlm)
                    print(f'VLM {index+1}/{len(pending)}',flush=True)
        del model,processor;gc.collect();torch.cuda.empty_cache()
    save(out/'clean_predictions.json',clean)
    # Actual payload noiseless roundtrip checks use full original decoder.
    sys.path.insert(0,str(PHY.parent));from phy import Digital
    digital=Digital();noiseless=[]
    for codec in ['bpg','learned_frozen']:
        for b in budgets:
            rr=[r for r in phy_records if r['codec']==codec and r['budget']==b]
            if not rr:continue
            r=rr[0];frame=Path(r['packet']).read_bytes();bits,coded,tx=digital.encode(frame)
            got,log=digital.decode(tx,100.,1+0j,bits,coded);assert got==frame and len(tx)==r['actual_symbols']
            noiseless.append({'codec':codec,'budget':b,'n':digital.n,'k':digital.k,'identity':True,'actual_symbols':len(tx)})
    save(out/'noiseless_phy.json',noiseless)
    task_meta={};tasks=[]
    for r in phy_records:
        for snr in snrs:
            for seed in seeds:
                key=f"{r['packet_sha256']}|{r['image_id']}|{seed}|{snr}|{before[str(PHY)]}"
                task_meta[r['id'],r['budget'],r['codec'],snr,seed]=key
                tasks.append((key,r['packet'],r['packet_sha256'],r['image_id'],seed,snr))
    outcomes={};journal=out/'transmissions.jsonl'
    if a.reuse_phy:
        reused_protocol=json.loads((a.reuse_phy/'protocol.json').read_text())
        for field in ['snrs','seeds','budgets','tier','delivery','energy','radio_rounding','early_stop','failed_packet']:
            assert reused_protocol[field]==protocol[field],f'Real-PHY reuse protocol differs: {field}'
        assert prior_hash(reused_protocol,PHY)==before[str(PHY)]
    for path in [a.reuse_phy/'transmissions.jsonl',a.reuse_bpg_smoke/'transmissions.jsonl' if a.reuse_bpg_smoke else None,journal]:
        if path and path.exists():
            for x in journal_rows(path,repair=(path==journal)):
                assert x['delivery']=='real_bitstream_phy';outcomes[x['task_key']]=x
    needed={t[0]:t for t in tasks};existing=set(needed)&outcomes.keys()
    learned_task_keys={key for (qid,budget,codec,snr,seed),key in task_meta.items() if codec=='learned_frozen'}
    assert learned_task_keys<=outcomes.keys(),'Missing JPEG2400 learned true-PHY tasks; refusing new baseline PHY'
    outcomes={key:value for key,value in outcomes.items() if key in needed}
    for key in existing:
        x=outcomes[key];task=needed[key]
        expected_ns=510*math.ceil(Path(task[1]).stat().st_size/48)
        assert x['packet_sha256']==task[2] and x['actual_symbols']==expected_ns
        assert abs(x['effective_snr_db']-(task[5]+10*math.log10(21420/expected_ns)))<1e-12
    # Journal reused real outcomes into this output for self-contained results.
    journal_tmp=journal.with_suffix('.jsonl.tmp')
    with journal_tmp.open('w') as f:
        for key in sorted(existing):f.write(json.dumps(outcomes[key])+'\n')
        f.flush();os.fsync(f.fileno())
    journal_tmp.replace(journal)
    pending_phy=[t for key,t in needed.items() if key not in outcomes]
    print(f'PHY pending={len(pending_phy)} real_reused={len(existing)}',flush=True)
    progress('NOISY_PHY',phy_completed=len(existing),phy_total=len(needed),new_phy_this_invocation=0,clean_completed=len(clean),clean_total=len(encoded),new_clean_this_invocation=new_vlm,clean_reused=reused)
    with journal.open('a') as f:
        with mp.get_context('spawn').Pool(a.workers,initializer=init_worker) as pool:
            for i,x in enumerate(pool.imap_unordered(transmission,pending_phy,chunksize=1),1):
                outcomes[x['task_key']]=x;f.write(json.dumps(x)+'\n');f.flush()
                if i%100==0:
                    os.fsync(f.fileno())
                    progress('NOISY_PHY',phy_completed=len(existing)+i,phy_total=len(needed),new_phy_this_invocation=i,clean_completed=len(clean),clean_total=len(encoded),new_clean_this_invocation=new_vlm,clean_reused=reused)
                    print(f'PHY {i}/{len(pending_phy)} elapsed={time.monotonic()-t0:.1f}s',flush=True)
    assert set(needed)<=outcomes.keys()
    # Load truth only after encoding/prediction/channel work; no model/codec tuning.
    truth={r['id']:r['answer'] for r in json.loads((DATA/'test_truth.sealed.json').read_text())}
    norm=lambda x:' '.join(str(x).strip().lower().replace('_',' ').split())
    clean_map={(r['id'],r['budget']):r for r in clean};phy_map={(r['id'],r['budget'],r['codec']):r for r in phy_records}
    metrics=[]
    for codec in ['bpg','learned_frozen']:
        for snr in snrs:
            for seed in seeds:
                for policy in ['fixed_2000_medium','fixed_4000_medium','proposed_snr_adaptive_v2']:
                    b=(2000 if snr<=10 else 4000) if policy=='proposed_snr_adaptive_v2' else (2000 if policy=='fixed_2000_medium' else 4000)
                    correct=delivered=failures=syms=tokens=byte_sum=0
                    for row in rows:
                        c=clean_map[row['id'],b] if codec=='bpg' else learned[row['id'],b]
                        r=phy_map.get((row['id'],b,codec))
                        if r is None:failures+=1;continue
                        tx=outcomes[task_meta[row['id'],b,codec,snr,seed]];ok=tx['success']
                        delivered+=ok;correct+=int(ok and norm(c['prediction'])==norm(truth[row['id']]))
                        syms+=r['actual_symbols'];tokens+=c['actual_visual_tokens'];byte_sum+=r['image_bytes']
                    metrics.append({'delivery':'real_bitstream_phy','codec':codec,'snr_db':snr,'channel_seed':seed,'policy':policy,'selected_budget':b,'n':len(rows),'strict_correct':correct,'delivered':delivered,'encode_failures':failures,'acc_pct':100*correct/len(rows),'pdr_pct':100*delivered/len(rows),'mean_symbols':syms/len(rows),'mean_tokens_configured':tokens/len(rows),'mean_framed_bytes':byte_sum/len(rows)})
    csvsave(out/'real_phy_per_seed.csv',metrics);means=[]
    for codec in ['bpg','learned_frozen']:
        for snr in snrs:
            for policy in ['fixed_2000_medium','fixed_4000_medium','proposed_snr_adaptive_v2']:
                rr=[x for x in metrics if x['codec']==codec and x['snr_db']==snr and x['policy']==policy]
                means.append({'delivery':'real_bitstream_phy','codec':codec,'snr_db':snr,'policy':policy,'n_per_seed':a.n,'acc_mean_pct':statistics.mean(x['acc_pct'] for x in rr),'acc_sd_pp':statistics.stdev(x['acc_pct'] for x in rr),'pdr_mean_pct':statistics.mean(x['pdr_pct'] for x in rr),'mean_symbols':rr[0]['mean_symbols'],'mean_tokens_configured':rr[0]['mean_tokens_configured'],'encode_failures':rr[0]['encode_failures']})
    csvsave(out/'real_phy_mean_sd.csv',means)
    after={str(x):sha(x) for x in paths};assert before==after
    report={'status':'PASS','scope':f'nested n={a.n} BPG real-bitstream grid','n':a.n,'seconds':time.monotonic()-t0,'new_vlm_predictions':new_vlm,'excluded_warmups':warmups,'reused_bpg_clean_predictions':reused,'new_noisy_phy':len(pending_phy),'reused_real_phy':len(existing),'noiseless_roundtrips':len(noiseless),'n_k':[digital.n,digital.k],'new_training':0,'source_inputs_unchanged':True,'hashes_after':after,'encode_failures':sum(r['encode_failure'] for r in encoded),'exact_identity_on_all_successful_packets':True,'all_unique_tasks_completed':len(needed),'logical_grid_evaluations':a.n*2*4*3*2,'logical_transmission_evaluations':len(tasks),'reused_learned_tasks':len(learned_task_keys),'historical_proxy_used':False,'load_seconds':load_seconds,'rule_b_frozen':True}
    save(out/'freeze_final.json',verify_scanner(scanner))
    save(out/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='hashes_after'},indent=2),flush=True)
    progress('COMPLETE',clean_completed=len(clean),clean_total=len(encoded),phy_completed=len(needed),phy_total=len(needed),new_clean_this_invocation=new_vlm,new_phy_this_invocation=len(pending_phy))

if __name__=='__main__':main()
