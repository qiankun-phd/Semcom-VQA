#!/usr/bin/env python3
"""Read-only source audit and CPU replay; never equates ideal-rate and explicit PHY."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
sys.path.append(str(Path(__file__).parent/'revision_20260907_independent'))
import revision_20260907 as base
import revision_20260907_crossreceiver as cross
from detector_decoder_controls_20260908_decoder import packet,predict,correct

def dump(path: Path, obj: object) -> None:
    base.dump(path,obj)

def run(args: argparse.Namespace) -> None:
    if shutil.disk_usage(args.repo).free<20*1024**3:raise RuntimeError('Less than20GiB free; no work started')
    args.out.mkdir(parents=True,exist_ok=False);start=time.time()
    protocol={'scope':'matched468 test questions/104images/6SNR; 9150train,2646validation,2808test decisions',
        'question_location':'known at both endpoints before choice, zero query delivery charged to ALL methods; delivery excluded, not physically free',
        'prior_test_exposure':True,'gpu':False,'training':False,'new_vlm_inference':False,
        'resource_families':{'TDeepSC':'explicit complex channel symbols, no physical time/watt conversion',
            'JPEG':'Shannon ergodic airtime proxy from canonical source codec replay, not finite-length coded uses',
            'detector':'nominal rate1/2 BPSK accounting separate from independent per-record outage corruption'},
        'correction':'replace256 outage-size constant by actual pre-channel serialized text bytes, retain stable per-record trial and logical decoder; refit ratio on train only; not a packet-PHY repair',
        'routing':'reuse frozen enhanced network scores; new validation-only resource thresholds; no test fitting; prediction not retrained after corrected labels',
        'route_price_grid':[0.]+np.logspace(-8,-3,31).tolist(),
        'validation_budget_fractions':[0.,.25,.5,.75,1.],
        'common_reference_window':{'bandwidth_hz':1e6,'seconds':.3,'dimensionless_Btau':300000,'strict_cross_family_equivalence':False},
        'power':'0.5W modeled radiated power only for original nominal airtime ledger; T physical W/time unknown; all new end-to-end energy unknown',
        'pilot_header_CSI':'not charged; no implicit claim of zero real cost; explicit T perfectCSI, JPEG ideal-rate adaptation/SE expectation',
        'script_sha256':base.sha(Path(__file__))}
    dump(args.out/'protocol.json',protocol);dump(args.out/'status.json',{'state':'RUNNING','pid':__import__('os').getpid()})
    sys.path.insert(0,str(args.repo/'src'))
    from vqa_semcom.detector.visdrone_yolo import DetectionRecord,build_detector_lightweight_evidence,_stable_score
    from vqa_semcom.degradation.digital_link import FadingConfig,sample_power_gain
    root=args.repo/'outputs/literature_baselines_20260908'
    keypath=args.repo/'outputs/revision_20260907_independent/crossreceiver_v2/common_keys.json'
    keys=json.loads(keypath.read_text());assert Counter(k['split'] for k in keys)=={'train':9150,'validation':2646,'test':2808}
    groups={};hashes={str(keypath):base.sha(keypath)}
    for receiver in cross.RECEIVERS:
        loaded,audit=cross.load_receiver(args.repo,receiver);groups[receiver]=[loaded[(k['image'],k['question'],k['snr'])] for k in keys]
        hashes.update(audit['source_sha256'])
    source=groups['qwen2'];payloadpath=args.repo/'outputs/revision_20260907_independent/formal_v1/rician/payload_audit.json'
    payload=json.loads(payloadpath.read_text());hashes[str(payloadpath)]=base.sha(payloadpath)
    codec=args.repo/'src/vqa_semcom/degradation/digital_link.py';hashes[str(codec)]=base.sha(codec)
    assert hashes[str(codec)]==payload['link_source_sha256'],'Canonical codec source changed'
    boxes=defaultdict(list);boxpath=args.repo/'outputs/detector/v2_0_snr_detections.csv';hashes[str(boxpath)]=base.sha(boxpath)
    with boxpath.open() as f:
        for r in csv.DictReader(f):boxes[r['image_id']].append(DetectionRecord(r['category'],*(int(r[k]) for k in ('bbox_x','bbox_y','bbox_w','bbox_h')),float(r['confidence'])))
    gains=sample_power_gain('rician',6.,40000,np.random.default_rng(1))
    capacities={s:np.sort(np.log2(1+10**(s/10)*gains)) for s in base.SNRS}
    def outage(nbytes: int,snr: int) -> float:
        return float(np.searchsorted(capacities[snr],nbytes*8/300000,side='left')/len(gains))
    def received(iid: str,snr: int,p: float) -> dict:
        out=Counter();label=f'{snr}dB'
        for i,b in enumerate(boxes[iid]):
            if _stable_score(iid,label,b.category,str(i),'frame')>=p:out[b.category]+=1
            elif _stable_score(iid,label,b.category,str(i),'mode')>=.5:out['unknown']+=1
        return dict(out)
    newgroups=copy.deepcopy(source);oldcounts=[];newcounts=[];ledger=[];imagehashes={};textcache={};rxcache={}
    for j,(g,new) in enumerate(zip(source,newgroups)):
        iid,snr=g['image'],g['snr'];task={'image_id':iid,'question_type':g['qt'],'target_class':g['class']}
        tk=(iid,g['qt'],g['class'])
        if tk not in textcache:
            # Non-SNR good label + empty legacy degradation preserves original records.
            textcache[tk]=build_detector_lightweight_evidence(task,boxes[iid],'good',{'vlm':{'channel_model':'none'}})
        nbytes=len(textcache[tk].encode());pold,pnew=outage(256,snr),outage(nbytes,snr)
        for p in (pold,pnew):
            if (iid,snr,p) not in rxcache:rxcache[iid,snr,p]=received(iid,snr,p)
        before=packet(g['1']['evidence']);after=rxcache[iid,snr,pnew]
        assert before==rxcache[iid,snr,pold],f'Old received records fail replay: {iid}/{snr}'
        oldcounts.append(before);newcounts.append(after);new['1']['transmitted']=after.get(g['class'],0)
        meta=payload['records'][f'{iid}|{snr}'];assert meta['saved_jpeg_matches']
        for receiver in groups:
            receivedpath=Path(groups[receiver][j]['2']['path']);canonical=Path(meta['logged_path'])
            for p in [receivedpath,canonical]:
                if str(p) not in imagehashes:imagehashes[str(p)]=base.sha(p)
            assert imagehashes[str(receivedpath)]==imagehashes[str(canonical)]
        row={k:g[k] for k in ('image','question','qt','snr','split','class')}
        row.update(detector_source_text_bytes=nbytes,detector_received_text_bytes=int(g['1']['bytes']),
            detector_legacy_outage_size_bytes=256,detector_legacy_outage_probability=pold,detector_source_size_outage_probability=pnew,
            detector_legacy_nominal_bpsk_uses=16*int(g['1']['bytes']),detector_source_size_nominal_bpsk_uses=16*nbytes,
            detector_source_size_nominal_airtime_s=16*nbytes/1e6,detector_source_size_modeled_radiated_j=.5*16*nbytes/1e6,
            detector_count_before_channel=len(boxes[iid]),detector_received_known_records=sum(v for k,v in before.items() if k!='unknown'),
            detector_corrected_known_records=sum(v for k,v in after.items() if k!='unknown'),
            jpeg_wire_bytes=meta['wire_payload_bytes'],jpeg_received_file_bytes=meta['logged_file_bytes'],jpeg_outage=meta['meta']['outage'],
            jpeg_ideal_airtime_s=meta['airtime_s'],jpeg_ideal_Btime_proxy=1e6*meta['airtime_s'],jpeg_modeled_radiated_j=.5*meta['airtime_s'],
            jpeg_exceeds_reference_window=meta['airtime_s']>.3,jpeg_code_rate=None,jpeg_explicit_channel_uses=None,
            query_uplink_uses_charged=0,header_pilot_CSI_cost=None,end_to_end_energy_j=None)
        for receiver in groups:row[f'{receiver}_image_correct']=int(groups[receiver][j]['2']['correct'])
        ledger.append(row)
    oldratio=base.fit_calibration([g for g in source if g['split']=='train']);newratio=base.fit_calibration([g for g in newgroups if g['split']=='train'])
    correctedpredictions=[]
    for i,(g,new,oldc,newc,row) in enumerate(zip(source,newgroups,oldcounts,newcounts,ledger)):
        oldanswer=predict(g,oldc,oldratio,'baseline');newanswer=predict(new,newc,newratio,'baseline')
        oldok=correct(g,oldanswer);newok=correct(new,newanswer)
        assert oldok==bool(base.labels([g],oldratio)[0,0])
        row.update(detector_old_correct=int(oldok),detector_corrected_correct=int(newok),detector_old_answer=oldanswer,detector_corrected_answer=newanswer,
            detector_received_counts_changed=oldc!=newc)
        correctedpredictions.append({'image':g['image'],'question':g['question'],'snr':g['snr'],'split':g['split'],'old_received_counts':oldc,'corrected_received_counts':newc})
    with gzip.open(args.out/'resource_ledger.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
    dump(args.out/'calibration_replay.json',{'original_train_only':oldratio,'corrected_train_only':newratio})
    with gzip.open(args.out/'received_counts_replay.json.gz','wt') as f:json.dump(correctedpredictions,f)
    correction={}
    for split in cross.SPLITS:
        rs=[r for r in ledger if r['split']==split]
        correction[split]={'n':len(rs),'old_accuracy':float(np.mean([r['detector_old_correct'] for r in rs])),
            'corrected_accuracy':float(np.mean([r['detector_corrected_correct'] for r in rs])),
            'changed_received_counts':sum(r['detector_received_counts_changed'] for r in rs),
            'changed_answers':sum(r['detector_old_answer']!=r['detector_corrected_answer'] for r in rs),
            'mean_source_bytes':float(np.mean([r['detector_source_text_bytes'] for r in rs])),
            'mean_received_bytes':float(np.mean([r['detector_received_text_bytes'] for r in rs])),
            'source_bytes_range':[min(r['detector_source_text_bytes'] for r in rs),max(r['detector_source_text_bytes'] for r in rs)],
            'per_type':{q:{'old_accuracy':float(np.mean([r['detector_old_correct'] for r in rs if r['qt']==q])),
                'corrected_accuracy':float(np.mean([r['detector_corrected_correct'] for r in rs if r['qt']==q]))} for q in base.QTYPES},
            'jpeg_airtime_above_cap':sum(r['jpeg_exceeds_reference_window'] for r in rs)}
    dump(args.out/'detector_size_correction.json',correction)
    dump(args.out/'jpeg_identity_audit.json',{'n_hashed_paths':len(imagehashes),'all_matched':True,'hashes':imagehashes})
    # Explicit symbol-ledger stays separate from ideal-rate accounting.
    literature=[];literature_status={}
    for name in ['tdeepsc_symbols24_seed7','tdeepsc_symbols48_seed7','tdeepsc_symbols96_seed7','rsvqa_d4_seed7']:
        d=root/'runs'/name;status=json.loads((d/'status.json').read_text()) if (d/'status.json').exists() else {'state':'NOT_STARTED'}
        literature_status[name]=status
        if status['state']!='COMPLETE':continue
        p=d/'test_matched.json';result=json.loads(p.read_text());hashes[str(p)]=base.sha(p)
        proto=json.loads((d/'protocol.json').read_text());hashes[str(d/'protocol.json')]=base.sha(d/'protocol.json')
        expected={(g['image'],g['question'],g['snr']) for g in source if g['split']=='test'}
        actual={(r['image'],r['question'],r['snr']) for r in result['records']};assert expected==actual
        for r in result['records']:
            n=proto['complex_symbols'];v={**r,'method':name,'complex_channel_symbols':n,'real_channel_coordinates':2*n if n else None,
                'normalized_transmit_energy_units':n,'physical_symbol_seconds':None,'physical_radiated_power_w':None,'physical_radiated_energy_j':None,
                'query_uplink_symbols_charged':0,'end_to_end_energy_j':None}
            literature.append(v)
    with gzip.open(args.out/'literature_symbol_ledger.json.gz','wt') as f:json.dump(literature,f)
    dump(args.out/'literature_status_snapshot.json',literature_status)
    # Freeze validation-derived resource thresholds BEFORE evaluating test routes.
    phases={s:[i for i,g in enumerate(source) if g['split']==s] for s in ['validation','test']}
    val=np.array([[ledger[i]['detector_source_size_nominal_bpsk_uses'],ledger[i]['jpeg_ideal_Btime_proxy']] for i in phases['validation']])
    budgets=[float(val[:,0].mean()+f*(val[:,1].mean()-val[:,0].mean())) for f in protocol['validation_budget_fractions']]
    routes={};selection={};routepoints=[]
    for receiver in groups:
        for seed in range(10):
            d=args.repo/f'outputs/accuracy_followup_20260908/phase_ab/{receiver}/enhanced_network_seed_{seed}'
            vp=d/'validation_outcomes.npz';hashes[str(vp)]=base.sha(vp)
            with np.load(vp) as z:score=z['score'].copy()
            yi=np.array([[ledger[i]['detector_corrected_correct'],ledger[i][f'{receiver}_image_correct']] for i in phases['validation']])
            records=[]
            for price in protocol['route_price_grid']:
                pick=(score>price*(val[:,1]-val[:,0])).astype(int);idx=np.arange(len(pick))
                records.append({'price':price,'accuracy':float(yi[idx,pick].mean()),'mean_hybrid_resource_proxy':float(val[idx,pick].mean())})
            chosen=[]
            for budget in budgets:
                candidates=[r for r in records if r['mean_hybrid_resource_proxy']<=budget+1e-8]
                chosen.append(max(candidates,key=lambda r:(r['accuracy'],-r['mean_hybrid_resource_proxy'])) if candidates else None)
            selection[f'{receiver}|{seed}']={'sweep':records,'selected':chosen}
    dump(args.out/'validation_selection.json',{'budgets_hybrid_proxy':budgets,'policies':selection,'test_used':False,
        'warning':'within original mixed-accounting family only; NOT physical alignment against TDeepSC'})
    for key,s in selection.items():
        receiver,seed=key.split('|');p=args.repo/f'outputs/accuracy_followup_20260908/phase_ab/{receiver}/enhanced_network_seed_{seed}/test_outcomes.npz';hashes[str(p)]=base.sha(p)
        with np.load(p) as z:score=z['score'].copy()
        ids=phases['test'];cost=np.array([[ledger[i]['detector_source_size_nominal_bpsk_uses'],ledger[i]['jpeg_ideal_Btime_proxy']] for i in ids])
        yi=np.array([[ledger[i]['detector_corrected_correct'],ledger[i][f'{receiver}_image_correct']] for i in ids]);ii=np.arange(len(ids))
        for j,selected in enumerate(s['selected']):
            if selected is None:continue
            pick=(score>selected['price']*(cost[:,1]-cost[:,0])).astype(np.int8)
            routes[f'{receiver}_seed{seed}_budget{j}']=pick
            routepoints.append({'receiver':receiver,'seed':int(seed),'budget_index':j,'validation_budget':budgets[j],
                'validation_price':selected['price'],'validation_accuracy':selected['accuracy'],
                'test_accuracy':float(yi[ii,pick].mean()),'test_actual_mean_hybrid_proxy':float(cost[ii,pick].mean()),
                'test_budget_exceeded':bool(cost[ii,pick].mean()>budgets[j]),'image_fraction':float(pick.mean())})
    np.savez_compressed(args.out/'selected_route_picks.npz',**routes);dump(args.out/'route_resource_points.json',routepoints)
    dump(args.out/'manifest.json',{'source_sha256':hashes,'completed_unix':time.time(),'disk_free_bytes':shutil.disk_usage(args.repo).free,
        'input_cache_copied':False,'strict_common_PHY_alignment':False,'no_test_training_or_threshold_selection':True})
    dump(args.out/'status.json',{'state':'CPU_AUDIT_AND_REPLAY_COMPLETE','seconds':time.time()-start,'strict_common_PHY_alignment':'requires supplemental common-link evaluation'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    try:run(args)
    except Exception as exc:
        if args.out.exists():dump(args.out/'failure.json',{'error':repr(exc)})
        raise
