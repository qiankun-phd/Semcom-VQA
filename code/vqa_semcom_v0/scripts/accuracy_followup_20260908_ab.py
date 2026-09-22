#!/usr/bin/env python3
"""Fair enhanced-input linear baselines and validation-frozen resource policies."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
sys.path.append(str(Path(__file__).parent/'revision_20260907_independent'))
import revision_20260907 as original
import router_network_revision_20260908 as b

KAPPAS=[0.0]+np.logspace(-4,0,25).tolist()
TARGET=.74
NETWORKS={'qwen2':'wide_bce','qwen25':'advantage_mse','smol':'advantage_mse'}

def logistic_objective(flat:np.ndarray,xb:np.ndarray,y:np.ndarray):
    w=flat.reshape(xb.shape[1],2);z=xb@w
    loss=float(np.sum(np.logaddexp(0,z)-y*z)/len(y)+.0005*np.sum(w*w))
    gradient=xb.T@(expit(z)-y)/len(y)+.001*w
    return loss,gradient.ravel()

def fit_linears(x:np.ndarray,y:np.ndarray,advantage:bool) -> dict:
    xb=np.column_stack([x,np.ones(len(x))]);start=time.perf_counter()
    w=original.fit_linear(x,y)
    loss,gradient=logistic_objective(w.ravel(),xb,y)
    models={'logistic_gd400':{'weights':w,'objective':'dual_bce','fit_seconds':time.perf_counter()-start,
        'train_objective':loss,'gradient_inf_norm':float(np.max(np.abs(gradient))),'iterations':400,
        'converged':bool(np.max(np.abs(gradient))<1e-5),'solver':'historical400steps step.5 L2.001 including intercept'}}
    tick=time.perf_counter()
    result=minimize(logistic_objective,np.zeros(xb.shape[1]*2),args=(xb,y),jac=True,method='L-BFGS-B',
        options={'maxiter':5000,'gtol':1e-8,'ftol':1e-12,'maxls':50})
    assert np.isfinite(result.x).all()
    models['logistic_converged']={'weights':result.x.reshape(xb.shape[1],2),'objective':'dual_bce',
        'fit_seconds':time.perf_counter()-tick,'train_objective':float(result.fun),'gradient_inf_norm':float(np.max(np.abs(result.jac))),
        'iterations':int(result.nit),'converged':bool(result.success),'message':str(result.message),
        'solver':'fixed L-BFGS-B same objective L2.001, maxiter5000/gtol1e-8/ftol1e-12'}
    if advantage:
        tick=time.perf_counter();target=y[:,1]-y[:,0];matrix=xb.T@xb/len(y)+.001*np.eye(xb.shape[1])
        w=np.linalg.solve(matrix,xb.T@target/len(y));gradient=xb.T@(xb@w-target)/len(y)+.001*w
        models['ridge_advantage']={'weights':w,'objective':'advantage_mse','fit_seconds':time.perf_counter()-tick,
            'train_objective':float(.5*np.mean((xb@w-target)**2)+.0005*np.sum(w*w)),
            'gradient_inf_norm':float(np.max(np.abs(gradient))),'iterations':1,'converged':bool(np.max(np.abs(gradient))<1e-8),
            'condition_number':float(np.linalg.cond(matrix)),'solver':'closed-form normalized ridge L2.001 including intercept'}
    return models

def score(model:dict,x:np.ndarray)->np.ndarray:
    if model['objective']=='dual_bce':
        p=original.linear_predict(model['weights'],x);return p[:,1]-p[:,0]
    return np.column_stack([x,np.ones(len(x))])@model['weights']

def sweep(y:np.ndarray,scores:np.ndarray,grid:list,energy:np.ndarray|None=None):
    difference=np.ones(len(y)) if energy is None else energy[:,1]-energy[:,0]
    picks=np.array([(scores>price*difference).astype(np.int8) for price in grid])
    records=[]
    for price,pick in zip(grid,picks):
        r={'penalty':price,'n':len(y),'accuracy':float(y[np.arange(len(y)),pick].mean()),'image_fraction':float(pick.mean())}
        if energy is not None:r['energy_j_original_accounting']=float(energy[np.arange(len(y)),pick].mean())
        records.append(r)
    return records,picks

def select(records:list,target:float,energy:bool=False)->int|None:
    feasible=[i for i,r in enumerate(records) if r['accuracy']>=target]
    if not feasible:return None
    metric='energy_j_original_accounting' if energy else 'image_fraction'
    return min(feasible,key=lambda i:(records[i][metric],records[i]['penalty']))

def energy_gate(repo:Path,keys:dict,out:Path)->dict|None:
    path=repo/'outputs/revision_20260907_independent/formal_v1/rician/payload_audit.json'
    payload=b.read(path);codec=repo/'src/vqa_semcom/degradation/digital_link.py'
    source=repo/'outputs/vlm/v3_0_rician_predictions.csv';allowed={(k['image'],k['question'],k['snr']) for s in keys.values() for k in s}
    rows={}
    with source.open() as f:
        for r in csv.DictReader(f):
            k=(r['image_id'],r['question'],int(float(r['sensed_snr_db'])))
            if k not in allowed or r['service_level'] not in ('1','2'):continue
            branch={'bytes':int(r['payload_bytes']),'path':r['image_path']}
            if r['service_level'] in rows.setdefault(k,{}):assert rows[k][r['service_level']]==branch
            rows[k][r['service_level']]=branch
    hashes={};mismatch=[];codec_match=b.sha(codec)==payload['link_source_sha256']
    for split,items in keys.items():
        for k in items:
            kk=(k['image'],k['question'],k['snr']);record=payload['records'].get(f'{k["image"]}|{k["snr"]}')
            if not record or set(rows.get(kk,{}))!={'1','2'}:
                mismatch.append({'key':k,'reason':'missing payload or paired branch'});continue
            actual=rows[kk]['2']['path'];audited=record['logged_path']
            for p in (actual,audited):
                if p not in hashes:hashes[p]=b.sha(Path(p))
            if hashes[actual]!=hashes[audited] or not record['saved_jpeg_matches']:
                mismatch.append({'key':k,'reason':'received JPEG differs from payload-audited JPEG','actual':actual,'payload_image':audited})
    report={'payload_audit_sha256':b.sha(path),'prediction_cache_sha256':b.sha(source),'codec_sha_matches':codec_match,
        'rows':sum(len(v) for v in keys.values()),'mismatched_rows':len(mismatch),'mismatches':mismatch,
        'full_energy_curve_permitted':not mismatch and codec_match,
        'caution':'Additional enhanced-feature/router processing costs unmeasured; original accounting only, not new hardware measurement'}
    b.dump(out/'energy_gate.json',report);b.dump(out/'energy_image_sha256.json',hashes)
    if mismatch or not codec_match:return None
    power=repo/'outputs/energy/gpu_power_phases.json';evlm=b.read(power)['phases']['vlm']['joule_per_item_incremental']
    report.update(vlm_joules=evlm,detector_joules=original.ENERGY_DET,power_sha256=b.sha(power));b.dump(out/'energy_gate.json',report)
    energy={}
    for split,items in keys.items():
        matrix=[]
        for k in items:
            r=rows[(k['image'],k['question'],k['snr'])]
            airtime=payload['records'][f'{k["image"]}|{k["snr"]}']['airtime_s']
            matrix.append([original.ENERGY_DET+.5*r['1']['bytes']*8/.5/1e6,original.ENERGY_DET+evlm+.5*airtime])
        energy[split]=np.array(matrix)
    return energy

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);started=time.time()
    prior=args.repo/'outputs/accuracy_upgrade_20260908';source=args.repo/'outputs/revision_20260907_independent/crossreceiver_v2'
    protocol={'pid':os.getpid(),'started_unix':started,'kappas_dimensionless':KAPPAS,'physical_lambdas_per_J':original.PRICES,
        'common_validation_accuracy_target':TARGET,'same_inputs':'Qwen2combined52, Qwen2.5/Smolsemantics42 from prior phaseA',
        'logistic':'retain historicalGD400 and same-objective convergedL-BFGS fixedalpha.001','ridge':'fixedalpha.001 for advantage receivers',
        'selection':'same absolute validation accuracy.74; minresource then minpenalty; explicitly infeasible if none',
        'relative_policy':'secondary ownzeroaccuracy minus.01; not matched-target evidence','no_nonlinear_training':True,
        'test_gate':'write all model/validation policies before loading test arrays','prior_test_exposure':True,'script_sha256':b.sha(Path(__file__))}
    b.dump(args.out/'protocol.json',protocol)
    try:
        assert b.read(prior/'phase_a/status.json')['state']=='COMPLETE'
        allkeys=b.read(source/'common_keys.json');keys={s:[k for k in allkeys if k['split']==s] for s in ('train','validation','test')}
        energy=energy_gate(args.repo,{'validation':keys['validation'],'test':keys['test']},args.out)
        validation={};models={};data={};hashes={};policies={}
        for receiver in b.RECEIVERS:
            p=prior/f'phase_a/{receiver}_evaluation_inputs.npz';hashes[str(p)]=b.sha(p)
            with np.load(p) as z:data[receiver]={k:z[k].copy() for k in ('train_x','train_y','validation_x','validation_y')}
            d=data[receiver];d['train_y']=d['train_y'].astype(float)
            assert d['train_x'].shape==(9150,52 if receiver=='qwen2' else 42)
            assert d['validation_x'].shape==(2646,d['train_x'].shape[1])
            models[receiver]=fit_linears(d['train_x'],d['train_y'],receiver!='qwen2');validation[receiver]={};policies[receiver]={}
            for name,model in models[receiver].items():
                dest=args.out/receiver/name;dest.mkdir(parents=True)
                weights=model['weights'];record={k:v for k,v in model.items() if k!='weights'}
                record['parameters']=weights.size;np.savez_compressed(dest/'weights.npz',weights=weights)
                b.dump(dest/'fit.json',record)
                validation[receiver][name]=score(model,d['validation_x'])
            with np.load(source/receiver/'evaluation_inputs.npz') as z:legacy=z['linear_weights'].copy()
            legacy_x=None
            with np.load(source/receiver/'evaluation_inputs.npz') as z:legacy_x=z['validation_x'].copy()
            models[receiver]['legacy18_linear']={'weights':legacy,'objective':'dual_bce'}
            validation[receiver]['legacy18_linear']=score(models[receiver]['legacy18_linear'],legacy_x)
            for seed in range(10):
                name=f'enhanced_network_seed_{seed}';path=prior/f'phase_a/final/{receiver}/seed_{seed}'
                network=joblib.load(path/'models.joblib');hashes[str(path/'models.joblib')]=b.sha(path/'models.joblib')
                val=b.route_score(network,b.CONFIGS[NETWORKS[receiver]],d['validation_x'])
                with np.load(path/'validation_outcomes.npz') as z:np.testing.assert_array_equal(val,z['score'])
                validation[receiver][name]=val
            for name,val in validation[receiver].items():
                dest=args.out/receiver/name;dest.mkdir(parents=True,exist_ok=True)
                vs,vp=sweep(d['validation_y'],val,KAPPAS)
                record={'kappa_sweep':vs,'common_index':select(vs,TARGET),'relative_index':select(vs,vs[0]['accuracy']-.01),
                    'relative_target':vs[0]['accuracy']-.01,'score_below_minus1':int((val< -1).sum()),'score_above_plus1':int((val>1).sum())}
                if receiver=='qwen2' and energy is not None:
                    es,ep=sweep(d['validation_y'],val,original.PRICES,energy['validation'])
                    record.update(energy_sweep=es,energy_common_index=select(es,TARGET,True),energy_relative_index=select(es,es[0]['accuracy']-.01,True))
                    np.savez_compressed(dest/'validation_energy_outcomes.npz',picks=ep,energy=energy['validation'])
                np.savez_compressed(dest/'validation_outcomes.npz',score=val,picks=vp)
                b.dump(dest/'validation.json',record);policies[receiver][name]=record
            print(f'[validation frozen {receiver}] '+str({n:r['kappa_sweep'][0]['accuracy'] for n,r in policies[receiver].items() if not n.startswith('enhanced_network')}),flush=True)
        b.dump(args.out/'source_sha256.json',hashes)
        b.dump(args.out/'validation_selection.json',{'policies':policies,'common_target':TARGET,'frozen_unix':time.time(),'test_used':False,'protocol_sha256':b.sha(args.out/'protocol.json')})
        selection_sha=b.sha(args.out/'validation_selection.json');summary={}
        for receiver in b.RECEIVERS:
            with np.load(prior/f'phase_a/{receiver}_evaluation_inputs.npz') as z:xt=z['test_x'].copy();yt=z['test_y'].copy()
            with np.load(source/receiver/'evaluation_inputs.npz') as z:legacy_x=z['test_x'].copy();legacy_pick=z['linear'].copy();np.testing.assert_array_equal(yt,z['test_y'])
            summary[receiver]={}
            for name,policy in policies[receiver].items():
                dest=args.out/receiver/name
                if name.startswith('enhanced_network'):
                    seed=int(name.rsplit('_',1)[1]);path=prior/f'phase_a/final/{receiver}/seed_{seed}'
                    network=joblib.load(path/'models.joblib');tscore=b.route_score(network,b.CONFIGS[NETWORKS[receiver]],xt)
                    with np.load(path/'test_outcomes.npz') as z:np.testing.assert_array_equal(tscore,z['score']);np.testing.assert_array_equal(yt,z['y'])
                else:tscore=score(models[receiver][name],legacy_x if name=='legacy18_linear' else xt)
                ts,tp=sweep(yt,tscore,KAPPAS)
                if name=='legacy18_linear':np.testing.assert_array_equal(tp[0],legacy_pick)
                assert b.sha(args.out/'validation_selection.json')==selection_sha
                ci,ri=policy['common_index'],policy['relative_index']
                record={'unpriced':b.test_summary(keys['test'],yt,tscore),'kappa_sweep':ts,
                    'common_selected':None if ci is None else ts[ci],'relative_selected':ts[ri],
                    'common_index':ci,'relative_index':ri,'selection_sha256':selection_sha,
                    'score_below_minus1':int((tscore< -1).sum()),'score_above_plus1':int((tscore>1).sum())}
                np.savez_compressed(dest/'test_outcomes.npz',score=tscore,picks=tp,y=yt)
                if 'energy_sweep' in policy:
                    es,ep=sweep(yt,tscore,original.PRICES,energy['test']);ei=policy['energy_common_index'];er=policy['energy_relative_index']
                    record.update(energy_sweep=es,energy_common_selected=None if ei is None else es[ei],energy_relative_selected=es[er])
                    np.savez_compressed(dest/'test_energy_outcomes.npz',picks=ep,energy=energy['test'])
                b.dump(dest/'test.json',record);summary[receiver][name]=record
            print(f'[test complete {receiver}] '+str({n:r['unpriced']['accuracy'] for n,r in summary[receiver].items() if not n.startswith('enhanced_network')}),flush=True)
        b.dump(args.out/'summary.json',summary)
        b.dump(args.out/'status.json',{'state':'COMPLETE','new_dual_logistic_fits':6,'new_ridge_fits':2,'reused_network_models':30,
            'reused_legacy_linears':3,'seconds':time.time()-started,'selection_sha256':selection_sha,'qwen2_energy_gate_passed':energy is not None})
    except Exception as exc:
        b.dump(args.out/'status.json',{'state':'FAILED','error':repr(exc)});raise

if __name__=='__main__':main()
