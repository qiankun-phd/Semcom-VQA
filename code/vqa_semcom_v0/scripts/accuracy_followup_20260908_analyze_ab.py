#!/usr/bin/env python3
"""Read-only A/B verification, complete controls, and exploratory reporting."""
import argparse
from pathlib import Path
import zipfile
import joblib
import numpy as np
import accuracy_followup_20260908_ab as a
from analyze_router_network_revision_20260908 import cluster_contrast

def describe(values):
    values=np.array(values,float)
    return {'n':len(values),'mean':float(values.mean()),'sd':float(values.std(ddof=1)) if len(values)>1 else 0,'all':values.tolist()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);b=a.b
    root=args.repo/'outputs/accuracy_followup_20260908';run=root/'phase_ab';prior=args.repo/'outputs/accuracy_upgrade_20260908'
    source=args.repo/'outputs/revision_20260907_independent/crossreceiver_v2'
    assert b.read(run/'status.json')['state']=='COMPLETE'
    selection=b.read(run/'validation_selection.json');selhash=b.sha(run/'validation_selection.json')
    assert selection['protocol_sha256']==b.sha(run/'protocol.json')
    assert b.read(run/'protocol.json')['script_sha256']==b.sha(Path(a.__file__))
    allkeys=b.read(source/'common_keys.json');keys={s:[k for k in allkeys if k['split']==s] for s in ('validation','test')}
    images=np.array([k['image'] for k in keys['test']]);qtypes=np.array([k['qt'] for k in keys['test']])
    summary=b.read(run/'summary.json');rows={};qa=0;contrasts={};curves={};cost={};rng=np.random.default_rng(20260908)
    for receiver,methods in summary.items():
        with np.load(prior/f'phase_a/{receiver}_evaluation_inputs.npz') as z:
            data={k:z[k].copy() for k in ('train_x','train_y','validation_x','validation_y','test_x','test_y')}
        with np.load(source/receiver/'evaluation_inputs.npz') as z:legacy={k:z[k].copy() for k in ('validation_x','test_x','linear_weights','test_y')}
        np.testing.assert_array_equal(data['test_y'],legacy['test_y'])
        correct={};curves[receiver]={};fits={};validated={}
        for name,record in methods.items():
            dest=run/receiver/name;policy=b.read(dest/'validation.json');stored_test=b.read(dest/'test.json')
            assert record==stored_test and stored_test['selection_sha256']==selhash
            if name.startswith('enhanced_network'):
                seed=int(name.rsplit('_',1)[1]);path=prior/f'phase_a/final/{receiver}/seed_{seed}'
                model=joblib.load(path/'models.joblib');cfg=b.CONFIGS[a.NETWORKS[receiver]]
                compute=lambda x:b.route_score(model,cfg,x)
            else:
                if name=='legacy18_linear':w=legacy['linear_weights']
                else:
                    with np.load(dest/'weights.npz') as z:w=z['weights'].copy()
                    fit=b.read(dest/'fit.json');fits[name]=fit
                    xb=np.column_stack([data['train_x'],np.ones(len(data['train_x']))]);y=data['train_y'].astype(float)
                    if name!='ridge_advantage':
                        loss,g=a.logistic_objective(w.ravel(),xb,y)
                        np.testing.assert_allclose(loss,fit['train_objective'],atol=1e-12)
                        np.testing.assert_allclose(np.max(np.abs(g)),fit['gradient_inf_norm'],atol=1e-12)
                    else:
                        gradient=xb.T@(xb@w-(y[:,1]-y[:,0]))/len(y)+.001*w
                        assert np.max(np.abs(gradient))<1e-8
                linear={'weights':w,'objective':'advantage_mse' if name=='ridge_advantage' else 'dual_bce'}
                compute=lambda x:a.score(linear,x)
            predicted={}
            for split in ('validation','test'):
                scores=compute((legacy if name=='legacy18_linear' else data)[f'{split}_x'])
                with np.load(dest/f'{split}_outcomes.npz') as z:
                    np.testing.assert_array_equal(scores,z['score'])
                    sweep,picks=a.sweep(data[f'{split}_y'],scores,a.KAPPAS)
                    np.testing.assert_array_equal(picks,z['picks'])
                assert sweep==(policy if split=='validation' else stored_test)['kappa_sweep']
                predicted[split]=scores
                ep=dest/f'{split}_energy_outcomes.npz'
                if ep.exists():
                    with np.load(ep) as z:
                        es,pk=a.sweep(data[f'{split}_y'],scores,a.original.PRICES,z['energy'])
                        np.testing.assert_array_equal(pk,z['picks'])
                    assert es==(policy if split=='validation' else stored_test)['energy_sweep']
            assert policy['common_index']==a.select(policy['kappa_sweep'],.74)
            assert policy['relative_index']==a.select(policy['kappa_sweep'],policy['kappa_sweep'][0]['accuracy']-.01)
            for typ in ('common','relative'):
                index=policy[f'{typ}_index'];expected=None if index is None else stored_test['kappa_sweep'][index]
                assert stored_test[f'{typ}_selected']==expected
            if 'energy_sweep' in policy:
                for typ,target in (('common',.74),('relative',policy['energy_sweep'][0]['accuracy']-.01)):
                    assert policy[f'energy_{typ}_index']==a.select(policy['energy_sweep'],target,True)
            pick=(predicted['test']>0).astype(int);correct[name]=data['test_y'][np.arange(2808),pick]
            assert b.test_summary(keys['test'],data['test_y'],predicted['test'])==stored_test['unpriced']
            validated[name]=policy;qa+=1
        grouped={n:[n] for n in methods if not n.startswith('enhanced_network')}
        grouped['enhanced_network']=[f'enhanced_network_seed_{s}' for s in range(10)]
        rows[receiver]={}
        for label,names in grouped.items():
            values=np.array([correct[n] for n in names]);entry={'unpriced_accuracy':describe(values.mean(axis=1)),
                'per_type':{qt:describe(values[:,qtypes==qt].mean(axis=1)) for qt in b.QTYPES},'policies':{}}
            for typ in ('common','relative'):
                feasible=[n for n in names if methods[n][f'{typ}_selected'] is not None]
                vals=[methods[n][f'{typ}_selected'] for n in feasible]
                entry['policies'][typ]={'feasible':len(feasible),'total':len(names),'infeasible':[n for n in names if n not in feasible],
                    'test_accuracy':describe([v['accuracy'] for v in vals]) if vals else None,
                    'test_image_fraction':describe([v['image_fraction'] for v in vals]) if vals else None,
                    'validation_accuracy':describe([validated[n]['kappa_sweep'][validated[n][f'{typ}_index']]['accuracy'] for n in feasible]) if vals else None,
                    'validation_image_fraction':describe([validated[n]['kappa_sweep'][validated[n][f'{typ}_index']]['image_fraction'] for n in feasible]) if vals else None,
                    'penalties':[v['penalty'] for v in vals]}
            if 'energy_sweep' in methods[names[0]]:
                for typ in ('common','relative'):
                    vals=[methods[n][f'energy_{typ}_selected'] for n in names if methods[n][f'energy_{typ}_selected'] is not None]
                    entry[f'energy_{typ}']={'feasible':len(vals),'total':len(names),
                        'test_accuracy':describe([v['accuracy'] for v in vals]) if vals else None,
                        'test_energy_j_original_accounting':describe([v['energy_j_original_accounting'] for v in vals]) if vals else None,
                        'lambda_per_J':[v['penalty'] for v in vals]}
            rows[receiver][label]=entry
            curves[receiver][label]={s:{metric:np.array([[((validated[n] if s=='validation' else methods[n])['kappa_sweep'])[i][metric] for i in range(26)] for n in names]).tolist()
                for metric in ('accuracy','image_fraction')} for s in ('validation','test')}
        reference=np.mean([correct[n] for n in grouped['enhanced_network']],axis=0)
        for method in ('logistic_gd400','logistic_converged','ridge_advantage'):
            if method in correct:contrasts[f'{receiver}_network_vs_{method}']=cluster_contrast(reference-correct[method],images,rng)
        cost[receiver]=fits
    ordered=sorted(contrasts,key=lambda k:contrasts[k]['two_sided_cluster_sign_flip_p']);running=0
    for i,k in enumerate(ordered):
        running=max(running,min(1,(len(ordered)-i)*contrasts[k]['two_sided_cluster_sign_flip_p']))
        contrasts[k]['holm_p_eight_contrasts']=running
    for p in run.rglob('*.npz'):
        with zipfile.ZipFile(p) as z:assert z.testzip() is None
        with np.load(p) as z:assert all(np.isfinite(z[k]).all() for k in z.files)
    b.dump(args.out/'qa.json',{'model_controls_verified':qa,'nonlinear_models_reused':30,'selection_sha256':selhash,'npz_crc_finite':True,'all26point_replays_exact':True})
    b.dump(args.out/'comparisons.json',rows);b.dump(args.out/'curves.json',curves);b.dump(args.out/'fit_convergence.json',cost);b.dump(args.out/'exploratory_stats.json',contrasts)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    figdir=args.out/'figures';figdir.mkdir()
    styles={'legacy18_linear':('#888888','x','--'),'logistic_gd400':('#56B4E9','s',':'),
            'logistic_converged':('#0072B2','o','-'),'ridge_advantage':('#009E73','^','-.'),'enhanced_network':('#D55E00','D','-')}
    def save(fig,name):
        fig.tight_layout()
        for ext in ('pdf','svg','png'):fig.savefig(figdir/f'{name}.{ext}',dpi=600)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,3.6))
    for j,(name,(color,marker,ls)) in enumerate(styles.items()):
        for i,r in enumerate(b.RECEIVERS):
            if name not in rows[r]:continue
            v=rows[r][name]['unpriced_accuracy'];ax.errorbar(i+(j-2)*.13,v['mean']*100,yerr=v['sd']*100,fmt=marker,color=color,capsize=3,label=name if i==(1 if name=='ridge_advantage' else 0) else None)
    ax.set_xticks(range(3),['Qwen2-VL','Qwen2.5-VL','SmolVLM']);ax.set_ylabel('Pooled test answer accuracy (%)');ax.set_ylim(67,74.8)
    ax.grid(axis='y',alpha=.2);ax.legend(ncol=3,fontsize=7,frameon=False,loc='upper left');save(fig,'matched_input_accuracy')
    for split in ('validation','test'):
        fig,axes=plt.subplots(1,3,figsize=(10,3.5),sharey=True)
        for ax,r,title in zip(axes,b.RECEIVERS,['Qwen2-VL','Qwen2.5-VL','SmolVLM']):
            for name,(color,marker,ls) in styles.items():
                if name not in curves[r]:continue
                v=curves[r][name][split];xs=np.array(v['image_fraction'])*100;ys=np.array(v['accuracy'])*100
                if name=='enhanced_network':
                    for xx,yy in zip(xs,ys):ax.plot(xx,yy,color=color,alpha=.15,linewidth=.7)
                ax.plot(xs.mean(axis=0),ys.mean(axis=0),color=color,linestyle=ls,linewidth=1.4,label=name)
                selected=rows[r][name]['policies']['common']
                if selected['feasible']:
                    ax.scatter(selected[f'{split}_image_fraction']['mean']*100,selected[f'{split}_accuracy']['mean']*100,color=color,marker=marker,s=38,zorder=4)
            if split=='validation':ax.axhline(74,color='black',linewidth=.7,linestyle='--')
            ax.set_title(title);ax.set_xlabel('Image-branch use (%)');ax.grid(alpha=.2)
        from matplotlib.lines import Line2D
        handles=[Line2D([0],[0],color=color,linestyle=ls,label=name) for name,(color,marker,ls) in styles.items()]
        axes[0].set_ylabel(f'{split.capitalize()} answer accuracy (%)');axes[0].legend(handles=handles,fontsize=6.5,frameon=False,loc='lower right')
        save(fig,f'{split}_accuracy_image_use')
    if b.read(run/'energy_gate.json')['full_energy_curve_permitted']:
        fig,ax=plt.subplots(figsize=(6,3.5));r='qwen2'
        for name,(color,marker,ls) in styles.items():
            if name not in rows[r]:continue
            names=[f'enhanced_network_seed_{s}' for s in range(10)] if name=='enhanced_network' else [name]
            xs=np.array([[p['energy_j_original_accounting'] for p in summary[r][n]['energy_sweep']] for n in names]);ys=np.array([[p['accuracy'] for p in summary[r][n]['energy_sweep']] for n in names])*100
            if len(names)>1:
                for xx,yy in zip(xs,ys):ax.plot(xx,yy,color=color,alpha=.15,linewidth=.7)
            ax.plot(xs.mean(axis=0),ys.mean(axis=0),color=color,linestyle=ls,label=name)
            selected=rows[r][name]['energy_common']
            if selected['feasible']:ax.scatter(selected['test_energy_j_original_accounting']['mean'],selected['test_accuracy']['mean']*100,color=color,marker=marker,s=35)
        ax.set_xlabel('Energy per query under original accounting (J)');ax.set_ylabel('Qwen2-VL test answer accuracy (%)');ax.grid(alpha=.2)
        ax.legend(fontsize=7,frameon=False);save(fig,'qwen2_original_energy_accounting')
    lines=['# Matched-input baseline and resource analysis','','Same augmented inputs and labels as the preceding feature study. No nonlinear retraining, feature search or test-driven model/price selection.104testimages,2808decisions; models/price policies frozen from2646validation decisions. The test benchmark has been used for development before; this is not pristine confirmation.','','## A: full accuracy controls','','All deterministic linear runs retained; enhanced network is mean±sample SD over10existing seeds. GD400 and convergedlogistic have identical full regularized training objective; ridge uses the same advantage target as the two MSE networks, not an identical architectural regularization penalty.','','| Receiver | Enhanced network (%) | Logistic GD400 (%) | Converged logistic (%) | Ridge advantage (%) | Legacy18linear (%) |','|---|---:|---:|---:|---:|---:|']
    for r,m in rows.items():
        v=m['enhanced_network']['unpriced_accuracy'];parts=[f'{v["mean"]*100:.4f} ± {v["sd"]*100:.4f}']
        for name in ('logistic_gd400','logistic_converged','ridge_advantage','legacy18_linear'):parts.append(f'{m[name]["unpriced_accuracy"]["mean"]*100:.4f}' if name in m else 'not applicable')
        lines.append(f'| {r} | '+' | '.join(parts)+' |')
    lines+=['','Qwen2linear controls match the enhanced network: the fixed400-step solver is slightly higher and the convergedsolver slightly lower. Its earlier gain is therefore mainly evidence for the added input information, not a demonstrated nonlinear-network advantage. Network gains remain larger onSmol and smaller onQwen2.5. Do not delete the strong linear results. Convergence diagnostics retain gradient norms, objective and stopping messages in fit_convergence.json; lower training objective need not imply higher validation/test accuracy.','','## B: common validation target, no test guarantee','','Kappa is dimensionless and decisions are estimated correctness-difference>kappa. The same absolute validation accuracy target74% applies to every method/seed. Selected points minimize validation image-use on the26point grid; feasible/infeasible counts remain explicit. The test accuracies below are outcomes of those frozen choices, not claims that74%was guaranteed ontest. The own-zero-minus1pp relative policy is retained separately in comparisons.json and must not be used to assert matched-target dominance.','','| Receiver | Method | Feasible | Validation accuracy (%) | Validation image use (%) | Test accuracy (%) | Test image use (%) |','|---|---|---:|---:|---:|---:|---:|']
    for r,m in rows.items():
        for name,entry in m.items():
            v=entry['policies']['common'];vals=[f'{v[k]["mean"]*100:.3f}' if v[k] is not None else 'infeasible' for k in ('validation_accuracy','validation_image_fraction','test_accuracy','test_image_fraction')]
            lines.append(f'| {r} | {name} | {v["feasible"]}/{v["total"]} | '+' | '.join(vals)+' |')
    lines+=['','Validation/test distributions differ: a validation constraint does not ensure equal test accuracy. Fullcurves are retained, not just favorable selectedpoints. Comparing methods requires considering both their actual accuracy and resource-use; no blanket Pareto-dominance claim follows from this table.','','## Original Qwen2 energy accounting','','All5454validation/test decisions pass exactJPEG/payload identity, sourcecodec and original power-artifact gates. Physical lambda uses the original26points and score>lambda*(Eimage-Edetection). Detector compute is charged onboth learned-model branches because all inputs depend on original detector output. Energy reuses the documented original power/airtime model, not a new measurement; feature processing and router costs are unmeasured additions. Other receivers receive only image-use curves, not borrowed32.31J values. No7B/newdetector answers enter these labels or costs.','','| Qwen2 method | Feasible | Selected test accuracy (%) | Energy/query, original accounting (J) |','|---|---:|---:|---:|']
    if b.read(run/'energy_gate.json')['full_energy_curve_permitted']:
        for name,m in rows['qwen2'].items():
            e=m['energy_common'];lines.append(f'| {name} | {e["feasible"]}/{e["total"]} | {e["test_accuracy"]["mean"]*100:.3f} | {e["test_energy_j_original_accounting"]["mean"]:.4f} |' if e['feasible'] else f'| {name} | 0/{e["total"]} | infeasible | infeasible |')
    lines+=['','## Every question type, including adverse results','','| Receiver | Method | Type | Accuracy (%) |','|---|---|---|---:|']
    for r,m in rows.items():
        for name,entry in m.items():
            for qt,v in entry['per_type'].items():lines.append(f'| {r} | {name} | {qt} | {v["mean"]*100:.3f} |')
    (args.out/'analysis-report.md').write_text('\n'.join(lines)+'\n')
    stats=['# Exploratory matched-input statistics','','Mean each network’s10seed correctness indicators per decision before comparing to each deterministic matched-input linear control.104imageclusters retain allquestions/SNRs;10000image-bootstrap draws and10000two-sided sign flips, RNG20260908. Effects are pooled percentage-point differences and percentile95%intervals. Holm covers8network-versus-newlinear contrasts(3GD,3converged,2ridge); legacy18is contextual only. No parametric normality-based test is used. Sign flips assume independent/exchangeable symmetric cluster effects; nearbyvideo frames and historicaltest exposure violate simple confirmation assumptions. These are conditional exploratory checks, not universal superiority evidence or proof of equality. No seed or adverse outcome is discarded.','','| Contrast | Effect pp | Image-bootstrap95%interval pp | Raw p | Holm p |','|---|---:|---:|---:|---:|']
    for name,v in contrasts.items():
        ci=v['cluster_bootstrap_95_percentile_ci_pp'];stats.append(f'| {name} | {v["effect_pp"]:+.4f} | [{ci[0]:+.4f},{ci[1]:+.4f}] | {v["two_sided_cluster_sign_flip_p"]:.5f} | {v["holm_p_eight_contrasts"]:.5f} |')
    (args.out/'stats-appendix.md').write_text('\n'.join(stats)+'\n')
    (args.out/'figure-catalog.md').write_text('# Figure catalog\n\n`matched_input_accuracy`: full2808testdecisions; points meanaccuracy, networkerrors10seedSD, deterministic linear noerror. Zoomedpercentageaxis is a dotplot. Observe Qwen2linear/network near equality, receiver-dependent nonlinear benefit. Decision: preserve linear and attribute input gains carefully.\n\n`validation_accuracy_image_use` and `test_accuracy_image_use`: complete26pointdimensionlesskappagrids; networkthinlines all10seeds, thickline mean; markers are policies selected only under common74%validation target. Horizontal74%line appears only onvalidation. Observe both usage and accuracy; testneednotmeetvalidationtarget. Do not infer dominance from different relative constraints or hide infeasibility.\n\n`qwen2_original_energy_accounting`: same testdecisions under originalphysical-lambda grid; allJPEG/payloadgates passed. Markers selectedusingcommonvalidation74%target. Energy excludes unmeasured extra feature/routerwork and is not a new measurement. NoJclaim for otherreceivers.\n')
    print({'verified_controls':qa,'same_feature_contrasts':contrasts},flush=True)

if __name__=='__main__':main()
