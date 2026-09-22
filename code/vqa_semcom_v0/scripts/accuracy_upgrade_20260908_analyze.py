#!/usr/bin/env python3
"""Verify saved models and summarize all feature candidates without retraining."""
import argparse
from pathlib import Path
import zipfile
import joblib
import numpy as np
import accuracy_upgrade_20260908 as a
from analyze_router_network_revision_20260908 import cluster_contrast

def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    root=args.repo/'outputs/accuracy_upgrade_20260908';run=root/'phase_a';old=args.repo/'outputs/router_network_revision_20260908'
    source=args.repo/'outputs/revision_20260907_independent/crossreceiver_v2';b=a.base
    assert b.read(run/'status.json')['state']=='COMPLETE'
    selection=b.read(run/'validation_selection.json');selhash=b.sha(run/'validation_selection.json')
    assert selection['protocol_sha256']==b.sha(run/'protocol.json')
    assert b.read(run/'protocol.json')['script_sha256']==b.sha(Path(a.__file__))
    keys=b.read(source/'common_keys.json');keysets={s:[k for k in keys if k['split']==s] for s in ('train','validation','test')}
    images=np.array([k['image'] for k in keysets['test']]);qtypes=np.array([k['qt'] for k in keysets['test']])
    boxes=a.load_boxes(args.repo/'outputs/detector/v2_0_snr_detections.csv')
    extra={s:a.enhanced_features(keysets[s],boxes,args.repo/'data/raw/visdrone/DET/val/images') for s in ('train','validation')}
    mean,sd=a.scaler(extra['train']);np.testing.assert_array_equal(mean,b.read(run/'scaler.json')['mean'])
    np.testing.assert_array_equal(sd,b.read(run/'scaler.json')['sd'])
    summary=b.read(run/'summary.json');rows={};contrasts={};cost={};verified=0;reproduced=0
    rng=np.random.default_rng(20260908)
    for receiver in b.RECEIVERS:
        with np.load(source/receiver/'evaluation_inputs.npz') as z:
            original={k:z[k].copy() for k in ('train_x','validation_x','validation_y','test_y','linear')}
        cfg=b.CONFIGS[a.NETWORKS[receiver]]
        screen={}
        for group in a.GROUPS:
            screen[group]=[]
            xv=a.compose(original['validation_x'],extra['validation'],group,mean,sd)
            for seed in range(3):
                dest=run/'screen'/receiver/group/f'seed_{seed}';record=b.read(dest/'record.json')
                models=joblib.load(dest/'models.joblib');score=b.route_score(models,cfg,xv)
                with np.load(dest/'validation_outcomes.npz') as z:
                    np.testing.assert_array_equal(score,z['score']);np.testing.assert_array_equal(score>0,z['pick'])
                assert b.accuracy(original['validation_y'],score)==record['validation_accuracy']
                if group=='base18':
                    with np.load(old/'screen'/receiver/a.NETWORKS[receiver]/f'seed_{seed}'/'validation_outcomes.npz') as z:
                        np.testing.assert_array_equal(score,z['score']);reproduced+=1
                screen[group].append(record);verified+=1
        selected=min(a.GROUPS,key=lambda g:(-np.mean([r['validation_accuracy'] for r in screen[g]]),len(a.GROUPS[g]),g))
        assert selected==selection['selected'][receiver]
        with np.load(run/f'{receiver}_evaluation_inputs.npz') as z:xt=z['test_x'].copy();yt=z['test_y'].copy()
        np.testing.assert_array_equal(yt,original['test_y'])
        outcomes={'features':[],'network':[],'original_mlp':[]};final=[]
        for seed in range(10):
            dest=run/'final'/receiver/f'seed_{seed}';record=b.read(dest/'record.json');models=joblib.load(dest/'models.joblib')
            assert record['selection_sha256']==selhash
            score=b.route_score(models,cfg,xt)
            with np.load(dest/'test_outcomes.npz') as z:
                np.testing.assert_array_equal(score,z['score']);np.testing.assert_array_equal(score>0,z['pick']);np.testing.assert_array_equal(yt,z['y'])
            assert record['test']==b.test_summary(keysets['test'],yt,score)
            outcomes['features'].append(yt[np.arange(len(yt)),(score>0).astype(int)])
            for name,path in [('network',old/'final'/receiver/f'seed_{seed}'/'test_outcomes.npz'),('original_mlp',source/receiver/f'seed_{seed}_outcomes.npz')]:
                with np.load(path) as z:outcomes[name].append(yt[np.arange(len(yt)),z['pick']])
            final.append(record);verified+=1
        outcomes={k:np.array(v) for k,v in outcomes.items()}
        outcomes['linear']=yt[np.arange(len(yt)),original['linear']][None,:]
        rows[receiver]={'selected_group':selected,'network':a.NETWORKS[receiver],'methods':{},'per_type':{}}
        for method,correct in outcomes.items():
            values=correct.mean(axis=1)
            rows[receiver]['methods'][method]={'mean':float(values.mean()),'sd':float(values.std(ddof=1)) if len(values)>1 else 0,'seeds':values.tolist()}
        np.testing.assert_allclose(rows[receiver]['methods']['features']['mean'],summary[receiver]['accuracy_mean'],atol=1e-15)
        for qt in b.QTYPES:
            mask=qtypes==qt;rows[receiver]['per_type'][qt]={'n':int(mask.sum())}
            for method,correct in outcomes.items():
                values=correct[:,mask].mean(axis=1)
                rows[receiver]['per_type'][qt][method]={'mean':float(values.mean()),'sd':float(values.std(ddof=1)) if len(values)>1 else 0}
        for method in ('network','original_mlp','linear'):
            contrasts[f'{receiver}_vs_{method}']=cluster_contrast(outcomes['features'].mean(axis=0)-outcomes[method].mean(axis=0),images,rng)
        cost[receiver]={}
        for method,records in [('same_network_base18_screen',screen['base18']),('selected_features_final',final)]:
            cost[receiver][method]={'n_runs':len(records),'parameters':records[0]['parameter_count'],'dimension':records[0]['dimension'],
                'fit_seconds_mean':float(np.mean([r['training_seconds'] for r in records])),
                'single_query_us_mean':float(np.mean([r['inference']['single_query_median_us'] for r in records])),
                'batch_us_per_query_mean':float(np.mean([r['inference']['batch_us_per_query'] for r in records]))}
    ordered=sorted(contrasts,key=lambda k:contrasts[k]['two_sided_cluster_sign_flip_p']);running=0
    for rank,k in enumerate(ordered):
        running=max(running,min(1,(len(ordered)-rank)*contrasts[k]['two_sided_cluster_sign_flip_p']))
        contrasts[k]['holm_p_nine_contrasts']=running
    for path in run.rglob('*.npz'):
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
        with np.load(path) as z:assert all(np.isfinite(z[k]).all() for k in z.files)
    b.dump(args.out/'qa.json',{'verified_models':verified,'exact_base18_reproductions':reproduced,'all_npz_crc_finite':True,'selection_sha256':selhash})
    b.dump(args.out/'comparisons.json',rows);b.dump(args.out/'cost_comparison.json',cost);b.dump(args.out/'exploratory_stats.json',contrasts)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    figdir=args.out/'figures';figdir.mkdir()
    def save(fig,name):
        fig.tight_layout()
        for ext in ('pdf','svg','png'):fig.savefig(figdir/f'{name}.{ext}',dpi=600)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,3.8))
    for j,(method,color,marker) in enumerate([('linear','#CC79A7','s'),('original_mlp','#0072B2','o'),('network','#009E73','^'),('features','#D55E00','D')]):
        vals=[rows[r]['methods'][method] for r in b.RECEIVERS]
        ax.errorbar(np.arange(3)+(j-1.5)*.13,[v['mean']*100 for v in vals],yerr=[v['sd']*100 for v in vals],fmt=marker,linestyle='none',color=color,capsize=3,label=method)
    ax.set_xticks(range(3),['Qwen2-VL','Qwen2.5-VL','SmolVLM']);ax.set_ylabel('Pooled test answer accuracy (%)')
    ax.set_ylim(67,75);ax.grid(axis='y',alpha=.2);ax.legend(ncol=4,fontsize=8,frameon=False);save(fig,'test_comparison')
    fig,ax=plt.subplots(figsize=(7.2,3.8))
    for j,(group,color,marker) in enumerate(zip(a.GROUPS,['#0072B2','#E69F00','#009E73','#D55E00'],['o','s','^','D'])):
        vals=[selection['screen_results'][r][group] for r in b.RECEIVERS]
        ax.errorbar(np.arange(3)+(j-1.5)*.13,[v['mean']*100 for v in vals],yerr=[v['sd']*100 for v in vals],fmt=marker,linestyle='none',color=color,capsize=3,label=group)
    ax.set_xticks(range(3),['Qwen2-VL','Qwen2.5-VL','SmolVLM']);ax.set_ylabel('Validation screening accuracy (%)');ax.grid(axis='y',alpha=.2)
    ax.legend(ncol=4,fontsize=8,frameon=False);save(fig,'all_feature_groups')
    lines=['# Accuracy upgrade: verified phase A analysis','','Question: can sender-observable detector quality and question semantics improve the same frozen routing networks? Labels, receiver answers, calibration, question/SNR keys and image splits are unchanged. 9150/2646/2808 train/validation/test decisions; 104 test images. Existing test exposure means subsequent development, not a pristine independent confirmation.','','## Final pooled accuracy','','Percent, mean +/- sample SD over10seeds (linear deterministic).','','| Receiver | Selected features | New | Same network, original18 | Original MLP | Linear | Delta vs same network (pp) |','|---|---|---:|---:|---:|---:|---:|']
    for r,row in rows.items():
        m=row['methods'];fmt=lambda x:f'{x["mean"]*100:.4f} ± {x["sd"]*100:.4f}'
        lines.append(f'| {r} | {row["selected_group"]} | {fmt(m["features"])} | {fmt(m["network"])} | {fmt(m["original_mlp"])} | {m["linear"]["mean"]*100:.4f} | {(m["features"]["mean"]-m["network"]["mean"])*100:+.4f} |')
    lines+=['','All nine base18 validation predictions reproduce the preceding network experiment exactly, isolating the feature comparison from network implementation changes. The added inputs are computed only from original predicted boxes, source image dimensions, and exact parsed question text. No ground-truth quantities or received outcomes enter features. Training-fitted standardization applies only to appended columns.','','The gains are larger than the preceding network-only refinement. Question semantics wins on two receivers; the combined group wins Qwen2. This supports the narrower conclusion that discarded sender-visible question/count information was useful. It does not identify which individual semantic field is causal; no post-test ablation or new search was performed. Quality-only results and unsuccessful candidates are retained.','','## All validation candidates','','Three screening seeds, no test input to selection.','','| Receiver | Group | Mean ± SD (%) |','|---|---|---:|']
    for r,groups in selection['screen_results'].items():
        for g,v in groups.items():lines.append(f'| {r} | {g} | {v["mean"]*100:.4f} ± {v["sd"]*100:.4f} |')
    lines+=['','## Every question type','','| Receiver | Type | N | Features | Same network | Original MLP | Linear |','|---|---|---:|---:|---:|---:|---:|']
    for r,row in rows.items():
        for qt,v in row['per_type'].items():lines.append(f'| {r} | {qt} | {v["n"]} | '+ ' | '.join(f'{v[m]["mean"]*100:.3f}' for m in ('features','network','original_mlp','linear'))+' |')
    lines+=['','## Costs and boundaries','','Cost comparison JSON contains parameter counts, CPU fitting and inference time under the same thread/host implementation. Feature extraction additionally reads original boxes and image dimensions; it invokes no detector or VLM and may be cached per image/question. Dataset-level feature construction time is in phase_a/feature_audit.json, not a per-query hardware energy measurement. Small timing differences include Python and filesystem overhead. No old Joule measurement is transferred.','','Question semantics are specific to the supported structured English templates. Unsupported/negated templates stop explicitly. Thresholds are extracted from user-visible text, not labels, but benchmark question-generation biases may limit generalization. Receiver-specific refitting is not zero-shot transfer. Detection pilot and7B pilot are validation-only separate comparisons, not evidence that these test numbers contain upgraded detector/VLM answers. No manuscript edits or DV run were made.']
    (args.out/'analysis-report.md').write_text('\n'.join(lines)+'\n')
    stats=['# Exploratory statistics','','Primary descriptive contrast: selected features versus the same previously selected network on original18. Also retain original MLP and linear: nine receiver×baseline contrasts form one Holm family. Average ten seed outcomes per decision; seeds are not independent test datasets. The resampling unit is104images, keeping each image’s question/SNR decisions together. 10000 cluster bootstrap draws, ratio of summed differences to summed decision counts, percentile95%interval; 10000 two-sided image-cluster sign flips, RNG20260908. Effect is pooled percentage-point difference. No normality-based t test is used. Sign flips assume independent clusters and symmetric/exchangeable paired effects under the null. Nearby frames may violate image independence. These calculations do not correct prior test exposure, historical searches or validation selection; no confirmatory/general superiority claim follows. All seeds retained, no outlier removal.','','| Contrast | Effect pp | 95% cluster interval pp | Raw p | Holm p |','|---|---:|---:|---:|---:|']
    for k,v in contrasts.items():
        ci=v['cluster_bootstrap_95_percentile_ci_pp'];stats.append(f'| {k} | {v["effect_pp"]:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {v["two_sided_cluster_sign_flip_p"]:.5f} | {v["holm_p_nine_contrasts"]:.5f} |')
    (args.out/'stats-appendix.md').write_text('\n'.join(stats)+'\n')
    (args.out/'figure-catalog.md').write_text('# Figure catalog\n\n## test_comparison\n\nPurpose: isolate feature gain from prior network gain and keep linear/original baselines visible. Three receivers,2808matched test decisions, ten seeds; dots are pooled accuracy, errors sample SD; linear has no seed error. Zoomed67–75percent axis is explicitly a dot plot, not truncated bars. Observe2.2–4.3pp gain over same-network base18. Decision: sender-visible semantic detail deserves more attention than further width search. Prior exposure and receiver-specific refitting limit confirmation.\n\n## all_feature_groups\n\nPurpose: disclose every predeclared feature candidate, including nonwinners. Validation2646decisions, three seeds, mean±sample SD, zoomed accuracy axis. Observe semantics/combined win while quality alone does not consistently improve. This motivates a future independent evaluation of the frozen schema, not further test-tuned search. Check that no test accuracy enters selection and that all12receiver×group entries remain.\n')
    print({'verified_models':verified,'base18_exact':reproduced,'rows':{r:v['methods']['features'] for r,v in rows.items()}},flush=True)

if __name__=='__main__':main()
