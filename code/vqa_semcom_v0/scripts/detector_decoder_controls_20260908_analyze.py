#!/usr/bin/env python3
"""Exploratory paired image-cluster analysis; do not treat SNR repeats as iid."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root;out=root/'analysis';out.mkdir(exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    summary=json.loads((root/'decoder/summary.json').read_text());rows=json.loads((root/'decoder/test_outcomes.json').read_text())
    models=json.loads((root/'decoder/models.json').read_text())
    model_sizes={'baseline':len(models['baseline']),'linear':2*len(models['linear']),
        'isotonic':sum(len(m['x'])+len(m['y']) for m in models['isotonic'].values())}
    (out/'model_sizes.json').write_text(json.dumps({'numeric_model_values':model_sizes,
        'definition':'baseline ratios; linear slope/intercept pairs; isotonic stored x/y interpolation knots; excludes metadata and sample counts'},indent=2)+'\n')
    methods=['baseline','linear','isotonic'];labels=['Original rules','Linear calibration','Isotonic calibration'];colors=['#666666','#0072B2','#D55E00']
    ids=sorted({r['image'] for r in rows});sums={};counts=[]
    for m in methods:
        subset=[r for r in rows if r['method']==m]
        sums[m]=np.array([sum(r['correct'] for r in subset if r['image']==i) for i in ids],dtype=float)
        if m=='baseline':counts=np.array([sum(r['image']==i for r in subset) for i in ids])
    rng=np.random.default_rng(20260908);draw=rng.integers(0,len(ids),(10000,len(ids)));denom=counts[draw].sum(axis=1)
    stats={};intervals={}
    for m in methods:
        boot=sums[m][draw].sum(axis=1)/denom
        intervals[m]=np.quantile(boot,[.025,.975]).tolist()
    for m in methods[1:]:
        diff=sums[m]-sums['baseline'];point=float(diff.sum()/counts.sum());boot=diff[draw].sum(axis=1)/denom
        signs=rng.choice([-1,1],size=(20000,len(ids)));null=(signs*diff).sum(axis=1)/counts.sum()
        p=float((1+np.count_nonzero(abs(null)>=abs(point)))/(len(null)+1))
        stats[m]={'difference_pp':100*point,'cluster_bootstrap_95_ci_pp':(100*np.quantile(boot,[.025,.975])).tolist(),
            'cluster_sign_flip_p':p,'bonferroni_p':min(1.,2*p),'images':len(ids),'decisions':int(counts.sum()),'fits':1}
    (out/'exploratory_statistics.json').write_text(json.dumps({'contrasts':stats,'accuracy_intervals':intervals},indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7,3.7));x=np.arange(3);values=[100*summary['test'][m]['accuracy'] for m in methods]
    ax.bar(x,values,color=colors,width=.6)
    for i,m in enumerate(methods):
        lo,hi=np.array(intervals[m])*100;ax.errorbar(i,values[i],yerr=[[values[i]-lo],[hi-values[i]]],color='black',capsize=4)
        ax.text(i,hi+1.2,f'{values[i]:.2f}',ha='center')
    ax.set_xticks(x,labels);ax.set_ylim(0,100);ax.set_ylabel('Test accuracy (%)');fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(out/f'figures/overall.{ext}',dpi=600)
    plt.close(fig)
    qtypes=list(summary['test']['baseline']['per_type']);fig,ax=plt.subplots(figsize=(8,4))
    for j,m in enumerate(methods):ax.bar(np.arange(5)+(j-1)*.25,[100*summary['test'][m]['per_type'][q]['accuracy'] for q in qtypes],width=.25,label=labels[j],color=colors[j])
    ax.set_xticks(np.arange(5),['Presence','Counting','Comparison','Co-presence','Threshold']);ax.set_ylim(0,100);ax.set_ylabel('Test accuracy (%)');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=3,frameon=False);fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(out/f'figures/task_breakdown.{ext}',dpi=600)
    plt.close(fig)
    table='| Decoder | Validation (%) | Test (%) | Counting (%) | CPU μs/query | Model JSON bytes |\n|---|---:|---:|---:|---:|---:|\n'
    for m,label in zip(methods,labels):
        s=summary['test'][m];table+=f"| {label} | {100*summary['validation'][m]['accuracy']:.4f} | {100*s['accuracy']:.4f} | {100*s['per_type']['counting']['accuracy']:.4f} | {1e6*s['mean_cpu_seconds_per_query']:.3f} | {s['serialized_model_bytes']} |\n"
    (out/'analysis-report.md').write_text('# Lightweight decoder controls\n\nNo improvement: validation selects original decoder. Two alternatives retain exactly the same received packets and deterministic answering logic, replacing counting calibration and applying it consistently across quantity-dependent questions. Zero evidence stays zero, so these candidates cannot recover an entirely missing category. Presence/co-presence accuracies happen to be unchanged in these results; this is not an independent benefit.\n\n'+table+'\nCounting worsens under both alternatives; some test comparison/threshold gains do not transfer to validation. Do not replace the decoder based on favorable test task slices. Learned count MSE calibration is not necessarily aligned with tolerance-based QA correctness. All candidates are retained; no test retuning.\n\n104 test images, 2808 decisions, 6 SNRs. One deterministic fit per candidate, no seed SD. The test has been inspected in earlier development: all inference is exploratory, not confirmatory. CPU timing is 20 repeated serial passes excluding packet parsing/I/O; it is not energy and not embedded-device latency. JSON bytes include serialization overhead, not transmission overhead; payload is unchanged. Ground-truth counts enter training targets only.\n\nDetection pilot is separately queued; no new detector results yet. No manuscript edits.\n')
    if (root/'detector/summary.json').exists():
        p=out/'analysis-report.md'
        p.write_text(p.read_text().replace('Detection pilot is separately queued; no new detector results yet.',
            'The detector pilot is now complete and reported separately in ../detector_analysis/.'))
    (out/'stats-appendix.md').write_text('# Exploratory uncertainty\n\n'+json.dumps(stats,indent=2)+'\n\nCluster bootstrap uses 10000 image resamples, preserving all queries/SNR repeats per image and task-weighted ratio denominators. Paired cluster sign-flip randomization uses 20000 sign draws under exchangeable algorithm labels per image; Bonferroni corrects two contrasts. No normal approximation is used, and no claim that images from shared drone scenes are fully independent. Prior test exposure and scene correlation limit interpretation. Descriptive difference is the main result. No independent seed runs were fabricated.\n')
    (out/'figure-catalog.md').write_text('# Figures\n\n## overall\nPurpose: test whether decoder replacement helps. Bars: accuracy on 2808 decisions/104 images; whiskers: exploratory 95% image-cluster bootstrap intervals, not seed SD. Notice none improves the point estimate. Decision: retain baseline, which validation also selected. Prior test exposure and within-scene correlations limit inference.\n\n## task_breakdown\nPurpose: locate error trade-offs. Deterministic task accuracies without error bars; counts differ by task (see summary JSON). Notice counting loss despite some comparison/threshold gains. Decision: do not describe universal improvement or select based on test slices. Zero preservation means missing categories cannot be recovered.\n')

if __name__=='__main__':main()
