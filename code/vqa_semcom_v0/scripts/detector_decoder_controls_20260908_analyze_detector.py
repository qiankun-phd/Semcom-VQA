#!/usr/bin/env python3
"""Matched detector pilot analysis, preserving original context and bad results."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from detector_decoder_controls_20260908_decoder import correct

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root
    det=root/'detector';summary=json.loads((det/'summary.json').read_text())
    old=json.loads((root/'decoder/summary.json').read_text())
    out=root/'detector_analysis';out.mkdir(exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    methods=['yolov8n','yolov8s'];labels=['YOLOv8n, 20 epochs','YOLOv8s, 20 epochs'];colors=['#0072B2','#D55E00']
    rows={m:[r for r in json.loads((det/m/'qa_outcomes.json').read_text()) if r['split']=='test'] for m in methods}
    keys=lambda rs:[(r['image'],r['question'],r['snr']) for r in rs]
    assert keys(rows[methods[0]])==keys(rows[methods[1]])
    ids=sorted({r['image'] for r in rows[methods[0]]});sums={}
    for m in methods:
        sums[m]=np.array([sum(correct(r,r['prediction']) for r in rows[m] if r['image']==i) for i in ids])
        assert abs(sums[m].sum()/len(rows[m])-summary[m]['test']['accuracy'])<1e-12
    counts=np.array([sum(r['image']==i for r in rows[methods[0]]) for i in ids]);delta=sums['yolov8s']-sums['yolov8n']
    rng=np.random.default_rng(20260908);draw=rng.integers(0,len(ids),(10000,len(ids)))
    boot=delta[draw].sum(axis=1)/counts[draw].sum(axis=1);point=float(delta.sum()/counts.sum())
    null=(rng.choice([-1,1],(20000,len(ids)))*delta).sum(axis=1)/counts.sum()
    stats={'difference_s_minus_n_pp':100*point,'95_image_cluster_bootstrap_ci_pp':(100*np.quantile(boot,[.025,.975])).tolist(),
        'cluster_sign_flip_p':float((1+(abs(null)>=abs(point)).sum())/(len(null)+1)),
        'images':len(ids),'decisions':int(counts.sum()),'seed_count':1,'comparisons':1,
        'interpretation':'exploratory previously inspected benchmark; no independent-scene or seed variance claim'}
    (out/'statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    table='| Detector | QA validation (%) | QA test (%) | Test counting (%) | Parameters | Inference ms/image | Mean received text bytes |\n|---|---:|---:|---:|---:|---:|---:|\n'
    table+=f"| Historical n50 (context only) | {100*old['validation']['baseline']['accuracy']:.4f} | {100*old['test']['baseline']['accuracy']:.4f} | {100*old['test']['baseline']['per_type']['counting']['accuracy']:.4f} | — | — | — |\n"
    for m in methods:
        s=summary[m];table+=f"| {m}, 20 epochs | {100*s['validation']['accuracy']:.4f} | {100*s['test']['accuracy']:.4f} | {100*s['test']['per_type']['counting']['accuracy']:.4f} | {s['parameters']} | {1000*s['mean_inference_seconds']:.4f} | {s['received_text_bytes_mean']:.4f} |\n"
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    qtypes=list(summary['yolov8n']['test']['per_type']);fig,ax=plt.subplots(figsize=(8,4))
    for j,m in enumerate(methods):ax.bar(np.arange(5)+(j-.5)*.34,[100*summary[m]['test']['per_type'][q]['accuracy'] for q in qtypes],width=.34,label=labels[j],color=colors[j])
    ax.set_xticks(np.arange(5),['Presence','Counting','Comparison','Co-presence','Threshold']);ax.set_ylim(0,100);ax.set_ylabel('Test QA accuracy (%)');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.16),ncol=2,frameon=False);fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(out/f'figures/qa_tasks.{ext}',dpi=600)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.6))
    for m,label,color in zip(methods,labels,colors):
        with (det/m/'results.csv').open() as f:history=[{k.strip():v for k,v in r.items()} for r in csv.DictReader(f)]
        assert len(history)==20
        ax.plot([int(r['epoch']) for r in history],[100*float(r['metrics/mAP50(B)']) for r in history],label=label,color=color,marker='o',markersize=3)
    ax.set_xlabel('Epoch');ax.set_xticks([1,5,10,15,20]);ax.set_ylabel('Internal detector-val mAP50 (%)');ax.set_ylim(0,50);ax.legend(frameon=False);fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(out/f'figures/training.{ext}',dpi=600)
    plt.close(fig)
    validation_winner=max(methods,key=lambda m:summary[m]['validation']['accuracy'])
    (out/'analysis-report.md').write_text('# Same-budget detector replacement pilot\n\n'+table+f'\nValidation ranking favors {validation_winner}; both configurations were frozen before evaluation. Test difference s minus n is {point*100:.4f} percentage points. This is symbolic-branch accuracy, NOT full router accuracy. No router was retrained or VLM rerun.\n\nBoth architectures used official COCO initialization, then identical 20-epoch VisDrone adaptation, 640 resolution, batch4, AdamW .001, seed0, same train/val split within official training images. Best checkpoints were selected only by internal detector validation. The public QA split is disjoint by image ID from detector fitting and checkpoint selection. Existing 11-category conversion is retained; tasks use ten categories. Historical n50 used official validation for checkpoint selection and a different training schedule, so it is context only, not a fair architecture comparator.\n\nThe same original decoder algorithm is used for n20 and s20; its class/SNR multiplier is fitted separately on QA training images because detection counts changed. Binary logic and counting tolerance are unchanged. Received records are regenerated with original LDPC/Rician simulation, not old correctness labels. The image branch remains unchanged. Candidate decoder comparisons are reported separately in ../analysis/.\n\nCounts and IoU diagnostics are in detector/summary.json; these are not official VisDrone AP, because historical ignored-region handling is retained. Training curves show internal detector validation, not QA validation. One training seed and 20 epochs are a bounded pilot, not proof of converged superiority. No test-based hyperparameter revision was performed; test had earlier development exposure.\n\nInference timing excludes first image, includes synchronous preprocessing/inference/postprocessing, and is not embedded timing. GPU power.draw was N/A; power limit is not consumption. No Joule estimate or old detector cost is reused. Text evidence bytes are recomputed, but original fixed nominal TOKEN_PAYLOAD_BYTES outage model remains: no packet-length-dependent PHY validation or new system energy claim is made.\n')
    report_path=out/'analysis-report.md'
    report_path.write_text(report_path.read_text()+'\nRuntime caution: the single sequential inference passes were not randomized/interleaved microbenchmarks; concurrent CPU feature extraction may affect preprocessing. The smaller measured time for s is NOT evidence that the larger architecture is faster. No new GPU timing run was added after handing the GPU to the literature-baseline task.\n\nRecommendation: do not automatically replace the manuscript detector. The controlled s20 result is a candidate for further validation, but historical n50 still has higher QA validation accuracy. Task-level trade-offs are material, and the exploratory image-cluster interval for s-minus-n crosses zero.\n')
    (out/'stats-appendix.md').write_text('# Exploratory paired uncertainty\n\n'+json.dumps(stats,indent=2)+'\n\n10000 image-cluster bootstrap resamples and 20000 paired image sign-flips; all questions/SNR repeats move together, task-weighted ratio is retained. One prespecified architecture contrast, no multiplicity correction. No normality or iid-SNR assumption. Shared drone-scene dependence and prior test exposure limit inference; no confirmatory significance claims.\n')
    (out/'figure-catalog.md').write_text('# Figures\n\n## qa_tasks\nPurpose: identify which task changes under detector replacement. Both 20-epoch detectors, fixed symbolic decoder, 104 test images/2808 decisions across six SNRs. Deterministic single-seed point estimates; no error bars representing nonexistent seeds. Notice task trade-offs, not only aggregate score. Decide whether stronger detection benefits the intended QA task before changing manuscript.\n\n## training\nPurpose: inspect internal detector validation trajectories over the equal training budget. 20 epochs, one seed per architecture, 670 held-out official training images. No smoothing or seed error bars. Notice whether improvement is still continuing; avoid conflating underconvergence with architecture limitations. Internal mAP curves are not public QA accuracy.\n')
    (out/'qa.json').write_text(json.dumps({'matched_keys':True,'20_epochs_each':True,'test_accuracy_recomputed':True,'image_branch_unchanged':True,'power_available':False,'visual_inspection':'pending'},indent=2)+'\n')

if __name__=='__main__':main()
