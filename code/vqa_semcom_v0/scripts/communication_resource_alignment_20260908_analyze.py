#!/usr/bin/env python3
"""Native-unit resource plots; explicitly refuses mixed-PHY dominance claims."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root
    out=root/'analysis';out.mkdir(exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    with gzip.open(root/'resource_ledger.csv.gz','rt') as f:allrows=list(csv.DictReader(f))
    rows=[r for r in allrows if r['split']=='test'];assert len(rows)==2808 and len({r['image'] for r in rows})==104
    with gzip.open(root/'literature_symbol_ledger.json.gz','rt') as f:lit=json.load(f)
    correction=json.loads((root/'detector_size_correction.json').read_text());route=json.loads((root/'route_resource_points.json').read_text())
    lookup={(r['image'],r['question'],int(r['snr'])):r for r in rows};by_method=defaultdict(list)
    for r in lit:
        g=lookup[(r['image'],r['question'],r['snr'])];assert g['qt']==r['qt']
        by_method[r['method']].append(r)
    native=[]
    for m,rs in by_method.items():
        for snr in sorted({r['snr'] for r in rs}):
            sub=[r for r in rs if r['snr']==snr];n=sub[0]['complex_channel_symbols']
            native.append({'method':m,'snr':snr,'accuracy':float(np.mean([r['correct'] for r in sub])),
                'resource_family':'explicit_complex_symbols' if n else 'same_received_JPEG',
                'complex_symbols':n,'samples':len(sub),'training_seeds':1,'channel_draws':len({r['channel_seed'] for r in sub})})
    summaries={}
    for s in [-5,0,5,10,15,20]:
        sub=[r for r in rows if int(r['snr'])==s]
        summaries[s]={'jpeg_wire_bytes_mean_nonoutage':float(np.mean([float(r['jpeg_wire_bytes']) for r in sub if r['jpeg_wire_bytes']])),
            'jpeg_received_file_bytes_mean':float(np.mean([float(r['jpeg_received_file_bytes']) for r in sub])),
            'jpeg_ideal_airtime_mean_s':float(np.mean([float(r['jpeg_ideal_airtime_s']) for r in sub])),
            'jpeg_ideal_airtime_max_s':max(float(r['jpeg_ideal_airtime_s']) for r in sub),
            'jpeg_outage_decisions':sum(r['jpeg_outage']=='True' for r in sub),
            'det_source_bytes_mean':float(np.mean([float(r['detector_source_text_bytes']) for r in sub])),
            'old_outage_probability':float(sub[0]['detector_legacy_outage_probability']),
            'new_outage_probability_mean':float(np.mean([float(r['detector_source_size_outage_probability']) for r in sub]))}
    (out/'native_resource_points.json').write_text(json.dumps(native,indent=2)+'\n');(out/'snr_resource_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    grouped=defaultdict(list)
    for r in route:grouped[r['receiver'],r['budget_index']].append(r)
    points=[]
    for (receiver,budget),rs in grouped.items():
        points.append({'receiver':receiver,'budget_index':budget,'runs':len(rs),'validation_budget_proxy':rs[0]['validation_budget'],
            'test_mean_proxy':float(np.mean([r['test_actual_mean_hybrid_proxy'] for r in rs])),
            'accuracy_mean':float(np.mean([r['test_accuracy'] for r in rs])),
            'accuracy_seed_sd':float(np.std([r['test_accuracy'] for r in rs],ddof=1)),
            'test_budget_exceeded_runs':sum(r['test_budget_exceeded'] for r in rs)})
    for p in points:
        p['within_family_empirical_nondominated']=not any(q['receiver']==p['receiver'] and q['test_mean_proxy']<=p['test_mean_proxy'] and q['accuracy_mean']>=p['accuracy_mean'] and (q['test_mean_proxy']<p['test_mean_proxy'] or q['accuracy_mean']>p['accuracy_mean']) for q in points)
    (out/'within_family_route_points.json').write_text(json.dumps(points,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for receiver,color in [('qwen2','#0072B2'),('qwen25','#D55E00'),('smol','#009E73')]:
        ps=sorted([p for p in points if p['receiver']==receiver],key=lambda p:p['test_mean_proxy'])
        axes[0].errorbar([p['test_mean_proxy']/1000 for p in ps],[p['accuracy_mean']*100 for p in ps],
            yerr=[p['accuracy_seed_sd']*100 for p in ps],fmt='o-',label=receiver,color=color,capsize=3)
    axes[0].set_xlabel('Mixed-accounting resource proxy (thousands)');axes[0].set_ylabel('Test accuracy (%)');axes[0].set_title('Original family only; not physical channel uses');axes[0].set_ylim(55,85);axes[0].legend(frameon=False)
    for m,rs in by_method.items():
        n=rs[0]['complex_channel_symbols']
        if n:axes[1].scatter(n,100*np.mean([r['correct'] for r in rs]),label=f'T{n}',s=55)
    axes[1].set_xlabel('Explicit complex channel symbols');axes[1].set_ylabel('Test accuracy (%)');axes[1].set_title('T-DeepSC family; completed points only');axes[1].set_ylim(55,85);axes[1].set_xlim(0,105);axes[1].legend(frameon=False)
    fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(out/f'figures/native_resource_families.{ext}',dpi=400)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));snrs=list(summaries)
    axes[0].plot(snrs,[summaries[s]['jpeg_wire_bytes_mean_nonoutage']/1000 for s in snrs],'o-',label='Wire JPEG, successful source coding')
    axes[0].plot(snrs,[summaries[s]['jpeg_received_file_bytes_mean']/1000 for s in snrs],'s--',label='Saved received JPEG (not wire payload)')
    axes[0].set_xlabel('SNR (dB)');axes[0].set_ylabel('Mean bytes (thousands)');axes[0].set_ylim(bottom=0);axes[0].legend(fontsize=8,frameon=False)
    axes[1].plot(snrs,[100*summaries[s]['old_outage_probability'] for s in snrs],'o-',label='Legacy256-byte constant')
    axes[1].plot(snrs,[100*summaries[s]['new_outage_probability_mean'] for s in snrs],'s--',label='Actual source-text length replay')
    axes[1].set_xlabel('SNR (dB)');axes[1].set_ylabel('Per-record loss probability (%)');axes[1].set_ylim(bottom=0);axes[1].legend(fontsize=8,frameon=False);fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(out/f'figures/payload_reliability_audit.{ext}',dpi=400)
    plt.close(fig)
    # Paired evidence for the genuinely communication-matched receiver comparison.
    rs=by_method.get('rsvqa_d4_seed7',[]);stats={}
    if rs:
        bykey={(r['image'],r['question'],r['snr']):r for r in rs};ids=sorted({r['image'] for r in rows})
        counts=np.array([sum(r['image']==i for r in rows) for i in ids]);rng=np.random.default_rng(20260908)
        draw=rng.integers(0,len(ids),(10000,len(ids)))
        for receiver in ['qwen2','qwen25','smol']:
            d=np.array([sum(int(bykey[(r['image'],r['question'],int(r['snr']))]['correct'])-int(r[f'{receiver}_image_correct']) for r in rows if r['image']==i) for i in ids])
            point=d.sum()/counts.sum();boot=d[draw].sum(axis=1)/counts[draw].sum(axis=1)
            null=(rng.choice([-1,1],(20000,len(ids)))*d).sum(axis=1)/counts.sum();p=(1+np.count_nonzero(abs(null)>=abs(point)))/(len(null)+1)
            stats[receiver]={'RSVQA_minus_fixed_image_pp':float(100*point),'95_cluster_bootstrap_pp':(100*np.quantile(boot,[.025,.975])).tolist(),'cluster_sign_flip_p':float(p),'bonferroni3_p':min(1.,3*float(p))}
    (out/'same_JPEG_receiver_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    table='| Method | Accuracy (%) | Communication resource meaning |\n|---|---:|---|\n'
    for receiver in ['qwen2','qwen25','smol']:table+=f"| Fixed-image {receiver} | {100*np.mean([int(r[f'{receiver}_image_correct']) for r in rows]):.4f} | Same canonical JPEGs / ideal-rate accounting |\n"
    for m,rs in by_method.items():table+=f"| {m} | {100*np.mean([r['correct'] for r in rs]):.4f} | {'Explicit '+str(rs[0]['complex_channel_symbols'])+' complex symbols; no physical time/J mapping' if rs[0]['complex_channel_symbols'] else 'Exactly same JPEG communication as fixed-image receivers'} |\n"
    table+=f"| Fixed detection, source-size loss replay | {100*correction['test']['corrected_accuracy']:.4f} | Nominal BPSK resource + separate application-level loss abstraction |\n"
    (out/'numeric-summary.md').write_text(table)
    (out/'stats-appendix.md').write_text('# Statistical scope\n\nSame-JPEG receiver differences use paired104-image cluster bootstrap (10000) and sign-flip tests(20000), with Bonferroni for three contrasts. Questions/SNR repeats remain clustered. Prior test exposure and same-flight frame dependence make inference exploratory. T three channel seeds are not three training runs. Router error bars are10 training-seed SD for frozen model scores; they do not repair the physical-model mismatch. No cross-PHY superiority test is performed. No normality approximation or iid-SNR assumption.\n')
    (out/'figure-catalog.md').write_text('# Figures\n\n## native_resource_families\nPurpose: show actual available trade-offs without mixing incompatible units. Left: validation-selected frozen-score routes under corrected detector loss, original hybrid accounting; error bars10 seed SD. Right: completed T symbols/accuracy points only; no unmeasured48/96 interpolation. The panels MUST NOT be read as a common physical Pareto frontier.\n\n## payload_reliability_audit\nPurpose: distinguish saved file bytes from transmitter JPEG bytes and expose fixed256-byte loss-size mismatch. Means weighted over2808 decisions/104images; SNR repeats are not independent seeds. Right curve recomputes record loss and downstream answers, not merely cost. No uncertainty bars for this deterministic replay.\n')

if __name__=='__main__':main()
