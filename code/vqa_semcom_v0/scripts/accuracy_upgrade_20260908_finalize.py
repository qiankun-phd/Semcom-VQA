#!/usr/bin/env python3
"""Final cache/provenance checks and validation-pilot report; never trains."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from PIL import Image
import router_network_revision_20260908 as b
from analyze_router_network_revision_20260908 import cluster_contrast

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);args=ap.parse_args()
    root=args.repo/'outputs/accuracy_upgrade_20260908';out=root/'pilot_analysis';out.mkdir(exist_ok=False)
    sys.path.insert(0,str(args.repo/'src'))
    from vqa_semcom.vlm.answer import check_answer
    for phase in ('phase_a','phase_b','phase_c'):
        assert b.read(root/phase/'status.json')['state']=='COMPLETE',(phase,'not complete')
    det=b.read(root/'phase_b/summary.json');diag=b.read(root/'diagnostics_v2/sender_no_channel_qa_diagnostic.json')
    c=b.read(root/'phase_c/summary.json');budget=b.read(root/'phase_c/budget_selection.json')
    results={k:b.read(root/f'phase_c/{k}_predictions.json') for k in ('7b_nf4','3b_nf4')}
    n=budget['decisions_per_model'];images=np.array([r['image'] for r in results['7b_nf4']])
    assert all(len(rows)==n for rows in results.values())
    for large,small in zip(results['7b_nf4'],results['3b_nf4']):
        for field in ('index','image','question','qt','snr','ground_truth','image_sha256','received_size','vision_size','image_grid_thw','prompt_sha256'):
            assert large[field]==small[field],field
        for row in (large,small):
            check=check_answer(row['qt'],row['prediction'],row['ground_truth'])
            assert check.correct==row['correct'] and check.normalized_prediction==row['normalized']
    correct={k:np.array([r['correct'] for r in rows],float) for k,rows in results.items()}
    diff=correct['7b_nf4']-correct['3b_nf4'];rng=np.random.default_rng(20260908)
    stats=cluster_contrast(diff,images,rng)
    sums=np.array([diff[images==iid].sum() for iid in np.unique(images)])
    exact=np.array(list(itertools.product([-1,1],repeat=len(sums))))@sums/len(diff)
    stats['exact_two_sided_image_sign_flip_p']=float(np.mean(np.abs(exact)>=abs(diff.mean())-1e-15))
    stats['scope']='single predeclared7B-NF4 vs3B-NF4 validation contrast;8images, correlated frames, exploratory only'
    b.dump(out/'vlm_pair_qa.json',{'paired_rows':n,'identical_images_prompts_grids_questions':True,'rescoring_exact':True})
    b.dump(out/'vlm_exploratory_stats.json',stats)
    cached=np.array([r['cached_3b_correct'] for r in results['3b_nf4']],float)
    b.dump(out/'vlm_cached_context.json',{'n':n,'cached_3b_original_precision_accuracy':float(cached.mean()),
        'new_3b_nf4_accuracy':float(correct['3b_nf4'].mean()),'caution':'cached precision differs, not a scale-only control'})
    rows=['# Validation detector and stronger-VLM pilots','','These pilots are separate from the full2808-key phaseA test evaluation. Fixed8validation images, including3frames from the same video; lexical selection is not representative sampling. No test-based choice or manuscript update.','','## Detector quality','','Same VisDrone YOLO checkpoint. Four predeclared settings, same NMSIoU.7/max_det300. Original640/.25 counts exactly reproduce the sender cache on all8images. IoU.5matching is class-aware greedy one-to-one, not official VisDroneAP; ignored-region suppression is absent. Unmatched detections include localization/class errors, so they are not called verified duplicates.','','| Setting | Recall (%) | Precision (%) | Count MAE/image-class | Count bias | Overlap pairs IoU>.7 | ms/image | Sender no-link QA (%) |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,v in det.items():rows.append(f'| {name} | {v["recall_iou05"]*100:.3f} | {v["precision_iou05"]*100:.3f} | {v["count_mae_per_image_class"]:.4f} | {v["count_bias_per_image_class"]:.4f} | {v["overlap_pairs_iou07"]} | {v["mean_seconds_per_image"]*1000:.3f} | {diag[name]["accuracy"]*100:.2f} |')
    rows+=['','Sender no-link QA contains40unique questions, no channel corruption and no counting calibration. It is a diagnostic only and does not use old received labels with new boxes. Higher resolution improves recall and counting MAE but does not improve this pilot’s overall symbolic QA; simply lowering confidence creates more false positives and can harm QA. These results do not warrant a full end-to-end replacement without further controlled validation.','','| Class | GT | 640/.25 predicted / MAE | 1280/.25 predicted / MAE | 640/.15 predicted / MAE | 1280/.15 predicted / MAE |','|---|---:|---:|---:|---:|---:|']
    for cls in next(iter(det.values()))['per_class']:
        values=[v['per_class'][cls] for v in det.values()]
        rows.append(f'| {cls} | {values[0]["gt"]} | '+' | '.join(f'{v["predicted"]} / {v["mae"]:.3f}' for v in values)+' |')
    rows+=['','## Stronger receiver, matched quantization','','Both Qwen2.5 models use NF4double quantization and BF16compute/nonquantized modules. Each receives the same JPEG, source prompt (including channel/SNR metadata), deterministic24new-token limit, official model-specific processor with min200704/max802816pixels. Actual grids, JPEG hashes, prompts and scoring were verified per decision. No OOM-driven resizing or new prompt was used.','','| Model | N | Accuracy (%) | Mean inference seconds/query | Unknown answers |','|---|---:|---:|---:|---:|']
    for name,v in c.items():rows.append(f'| {name} | {v["n"]} | {v["accuracy"]*100:.3f} | {v["mean_inference_seconds"]:.4f} | {v["unknown_answers"]} |')
    rows+=['',f'Paired7Bminus3B effect: {diff.mean()*100:+.3f}pp. Cached3B original-precision accuracy on these same keys is {cached.mean()*100:.3f}%, provided only as context: comparing7B-NF4 directly with cached3B-BF16 is not precision-controlled. Eight images are too few, and include related frames, for general model-superiority claims. The full pooled task composition differs from this balanced5type pilot.','','| Type | 7B NF4 (%) | 3B NF4 (%) |','|---|---:|---:|']
    for qt in c['7b_nf4']['per_type']:rows.append(f'| {qt} | {c["7b_nf4"]["per_type"][qt]*100:.3f} | {c["3b_nf4"]["per_type"][qt]*100:.3f} |')
    rows+=['','## Practical boundaries','','Recorded inference wall time synchronizesCUDA but excludes preprocessing and model load (these are separately saved). GPUpeak allocation counters are cumulative within the pilot process, not independently reset per model; use model-specific get_memory_footprint values for model-storage comparison and do not compare cumulative peaks as isolated working-memory measurements. The RTX4060 power sensor returnsN/A, so no Joule number or measured onboard energy is claimed. Old32.31J/.4275J values were not reused. No changed detector/7B answers are injected into phaseA or manuscript results.','','## Parsing and image-path audit','','Training/validation old predictions for all3receivers reproduce the original parser and correctness exactly. Qwen2 threshold has158/2058training and46/606validation unknown outputs; inspected examples answer an integer to a yes/no threshold question. The original prompt explicitly specifies output forms only for presence/counting; this is a plausible issue for a future prompt-controlled validation, not grounds to rewrite cached scores. No parser/prompt change was made in this experiment. The audit stores actual source/received dimensions, qwen-utils resize and processor grids.','','## Next decision','','The clear completed improvement is phaseA: sender-visible question/detector features add2.20–4.31pp without new detector/VLM calls. Freeze that schema and validate independently before a manuscript replacement. Detector resolution warrants narrow further investigation, but lower thresholds and overall symbolic QA do not support blind deployment. Interpret the7B result only as this matched-precision validation pilot, not a full-test gain or an energy saving.']
    (out/'analysis-report.md').write_text('\n'.join(rows)+'\n')
    ci=stats['cluster_bootstrap_95_percentile_ci_pp']
    (out/'stats-appendix.md').write_text(f'# Pilot uncertainty\n\nOne predeclared7Bvs3Bcontrast,120decisions within8images. Pooled effect {stats["effect_pp"]:+.4f}pp; image-bootstrap10000replications95%percentile interval [{ci[0]:+.4f},{ci[1]:+.4f}]pp. Exact two-sided image sign-flip overall256patterns p={stats["exact_two_sided_image_sign_flip_p"]:.6f}. No multi-contrast adjustment for this single pilot contrast. Independence/exchangeability of images is doubtful because3frames share a video, and8clusters is small; descriptive pilot evidence, not confirmatory inference. Detector comparisons are descriptive onthe same8images with no repeat-trained detector variation. OfficialAP/inferential claims are not supported by this diagnostic scope.\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(8,3.5))
    names=list(det);colors=['#0072B2','#E69F00','#009E73','#D55E00']
    for i,(name,color) in enumerate(zip(names,colors)):
        axes[0].scatter(det[name]['recall_iou05']*100,det[name]['count_mae_per_image_class'],color=color,marker=['o','s','^','D'][i],s=55,label=name)
    axes[0].set_xlabel('Validation detection recall at IoU0.5 (%)');axes[0].set_ylabel('Count MAE/image-class');axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=7,frameon=False,loc='upper right')
    qts=list(c['7b_nf4']['per_type'])
    for i,(name,color,marker) in enumerate([('3b_nf4','#0072B2','o'),('7b_nf4','#D55E00','D')]):
        axes[1].plot(range(5),[c[name]['per_type'][q]*100 for q in qts],color=color,marker=marker,label=name)
    axes[1].set_xticks(range(5),['Presence','Count','Compare','Co-pres.','Threshold'],rotation=35,ha='right');axes[1].set_ylim(0,105)
    axes[1].set_ylabel('Validation answer accuracy (%)');axes[1].grid(alpha=.2);axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/f'pilot_comparisons.{ext}',dpi=600)
    plt.close(fig)
    (out/'figure-catalog.md').write_text('# Pilot figure catalog\n\n`pilot_comparisons`: Leftshows recall/countMAE tradeoff for4predeclared YOLOsettings on8validationimages; dots do not imply end-to-end QA improvements. Rightshows matchedNF4 3B/7B pertype accuracy,24decisions/type (8images×3SNR), no seed-based errorbars because eachmodel is deterministic. Caption must state pilotcomposition differs from fullpooled mainbenchmark and relatedimages limit generalization. Decision: evaluate detector and receiver improvements separately; neither should silently changephaseAlabels. No statistical stars.\n')
    # Source-image hashes support the extra area-feature provenance without copying imagery.
    common=b.read(args.repo/'outputs/revision_20260907_independent/crossreceiver_v2/common_keys.json')
    image_manifest={}
    for iid in sorted({k['image'] for k in common}):
        path=args.repo/f'data/raw/visdrone/DET/val/images/{iid}.jpg'
        with Image.open(path) as im:size=list(im.size)
        image_manifest[iid]={'size':size,'sha256':b.sha(path),'bytes':path.stat().st_size}
    b.dump(root/'source_image_manifest.json',image_manifest)
    # Checkpoint files are large; retain hashes and paths, not copies in the manuscript folder.
    checkpoint=b.read(root/'download_status.json')['path'];weight_manifest={}
    for path in sorted(Path(checkpoint).iterdir()):
        if path.is_file():weight_manifest[path.name]={'sha256':b.sha(path),'bytes':path.stat().st_size}
    b.dump(root/'7b_checkpoint_manifest.json',{'snapshot':checkpoint,'files':weight_manifest})
    code=root/'code';code.mkdir()
    for path in (args.repo/'outputs').glob('*accuracy_upgrade_20260908*'):
        if path.is_file() and path.suffix in ('.py','.sh'):shutil.copy2(path,code/path.name)
    for name in ('router_network_revision_20260908.py','analyze_router_network_revision_20260908.py'):
        shutil.copy2(args.repo/'outputs'/name,code/name)
    shutil.copy2(args.repo/'outputs/accuracy-upgrade-2026-09-08.md',root/'plan_snapshot.md')
    shutil.copy2(args.repo/'outputs/accuracy_upgrade_20260908_phase_a.log',root/'phase_a.log')
    b.dump(root/'attempts.json',{'smoke_a':'retained fail-before-fit: known others category','smoke_a2':'complete12twoepochfits',
        'diagnostics':'retained metadata-only attempt stopped after optional online lookup stalled','diagnostics_v2':'complete explicit cachedsnapshot offline',
        'slow_pip_transfer':'own exactPID stopped; samewheel mirror copy officialPyPIsha verified before isolatedinstall',
        'old_environment_modified':False,'manuscript_modified':False,'new_VLM_test_run':False})
    b.dump(root/'status.json',{'state':'COMPLETE','phase_a':'COMPLETE66fits verified','phase_b':'COMPLETE32validationdetectorpasses',
        'phase_c':f'COMPLETE{2*n}validationVLManswers matchedNF4','analysis':'COMPLETE','completed_unix':time.time()})
    manifest={str(p.relative_to(root)):b.sha(p) for p in sorted(root.rglob('*')) if p.is_file()
        and not any(part in ('quant_env','wheels') for part in p.relative_to(root).parts) and p.name!='bundle_manifest.json'}
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    b.dump(root/'bundle_manifest.json',{'files':manifest,'digest':digest,'exclusions':['quant_env','wheels'],'file_count':len(manifest)})
    print(json.dumps({'state':'COMPLETE','files':len(manifest),'digest':digest,'pilot':c,'effect':stats},indent=2),flush=True)

if __name__=='__main__':main()
