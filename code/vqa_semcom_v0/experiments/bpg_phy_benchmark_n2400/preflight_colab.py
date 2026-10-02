"""No-inference asset/GPU preflight; missing asset list, original bytes untouched."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from verify_rule_b import verify_scanner

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--reuse-phy',type=Path,required=True);a=p.parse_args()
    root=Path(os.environ.get('VQA_SEMCOM_ROOT','/content/vqa_semcom')).resolve()
    scanner=root/'code/vqa_semcom_v0/experiments/rgb_channel_snr_scan/run_snr_scan.py'
    assets=['outputs/rgb_selector_large_20260922/test_manifest.json','outputs/rgb_selector_large_20260922/test_records.json',
            'outputs/rgb_selector_large_20260922/test_truth.sealed.json','outputs/rgb_selector_large_20260922/grid_test/representations.json',
            'outputs/prerun_protocol_finalization_20260908/phy.py','outputs/rgb_rate_visual_budget_20260922/code/runtime.py',
            'outputs/qwen_compression_joint_stage1_20260921/run_stage1.py','outputs/tdiuc_qlora_pilot_20260916/run_training.py',
            'outputs/rgb_joint_selector_20260922/code/route_frame.py','outputs/tdiuc_qlora_pilot_20260916/standard/final_adapter/adapter_model.safetensors']
    required=[root/x for x in assets]+[scanner]+[a.reuse_phy/x for x in ['protocol.json','report.json','transmissions.jsonl']]
    missing=[str(x) for x in required if not x.is_file()]
    result={'root':str(root),'missing_assets':missing,'new_phy':0,'new_vlm':0}
    if missing:
        result['status']='ASSET_BUNDLE_NEEDED';print(json.dumps(result,indent=2));return 2
    result['freeze']=verify_scanner(scanner)
    def mapped(path):
        old=Path('/home/qiankun/phd_research/vqa_semcom');path=Path(path)
        return root/path.relative_to(old) if path.is_relative_to(old) else path
    data=root/'outputs/rgb_selector_large_20260922'
    manifest=json.loads((data/'test_manifest.json').read_text())
    reps=json.loads((data/'grid_test/representations.json').read_text())
    missing=[str(mapped(r['file'])) for r in manifest if not mapped(r['file']).is_file()]
    missing += [str(mapped(r[f])) for r in reps if r['budget'] in [2000,4000] for f in ['packet','decoded'] if not mapped(r[f]).is_file()]
    result['missing_assets']=missing
    if missing:
        result['status']='ASSET_BUNDLE_NEEDED';print(json.dumps(result,indent=2));return 2
    import torch
    result['cuda_available']=torch.cuda.is_available()
    result['gpu']=torch.cuda.get_device_name(0) if result['cuda_available'] else None
    result['native_bf16_supported']=torch.cuda.is_bf16_supported() if result['cuda_available'] else False
    result['status']='ASSETS_PRESENT_RECEIVER_SMOKE_STILL_REQUIRED' if result['cuda_available'] else 'GPU_NEEDED'
    result['note']='L4 preferred for original BF16/NF4 receiver; T4 requires bounded dtype compatibility check, no silent dtype change.'
    print(json.dumps(result,indent=2));return 0 if result['cuda_available'] else 2

if __name__=='__main__':raise SystemExit(main())
