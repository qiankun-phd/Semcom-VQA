"""Six generated images, real BPG encode/decode; no benchmark, PHY, VLM or labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import budgeted_bpg as bpg

def main():
    from PIL import Image, ImageDraw
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bpgenc',default='bpgenc');p.add_argument('--bpgdec',default='bpgdec');a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing existing fixture output')
    a.output.mkdir(parents=True);start=time.monotonic()
    binary=bpg.fingerprints(a.bpgenc,a.bpgdec);rows=[];calls=0
    sizes=[(512,384),(400,300),(384,256),(192,160),(640,480),(320,240)]
    for i,size in enumerate(sizes):
        if i==5:
            rng=random.Random(1701);im=Image.frombytes('RGB',size,bytes(rng.randrange(256) for _ in range(size[0]*size[1]*3)))
        else:
            im=Image.new('RGB',size,(20+i*25,80,160));draw=ImageDraw.Draw(im)
            for j in range(12):draw.rectangle((j*size[0]//12,0,(j+1)*size[0]//12,size[1]),fill=(j*20,(i*50+j*11)%256,255-j*15))
            draw.ellipse((20,20,size[0]-20,size[1]-20),outline='white',width=5)
        im.save(a.output/f'fixture_{i}.png')
        first=bpg.encode_bpg_caps(im,[1999,3999],bpgenc=a.bpgenc)
        repeated=bpg.encode_bpg_caps(im,[1999,3999],bpgenc=a.bpgenc)
        calls+=max(x.attempts for x in first.values())+max(x.attempts for x in repeated.values())
        for cap in [1999,3999]:
            result=first[cap];assert result.payload is not None and result.payload==repeated[cap].payload
            packet=bpg.frame(result.payload);assert len(packet)<=cap+1 and bpg.unframe(packet)==result.payload
            decoded=bpg.decode_bpg(bpg.unframe(packet),bpgdec=a.bpgdec)
            assert decoded.mode=='RGB' and max(decoded.size)<=result.side
            (a.output/f'fixture_{i}_{cap+1}.bin').write_bytes(packet)
            decoded.save(a.output/f'fixture_{i}_{cap+1}_decoded.png')
            rows.append({'fixture':i,'budget_bytes':cap+1,'framed_bytes':len(packet),'bpg_side':result.side,
                'bpg_qp':result.qp,'candidate_attempts':result.attempts,'first_fit_seconds':result.seconds,
                'actual_symbols':510*((len(packet)+47)//48),'packet_sha256':hashlib.sha256(packet).hexdigest(),
                'decoded_width':decoded.width,'decoded_height':decoded.height,'deterministic_reencode':True})
    for invalid in [b'J'+bpg.NATIVE_MAGIC,b'Bbad']:
        try:bpg.unframe(invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid transport prefix accepted')
    with (a.output/'fixture_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'status':'PASS','fixture_images':6,'framed_cases':12,'candidate_encoder_invocations_including_determinism_recheck':calls,
        'native_decoder_invocations':12,'seconds':time.monotonic()-start,'bpg_binaries':binary,
        'new_phy':0,'new_vlm':0,'new_training':0,'benchmark_queries_evaluated':0,
        'warning':'Generated fixture codec smoke only; not n=6 semantic/PHY benchmark',
        'all_caps_met':True,'deterministic_payloads':True,'invalid_transport_frames_rejected':True}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
