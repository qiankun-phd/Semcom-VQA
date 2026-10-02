"""Independent completed-artifact checks; does not run PHY or VLM."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();o=a.output
    protocol=json.loads((o/'protocol.json').read_text());report=json.loads((o/'report.json').read_text())
    assert report['status']=='PASS' and report['historical_proxy_used'] is False
    assert protocol['codecs']==['bpg','learned_frozen'] and protocol['snrs']==[0.,5.,10.,20.]
    tx=[json.loads(line) for line in (o/'transmissions.jsonl').read_text().splitlines()]
    assert len(tx)==len({r['task_key'] for r in tx})==report['all_unique_tasks_completed']
    fades={}
    for r in tx:
        assert r['delivery']=='real_bitstream_phy' and r['success']==r['payload_identity']
        assert r['actual_symbols']==510*math.ceil(r['payload_bytes']/48)
        assert abs(r['effective_snr_db']-r['nominal_snr_db']-10*math.log10(21420/r['actual_symbols']))<1e-12
        if r['success']:assert r['blocks_decoded']==r['total_blocks']
        key=r['image_id'],r['channel_seed'];fades.setdefault(key,set()).add(r['fade_power'])
    assert all(len(v)==1 for v in fades.values())
    rows=list(csv.DictReader((o/'real_phy_per_seed.csv').open()));means=list(csv.DictReader((o/'real_phy_mean_sd.csv').open()))
    assert len(rows)==72 and len(means)==24
    for r in rows:
        assert int(r['n'])==protocol['n']
        assert 0<=int(r['strict_correct'])<=int(r['delivered'])<=int(r['n'])
        assert abs(float(r['acc_pct'])-100*int(r['strict_correct'])/int(r['n']))<1e-12
        if r['policy']=='proposed_snr_adaptive_v2':
            fixed='fixed_2000_medium' if float(r['snr_db'])<=10 else 'fixed_4000_medium'
            other=next(v for v in rows if v['codec']==r['codec'] and v['snr_db']==r['snr_db'] and v['channel_seed']==r['channel_seed'] and v['policy']==fixed)
            assert all(r[k]==other[k] for k in r if k!='policy')
    for m in means:
        rr=[r for r in rows if r['codec']==m['codec'] and r['snr_db']==m['snr_db'] and r['policy']==m['policy']]
        assert abs(float(m['acc_mean_pct'])-statistics.mean(float(r['acc_pct']) for r in rr))<1e-12
        assert abs(float(m['acc_sd_pp'])-statistics.stdev(float(r['acc_pct']) for r in rr))<1e-12
    encoded=json.loads((o/'encoded.json').read_text());assert len(encoded)==protocol['n']*2
    for r in encoded:
        if r['encode_failure']:continue
        packet=Path(r['packet']).read_bytes();assert packet.startswith(b'BBPG\xfb') and len(packet)==r['image_bytes']<=r['budget']
    freeze=json.loads((o/'freeze_final.json').read_text());assert freeze['passed'] and freeze['status']=='PASS'
    result={'status':'PASS','unique_phy_tasks':len(tx),'logical_grid_evaluations':report['logical_grid_evaluations'],
        'actual_ns_ppc_verified':True,'paired_fading_verified':True,'compositional_rule_b_verified':True,
        'mean_sd_denominators_verified':True,'no_proxy_rows':True,'native_bpg_frames_verified':True}
    (o/'artifact_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
