#!/usr/bin/env python3
"""Would a cliff-prone digital image path make SNR 'live'? (reuse existing preds)
Compare token(s1) vs naive-cliff-image(M0_naive s2) per (qtype, snr); does the winner flip?
Then: qtype-only routing vs (qtype+snr) routing when the image path cliffs -> accuracy gained."""
import csv, zlib
from collections import defaultdict
CH = ["awgn", "rayleigh", "rician"]
def ok(v): return str(v).strip().lower() in ("true","1","yes")
def is_test(i, f=0.2): return (zlib.crc32(str(i).encode())%100) < int(f*100)
def snrv(s): return float(str(s).replace("dB",""))

# token (s1) from v3_0 ; naive cliff image (s2) from v2_0_{ch}_naive
tok = defaultdict(lambda: defaultdict(lambda:[0,0]))   # qt -> snr -> [k,n]
for c in CH:
    for r in csv.DictReader(open(f"outputs/vlm/v3_0_{c}_predictions.csv")):
        if r["service_level"]=="1" and is_test(r["image_id"]):
            a=tok[r["question_type"]][r["snr_bin"]]; a[1]+=1; a[0]+=int(ok(r["correct"]))
cliff = defaultdict(lambda: defaultdict(lambda:[0,0]))
for c in CH:
    try: rows=list(csv.DictReader(open(f"outputs/vlm/v2_0_{c}_naive_predictions.csv")))
    except FileNotFoundError: continue
    for r in rows:
        if r["service_level"]=="2" and is_test(r["image_id"]):
            a=cliff[r["question_type"]][r["snr_bin"]]; a[1]+=1; a[0]+=int(ok(r["correct"]))

qts = sorted(set(tok)&set(cliff))
print("naive-image qtypes:", sorted(cliff))
for qt in ["presence","counting"]:
    if qt not in cliff:
        print(f"[{qt}] no naive-image data"); continue
    print(f"\n=== {qt}: token vs cliff-image by SNR (test) ===")
    print("  snr  token  cliffImg  winner")
    for s in sorted(set(tok[qt])|set(cliff[qt]), key=snrv):
        t=tok[qt][s]; i=cliff[qt][s]
        tr=t[0]/t[1] if t[1] else 0; ir=i[0]/i[1] if i[1] else 0
        print("  %4s  %.3f   %.3f    %s"%(s,tr,ir,"token" if tr>=ir else "IMAGE"))

# accuracy: qtype-only routing vs qtype+snr routing, when image = cliff path
# build per (qt[,snr]) best on pooled test (proxy: use the same test as report)
def route_acc(use_snr):
    k=n=0
    for qt in qts:
        snrs = sorted(set(tok[qt])&set(cliff[qt]), key=snrv)
        # qtype-only: pick globally better modality for this qt (pooled)
        if not use_snr:
            T=sum(tok[qt][s][0] for s in snrs); Tn=sum(tok[qt][s][1] for s in snrs)
            I=sum(cliff[qt][s][0] for s in snrs); In=sum(cliff[qt][s][1] for s in snrs)
            pick_tok_all = (T/Tn if Tn else 0) >= (I/In if In else 0)
        for s in snrs:
            t=tok[qt][s]; i=cliff[qt][s]
            if use_snr:
                pick_tok = (t[0]/t[1] if t[1] else 0) >= (i[0]/i[1] if i[1] else 0)
            else:
                pick_tok = pick_tok_all
            sel = t if pick_tok else i
            k+=sel[0]; n+=sel[1]
    return k/n if n else 0

print("\n=== accuracy: routing when image path = cliff digital ===")
print("  qtype-only routing : %.4f"%route_acc(False))
print("  qtype+SNR routing  : %.4f"%route_acc(True))
