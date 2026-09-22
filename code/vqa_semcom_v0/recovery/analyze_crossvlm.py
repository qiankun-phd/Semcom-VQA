#!/usr/bin/env python3
import csv, zlib
from collections import defaultdict

def is_test(i, frac=0.2): return (zlib.crc32(str(i).encode()) % 100) < int(frac*100)
def ok(v): return str(v).strip().lower() in ("true","1","yes")

def per_qtype(paths):
    # qtype -> service -> [k,n]
    acc = defaultdict(lambda: defaultdict(lambda: [0,0]))
    for p in paths:
        try: rows = list(csv.DictReader(open(p)))
        except FileNotFoundError: continue
        for r in rows:
            if not is_test(r["image_id"]): continue
            s = r["service_level"]
            if s not in ("1","2"): continue
            a = acc[r["question_type"]][s]; a[1]+=1; a[0]+=int(ok(r["correct"]))
    return acc

# 2B: from v3_0_rician (has all 5 qtypes, services 0/1/2)
b2 = per_qtype(["outputs/vlm/v3_0_rician_predictions.csv"])
# 3B: v25 main (presence,counting) + cmp (comparison)
b3 = per_qtype(["outputs/vlm/v25_rician_main_predictions.csv","outputs/vlm/v25_rician_cmp_predictions.csv"])

def rate(a,s): k,n=a[s]; return k/n if n else None
print("%-12s | %-19s | %-19s | routing" % ("qtype","2B  token / image","3B  token / image"))
print("-"*72)
for qt in ["presence","counting","comparison","co_presence","threshold"]:
    t2,i2 = rate(b2[qt],"1"), rate(b2[qt],"2")
    t3,i3 = rate(b3[qt],"1"), rate(b3[qt],"2")
    def r(t,i):
        if t is None or i is None: return "-"
        return "token" if t>=i else "image"
    def f(x): return "%.3f"%x if x is not None else "  -  "
    print("%-12s | %s / %s | %s / %s | 2B:%s 3B:%s" % (
        qt, f(t2),f(i2), f(t3),f(i3), r(t2,i2), r(t3,i3)))
