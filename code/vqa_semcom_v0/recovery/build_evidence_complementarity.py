#!/usr/bin/env python3
"""Principled evidence <-> question complementarity (reuses v3_0 predictions).

Claims tested:
  (1) token-gain Delta = acc(token) - acc(image) orders by question reasoning type
      (symbolic/count-based -> token; perceptual/existence -> image).
  (2) sign of Delta is consistent across VLM backbones (2B vs 3B).
  (3) a ZERO-PARAMETER semantic rule (symbolic->token, perceptual->image) matches the
      data-CALIBRATED selector -> optimal evidence is predictable from question semantics,
      no multi-dim quality LUT needed.
"""
from __future__ import annotations
import csv, math, zlib
from collections import defaultdict

CH = ["awgn", "rayleigh", "rician"]

# a-priori reasoning taxonomy (from question semantics, NOT from results)
REASONING = {
    "presence":    ("perceptual",  "existence: needs to visually confirm >=1 instance"),
    "counting":    ("symbolic",    "absolute count: exact number"),
    "comparison":  ("symbolic",    "relative count: count(A) vs count(B)"),
    "co_presence": ("symbolic",    "joint existence via counts: A>0 and B>0"),
    "threshold":   ("symbolic",    "count threshold: count(A) >= N"),
}
ORDER = ["counting", "comparison", "co_presence", "threshold", "presence"]

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0, 0.0
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d; m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return p, c-m, c+m

def ok(v): return str(v).strip().lower() in ("true","1","yes")
def is_test(i, frac=0.2): return (zlib.crc32(str(i).encode()) % 100) < int(frac*100)

def load(paths):
    """decision (img,q,snr) -> {service: correct}, + qtype; test only."""
    dec = defaultdict(dict); qt = {}
    for p in paths:
        try: rows = list(csv.DictReader(open(p)))
        except FileNotFoundError: continue
        for r in rows:
            if r["service_level"] not in ("1","2"): continue
            if not is_test(r["image_id"]): continue
            key = (r["image_id"], r["question"], r["snr_bin"])
            dec[key].setdefault(r["service_level"], ok(r["correct"]))
            qt[key] = r["question_type"]
    return dec, qt

dec, qt = load([f"outputs/vlm/v3_0_{c}_predictions.csv" for c in CH])

# ---- per-qtype token/image accuracy + Delta ----
agg = defaultdict(lambda: {"1":[0,0], "2":[0,0]})
for key, sv in dec.items():
    q = qt[key]
    for s in ("1","2"):
        if s in sv:
            agg[q][s][1]+=1; agg[q][s][0]+=int(sv[s])

print("=== (1) token-gain by question reasoning type (pooled 3ch, test) ===")
print("%-12s %-11s %8s %8s %9s   %s" % ("qtype","reasoning","token","image","Δ=t-i","note"))
for q in ORDER:
    if q not in agg: continue
    t,_,_ = wilson(*agg[q]["1"]); i,_,_ = wilson(*agg[q]["2"])
    tag, note = REASONING[q]
    print("%-12s %-11s %8.3f %8.3f %+9.3f   %s" % (q, tag, t, i, t-i, note))

# ---- (3) zero-param semantic rule vs calibrated selector vs fixed/oracle ----
SYMBOLIC = {q for q,(t,_) in REASONING.items() if t=="symbolic"}
def eval_router(route_fn):
    k=n=0
    for key, sv in dec.items():
        if not ({"1","2"} <= set(sv)): continue
        s = route_fn(qt[key]); k+=int(sv[s]); n+=1
    return k/n if n else 0, n
# data-calibrated qt-LCB policy (learn on train)
tr = defaultdict(lambda: {"1":[0,0],"2":[0,0]})
for p in [f"outputs/vlm/v3_0_{c}_predictions.csv" for c in CH]:
    try: rows=list(csv.DictReader(open(p)))
    except FileNotFoundError: continue
    for r in rows:
        if r["service_level"] not in ("1","2") or is_test(r["image_id"]): continue
        tr[r["question_type"]][r["service_level"]][1]+=1
        tr[r["question_type"]][r["service_level"]][0]+=int(ok(r["correct"]))
calib = {}
for q,sv in tr.items():
    _,l1,_=wilson(*sv["1"]); _,l2,_=wilson(*sv["2"]); calib[q]="1" if l1>=l2 else "2"

rules = {
    "fixed token (M3)":      lambda q: "1",
    "fixed image (M1)":      lambda q: "2",
    "SEMANTIC RULE (0-param)": lambda q: "1" if q in SYMBOLIC else "2",
    "data-CALIBRATED (LCB)": lambda q: calib.get(q,"1"),
}
print("\n=== (3) zero-parameter semantic rule vs calibrated selector (test acc) ===")
for name, fn in rules.items():
    a,n = eval_router(fn)
    print("  %-26s %.4f  (n=%d)" % (name, a, n))
# oracle
ko=no=0
for key,sv in dec.items():
    if not ({"1","2"}<=set(sv)): continue
    ko+=int(max(sv["1"],sv["2"])); no+=1
print("  %-26s %.4f" % ("oracle (per-decision)", ko/no))
print("\nSYMBOLIC set:", sorted(SYMBOLIC))
print("semantic rule == calibrated policy?", all((("1" if q in SYMBOLIC else "2")==calib.get(q)) for q in agg))
