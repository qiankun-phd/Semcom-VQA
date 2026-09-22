#!/usr/bin/env python3
"""Idempotent patch: add co_presence + threshold question types (yes/no, count-derived)."""
import sys, py_compile
REPO = "/home/qiankun/phd_research/uav-vqa-semantic-rl"

# 1) answer.py yes/no branch
ap = f"{REPO}/src/vqa_semcom/vlm/answer.py"
s = open(ap).read()
old = 'if question_type in {"presence", "risk", "comparison"}:'
new = 'if question_type in {"presence", "risk", "comparison", "co_presence", "threshold"}:'
if new in s:
    print("answer.py already has extra types")
elif old in s:
    open(ap, "w").write(s.replace(old, new)); print("answer.py patched")
else:
    print("ERROR answer.py anchor missing"); sys.exit(1)

# 2) eval s1 decode: insert elif for co_presence/threshold before the calibrated branch
ep = f"{REPO}/scripts/run_v1_detector_eval.py"
s = open(ep).read()
anchor = "                                    semantic_decoded = True\n                                elif use_semantic_decoder and (\n"
ins = (
    "                                    semantic_decoded = True\n"
    '                                elif task.get("question_type") in ("co_presence", "threshold"):\n'
    '                                    ca = sum(1 for rr in transmitted_records if rr.category == task.get("target_class", ""))\n'
    '                                    if task.get("question_type") == "co_presence":\n'
    '                                        cb2 = sum(1 for rr in transmitted_records if rr.category == task.get("target_class_b", ""))\n'
    '                                        xpred = "yes" if (ca > 0 and cb2 > 0) else "no"\n'
    "                                    else:\n"
    "                                        try:\n"
    '                                            thr = int(float(task.get("threshold_n", "1")))\n'
    "                                        except ValueError:\n"
    "                                            thr = 1\n"
    '                                        xpred = "yes" if ca >= thr else "no"\n'
    '                                    _xchk = check_answer(task["question_type"], xpred, task["answer"], tolerance_ratio=float(vlm_cfg.get("count_tolerance_ratio", 0.10)))\n'
    "                                    predicted, normalized, correct = xpred, _xchk.normalized_prediction, _xchk.correct\n"
    '                                    payload_bytes = _payload_bytes(service_level, evidence_type, evidence_repr, "")\n'
    "                                    prediction_cache[cache_key] = (\n"
    '                                        predicted, normalized, correct, detector_latency, "semantic-token-decoder",\n'
    "                                        evidence_type, evidence_repr, payload_bytes, detector_latency, detector_model, ca,\n"
    "                                        ca, ca, ca, str(bool(correct)),\n"
    "                                    )\n"
    "                                    semantic_decoded = True\n"
    "                                elif use_semantic_decoder and (\n"
)
if 'in ("co_presence", "threshold")' in s:
    print("eval already has extra types")
elif anchor in s:
    open(ep, "w").write(s.replace(anchor, ins, 1)); print("eval patched")
else:
    print("ERROR eval anchor missing (comparison patch present?)"); sys.exit(1)

for f in (ap, ep):
    py_compile.compile(f, doraise=True)
print("compile OK")
