#!/usr/bin/env python3
"""Idempotent patch: add 'comparison' question type to answer scoring + s1 token decode."""
import io, sys

REPO = "/home/qiankun/phd_research/uav-vqa-semantic-rl"

# ---- 1) answer.py: add 'comparison' to the yes/no branch ----
ap = f"{REPO}/src/vqa_semcom/vlm/answer.py"
s = open(ap).read()
old1 = 'if question_type in {"presence", "risk"}:'
new1 = 'if question_type in {"presence", "risk", "comparison"}:'
if new1 in s:
    print("answer.py already patched")
elif old1 in s:
    open(ap, "w").write(s.replace(old1, new1))
    print("answer.py patched")
else:
    print("ERROR: answer.py anchor not found"); sys.exit(1)

# ---- 2) run_v1_detector_eval.py: insert comparison branch in s1 block ----
ep = f"{REPO}/scripts/run_v1_detector_eval.py"
s = open(ep).read()
anchor = (
    "                                detector_count = target_count\n"
    "                                if use_semantic_decoder and (\n"
)
branch = (
    "                                detector_count = target_count\n"
    '                                if task.get("question_type") == "comparison":\n'
    '                                    cb = task.get("target_class_b", "")\n'
    '                                    count_a = sum(1 for rr in transmitted_records if rr.category == task.get("target_class", ""))\n'
    "                                    count_b = sum(1 for rr in transmitted_records if rr.category == cb)\n"
    '                                    cmp_pred = "yes" if count_a > count_b else "no"\n'
    '                                    _chk = check_answer("comparison", cmp_pred, task["answer"], tolerance_ratio=float(vlm_cfg.get("count_tolerance_ratio", 0.10)))\n'
    "                                    predicted, normalized, correct = cmp_pred, _chk.normalized_prediction, _chk.correct\n"
    '                                    payload_bytes = _payload_bytes(service_level, evidence_type, evidence_repr, "")\n'
    "                                    prediction_cache[cache_key] = (\n"
    '                                        predicted, normalized, correct, detector_latency, "semantic-token-decoder",\n'
    "                                        evidence_type, evidence_repr, payload_bytes, detector_latency, detector_model, count_a,\n"
    "                                        count_a, count_b, count_a, str(bool(correct)),\n"
    "                                    )\n"
    "                                    semantic_decoded = True\n"
    "                                elif use_semantic_decoder and (\n"
)
if 'if task.get("question_type") == "comparison":' in s:
    print("run_v1_detector_eval.py already patched")
elif anchor in s:
    open(ep, "w").write(s.replace(anchor, branch, 1))
    print("run_v1_detector_eval.py patched")
else:
    print("ERROR: eval anchor not found"); sys.exit(1)

# ---- sanity: compile both ----
import py_compile
for f in (ap, ep):
    py_compile.compile(f, doraise=True)
print("both files compile OK")
