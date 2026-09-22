"""Freeze an eligible fixed policy for later validation; never evaluates test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_fixed(summary: dict[str, Any], decision: dict[str, Any]) -> str | None:
    """Prefer a simple fixed candidate over training an oracle-mimicking policy."""
    if not decision.get("decision_eligible") or not summary["audit"].get("decision_eligible"):
        raise ValueError("Only a complete development screen can freeze a policy")
    if not decision.get("efficient_fixed_candidate"):
        return None
    candidates = decision.get("efficient_fixed_candidates", [])
    if not candidates:
        raise ValueError("Positive fixed screen lacks candidates")
    reference = summary["cells"][summary["primary"]]
    for candidate in candidates:
        row = summary["cells"][candidate["cell"]]
        if (row["correct"] < reference["correct"] - 1
                or row["image_bytes"]["mean"] > reference["image_bytes"]["mean"]
                or row["receiver_seconds"]["median"] > .8 * reference["receiver_seconds"]["median"]):
            raise ValueError("Candidate does not meet the frozen fixed-efficiency screen")
    return min((candidate["cell"] for candidate in candidates), key=lambda cell: (
        -summary["cells"][cell]["correct"],
        summary["cells"][cell]["image_bytes"]["mean"],
        summary["cells"][cell]["receiver_seconds"]["median"], cell))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary_path, decision_path = args.analysis / "summary.json", args.analysis / "decision.json"
    summary, decision, protocol = [json.loads(path.read_text()) for path in
                                   (summary_path, decision_path, args.protocol)]
    if summary["audit"]["protocol_sha256"] != digest(args.protocol):
        raise ValueError("Analysis does not match frozen protocol")
    records = json.loads(args.records.read_text())
    if len(records) != protocol["expected_questions"] * 9:
        raise ValueError("The complete paired development grid is required")
    if any(row["protocol_sha256"] != digest(args.protocol)
           or row["receiver_sha256"] != protocol["receiver_sha256"] for row in records):
        raise ValueError("Prediction provenance mismatch")
    # Recompute to ensure summary/candidate cannot be swapped while retaining a
    # valid protocol hash. This reads only already-scored DEVELOPMENT answers.
    from analyze_grid import analyze, validate
    scored_path = args.analysis / "scored.json"
    scored = json.loads(scored_path.read_text())
    truth = [{"id": row["id"], "answer": row["answer"]} for row in scored
             if row["budget"] == 2000 and row["tier"] == "low"]
    checked, audit = validate(records, truth, protocol, digest(args.protocol), allow_partial=False)
    recomputed, checked_decision = analyze(checked, audit)
    if recomputed != summary or checked_decision != decision:
        raise ValueError("Analysis no longer matches its input predictions")
    selected = choose_fixed(summary, decision)
    if selected is None:
        state = ("CONTROLLER_TRAINING_PROTOCOL_NEEDED" if decision.get("potential_routing_headroom")
                 else "STOP_NO_PRESPECIFIED_SIGNAL")
    else:
        state = "FIXED_POLICY_FROZEN_VALIDATION_PENDING"
    result = {"state": state, "selected_cell": selected,
              "policy_kind": "globally_fixed" if selected else None,
              "selection_rule": "screened fixed candidate: max development correct, tie min bytes then median latency",
              "reference_cell": summary["primary"], "protocol_sha256": digest(args.protocol),
              "input_sha256": {"summary": digest(summary_path), "decision": digest(decision_path),
                               "records": digest(args.records), "scored": digest(scored_path)},
              "receiver_sha256": protocol["receiver_sha256"],
              "channel_protocol": protocol["stage3"],
              "test_opened": False, "wireless_evaluated": False,
              "primary_test_comparison": "selected fixed versus 4k/high on all 300 sealed questions; same receiver and PHY",
              "energy_j": None,
              "limits": "Freezing a candidate is not test success or a trained routing model. Independent validation remains pending."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output.exists() and args.output.read_text() != serialized:
        raise ValueError("Refusing to overwrite a different frozen policy")
    args.output.write_text(serialized)
    print(json.dumps({"state": state, "selected_cell": selected, "test_opened": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
