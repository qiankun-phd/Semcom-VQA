#!/usr/bin/env python3
"""Derive co_presence + threshold yes/no tasks from v1_7 counting rows (counts only)."""
from __future__ import annotations
import argparse, csv
from collections import defaultdict

FIELDS = ["image_id", "question_type", "question", "answer", "target_class", "object_count",
          "risk_level", "epsilon_k", "tau_k", "view_quality_bin", "scale_proxy", "occlusion_score",
          "truncation_score", "density_score", "presence_polarity", "target_class_b", "threshold_n"]
# countable classes (same family the detector counts)
CANDS = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
         "awning-tricycle", "bus", "motor"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/tasks/v1_7_tasks.csv")
    ap.add_argument("--out", default="outputs/tasks/v2_extra_tasks.csv")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.inp)))
    counts = defaultdict(dict); meta = {}
    for r in rows:
        if r["question_type"] == "counting":
            try: c = int(float(r["object_count"]))
            except ValueError: c = 0
            counts[r["image_id"]][r["target_class"]] = c
            meta.setdefault(r["image_id"], r)

    out = []

    def base(img):
        m = meta[img]
        return {"image_id": img, "risk_level": m.get("risk_level", "normal"), "epsilon_k": m.get("epsilon_k", "0.8"),
                "tau_k": m.get("tau_k", "3.0"), "view_quality_bin": m.get("view_quality_bin", "poor"),
                "scale_proxy": m.get("scale_proxy", "0"), "occlusion_score": m.get("occlusion_score", "0"),
                "truncation_score": m.get("truncation_score", "0"), "density_score": m.get("density_score", "0"),
                "presence_polarity": "", "target_class_b": "", "threshold_n": ""}

    flip = False
    for img in sorted(counts):
        present = [c for c, n in counts[img].items() if n > 0]
        absent = [c for c in CANDS if counts[img].get(c, 0) == 0]
        # ---- co_presence: alternate yes (two present) / no (present + absent) ----
        if flip and len(present) >= 2:
            A, B = present[0], present[1]; ans = "yes"
        elif absent and present:
            A, B = present[0], absent[0]; ans = "no"
        elif len(present) >= 2:
            A, B = present[0], present[1]; ans = "yes"
        else:
            A = B = None
        if A:
            r = base(img); r.update({"question_type": "co_presence",
                "question": f"Are there both {A} and {B} objects in this area?",
                "answer": ans, "target_class": A, "object_count": counts[img].get(A, 0), "target_class_b": B})
            out.append(r)
        # ---- threshold: alternate yes (N<=c) / no (N>c) ----
        if present:
            A = present[0]; c = counts[img][A]
            if flip:
                N = max(1, c); ans = "yes"          # at least c -> yes
            else:
                N = c + 3; ans = "no"               # at least c+3 -> no
            r = base(img); r.update({"question_type": "threshold",
                "question": f"Are there at least {N} {A} objects in this area?",
                "answer": ans, "target_class": A, "object_count": c, "threshold_n": str(N)})
            out.append(r)
        flip = not flip

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(out)
    from collections import Counter
    qt = Counter(r["question_type"] for r in out)
    ya = Counter((r["question_type"], r["answer"]) for r in out)
    print(f"extra tasks={len(out)} {dict(qt)}")
    print(f"balance: {dict(ya)}")


if __name__ == "__main__":
    main()
