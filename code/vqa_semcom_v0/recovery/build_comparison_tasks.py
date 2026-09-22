#!/usr/bin/env python3
"""Derive 'comparison' VQA tasks from existing counting rows in v1_7_tasks.csv.

Question: "Are there more {A} than {B} objects in this area?"  answer yes/no from
GT counts.  Both A,B are detector-countable classes, so the s1 token path stays
meaningful (count A vs count B).  Deterministic yes/no balancing.
"""
from __future__ import annotations
import argparse, csv
from collections import defaultdict

FIELDS = ["image_id", "question_type", "question", "answer", "target_class", "object_count",
          "risk_level", "epsilon_k", "tau_k", "view_quality_bin", "scale_proxy", "occlusion_score",
          "truncation_score", "density_score", "presence_polarity", "target_class_b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/tasks/v1_7_tasks.csv")
    ap.add_argument("--out", default="outputs/tasks/v2_comparison_tasks.csv")
    ap.add_argument("--max-per-image", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.inp)))
    by_img = defaultdict(dict)   # image_id -> {category: count}
    meta = {}                    # image_id -> a representative counting row (for field copy)
    for r in rows:
        if r["question_type"] == "counting":
            try:
                c = int(float(r["object_count"]))
            except ValueError:
                c = 0
            by_img[r["image_id"]][r["target_class"]] = c
            meta.setdefault(r["image_id"], r)

    out = []
    flip = False  # alternate orientation to balance yes/no
    for img in sorted(by_img):
        cats = sorted(by_img[img].items(), key=lambda kv: (-kv[1], kv[0]))  # by count desc
        cats = [(c, n) for c, n in cats]
        made = 0
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                (ca, na), (cb, nb) = cats[i], cats[j]
                if na == nb:
                    continue  # need a clear winner
                m = meta[img]
                # alternate which class is asked first to balance yes/no
                if flip:
                    A, nA, B, nB = cb, nb, ca, na
                else:
                    A, nA, B, nB = ca, na, cb, nb
                flip = not flip
                out.append({
                    "image_id": img, "question_type": "comparison",
                    "question": f"Are there more {A} than {B} objects in this area?",
                    "answer": "yes" if nA > nB else "no",
                    "target_class": A, "object_count": nA, "risk_level": m.get("risk_level", "normal"),
                    "epsilon_k": m.get("epsilon_k", "0.8"), "tau_k": m.get("tau_k", "3.0"),
                    "view_quality_bin": m.get("view_quality_bin", "poor"),
                    "scale_proxy": m.get("scale_proxy", "0"), "occlusion_score": m.get("occlusion_score", "0"),
                    "truncation_score": m.get("truncation_score", "0"), "density_score": m.get("density_score", "0"),
                    "presence_polarity": "", "target_class_b": B,
                })
                made += 1
                if made >= args.max_per_image:
                    break
            if made >= args.max_per_image:
                break
        if args.limit and len(out) >= args.limit:
            break

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(out)
    yes = sum(1 for r in out if r["answer"] == "yes")
    print(f"comparison tasks={len(out)} yes={yes} no={len(out)-yes} images={len(by_img)} -> {args.out}")


if __name__ == "__main__":
    main()
