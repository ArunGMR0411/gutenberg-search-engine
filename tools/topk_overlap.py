#!/usr/bin/env python3
"""Top-k overlap and Jaccard similarity between models."""

from __future__ import annotations

import csv
import os
from pathlib import Path


def load_top_ids(path: str, k: int = 10) -> list[str]:
    ids = []
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return ids
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= k:
                break
            ids.append(row[1])
    return ids


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    tsv_dir = base / "outputs" / "tsv"
    out_dir = base / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = ["1", "2", "3", "4", "5", "6"]
    pairs = [
        ("structured", "vsm"),
        ("structured", "bm25"),
        ("vsm", "bm25"),
    ]

    rows = []
    for qnr in queries:
        model_ids = {}
        for model in ["structured", "vsm", "bm25"]:
            path = tsv_dir / f"{qnr}_{model}.tsv"
            model_ids[model] = set(load_top_ids(str(path)))

        for m1, m2 in pairs:
            s1 = model_ids[m1]
            s2 = model_ids[m2]
            ov = len(s1 & s2)
            jac = jaccard(s1, s2)
            rows.append({
                "query_nr": qnr,
                "pair": f"{m1}_vs_{m2}",
                "overlap_top10": ov,
                "jaccard_top10": f"{jac:.4f}",
            })

    out_path = out_dir / "top10_overlap.csv"
    fields = ["query_nr", "pair", "overlap_top10", "jaccard_top10"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
