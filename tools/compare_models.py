#!/usr/bin/env python3
"""Compare model outputs across queries."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

QUERIES = {
    "1": "to be, or not to be",
    "2": "English Grammar",
    "3": "Philip K Dick",
    "4": "Jabberwocky",
    "5": "Gutenberg",
    "6": "Dornröschen",
}
MODELS = ["structured", "vsm", "bm25"]


def load_tsv_ids(path: str, topk: int = 10) -> list[str]:
    """Load gutenberg IDs from a TSV file."""
    ids = []
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return ids
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= topk:
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
    analysis_dir = base / "outputs" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    comparison: dict = {}
    table_rows: list[dict] = []

    for qnr, qtext in sorted(QUERIES.items()):
        qdata: dict = {"query_text": qtext, "models": {}}
        model_sets: dict[str, set[str]] = {}

        for model in MODELS:
            path = tsv_dir / f"{qnr}_{model}.tsv"
            total_path = str(path)
            total_count = 0
            if path.exists() and path.stat().st_size > 0:
                with open(path, "r") as f:
                    total_count = sum(1 for _ in f)

            top10_ids = load_tsv_ids(str(path))
            model_sets[model] = set(top10_ids)
            qdata["models"][model] = {
                "result_count": total_count,
                "top10_ids": top10_ids,
            }

        # Pairwise overlap
        pairs = [
            ("structured", "vsm"),
            ("structured", "bm25"),
            ("vsm", "bm25"),
        ]
        qdata["pairwise"] = {}
        for m1, m2 in pairs:
            s1 = model_sets.get(m1, set())
            s2 = model_sets.get(m2, set())
            overlap = len(s1 & s2)
            jac = jaccard(s1, s2)
            key = f"{m1}_vs_{m2}"
            qdata["pairwise"][key] = {
                "overlap": overlap,
                "jaccard": round(jac, 4),
            }
            table_rows.append({
                "query_nr": qnr,
                "query_text": qtext,
                "model_1": m1,
                "model_2": m2,
                "result_count_1": qdata["models"][m1]["result_count"],
                "result_count_2": qdata["models"][m2]["result_count"],
                "overlap_top10": overlap,
                "jaccard_top10": f"{jac:.4f}",
            })

        comparison[qnr] = qdata

    # Write JSON
    out_json = analysis_dir / "tsv_comparison.json"
    with open(out_json, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_json}")

    # Write CSV
    out_csv = analysis_dir / "tsv_comparison_table.csv"
    fields = ["query_nr", "query_text", "model_1", "model_2",
              "result_count_1", "result_count_2", "overlap_top10", "jaccard_top10"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table_rows)
    print(f"Wrote {out_csv}  ({len(table_rows)} rows)")


if __name__ == "__main__":
    main()
