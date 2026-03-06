#!/usr/bin/env python3
"""Score distribution summary per query and model."""

from __future__ import annotations

import csv
import os
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    """Compute p-th percentile (0-100) using linear interpolation."""
    if not values:
        return 0.0
    n = len(values)
    k = (n - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= n:
        return values[-1]
    return values[f] + (k - f) * (values[c] - values[f])


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    tsv_dir = base / "outputs" / "tsv"
    out_dir = base / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = ["1", "2", "3", "4", "5", "6"]
    models = ["structured", "vsm", "bm25"]

    rows = []
    for qnr in queries:
        for model in models:
            path = tsv_dir / f"{qnr}_{model}.tsv"
            if not path.exists() or path.stat().st_size == 0:
                rows.append({
                    "query_nr": qnr,
                    "model": model,
                    "result_count": 0,
                    "min_score": "",
                    "p25": "",
                    "median": "",
                    "p75": "",
                    "max_score": "",
                })
                continue

            scores = []
            with open(path, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    scores.append(float(row[2]))

            scores.sort()
            rows.append({
                "query_nr": qnr,
                "model": model,
                "result_count": len(scores),
                "min_score": f"{scores[0]:.6f}" if scores else "",
                "p25": f"{percentile(scores, 25):.6f}" if scores else "",
                "median": f"{percentile(scores, 50):.6f}" if scores else "",
                "p75": f"{percentile(scores, 75):.6f}" if scores else "",
                "max_score": f"{scores[-1]:.6f}" if scores else "",
            })

    out_path = out_dir / "score_stats_by_query_model.csv"
    fields = ["query_nr", "model", "result_count", "min_score", "p25",
              "median", "p75", "max_score"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
