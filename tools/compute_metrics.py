#!/usr/bin/env python3
"""Compute P@10, AP@10, and NDCG@10 from judgments.csv."""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

# ---- Metric functions ----

def precision_at_k(labels: list[int], k: int, partial_weight: float = 0.0) -> float:
    """Precision@K.  label 1=relevant, 2=partial, 0=not-relevant."""
    top = labels[:k]
    count = sum(1 for l in top if l == 1) + partial_weight * sum(1 for l in top if l == 2)
    return count / k if k > 0 else 0.0


def average_precision_at_k(labels: list[int], k: int) -> float:
    """AP@K (binary: only label==1 is relevant)."""
    top = labels[:k]
    rel_count = 0
    sum_prec = 0.0
    for i, l in enumerate(top):
        if l == 1:
            rel_count += 1
            sum_prec += rel_count / (i + 1)
    return sum_prec / rel_count if rel_count > 0 else 0.0


def ndcg_at_k(labels: list[int], k: int) -> float:
    """NDCG@K with gains: gain(1)=1.0, gain(2)=0.5, gain(0)=0.0."""
    gain_map = {0: 0.0, 1: 1.0, 2: 0.5}

    def dcg(scores: list[float], n: int) -> float:
        return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:n]))

    gains = [gain_map.get(l, 0.0) for l in labels[:k]]
    actual = dcg(gains, k)
    ideal_gains = sorted(gains, reverse=True)
    ideal = dcg(ideal_gains, k)
    return actual / ideal if ideal > 0 else 0.0


# ---- Main ----

QUERIES = {
    "1": "to be, or not to be",
    "2": "English Grammar",
    "3": "Philip K Dick",
    "4": "Jabberwocky",
    "5": "Gutenberg",
    "6": "Dornröschen",
}


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    judgments_path = base / "outputs" / "judgments" / "judgments.csv"
    analysis_dir = base / "outputs" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Load judgments grouped by (query_nr, model) -> ordered list of labels
    data: dict[tuple[str, str], list[int]] = defaultdict(list)
    with open(judgments_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["query_nr"], row["model"])
            data[key].append(int(row["label"]))

    k = 10

    # Compute per-query-model metrics
    rows = []
    for (qnr, model), labels in sorted(data.items()):
        p10 = precision_at_k(labels, k, partial_weight=0.0)
        p10p = precision_at_k(labels, k, partial_weight=0.5)
        ap10 = average_precision_at_k(labels, k)
        ndcg10 = ndcg_at_k(labels, k)
        rows.append({
            "query_nr": qnr,
            "query_text": QUERIES.get(qnr, ""),
            "model": model,
            "P@10": f"{p10:.4f}",
            "P@10_partial": f"{p10p:.4f}",
            "AP@10": f"{ap10:.4f}",
            "NDCG@10": f"{ndcg10:.4f}",
            "result_count": len(labels),
        })

    # Write per-query-model CSV
    fields = ["query_nr", "query_text", "model", "P@10", "P@10_partial",
              "AP@10", "NDCG@10", "result_count"]
    out1 = analysis_dir / "metrics_by_query_model.csv"
    with open(out1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out1}  ({len(rows)} rows)")

    # Aggregate mean per model
    model_metrics: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        model_metrics[row["model"]].append(row)

    summary_rows = []
    for model in ["structured", "vsm", "bm25"]:
        mrows = model_metrics.get(model, [])
        if not mrows:
            continue
        n = len(mrows)
        mean_p10 = sum(float(r["P@10"]) for r in mrows) / n
        mean_p10p = sum(float(r["P@10_partial"]) for r in mrows) / n
        mean_ap10 = sum(float(r["AP@10"]) for r in mrows) / n
        mean_ndcg10 = sum(float(r["NDCG@10"]) for r in mrows) / n
        summary_rows.append({
            "model": model,
            "mean_P@10": f"{mean_p10:.4f}",
            "mean_P@10_partial": f"{mean_p10p:.4f}",
            "mean_AP@10": f"{mean_ap10:.4f}",
            "mean_NDCG@10": f"{mean_ndcg10:.4f}",
            "queries_judged": n,
        })

    out2 = analysis_dir / "metrics_summary.csv"
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "mean_P@10", "mean_P@10_partial",
            "mean_AP@10", "mean_NDCG@10", "queries_judged",
        ])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {out2}  ({len(summary_rows)} rows)")

    # Also write the report results table
    out3 = analysis_dir / "report_results_table.csv"
    with open(out3, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out3}  ({len(rows)} rows)")

    # Print summary to stdout
    print("\n=== Metrics Summary (Mean per Model) ===")
    print(f"{'Model':<12} {'P@10':>6} {'P@10p':>6} {'AP@10':>6} {'NDCG':>6}")
    for sr in summary_rows:
        print(f"{sr['model']:<12} {sr['mean_P@10']:>6} {sr['mean_P@10_partial']:>6} "
              f"{sr['mean_AP@10']:>6} {sr['mean_NDCG@10']:>6}")


if __name__ == "__main__":
    main()
