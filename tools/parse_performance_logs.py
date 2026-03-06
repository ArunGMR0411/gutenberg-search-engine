#!/usr/bin/env python3
"""Parse performance logs into CSV tables and LaTeX."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    logs = base / "outputs" / "logs"
    out = base / "outputs" / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    # --- Build metrics ---
    bm_path = logs / "index_build_metrics.json"
    with open(bm_path, "r") as f:
        bm = json.load(f)

    perf_rows = [{
        "metric": "Documents indexed",
        "value": str(bm["N"]),
    }, {
        "metric": "Total text read (GB)",
        "value": f"{bm['total_text_bytes_read'] / 1e9:.1f}",
    }, {
        "metric": "Build time (s)",
        "value": f"{bm['elapsed_seconds']:.1f}",
    }, {
        "metric": "Throughput (docs/s)",
        "value": f"{bm['docs_per_second']:.2f}",
    }, {
        "metric": "Throughput (MB/s)",
        "value": f"{bm['mb_per_second']:.2f}",
    }, {
        "metric": "Peak memory (MB)",
        "value": f"{bm['peak_tracemalloc_mb']:.0f}",
    }, {
        "metric": "SPIMI blocks",
        "value": str(bm["blocks_written"]),
    }, {
        "metric": "Vocabulary size",
        "value": f"{bm['final_vocab_size']:,}",
    }, {
        "metric": "Avg doc length (tokens)",
        "value": f"{bm['avgdl']:.0f}",
    }, {
        "metric": "postings.bin (GB)",
        "value": f"{bm['postings_bytes'] / 1e9:.2f}",
    }, {
        "metric": "index.sqlite (GB)",
        "value": f"{bm['sqlite_bytes'] / 1e9:.2f}",
    }]

    perf_csv = out / "performance_table.csv"
    with open(perf_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerows(perf_rows)
    print(f"Wrote {perf_csv}")

    # --- Query latency ---
    qm_path = logs / "query_metrics.jsonl"
    latency_rows = []
    with open(qm_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            latency_rows.append({
                "query": entry.get("query", ""),
                "model": entry.get("model", ""),
                "elapsed_ms": f"{entry.get('elapsed_ms', 0.0):.2f}",
                "topk": entry.get("topk", 0),
                "candidates_scored": entry.get("candidates_scored", 0),
                "phrase_fallback": entry.get("phrase_fallback_used", False),
            })

    ql_csv = out / "query_latency_table.csv"
    fields = ["query", "model", "elapsed_ms", "topk", "candidates_scored", "phrase_fallback"]
    with open(ql_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(latency_rows)
    print(f"Wrote {ql_csv}  ({len(latency_rows)} rows)")

    # --- LaTeX perf table ---
    tex_path = out / "latex_perf_table.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Index Build Performance}\n")
        f.write("\\label{tab:build-perf}\n")
        f.write("\\begin{tabular}{lr}\n")
        f.write("\\toprule\n")
        f.write("Metric & Value \\\\\n")
        f.write("\\midrule\n")
        for row in perf_rows:
            val = row["value"].replace(",", "{,}")
            f.write(f"{row['metric']} & {val} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
