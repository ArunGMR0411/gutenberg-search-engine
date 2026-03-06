#!/usr/bin/env python3
"""Emit LaTeX tables from analysis CSVs."""

from __future__ import annotations

import csv
from pathlib import Path


def escape_latex(s: str) -> str:
    """Escape special LaTeX chars."""
    for old, new in [("&", "\\&"), ("%", "\\%"), ("_", "\\_"),
                     ("#", "\\#"), ("ö", "\\\"{o}")]:
        s = s.replace(old, new)
    return s


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    analysis = base / "outputs" / "analysis"

    # --- Metrics table ---
    in_path = analysis / "report_results_table.csv"
    out_path = analysis / "latex_metrics_table.tex"
    with open(in_path, "r") as f:
        rows = list(csv.DictReader(f))

    with open(out_path, "w") as f:
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Evaluation Metrics per Query and Model (top-10 judged)}\n")
        f.write("\\label{tab:eval-metrics}\n")
        f.write("\\begin{tabular}{clccccc}\n")
        f.write("\\toprule\n")
        f.write("Q\\# & Model & P@10 & P@10\\textsubscript{partial} & AP@10 & NDCG@10 & \\#Results \\\\\n")
        f.write("\\midrule\n")
        prev_qnr = None
        for row in rows:
            qnr = row["query_nr"]
            if prev_qnr and qnr != prev_qnr:
                f.write("\\midrule\n")
            prev_qnr = qnr
            model = escape_latex(row["model"])
            f.write(f"{qnr} & {model} & {row['P@10']} & {row['P@10_partial']} & "
                    f"{row['AP@10']} & {row['NDCG@10']} & {row['result_count']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    print(f"Wrote {out_path}")

    # --- Overlap table ---
    in_path = analysis / "top10_overlap.csv"
    out_path = analysis / "latex_overlap_table.tex"
    with open(in_path, "r") as f:
        rows = list(csv.DictReader(f))

    with open(out_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Top-10 Overlap (Jaccard) Between Models}\n")
        f.write("\\label{tab:overlap}\n")
        f.write("\\begin{tabular}{clcc}\n")
        f.write("\\toprule\n")
        f.write("Q\\# & Model Pair & Overlap & Jaccard \\\\\n")
        f.write("\\midrule\n")
        prev_qnr = None
        for row in rows:
            qnr = row["query_nr"]
            if prev_qnr and qnr != prev_qnr:
                f.write("\\midrule\n")
            prev_qnr = qnr
            pair = escape_latex(row["pair"].replace("_", " "))
            f.write(f"{qnr} & {pair} & {row['overlap_top10']} & {row['jaccard_top10']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
