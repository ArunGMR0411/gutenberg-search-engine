"""Manual relevance judging and Precision@K computation."""

from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path
from typing import List, Tuple


def parse_tsv_filename(tsv_path: str) -> Tuple[str, str]:
    """Extract (query_nr, model_name) from a TSV filename."""
    stem = Path(tsv_path).stem  # e.g. "1_bm25"
    match = re.match(r"^(\d+)_(.+)$", stem)
    if match:
        return match.group(1), match.group(2)
    return stem, "unknown"


def load_tsv_results(tsv_path: str, k: int) -> List[dict]:
    """Load top-k results from a TSV file."""
    results = []
    filepath = Path(tsv_path)
    if not filepath.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= k:
                break
            line = line.rstrip("\n")
            cols = line.split("\t")
            if len(cols) < 5:
                continue
            results.append({
                "rank": int(cols[0]),
                "gutenberg_id": cols[1],
                "score": float(cols[2]),
                "snippet": cols[3],
                "start_line": cols[4],
            })
    return results


def load_titles_from_index(
    index_dir: str, gutenberg_ids: List[str]
) -> dict[str, str]:
    """Look up titles for the given gutenberg_ids from SQLite."""
    titles: dict[str, str] = {}
    db_path = Path(index_dir) / "index.sqlite"
    if not db_path.exists():
        return titles

    with sqlite3.connect(db_path) as conn:
        for gid in gutenberg_ids:
            row = conn.execute(
                "SELECT title FROM metadata WHERE gutenberg_id=?", (gid,)
            ).fetchone()
            if row:
                titles[gid] = row[0] or ""
            else:
                titles[gid] = ""
    return titles


def compute_precision_at_k(
    labels: List[int], partial_value: float = 0.0
) -> float:
    """Compute Precision@K from relevance labels."""
    if not labels:
        return 0.0
    total = 0.0
    for label in labels:
        if label == 1:
            total += 1.0
        elif label == 2:
            total += partial_value
    return total / len(labels)


def append_judgments(
    judgments_path: Path,
    query_nr: str,
    model: str,
    results: List[dict],
    labels: List[int],
) -> None:
    """Append judgment rows to the judgments CSV, creating it if needed."""
    judgments_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = judgments_path.exists() and judgments_path.stat().st_size > 0

    with open(judgments_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["query_nr", "model", "gutenberg_id", "rank", "label"])
        for result, label in zip(results, labels):
            writer.writerow([
                query_nr,
                model,
                result["gutenberg_id"],
                result["rank"],
                label,
            ])


def run_judge(
    tsv_path: str,
    k: int,
    index_dir: str | None = None,
    input_fn=None,
    print_fn=None,
) -> Tuple[float, float]:
    """Run the interactive judging session. Returns (p@k, p@k_partial)."""
    if input_fn is None:
        input_fn = input
    if print_fn is None:
        print_fn = print

    results = load_tsv_results(tsv_path, k)
    if not results:
        print_fn("No results to judge.")
        return 0.0, 0.0

    query_nr, model = parse_tsv_filename(tsv_path)

    # Try to load titles
    titles: dict[str, str] = {}
    if index_dir:
        gids = [r["gutenberg_id"] for r in results]
        titles = load_titles_from_index(index_dir, gids)

    print_fn(f"\nJudging: {Path(tsv_path).name} (query={query_nr}, model={model})")
    print_fn(f"Top {len(results)} results:\n")

    labels: List[int] = []
    for result in results:
        gid = result["gutenberg_id"]
        title = titles.get(gid, "")
        snippet = result["snippet"][:80] if result["snippet"] else ""

        print_fn(f"  Rank {result['rank']}: {gid}")
        if title:
            print_fn(f"    Title: {title}")
        if snippet:
            print_fn(f"    Snippet: {snippet}...")

        while True:
            label_str = input_fn("    Label (1=relevant, 2=partial, 0=not relevant): ")
            label_str = label_str.strip()
            if label_str in {"0", "1", "2"}:
                labels.append(int(label_str))
                break
            print_fn("    Invalid input. Enter 0, 1, or 2.")

    # Compute Precision@K
    p_at_k = compute_precision_at_k(labels, partial_value=0.0)
    p_at_k_partial = compute_precision_at_k(labels, partial_value=0.5)

    print_fn(f"\nPrecision@{len(labels)} (partial=0):      {p_at_k:.2f}")
    print_fn(f"Precision@{len(labels)}_partial (partial=0.5):  {p_at_k_partial:.2f}")

    # Append judgments
    judgments_path = Path("outputs") / "judgments" / "judgments.csv"
    append_judgments(judgments_path, query_nr, model, results, labels)
    print_fn(f"\nJudgments appended to {judgments_path}")

    return p_at_k, p_at_k_partial
