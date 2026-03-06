"""Batch evaluation runner and TSV validator."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple

from ir_system.phrase_fallback import (
    apply_phrase_fallback_bm25,
    is_phrase_fallback_query,
)
from ir_system.preprocessing import preprocess_query
from ir_system.retrieval_bm25 import BM25Search
from ir_system.retrieval_structured import StructuredSearch
from ir_system.retrieval_vsm import VSMSearch
from ir_system.snippets import generate_snippets_batch
from ir_system.utils import load_index_globals

# ---------------------------------------------------------------------------
# Hardcoded evaluation queries
# ---------------------------------------------------------------------------

EVAL_QUERIES = {
    1: "to be, or not to be",
    2: "English Grammar",
    3: "Philip K Dick",
    4: "Jabberwocky",
    5: "Gutenberg",
    6: "Dornröschen",
}

MODEL_NAMES = ["structured", "vsm", "bm25"]


# ---------------------------------------------------------------------------
# Metadata loader (same as cli.py helper)
# ---------------------------------------------------------------------------


def _load_metadata_map(index_dir: str) -> Dict[str, dict]:
    """Load metadata from SQLite for snippet generation."""
    db_path = Path(index_dir) / "index.sqlite"
    metadata: Dict[str, dict] = {}
    with sqlite3.connect(db_path) as conn:
        for row in conn.execute(
            "SELECT gutenberg_id, title, authors, has_text FROM metadata"
        ):
            metadata[str(row[0])] = {
                "title": row[1] or "",
                "authors": row[2] or "",
                "has_text": int(row[3]),
            }
    return metadata


# ---------------------------------------------------------------------------
# TSV writer
# ---------------------------------------------------------------------------


def write_tsv(
    filepath: Path,
    results: List[Tuple[str, float]],
    snippets: List[Tuple[str, str]],
    max_rows: int = 100,
) -> None:
    """Write ranked results to a 5-column TSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        for i, ((gutid, score), (snippet, start_line)) in enumerate(
            zip(results[:max_rows], snippets[:max_rows]), start=1
        ):
            # Sanitize snippet: replace tabs and newlines to preserve TSV structure
            clean_snippet = snippet.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            line = f"{i}\t{gutid}\t{score:.6f}\t{clean_snippet}\t{start_line}\n"
            f.write(line)


# ---------------------------------------------------------------------------
# Query metrics logger
# ---------------------------------------------------------------------------


def _log_eval_query_metrics(
    model: str,
    query: str,
    topk: int,
    elapsed_ms: float,
    candidates_scored: int,
    phrase_fallback_used: bool,
) -> None:
    """Append one JSON line to outputs/logs/query_metrics.jsonl."""
    log_dir = Path("outputs") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "query": query,
        "topk": topk,
        "elapsed_ms": round(elapsed_ms, 2),
        "postings_accessed": 0,
        "candidates_scored": candidates_scored,
        "phrase_fallback_used": phrase_fallback_used,
    }
    with (log_dir / "query_metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


# ---------------------------------------------------------------------------
# Batch evaluation runner
# ---------------------------------------------------------------------------


def run_eval_queries(
    index_dir: str,
    tsv_dir: str,
    allow_full_scan: bool = True,
    text_dir: str | None = None,
) -> None:
    """Run all 6 queries x 3 models and write 18 TSV files."""
    print(f"[INFO] Loading index globals from {index_dir}...")
    docid_to_gutid, doc_len, doc_norm, N, avgdl = load_index_globals(index_dir)
    metadata_map = _load_metadata_map(index_dir)

    print("[INFO] Initializing retrieval models...")
    structured = StructuredSearch(index_dir)
    vsm = VSMSearch(index_dir, docid_to_gutid, doc_norm, N)
    bm25 = BM25Search(index_dir, docid_to_gutid, doc_len, N, avgdl)

    tsv_path = Path(tsv_dir)
    tsv_path.mkdir(parents=True, exist_ok=True)

    for query_nr, raw_query in EVAL_QUERIES.items():
        query_tokens = preprocess_query(raw_query)

        for model_name in MODEL_NAMES:
            filename = f"{query_nr}_{model_name}.tsv"
            filepath = tsv_path / filename
            start = time.perf_counter()
            phrase_fallback_used = False

            if model_name == "structured":
                results = structured.search(raw_query)
            elif model_name == "vsm":
                results = vsm.search(raw_query)
            elif model_name == "bm25":
                results = bm25.search(raw_query)

                # Check phrase fallback trigger for BM25
                db_path = Path(index_dir) / "index.sqlite"
                Q = set(query_tokens)
                df_map: Dict[str, int] = {}
                with sqlite3.connect(db_path) as conn:
                    for term in Q:
                        row = conn.execute(
                            "SELECT df FROM lexicon WHERE term=?",
                            (term,),
                        ).fetchone()
                        if row is not None:
                            df_map[term] = int(row[0])

                df_lookup = lambda t, _df_map=df_map: _df_map.get(t)
                if is_phrase_fallback_query(Q, df_lookup, N, raw_query=raw_query):
                    phrase_fallback_used = True
                    results, _used, _full = apply_phrase_fallback_bm25(
                        raw_query,
                        results,
                        index_dir,
                        allow_full_scan=allow_full_scan,
                        text_dir=Path(text_dir) if text_dir else None,
                    )
            else:
                results = []

            # Truncate to 100
            results = results[:100]

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # Generate snippets
            snippets = generate_snippets_batch(
                results=results,
                query_tokens=query_tokens,
                N=N,
                index_dir=index_dir,
                metadata_map=metadata_map,
                text_dir=text_dir,
                max_results=100,
            )

            write_tsv(filepath, results, snippets)

            # log metrics
            _log_eval_query_metrics(
                model=model_name,
                query=raw_query,
                topk=len(results),
                elapsed_ms=elapsed_ms,
                candidates_scored=len(results),
                phrase_fallback_used=phrase_fallback_used,
            )

            print(
                f"  [{model_name:>10}] Q{query_nr}: "
                f"{len(results):>4} results, "
                f"{elapsed_ms:>8.1f} ms -> {filename}"
            )

    print(f"[INFO] All 18 TSVs written to {tsv_dir}/")


# ---------------------------------------------------------------------------
# TSV validator
# ---------------------------------------------------------------------------


def validate_tsv(tsv_dir: str) -> Tuple[bool, List[str]]:
    """Validate all 18 TSV result files. Returns (all_pass, error_messages)."""
    failures: List[str] = []
    tsv_path = Path(tsv_dir)

    expected_files = [
        f"{qn}_{mn}.tsv"
        for qn in range(1, 7)
        for mn in MODEL_NAMES
    ]

    # Check a: all 18 files exist
    for filename in expected_files:
        filepath = tsv_path / filename
        if not filepath.exists():
            failures.append(f"MISSING: {filename}")
            continue

        lines = filepath.read_text(encoding="utf-8").splitlines()

        # Check c: <= 100 rows
        if len(lines) > 100:
            failures.append(f"{filename}: too many rows ({len(lines)} > 100)")

        prev_score = None
        prev_gutid = None

        for line_nr, line in enumerate(lines, start=1):
            cols = line.split("\t")

            # Check b: exactly 5 columns
            if len(cols) != 5:
                failures.append(
                    f"{filename} line {line_nr}: "
                    f"expected 5 columns, got {len(cols)}"
                )
                continue

            rank_str, gutid, score_str, _snippet, _start_line = cols

            # Check d: rank is 1-based sequential
            try:
                rank = int(rank_str)
            except ValueError:
                failures.append(
                    f"{filename} line {line_nr}: "
                    f"rank is not an integer: {rank_str!r}"
                )
                continue

            if rank != line_nr:
                failures.append(
                    f"{filename} line {line_nr}: "
                    f"expected rank {line_nr}, got {rank}"
                )

            # Check e: score non-increasing + tie-break
            try:
                score = float(score_str)
            except ValueError:
                failures.append(
                    f"{filename} line {line_nr}: "
                    f"score is not a float: {score_str!r}"
                )
                continue

            if prev_score is not None:
                if score > prev_score:
                    failures.append(
                        f"{filename} line {line_nr}: "
                        f"score {score} > previous {prev_score} "
                        f"(not non-increasing)"
                    )
                elif score == prev_score and prev_gutid is not None:
                    if gutid < prev_gutid:
                        failures.append(
                            f"{filename} line {line_nr}: "
                            f"tied score but gutenberg_id {gutid!r} < "
                            f"previous {prev_gutid!r} "
                            f"(tie-break should be ascending)"
                        )

            prev_score = score
            prev_gutid = gutid

    all_pass = len(failures) == 0
    return all_pass, failures


def validate_tsv_determinism(
    index_dir: str,
    tsv_dir: str,
    allow_full_scan: bool = True,
    text_dir: str | None = None,
) -> Tuple[bool, List[str]]:
    """Run eval queries twice and check byte-identical TSV outputs."""
    import shutil
    import tempfile

    tsv_path = Path(tsv_dir)
    failures: List[str] = []

    # Save first-run files
    first_run_dir = Path(tempfile.mkdtemp(prefix="tsv_run1_"))
    try:
        if tsv_path.exists():
            for f in tsv_path.iterdir():
                if f.suffix == ".tsv":
                    shutil.copy2(f, first_run_dir / f.name)

        # Run second time
        run_eval_queries(
            index_dir=index_dir,
            tsv_dir=tsv_dir,
            allow_full_scan=allow_full_scan,
            text_dir=text_dir,
        )

        # Compare
        expected_files = [
            f"{qn}_{mn}.tsv"
            for qn in range(1, 7)
            for mn in MODEL_NAMES
        ]

        for filename in expected_files:
            first = first_run_dir / filename
            second = tsv_path / filename

            if not first.exists():
                failures.append(f"MISSING from first run: {filename}")
                continue
            if not second.exists():
                failures.append(f"MISSING from second run: {filename}")
                continue

            if first.read_bytes() != second.read_bytes():
                failures.append(f"NOT IDENTICAL: {filename}")
    finally:
        shutil.rmtree(first_run_dir, ignore_errors=True)

    return len(failures) == 0, failures
