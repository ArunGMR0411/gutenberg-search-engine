"""CLI entry point for the Gutenberg IR system."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ir_system",
        description="Gutenberg IR Search Engine (Structured + VSM + BM25)",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ---- validate_dataset ----
    validate_p = sub.add_parser(
        "validate_dataset",
        help="Count files, deduplicate metadata, log missing IDs",
    )
    validate_p.add_argument(
        "--sample", action="store_true",
        help="Use index_sample/ instead of index/",
    )

    # ---- index ----
    index_p = sub.add_parser(
        "index",
        help="Build inverted index (SPIMI + merge + doc_norm)",
    )
    index_p.add_argument(
        "--sample", action="store_true",
        help="Build index_sample/ (subset) instead of index/",
    )
    index_p.add_argument(
        "--k1", type=float, default=1.2,
        help="BM25 k1 parameter (default: 1.2)",
    )
    index_p.add_argument(
        "--b", type=float, default=0.75,
        help="BM25 b parameter (default: 0.75)",
    )

    # ---- validate_index ----
    validate_idx_p = sub.add_parser(
        "validate_index",
        help="Validate that index is complete and ready",
    )
    validate_idx_p.add_argument(
        "--sample", action="store_true",
        help="Validate index_sample/ instead of index/",
    )

    # ---- search ----
    search_p = sub.add_parser(
        "search",
        help="Run a single query against one model",
    )
    search_p.add_argument(
        "--model", required=True, choices=["structured", "vsm", "bm25"],
        help="Retrieval model to use",
    )
    search_p.add_argument(
        "--query", required=True,
        help="Query string",
    )
    search_p.add_argument(
        "--topk", type=int, default=10,
        help="Number of results to return (default: 10)",
    )
    search_p.add_argument(
        "--sample", action="store_true",
        help="Use index_sample/ instead of index/",
    )
    search_p.add_argument(
        "--allow-full-scan", action="store_true",
        help="Allow full-corpus scan for phrase fallback",
    )

    # ---- interactive ----
    interactive_p = sub.add_parser(
        "interactive",
        help="Interactive REPL for querying the index",
    )
    interactive_p.add_argument(
        "--sample", action="store_true",
        help="Use index_sample/ instead of index/",
    )
    interactive_p.add_argument(
        "--allow-full-scan", action="store_true",
        help="Allow full-corpus scan for phrase fallback",
    )

    # ---- run_eval_queries ----
    eval_p = sub.add_parser(
        "run_eval_queries",
        help="Generate 18 TSVs (6 queries x 3 models)",
    )
    eval_p.add_argument(
        "--sample", action="store_true",
        help="Use index_sample/ and outputs/tsv_sample/",
    )
    eval_p.add_argument(
        "--allow-full-scan", action="store_true",
        help="Allow full-corpus scan for phrase fallback",
    )

    # ---- validate_tsv ----
    val_tsv_p = sub.add_parser(
        "validate_tsv",
        help="Validate all 18 TSV files",
    )
    val_tsv_p.add_argument(
        "--sample", action="store_true",
        help="Validate outputs/tsv_sample/ instead of outputs/tsv/",
    )

    # ---- judge ----
    judge_p = sub.add_parser(
        "judge",
        help="Manual relevance labeling + Precision@K",
    )
    judge_p.add_argument(
        "--tsv", required=True,
        help="Path to TSV file to judge",
    )
    judge_p.add_argument(
        "--k", type=int, required=True,
        help="Number of results to judge",
    )
    judge_p.add_argument(
        "--sample", action="store_true",
        help="Use index_sample/ for title lookup",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Path routing
    index_dir = "index_sample" if getattr(args, "sample", False) else "index"
    tsv_dir = "outputs/tsv_sample" if getattr(args, "sample", False) else "outputs/tsv"
    text_dir = "datasets/sample" if getattr(args, "sample", False) else "datasets/texts"

    if args.command == "validate_dataset":
        from ir_system.dataset import validate_dataset

        validate_dataset(
            csv_path="datasets/metadata.csv",
            text_dir=text_dir,
            index_dir=index_dir,
        )

    elif args.command == "index":
        from ir_system.indexing import build_index

        stats = build_index(
            csv_path="datasets/metadata.csv",
            text_dir=text_dir,
            index_dir=index_dir,
            k1=getattr(args, "k1", 1.2),
            b=getattr(args, "b", 0.75),
        )
        _write_index_build_metrics(stats, index_dir)

    elif args.command == "validate_index":
        from ir_system.indexing import validate_index

        validate_index(index_dir=index_dir)

    elif args.command == "search":
        _run_single_search(
            index_dir=index_dir,
            model=args.model,
            query=args.query,
            topk=args.topk,
            allow_full_scan=getattr(args, "allow_full_scan", False),
            text_dir=text_dir,
        )

    elif args.command == "interactive":
        _run_interactive(
            index_dir=index_dir,
            allow_full_scan=getattr(args, "allow_full_scan", False),
            text_dir=text_dir,
        )

    elif args.command == "run_eval_queries":
        from ir_system.evaluation import run_eval_queries

        run_eval_queries(
            index_dir=index_dir,
            tsv_dir=tsv_dir,
            allow_full_scan=True,  # eval mode always allows full scan
            text_dir=text_dir,
        )

    elif args.command == "validate_tsv":
        from ir_system.evaluation import validate_tsv

        all_pass, failures = validate_tsv(tsv_dir=tsv_dir)
        if all_pass:
            print("PASS: All 18 TSV files are valid.")
            sys.exit(0)
        else:
            for msg in failures:
                print(f"FAIL: {msg}")
            sys.exit(1)

    elif args.command == "judge":
        from ir_system.judge import run_judge

        run_judge(
            tsv_path=args.tsv,
            k=args.k,
            index_dir=index_dir,
        )

    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


def _write_index_build_metrics(stats: dict, index_dir: str) -> None:
    """Write index_build_metrics.json from build_index return value."""
    log_dir = Path("outputs") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    index_path = Path(index_dir)
    postings_bytes = 0
    postings_file = index_path / "postings.bin"
    if postings_file.exists():
        postings_bytes = postings_file.stat().st_size
    sqlite_bytes = 0
    sqlite_file = index_path / "index.sqlite"
    if sqlite_file.exists():
        sqlite_bytes = sqlite_file.stat().st_size

    metrics = {
        "elapsed_seconds": stats.get("elapsed_seconds", 0.0),
        "docs_per_second": stats.get("docs_per_second", 0.0),
        "mb_per_second": stats.get("mb_per_second", 0.0),
        "peak_tracemalloc_mb": stats.get("peak_tracemalloc_mb", 0.0),
        "blocks_written": stats.get("block_count", 0),
        "final_vocab_size": stats.get("vocab_size", 0),
        "N": stats.get("N", 0),
        "avgdl": stats.get("avgdl", 0.0),
        "postings_bytes": postings_bytes,
        "sqlite_bytes": sqlite_bytes,
        "total_text_bytes_read": stats.get("total_text_bytes", 0),
    }

    metrics_path = log_dir / "index_build_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print(f"[INFO] Index build metrics written to {metrics_path}")


def _load_metadata_map(index_dir: str) -> dict[str, dict]:
    import sqlite3

    db_path = Path(index_dir) / "index.sqlite"
    metadata: dict[str, dict] = {}
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


def _log_query_metrics(
    model: str,
    query: str,
    topk: int,
    elapsed_ms: float,
    postings_accessed: int,
    candidates_scored: int,
    phrase_fallback_used: bool,
) -> None:
    log_dir = Path("outputs") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "query": query,
        "topk": topk,
        "elapsed_ms": elapsed_ms,
        "postings_accessed": postings_accessed,
        "candidates_scored": candidates_scored,
        "phrase_fallback_used": phrase_fallback_used,
    }
    with (log_dir / "query_metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _run_query(
    model: str,
    raw_query: str,
    structured: object,
    vsm: object,
    bm25: object,
    N: int,
    index_dir: str,
    allow_full_scan: bool,
    text_dir: str,
) -> tuple[list, bool]:
    """Execute a single query against the chosen model.

    Returns (results, phrase_fallback_used).
    """
    from ir_system.phrase_fallback import apply_phrase_fallback_bm25, is_phrase_fallback_query
    from ir_system.preprocessing import preprocess_query
    import sqlite3

    phrase_fallback_used = False

    if model == "structured":
        results = structured.search(raw_query)
    elif model == "vsm":
        Q = set(preprocess_query(raw_query))
        df_map: dict[str, int] = {}
        with sqlite3.connect(Path(index_dir) / "index.sqlite") as conn:
            for term in Q:
                row = conn.execute(
                    "SELECT df FROM lexicon WHERE term=?", (term,),
                ).fetchone()
                if row is not None:
                    df_map[term] = int(row[0])
        df_lookup = lambda t: df_map.get(t)
        if is_phrase_fallback_query(Q, df_lookup, N, raw_query=raw_query):
            phrase_fallback_used = True
            results = []
            print("VSM skipped - phrase fallback trigger fired.")
        else:
            results = vsm.search(raw_query)
    else:  # bm25
        results = bm25.search(raw_query)
        Q = set(preprocess_query(raw_query))
        df_map: dict[str, int] = {}
        with sqlite3.connect(Path(index_dir) / "index.sqlite") as conn:
            for term in Q:
                row = conn.execute(
                    "SELECT df FROM lexicon WHERE term=?", (term,),
                ).fetchone()
                if row is not None:
                    df_map[term] = int(row[0])
        df_lookup = lambda t: df_map.get(t)
        if is_phrase_fallback_query(Q, df_lookup, N, raw_query=raw_query):
            results, phrase_fallback_used, _ = apply_phrase_fallback_bm25(
                raw_query, results, index_dir,
                allow_full_scan=allow_full_scan,
                text_dir=Path(text_dir),
            )

    return results, phrase_fallback_used


def _run_single_search(
    index_dir: str,
    model: str,
    query: str,
    topk: int,
    allow_full_scan: bool,
    text_dir: str,
) -> None:
    """Run a single search query and print results."""
    from ir_system.retrieval_bm25 import BM25Search
    from ir_system.retrieval_structured import StructuredSearch
    from ir_system.retrieval_vsm import VSMSearch
    from ir_system.utils import load_index_globals

    docid_to_gutid, doc_len, doc_norm, N, avgdl = load_index_globals(index_dir)
    metadata = _load_metadata_map(index_dir)

    structured = StructuredSearch(index_dir)
    bm25 = BM25Search(index_dir, docid_to_gutid, doc_len, N, avgdl)
    vsm = VSMSearch(index_dir, docid_to_gutid, doc_norm, N)

    start = time.perf_counter()
    results, phrase_fallback_used = _run_query(
        model, query, structured, vsm, bm25, N,
        index_dir, allow_full_scan, text_dir,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    results = results[:topk]
    for i, (gutid, score) in enumerate(results, start=1):
        meta = metadata.get(gutid, {})
        title = meta.get("title", "")
        authors = meta.get("authors", "")
        print(f"{i} | {gutid} | {score:.4f} | {title} | {authors}")

    _log_query_metrics(
        model=model, query=query, topk=topk, elapsed_ms=elapsed_ms,
        postings_accessed=0, candidates_scored=len(results),
        phrase_fallback_used=phrase_fallback_used,
    )


def _run_interactive(index_dir: str, allow_full_scan: bool, text_dir: str) -> None:
    from ir_system.retrieval_bm25 import BM25Search
    from ir_system.retrieval_structured import StructuredSearch
    from ir_system.retrieval_vsm import VSMSearch
    from ir_system.utils import load_index_globals

    docid_to_gutid, doc_len, doc_norm, N, avgdl = load_index_globals(index_dir)
    metadata = _load_metadata_map(index_dir)

    structured = StructuredSearch(index_dir)
    bm25 = BM25Search(index_dir, docid_to_gutid, doc_len, N, avgdl)
    vsm = VSMSearch(index_dir, docid_to_gutid, doc_norm, N)

    print("Interactive mode. Type 'quit' or 'exit' to leave.")
    while True:
        model = input("Model (structured/vsm/bm25): ").strip().lower()
        if model in {"quit", "exit"}:
            break
        if model not in {"structured", "vsm", "bm25"}:
            print("Invalid model. Choose structured, vsm, or bm25.")
            continue

        raw_query = input("Query: ").strip()
        if raw_query.lower() in {"quit", "exit"}:
            break

        topk_input = input("TopK [10]: ").strip()
        topk = 10
        if topk_input:
            try:
                topk = int(topk_input)
            except ValueError:
                print("Invalid TopK; using default 10.")
                topk = 10

        start = time.perf_counter()
        results, phrase_fallback_used = _run_query(
            model, raw_query, structured, vsm, bm25, N,
            index_dir, allow_full_scan, text_dir,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        results = results[:topk]
        for i, (gutid, score) in enumerate(results, start=1):
            meta = metadata.get(gutid, {})
            title = meta.get("title", "")
            authors = meta.get("authors", "")
            snippet = ""
            print(
                f"{i} | {gutid} | {score:.4f} | {title} | {authors} | {snippet}"
            )

        _log_query_metrics(
            model=model, query=raw_query, topk=topk, elapsed_ms=elapsed_ms,
            postings_accessed=0, candidates_scored=len(results),
            phrase_fallback_used=phrase_fallback_used,
        )


if __name__ == "__main__":
    main()
