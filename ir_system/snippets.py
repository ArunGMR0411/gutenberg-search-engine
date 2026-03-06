"""Snippet generation for search results."""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

from ir_system.postings import PostingsReader
from ir_system.preprocessing import preprocess_query


def _vsm_idf(df_t: int, N: int) -> float:
    """Compute VSM IDF: log((N+1)/(df(t)+1)) + 1 (always positive)."""
    return math.log((N + 1) / (df_t + 1)) + 1


def _normalize_for_snippet(text: str) -> str:
    """Normalize text for character-level snippet matching.

    Applies NFKC + casefold only (preserves whitespace structure for
    line-number counting).
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    return text


def select_best_token(
    query_tokens: List[str],
    N: int,
    doc_postings_docids: Dict[str, set],
    term_df: Dict[str, int],
    docid: int,
) -> str | None:
    """Pick the highest-IDF query token that appears in this document."""
    best_token = None
    best_idf = -1.0

    seen = set()
    for token in query_tokens:
        if token in seen:
            continue
        seen.add(token)

        if token not in term_df:
            continue
        if docid not in doc_postings_docids.get(token, set()):
            continue

        idf = _vsm_idf(term_df[token], N)
        if idf > best_idf:
            best_idf = idf
            best_token = token

    return best_token


def generate_snippet(
    text: str,
    best_token: str,
    window: int = 250,
) -> Tuple[str, int]:
    """Extract a 250-char window around the best token. Returns (snippet, start_line)."""
    norm_text = _normalize_for_snippet(text)

    pos = norm_text.find(best_token)
    if pos == -1:
        snippet = text[:window]
        snippet = re.sub(r"\s+", " ", snippet).strip()
        return snippet, 1

    half = window // 2
    center = pos + len(best_token) // 2
    start = max(0, center - half)
    end = min(len(text), start + window)
    if end - start < window:
        start = max(0, end - window)

    start_line = 1 + text[:start].count("\n")

    snippet = text[start:end]
    snippet = re.sub(r"\s+", " ", snippet).strip()

    return snippet, start_line


def generate_snippet_for_result(
    gutenberg_id: str,
    query_tokens: List[str],
    N: int,
    index_dir: str,
    text_dir: str | None = None,
    has_text: int = 1,
    docid: int | None = None,
    gutid_to_docid: Dict[str, int] | None = None,
) -> Tuple[str, str]:
    """Generate a snippet and start_line for one search result."""
    if has_text == 0:
        return "", ""

    index_path = Path(index_dir)
    if text_dir is None:
        base = Path(__file__).resolve().parents[1]
        if index_path.name == "index_sample":
            text_dir_path = base / "datasets" / "sample"
        else:
            text_dir_path = base / "datasets" / "texts"
    else:
        text_dir_path = Path(text_dir)

    text_path = text_dir_path / f"{gutenberg_id}.txt"
    if not text_path.exists():
        return "", ""

    # Resolve docid if not provided
    if docid is None:
        if gutid_to_docid is not None:
            docid = gutid_to_docid.get(gutenberg_id)
        else:
            db_path = index_path / "index.sqlite"
            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT docid FROM docmap WHERE gutenberg_id=?",
                    (gutenberg_id,),
                ).fetchone()
                if row is None:
                    return "", ""
                docid = int(row[0])

    if docid is None:
        return "", ""

    # Look up query terms in lexicon and collect postings info
    db_path = index_path / "index.sqlite"
    postings_path = index_path / "postings.bin"

    term_df: Dict[str, int] = {}
    doc_postings_docids: Dict[str, set] = {}

    unique_tokens = []
    seen = set()
    for t in query_tokens:
        if t not in seen:
            unique_tokens.append(t)
            seen.add(t)

    with sqlite3.connect(db_path) as conn, PostingsReader(postings_path) as reader:
        cache: Dict[tuple[int, int], List[Tuple[int, int]]] = {}
        for term in unique_tokens:
            row = conn.execute(
                "SELECT df, offset, length FROM lexicon WHERE term=?",
                (term,),
            ).fetchone()
            if row is None:
                continue
            df_t, offset, length = row
            term_df[term] = df_t

            postings = reader.decode_postings_cached(offset, length, cache)
            doc_postings_docids[term] = {d for d, _ in postings}

    best_token = select_best_token(
        query_tokens, N, doc_postings_docids, term_df, docid
    )
    if best_token is None:
        return "", ""

    text = text_path.read_text(encoding="utf-8", errors="replace")
    snippet, start_line = generate_snippet(text, best_token)

    return snippet, str(start_line)


def generate_snippets_batch(
    results: List[Tuple[str, float]],
    query_tokens: List[str],
    N: int,
    index_dir: str,
    metadata_map: Dict[str, dict],
    text_dir: str | None = None,
    max_results: int = 100,
) -> List[Tuple[str, str]]:
    """Generate snippets for a ranked result list. Returns [(snippet, start_line), ...]."""
    # Build reverse mapping gutid -> docid
    index_path = Path(index_dir)
    db_path = index_path / "index.sqlite"
    gutid_to_docid: Dict[str, int] = {}
    with sqlite3.connect(db_path) as conn:
        for row in conn.execute("SELECT docid, gutenberg_id FROM docmap"):
            gutid_to_docid[str(row[1])] = int(row[0])

    # Pre-load all term data for the query tokens (single pass)
    postings_path = index_path / "postings.bin"
    term_df: Dict[str, int] = {}
    doc_postings_docids: Dict[str, set] = {}

    unique_tokens = []
    seen_tokens = set()
    for t in query_tokens:
        if t not in seen_tokens:
            unique_tokens.append(t)
            seen_tokens.add(t)

    with sqlite3.connect(db_path) as conn, PostingsReader(postings_path) as reader:
        cache: Dict[tuple[int, int], List[Tuple[int, int]]] = {}
        for term in unique_tokens:
            row = conn.execute(
                "SELECT df, offset, length FROM lexicon WHERE term=?",
                (term,),
            ).fetchone()
            if row is None:
                continue
            df_t, offset, length = row
            term_df[term] = df_t
            postings = reader.decode_postings_cached(offset, length, cache)
            doc_postings_docids[term] = {d for d, _ in postings}

    # Resolve text directory
    if text_dir is None:
        base = Path(__file__).resolve().parents[1]
        if index_path.name == "index_sample":
            text_dir_path = base / "datasets" / "sample"
        else:
            text_dir_path = base / "datasets" / "texts"
    else:
        text_dir_path = Path(text_dir)

    snippets: List[Tuple[str, str]] = []
    for gutid, _score in results[:max_results]:
        meta = metadata_map.get(gutid, {})
        has_text = meta.get("has_text", 0)

        if has_text == 0:
            snippets.append(("", ""))
            continue

        docid = gutid_to_docid.get(gutid)
        if docid is None:
            snippets.append(("", ""))
            continue

        best_token = select_best_token(
            query_tokens, N, doc_postings_docids, term_df, docid
        )
        if best_token is None:
            snippets.append(("", ""))
            continue

        text_path = text_dir_path / f"{gutid}.txt"
        if not text_path.exists():
            snippets.append(("", ""))
            continue

        text = text_path.read_text(encoding="utf-8", errors="replace")
        snippet, start_line = generate_snippet(text, best_token)
        snippets.append((snippet, str(start_line)))

    return snippets
