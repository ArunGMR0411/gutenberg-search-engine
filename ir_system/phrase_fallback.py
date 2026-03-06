"""Phrase fallback for stopword-heavy queries."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import unicodedata


def normalize_for_phrase(text: str) -> str:
    """NFKC + casefold + strip non-alphanumeric chars for substring matching."""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    # Replace every non-letter, non-digit character (including _) with a space
    text = re.sub(r"[^\w]|_", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_quoted_phrase(raw_query: str) -> str:
    """Extract and normalize quoted spans for phrase matching."""
    spans = re.findall(r'"(.*?)"', raw_query)
    if not spans:
        return ""
    joined = " ".join(span.strip() for span in spans if span.strip())
    return normalize_for_phrase(joined)


def build_phrase_query(raw_query: str) -> str:
    """Build normalized phrase query for substring matching."""
    q_norm = extract_quoted_phrase(raw_query)
    if q_norm:
        return q_norm
    return normalize_for_phrase(raw_query)


def is_phrase_fallback_query(
    Q: Iterable[str],
    df_lookup: Callable[[str], int | None],
    N: int,
    raw_query: str = "",
) -> bool:
    """Return True if phrase fallback trigger fires."""
    if '"' in raw_query:
        return True
    Q_list = list(Q)
    if N <= 0 or len(Q_list) < 4:
        return False
    high_df = 0
    for term in Q_list:
        df_t = df_lookup(term)
        if df_t is not None and df_t / N > 0.8:
            high_df += 1
    return (high_df / len(Q_list)) > 0.6


def _resolve_text_dir(index_dir: str) -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    if Path(index_dir).name == "index_sample":
        return base_dir / "datasets" / "sample"
    return base_dir / "datasets" / "texts"


def _iter_text_ids(text_dir: Path) -> Iterable[str]:
    for path in sorted(text_dir.glob("*.txt")):
        yield path.stem


def apply_phrase_fallback_bm25(
    raw_query: str,
    bm25_results: List[Tuple[str, float]],
    index_dir: str,
    allow_full_scan: bool,
    topn: int = 5000,
    text_dir: Path | None = None,
) -> Tuple[List[Tuple[str, float]], bool, bool]:
    """Re-rank (or full-scan) BM25 results using exact phrase matching."""
    q_norm = build_phrase_query(raw_query)
    if not q_norm:
        return bm25_results, False, False

    text_dir = text_dir or _resolve_text_dir(index_dir)
    candidates = bm25_results[:topn]

    # full-corpus scan when BM25 returned nothing
    if not candidates:
        if not allow_full_scan:
            print("[WARN] BM25 returned zero candidates; full scan not allowed.")
            return bm25_results, False, False
        updated: List[Tuple[str, float]] = []
        for gutid in _iter_text_ids(text_dir):
            text_path = text_dir / f"{gutid}.txt"
            if not text_path.exists():
                continue
            text = text_path.read_text(encoding="utf-8", errors="replace")
            if q_norm in normalize_for_phrase(text):
                updated.append((gutid, 5.0))
                if len(updated) >= 200:  # pragmatic cap for perf
                    break
        updated.sort(key=lambda x: (-x[1], x[0]))
        return updated, True, True

    # re-rank existing BM25 candidates
    updated = []
    for gutid, score in candidates:
        text_path = text_dir / f"{gutid}.txt"
        if not text_path.exists():
            updated.append((gutid, score))
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace")
        if q_norm in normalize_for_phrase(text):
            score = score + 5.0
        updated.append((gutid, score))

    updated.sort(key=lambda x: (-x[1], x[0]))
    return updated, True, False
