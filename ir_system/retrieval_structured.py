"""Structured metadata retrieval."""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple

from ir_system.preprocessing import preprocess, preprocess_query


class StructuredSearch:
    """Metadata-only search over title, author, and subject fields."""

    def __init__(self, index_dir: str, include_missing_text: bool = False) -> None:
        self.index_dir = Path(index_dir)
        self.include_missing_text = include_missing_text
        self._metadata = self._load_metadata()
        self._token_to_gutid = self._build_token_index()

    def _load_metadata(self) -> Dict[str, dict]:
        db_path = self.index_dir / "index.sqlite"
        metadata: Dict[str, dict] = {}
        with sqlite3.connect(db_path) as conn:
            for row in conn.execute(
                "SELECT gutenberg_id, title, authors, language, bookshelf, rights, has_text "
                "FROM metadata"
            ):
                metadata[str(row[0])] = {
                    "title": row[1] or "",
                    "authors": row[2] or "",
                    "language": row[3] or "",
                    "bookshelf": row[4] or "",
                    "rights": row[5] or "",
                    "has_text": int(row[6]),
                }
        return metadata

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.casefold()
        for h in [
            "\u002D",
            "\u2010",
            "\u2011",
            "\u2012",
            "\u2013",
            "\u2014",
        ]:
            text = text.replace(h, " ")
        for apos in ["\u0027", "\u2019"]:
            text = text.replace(apos, "")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _build_token_index(self) -> Dict[str, Set[str]]:
        token_map: Dict[str, Set[str]] = {}
        for gutid, meta in self._metadata.items():
            title_tokens = preprocess(meta["title"], include_aliases=False)
            author_tokens = preprocess(meta["authors"], include_aliases=False)
            for token in set(title_tokens + author_tokens):
                token_map.setdefault(token, set()).add(gutid)
        return token_map

    @staticmethod
    def _sorted_tokens(text: str) -> str:
        tokens = preprocess(text, include_aliases=False)
        return " ".join(sorted(tokens))

    def search(self, raw_query: str) -> List[Tuple[str, float]]:
        q_norm = self._normalize_text(raw_query)
        if not q_norm:
            return []

        query_tokens = preprocess_query(raw_query)
        Q = set(query_tokens)

        if Q:
            candidates: Set[str] = set()
            for token in Q:
                candidates |= self._token_to_gutid.get(token, set())
            if not candidates:
                candidates = set(self._metadata.keys())
        else:
            candidates = set(self._metadata.keys())

        q_sorted = self._sorted_tokens(raw_query)
        results: List[Tuple[str, float]] = []

        for gutid in candidates:
            meta = self._metadata[gutid]
            if not self.include_missing_text and meta["has_text"] == 0:
                continue

            title_norm = self._normalize_text(meta["title"])
            authors_norm = self._normalize_text(meta["authors"])
            other_norm = self._normalize_text(
                f"{meta['language']} {meta['bookshelf']} {meta['rights']}"
            )

            f_title_exact = 1 if q_norm and q_norm in title_norm else 0

            f_author_exact = 0
            if q_sorted:
                for author in meta["authors"].split(";"):
                    if not author.strip():
                        continue
                    author_sorted = self._sorted_tokens(author)
                    if q_sorted == author_sorted:
                        f_author_exact = 1
                        break

            title_tokens = set(preprocess(meta["title"], include_aliases=False))
            author_tokens = set(preprocess(meta["authors"], include_aliases=False))
            other_tokens = set(preprocess(other_norm, include_aliases=False))

            h_title = len(Q & title_tokens)
            h_author = len(Q & author_tokens)
            h_other = len(Q & other_tokens)

            score = (
                10 * f_title_exact
                + 5 * f_author_exact
                + 2 * h_title
                + 1 * h_author
                + 1 * h_other
            )

            if score > 0:
                results.append((gutid, float(score)))

        results = [(gid, round(s, 6)) for gid, s in results]
        results.sort(key=lambda x: (-x[1], x[0]))
        return results
