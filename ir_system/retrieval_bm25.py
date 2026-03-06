"""BM25 retrieval."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

from ir_system.postings import PostingsReader
from ir_system.preprocessing import preprocess_query


class BM25Search:
    """Okapi BM25 search with in-memory doc_len and globals."""

    def __init__(
        self,
        index_dir: str,
        docid_to_gutid: Dict[int, str],
        doc_len: Dict[int, int],
        N: int,
        avgdl: float,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.docid_to_gutid = docid_to_gutid
        self.doc_len = doc_len
        self.N = N
        self.avgdl = avgdl
        self.k1 = k1
        self.b = b

    def search(self, raw_query: str) -> List[Tuple[str, float]]:
        query_tokens = preprocess_query(raw_query)
        if not query_tokens:
            return []

        Q = set(query_tokens)
        acc: Dict[int, float] = {}

        db_path = self.index_dir / "index.sqlite"
        postings_path = self.index_dir / "postings.bin"
        cache: Dict[tuple[int, int], List[Tuple[int, int]]] = {}

        with sqlite3.connect(db_path) as conn, PostingsReader(postings_path) as reader:
            for term in Q:
                row = conn.execute(
                    "SELECT df, offset, length FROM lexicon WHERE term=?",
                    (term,),
                ).fetchone()
                if row is None:
                    continue

                df_t, offset, length = row
                idf_t = math.log((self.N - df_t + 0.5) / (df_t + 0.5))
                if idf_t <= 0 or df_t / self.N > 0.8:
                    idf_t = 0.0
                if idf_t == 0.0:
                    continue

                for docid, tf_d in reader.decode_postings_cached(
                    offset, length, cache
                ):
                    if tf_d <= 0:
                        continue
                    denom = tf_d + self.k1 * (
                        1 - self.b + self.b * self.doc_len[docid] / self.avgdl
                    )
                    acc[docid] = acc.get(docid, 0.0) + idf_t * tf_d * (
                        self.k1 + 1
                    ) / denom

        results = [
            (self.docid_to_gutid[docid], round(score, 6)) for docid, score in acc.items()
        ]
        results.sort(key=lambda x: (-x[1], x[0]))
        return results
