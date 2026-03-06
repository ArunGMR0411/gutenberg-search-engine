"""VSM TF-IDF cosine retrieval."""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from ir_system.phrase_fallback import is_phrase_fallback_query
from ir_system.postings import PostingsReader
from ir_system.preprocessing import preprocess_query


def build_tf_query(query_tokens: List[str]) -> Counter:
    """Build raw term-frequency map for query tokens."""
    return Counter(query_tokens)


class VSMSearch:
    """TF-IDF cosine similarity search with early-abort rule."""

    def __init__(
        self,
        index_dir: str,
        docid_to_gutid: Dict[int, str],
        doc_norm: Dict[int, float],
        N: int,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.docid_to_gutid = docid_to_gutid
        self.doc_norm = doc_norm
        self.N = N

    def search(self, raw_query: str) -> List[Tuple[str, float]]:
        query_tokens = preprocess_query(raw_query)
        if not query_tokens:
            return []

        tf_query = build_tf_query(query_tokens)
        Q = set(tf_query.keys())

        db_path = self.index_dir / "index.sqlite"
        postings_path = self.index_dir / "postings.bin"
        cache: Dict[tuple[int, int], List[Tuple[int, int]]] = {}

        term_rows: Dict[str, Tuple[int, int, int]] = {}
        high_df_count = 0

        with sqlite3.connect(db_path) as conn:
            for term in Q:
                row = conn.execute(
                    "SELECT df, offset, length FROM lexicon WHERE term=?",
                    (term,),
                ).fetchone()
                if row is None:
                    continue
                df_t, offset, length = row
                term_rows[term] = (df_t, offset, length)
                if df_t / self.N > 0.8:
                    high_df_count += 1

            df_lookup = lambda t: term_rows.get(t, (None, 0, 0))[0]
            if is_phrase_fallback_query(Q, df_lookup, self.N, raw_query=raw_query):
                print("VSM skipped - phrase fallback trigger fired.")
                return []

            w_tq_map: Dict[str, float] = {}
            acc: Dict[int, float] = {}

            with PostingsReader(postings_path) as reader:
                for term in Q:
                    if term not in term_rows:
                        continue
                    df_t, offset, length = term_rows[term]
                    if tf_query[term] <= 0:
                        continue

                    idf_t = math.log((self.N + 1) / (df_t + 1)) + 1
                    w_tq = (1 + math.log(tf_query[term])) * idf_t
                    w_tq_map[term] = w_tq

                    for docid, tf_d in reader.decode_postings_cached(
                        offset, length, cache
                    ):
                        if tf_d <= 0:
                            continue
                        w_td = (1 + math.log(tf_d)) * idf_t
                        acc[docid] = acc.get(docid, 0.0) + w_tq * w_td

        norm_q = math.sqrt(sum(v ** 2 for v in w_tq_map.values()))
        if norm_q <= 0:
            return []

        results = []
        for docid, score in acc.items():
            norm_d = self.doc_norm.get(docid, 0.0)
            if norm_d <= 0:
                continue
            results.append((self.docid_to_gutid[docid], round(score / (norm_q * norm_d), 6)))

        results.sort(key=lambda x: (-x[1], x[0]))
        return results
