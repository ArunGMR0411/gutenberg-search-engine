"""Shared utilities for index loading and configuration."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Tuple


def load_index_globals(index_dir: str) -> tuple[
    Dict[int, str], Dict[int, int], Dict[int, float], int, float
]:
    """Load docid_to_gutid, doc_len, doc_norm, N, avgdl from SQLite."""
    index_path = Path(index_dir) / "index.sqlite"

    docid_to_gutid: Dict[int, str] = {}
    doc_len: Dict[int, int] = {}
    doc_norm: Dict[int, float] = {}

    with sqlite3.connect(index_path) as conn:
        # Single JOIN query loads docid_to_gutid, doc_len, doc_norm
        for row in conn.execute(
            "SELECT d.docid, d.gutenberg_id, s.doc_len, s.doc_norm "
            "FROM docmap d JOIN docstats s USING (docid)"
        ):
            docid = int(row[0])
            docid_to_gutid[docid] = str(row[1])
            doc_len[docid] = int(row[2])
            doc_norm[docid] = float(row[3])

        # Scalar globals
        n_row = conn.execute(
            "SELECT value FROM globals WHERE key='N'"
        ).fetchone()
        avgdl_row = conn.execute(
            "SELECT value FROM globals WHERE key='avgdl'"
        ).fetchone()

    if n_row is None or avgdl_row is None:
        raise ValueError("globals table missing N and/or avgdl")

    N = int(n_row[0])
    avgdl = float(avgdl_row[0])

    return docid_to_gutid, doc_len, doc_norm, N, avgdl
