"""Tests for retrieval models and index loading."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from ir_system.phrase_fallback import (
    apply_phrase_fallback_bm25,
    build_phrase_query,
    is_phrase_fallback_query,
    normalize_for_phrase,
)
from ir_system.postings import gap_encode, varint_encode
from ir_system.preprocessing import preprocess_query
from ir_system.retrieval_bm25 import BM25Search
from ir_system.retrieval_structured import StructuredSearch
from ir_system.retrieval_vsm import VSMSearch, build_tf_query
from ir_system.utils import load_index_globals


def write_postings_file(
    postings_by_term: Dict[str, List[Tuple[int, int]]], postings_path: Path
) -> Dict[str, Tuple[int, int, int]]:
    """Write postings.bin and return lexicon entries."""
    lexicon: Dict[str, Tuple[int, int, int]] = {}
    with open(postings_path, "wb") as f:
        for term in sorted(postings_by_term.keys()):
            postings = sorted(postings_by_term[term], key=lambda x: x[0])
            docids = [docid for docid, _ in postings]
            tfs = [tf for _, tf in postings]
            gaps = gap_encode(docids)

            blob = bytearray()
            blob.extend(varint_encode(len(postings)))
            for gap, tf in zip(gaps, tfs):
                blob.extend(varint_encode(gap))
                blob.extend(varint_encode(tf))

            offset = f.tell()
            f.write(blob)
            lexicon[term] = (len(postings), offset, len(blob))
    return lexicon


def compute_doc_norm(
    postings_by_term: Dict[str, List[Tuple[int, int]]], N: int
) -> Dict[int, float]:
    """Compute doc_norm values for test fixtures."""
    norm_acc: Dict[int, float] = {}
    for term, postings in postings_by_term.items():
        df_t = len(postings)
        idf_t = math.log((N + 1) / (df_t + 1)) + 1
        for docid, tf_d in postings:
            if tf_d <= 0:
                continue
            w = (1 + math.log(tf_d)) * idf_t
            norm_acc[docid] = norm_acc.get(docid, 0.0) + w * w
    return {docid: math.sqrt(v) for docid, v in norm_acc.items()}


@pytest.fixture
def tiny_index(tmp_path: Path) -> Path:
    """Create a tiny on-disk index with 3 docs and 4 terms."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    postings_by_term = {
        "apple": [(0, 2), (1, 1)],
        "banana": [(0, 1), (2, 3)],
        "cherry": [(2, 2)],
        "corrupt": [(0, 0)],
    }

    postings_path = index_dir / "postings.bin"
    lexicon = write_postings_file(postings_by_term, postings_path)

    docid_to_gutid = {0: "1", 1: "2", 2: "3"}
    doc_len = {0: 3, 1: 1, 2: 5}
    N = 3
    avgdl = sum(doc_len.values()) / N
    doc_norm = compute_doc_norm(postings_by_term, N)

    db_path = index_dir / "index.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            "CREATE TABLE lexicon (term TEXT PRIMARY KEY, df INTEGER, offset INTEGER, length INTEGER);"
            "CREATE TABLE docstats (docid INTEGER PRIMARY KEY, doc_len INTEGER, doc_norm REAL);"
            "CREATE TABLE globals (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE docmap (docid INTEGER PRIMARY KEY, gutenberg_id TEXT UNIQUE NOT NULL);"
            "CREATE TABLE metadata (gutenberg_id TEXT PRIMARY KEY, title TEXT, authors TEXT, language TEXT, bookshelf TEXT, rights TEXT, has_text INTEGER);"
        )

        for term, (df_t, offset, length) in lexicon.items():
            conn.execute(
                "INSERT INTO lexicon (term, df, offset, length) VALUES (?, ?, ?, ?)",
                (term, df_t, offset, length),
            )

        for docid, gutid in docid_to_gutid.items():
            conn.execute(
                "INSERT INTO docmap (docid, gutenberg_id) VALUES (?, ?)",
                (docid, gutid),
            )
            conn.execute(
                "INSERT INTO docstats (docid, doc_len, doc_norm) VALUES (?, ?, ?)",
                (docid, doc_len[docid], doc_norm.get(docid, 0.0)),
            )

        conn.execute("INSERT INTO globals (key, value) VALUES ('N', ?)", (N,))
        conn.execute(
            "INSERT INTO globals (key, value) VALUES ('avgdl', ?)", (avgdl,)
        )

        conn.execute(
            "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "1",
                "Do Androids Dream",
                "Dick, Philip K.",
                "en",
                "",
                "",
                1,
            ),
        )
        conn.execute(
            "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "2",
                "Another Book",
                "Someone Else",
                "en",
                "",
                "",
                0,
            ),
        )
        conn.execute(
            "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "3",
                "More Stories",
                "Other Author",
                "en",
                "",
                "",
                1,
            ),
        )
        conn.commit()

    return index_dir


def test_load_index_globals_one_pass(monkeypatch: pytest.MonkeyPatch, tiny_index: Path):
    statements: List[str] = []
    original_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        conn.set_trace_callback(lambda stmt: statements.append(stmt))
        return conn

    monkeypatch.setattr("ir_system.utils.sqlite3.connect", traced_connect)

    docid_to_gutid, doc_len, doc_norm, N, avgdl = load_index_globals(str(tiny_index))

    join_selects = [s for s in statements if "FROM docmap d JOIN docstats s" in s]
    assert len(join_selects) == 1
    assert N == 3
    assert round(avgdl, 6) > 0
    assert docid_to_gutid[0] == "1"
    assert doc_len[2] == 5
    assert doc_norm[2] > 0


def test_bm25_score(tiny_index: Path):
    docid_to_gutid, doc_len, doc_norm, N, avgdl = load_index_globals(str(tiny_index))
    bm25 = BM25Search(str(tiny_index), docid_to_gutid, doc_len, N, avgdl)

    results = bm25.search("cherry")
    assert len(results) == 1
    gutid, score = results[0]
    assert gutid == "3"

    df_t = 1
    idf_t = math.log((N - df_t + 0.5) / (df_t + 0.5))
    tf_d = 2
    denom = tf_d + bm25.k1 * (1 - bm25.b + bm25.b * doc_len[2] / avgdl)
    expected = idf_t * tf_d * (bm25.k1 + 1) / denom
    assert round(score, 4) == round(expected, 4)


def test_vsm_cosine_scores(tiny_index: Path):
    docid_to_gutid, doc_len, doc_norm, N, avgdl = load_index_globals(str(tiny_index))
    vsm = VSMSearch(str(tiny_index), docid_to_gutid, doc_norm, N)

    results = vsm.search("apple banana")
    scores = {gutid: score for gutid, score in results}

    idf = math.log((N + 1) / (2 + 1)) + 1
    wq = idf
    norm_q = math.sqrt(wq ** 2 + wq ** 2)

    w_apple_d0 = (1 + math.log(2)) * idf
    w_banana_d0 = (1 + math.log(1)) * idf
    dot_d0 = wq * w_apple_d0 + wq * w_banana_d0
    expected_d0 = dot_d0 / (norm_q * doc_norm[0])

    w_apple_d1 = (1 + math.log(1)) * idf
    dot_d1 = wq * w_apple_d1
    expected_d1 = dot_d1 / (norm_q * doc_norm[1])

    w_banana_d2 = (1 + math.log(3)) * idf
    dot_d2 = wq * w_banana_d2
    expected_d2 = dot_d2 / (norm_q * doc_norm[2])

    assert round(scores["1"], 4) == round(expected_d0, 4)
    assert round(scores["2"], 4) == round(expected_d1, 4)
    assert round(scores["3"], 4) == round(expected_d2, 4)


def test_vsm_tf_query_counter():
    tokens = preprocess_query("to be or not to be")
    tf_query = build_tf_query(tokens)
    assert tf_query["be"] == 2


def test_vsm_single_lexicon_lookup(monkeypatch: pytest.MonkeyPatch, tiny_index: Path):
    statements: List[str] = []
    original_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        conn.set_trace_callback(lambda stmt: statements.append(stmt))
        return conn

    monkeypatch.setattr("ir_system.retrieval_vsm.sqlite3.connect", traced_connect)

    docid_to_gutid, doc_len, doc_norm, N, avgdl = load_index_globals(str(tiny_index))
    vsm = VSMSearch(str(tiny_index), docid_to_gutid, doc_norm, N)
    vsm.search("apple banana")

    lexicon_selects = [
        s for s in statements if "SELECT df, offset, length FROM lexicon" in s
    ]
    assert len(lexicon_selects) == 2


def test_structured_author_exact(tiny_index: Path):
    structured = StructuredSearch(str(tiny_index))
    results = structured.search("Philip K Dick")
    assert results
    gutid, score = results[0]
    assert gutid == "1"
    assert score == 8.0


def test_tf_guard_bm25_vsm(tiny_index: Path):
    docid_to_gutid, doc_len, doc_norm, N, avgdl = load_index_globals(str(tiny_index))

    bm25 = BM25Search(str(tiny_index), docid_to_gutid, doc_len, N, avgdl)
    vsm = VSMSearch(str(tiny_index), docid_to_gutid, doc_norm, N)

    assert bm25.search("corrupt") == []
    assert vsm.search("corrupt") == []


def test_doc_len_doc_norm_dict_access(tiny_index: Path):
    docid_to_gutid, doc_len, doc_norm, N, avgdl = load_index_globals(str(tiny_index))

    class SpyLenDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.getitem_count = 0

        def __getitem__(self, key):
            self.getitem_count += 1
            return super().__getitem__(key)

    class SpyNormDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.get_count = 0

        def get(self, key, default=None):
            self.get_count += 1
            return super().get(key, default)

    spy_len = SpyLenDict(doc_len)
    spy_norm = SpyNormDict(doc_norm)

    bm25 = BM25Search(str(tiny_index), docid_to_gutid, spy_len, N, avgdl)
    vsm = VSMSearch(str(tiny_index), docid_to_gutid, spy_norm, N)

    bm25.search("cherry")
    vsm.search("apple")

    assert spy_len.getitem_count > 0
    assert spy_norm.get_count > 0


def test_phrase_fallback_trigger_quotes():
    df_lookup = lambda t: None
    assert is_phrase_fallback_query(set(), df_lookup, 10, raw_query='"to be"')


def test_phrase_fallback_trigger_stopword_heavy():
    Q = {"a", "b", "c", "d"}
    df_lookup = lambda t: 9
    assert is_phrase_fallback_query(Q, df_lookup, 10, raw_query="")


def test_phrase_query_extraction():
    raw = '"to be" or "not to be"'
    assert build_phrase_query(raw) == "to be not to be"


def test_phrase_fallback_scans_zero_score_candidates(tmp_path: Path):
    text_dir = tmp_path / "texts"
    text_dir.mkdir()
    (text_dir / "1.txt").write_text("to be not to be")

    results, used, full_scan = apply_phrase_fallback_bm25(
        '"to be" or "not to be"',
        [("1", 0.0)],
        index_dir="index",
        allow_full_scan=False,
        text_dir=text_dir,
    )

    assert used is True
    assert full_scan is False
    assert results[0][1] == 5.0


def test_phrase_fallback_no_full_scan_without_flag(tmp_path: Path):
    results, used, full_scan = apply_phrase_fallback_bm25(
        '"to be"',
        [],
        index_dir="index",
        allow_full_scan=False,
        text_dir=tmp_path,
    )

    assert results == []
    assert used is False
    assert full_scan is False


def test_phrase_normalize_strips_commas():
    """Phrase with commas must match phrase without commas."""
    assert normalize_for_phrase("to be, or not to be") == "to be or not to be"
    assert normalize_for_phrase("to be or not to be") == "to be or not to be"


def test_phrase_normalize_strips_semicolons():
    assert normalize_for_phrase("hello; world") == "hello world"


def test_phrase_comma_match_in_document(tmp_path: Path):
    """Q1-style query 'to be, or not to be' must find the phrase in text."""
    text_dir = tmp_path / "texts"
    text_dir.mkdir()
    (text_dir / "1524.txt").write_text(
        "To be, or not to be, that is the question."
    )

    results, used, full_scan = apply_phrase_fallback_bm25(
        "to be, or not to be",
        [],                    # zero BM25 candidates
        index_dir="index",
        allow_full_scan=True,
        text_dir=text_dir,
    )

    assert used is True
    assert full_scan is True
    assert len(results) == 1
    assert results[0][0] == "1524"
    assert results[0][1] == 5.0


def test_phrase_quotes_consistency():
    """Quoted and unquoted forms produce the same normalized phrase."""
    from ir_system.phrase_fallback import build_phrase_query

    q_quoted = build_phrase_query('"to be, or not to be"')
    q_plain = build_phrase_query("to be, or not to be")
    assert q_quoted == q_plain
