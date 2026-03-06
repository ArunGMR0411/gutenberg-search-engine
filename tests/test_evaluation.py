"""Tests for evaluation runner, snippets, and TSV validator."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from ir_system.evaluation import (
    EVAL_QUERIES,
    MODEL_NAMES,
    run_eval_queries,
    validate_tsv,
    write_tsv,
)
from ir_system.postings import gap_encode, varint_encode
from ir_system.preprocessing import preprocess_query
from ir_system.snippets import (
    _vsm_idf,
    generate_snippet,
    generate_snippets_batch,
    select_best_token,
)


# ---------------------------------------------------------------------------
# Helpers: reuse tiny_index fixture pattern from test_retrieval.py
# ---------------------------------------------------------------------------


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
def eval_index(tmp_path: Path) -> Path:
    """Create a tiny on-disk index with text files for eval testing."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    # Create text directory with sample documents
    text_dir = tmp_path / "texts"
    text_dir.mkdir()

    doc_texts = {
        "1": "The apple is a fruit. Apples are delicious and healthy.\n"
             "Many people enjoy eating apples every day.\n"
             "Apple pie is a classic dessert.",
        "2": "Banana smoothies are refreshing.\n"
             "A banana contains potassium and vitamins.",
        "3": "Cherry blossoms bloom in spring.\n"
             "The cherry tree is beautiful in April.\n"
             "Cherry picking is a fun activity.\n"
             "Wild cherries grow in the forest.",
    }

    for gutid, text in doc_texts.items():
        (text_dir / f"{gutid}.txt").write_text(text, encoding="utf-8")

    postings_by_term = {
        "apple": [(0, 3), (1, 1)],
        "banana": [(1, 2), (2, 1)],
        "cherry": [(2, 4)],
        "fruit": [(0, 1)],
        "delicious": [(0, 1)],
        "smoothie": [(1, 1)],
        "blossom": [(2, 1)],
    }

    postings_path = index_dir / "postings.bin"
    lexicon = write_postings_file(postings_by_term, postings_path)

    docid_to_gutid = {0: "1", 1: "2", 2: "3"}
    doc_len = {0: 20, 1: 10, 2: 25}
    N = 3
    avgdl = sum(doc_len.values()) / N
    doc_norm = compute_doc_norm(postings_by_term, N)

    db_path = index_dir / "index.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            "CREATE TABLE lexicon (term TEXT PRIMARY KEY, df INTEGER, "
            "offset INTEGER, length INTEGER);"
            "CREATE TABLE docstats (docid INTEGER PRIMARY KEY, "
            "doc_len INTEGER, doc_norm REAL);"
            "CREATE TABLE globals (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE docmap (docid INTEGER PRIMARY KEY, "
            "gutenberg_id TEXT UNIQUE NOT NULL);"
            "CREATE TABLE metadata (gutenberg_id TEXT PRIMARY KEY, "
            "title TEXT, authors TEXT, language TEXT, bookshelf TEXT, "
            "rights TEXT, has_text INTEGER);"
        )

        for term, (df_t, offset, length) in lexicon.items():
            conn.execute(
                "INSERT INTO lexicon VALUES (?, ?, ?, ?)",
                (term, df_t, offset, length),
            )

        for docid, gutid in docid_to_gutid.items():
            conn.execute(
                "INSERT INTO docmap VALUES (?, ?)",
                (docid, gutid),
            )
            conn.execute(
                "INSERT INTO docstats VALUES (?, ?, ?)",
                (docid, doc_len[docid], doc_norm.get(docid, 0.0)),
            )

        conn.execute("INSERT INTO globals VALUES ('N', ?)", (N,))
        conn.execute("INSERT INTO globals VALUES ('avgdl', ?)", (avgdl,))

        conn.execute(
            "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("1", "Apple Tales", "Author A", "en", "", "", 1),
        )
        conn.execute(
            "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2", "Banana Book", "Author B", "en", "", "", 1),
        )
        conn.execute(
            "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("3", "Cherry Stories", "Author C", "en", "", "", 1),
        )
        conn.commit()

    return tmp_path


# ---------------------------------------------------------------------------
# Snippet tests
# ---------------------------------------------------------------------------


class TestSnippetBestToken:
    """Test best-token selection uses VSM IDF (always positive)."""

    def test_best_token_highest_vsm_idf(self):
        """Best token = highest VSM IDF among query terms present in doc."""
        N = 100
        # "rare" has df=1 → high IDF; "common" has df=90 → low IDF
        term_df = {"rare": 1, "common": 90}
        doc_postings_docids = {"rare": {0, 1}, "common": {0, 1, 2}}

        # Query: "common rare" → best for doc 0 should be "rare"
        best = select_best_token(
            ["common", "rare"], N, doc_postings_docids, term_df, docid=0
        )
        assert best == "rare"

        # Verify VSM IDF is always positive, even for very common terms
        idf_common = _vsm_idf(90, 100)
        idf_rare = _vsm_idf(1, 100)
        assert idf_common > 0, "VSM IDF must always be positive"
        assert idf_rare > idf_common

    def test_best_token_not_bm25_idf_zero(self):
        """VSM IDF is used (never zero), not BM25 IDF (can be zero)."""
        N = 100
        # "stopword" has df=95 → BM25 IDF would be negative/zero
        # But VSM IDF is always positive
        term_df = {"stopword": 95, "content": 5}
        doc_postings_docids = {"stopword": {0}, "content": {0}}

        # BM25 IDF for "stopword": log((100-95+0.5)/(95+0.5)) < 0 → clamped to 0
        bm25_idf = math.log((N - 95 + 0.5) / (95 + 0.5))
        assert bm25_idf < 0, "BM25 IDF should be negative for high-df term"

        # VSM IDF for "stopword": log((101)/(96)) + 1 > 0 → still positive
        vsm_idf = _vsm_idf(95, N)
        assert vsm_idf > 0, "VSM IDF must be positive even for stopwords"

        # Best token should be "content" (higher VSM IDF)
        best = select_best_token(
            ["stopword", "content"],
            N,
            doc_postings_docids,
            term_df,
            docid=0,
        )
        assert best == "content"

    def test_best_token_tie_break_first_in_query_order(self):
        """Tie-break: first token in query order wins."""
        N = 100
        # Both terms have same df → same VSM IDF
        term_df = {"alpha": 10, "beta": 10}
        doc_postings_docids = {"alpha": {0}, "beta": {0}}

        best = select_best_token(
            ["alpha", "beta"], N, doc_postings_docids, term_df, docid=0
        )
        assert best == "alpha"

        # Reverse order → beta wins
        best2 = select_best_token(
            ["beta", "alpha"], N, doc_postings_docids, term_df, docid=0
        )
        assert best2 == "beta"

    def test_best_token_term_not_in_doc(self):
        """Token only selected if present in document's postings."""
        N = 100
        term_df = {"present": 5, "absent": 2}
        doc_postings_docids = {"present": {0}, "absent": {1}}

        best = select_best_token(
            ["absent", "present"], N, doc_postings_docids, term_df, docid=0
        )
        assert best == "present"


class TestSnippetGeneration:
    """Test snippet window extraction and whitespace collapsing."""

    def test_whitespace_collapsed(self):
        """Internal whitespace in snippet is collapsed to single spaces."""
        text = "Hello   world\t\tthis  is\n\na   test  document."
        snippet, _ = generate_snippet(text, "world", window=250)
        assert "   " not in snippet
        assert "\t" not in snippet
        assert "\n" not in snippet
        assert "world" in snippet.lower()

    def test_250_char_window_clamped(self):
        """Snippet does not exceed 250 chars (before whitespace collapse)."""
        text = "a " * 500  # 1000 chars
        snippet, _ = generate_snippet(text, "a", window=250)
        # After whitespace collapse, should be reasonable length
        assert len(snippet) <= 250

    def test_start_line_computation(self):
        """start_line = 1 + newlines before snippet start."""
        text = "line1\nline2\nline3\ntarget word here\nline5"
        snippet, start_line = generate_snippet(text, "target", window=250)
        assert "target" in snippet.lower()
        # "target" appears after 3 newlines → start_line depends on window
        assert start_line >= 1

    def test_snippet_at_document_start(self):
        """Snippet at doc start has start_line = 1."""
        text = "target is at the beginning of this document"
        snippet, start_line = generate_snippet(text, "target", window=250)
        assert start_line == 1
        assert "target" in snippet.lower()

    def test_snippet_near_end(self):
        """Snippet near end clamps correctly."""
        text = "x" * 200 + " target"
        snippet, start_line = generate_snippet(text, "target", window=250)
        assert "target" in snippet.lower()

    def test_snippet_missing_text_returns_empty(self):
        """has_text == 0 gives empty snippet (tested via generate_snippets_batch)."""
        # This is tested via the batch function with has_text=0 metadata
        pass


class TestSnippetBatch:
    """Test batch snippet generation."""

    def test_batch_respects_has_text(self, eval_index: Path):
        """Documents with has_text=0 get empty snippet and start_line."""
        index_dir = str(eval_index / "index")
        text_dir = str(eval_index / "texts")

        metadata_map = {
            "1": {"has_text": 1},
            "2": {"has_text": 0},
        }

        results = [("1", 1.0), ("2", 0.5)]
        query_tokens = preprocess_query("apple")

        snippets = generate_snippets_batch(
            results=results,
            query_tokens=query_tokens,
            N=3,
            index_dir=index_dir,
            metadata_map=metadata_map,
            text_dir=text_dir,
        )

        assert len(snippets) == 2
        # Doc "1" has text → snippet non-empty
        assert snippets[0][0] != ""
        assert snippets[0][1] != ""
        # Doc "2" has_text=0 → empty
        assert snippets[1] == ("", "")


# ---------------------------------------------------------------------------
# TSV writer + validator tests
# ---------------------------------------------------------------------------


class TestTSVWriter:
    """Test TSV file writing."""

    def test_write_tsv_format(self, tmp_path: Path):
        """TSV has correct format: 5 columns, tab-separated, Unix newlines."""
        filepath = tmp_path / "test.tsv"
        results = [("42", 1.5), ("7", 0.8)]
        snippets = [("some snippet text", "10"), ("", "")]

        write_tsv(filepath, results, snippets)

        content = filepath.read_text(encoding="utf-8")
        lines = content.split("\n")
        # Last element after split is empty string from trailing newline
        assert lines[-1] == ""
        data_lines = lines[:-1]
        assert len(data_lines) == 2

        cols1 = data_lines[0].split("\t")
        assert len(cols1) == 5
        assert cols1[0] == "1"
        assert cols1[1] == "42"
        assert cols1[2] == "1.500000"
        assert cols1[3] == "some snippet text"
        assert cols1[4] == "10"

        cols2 = data_lines[1].split("\t")
        assert cols2[0] == "2"
        assert cols2[4] == ""

    def test_write_tsv_max_100_rows(self, tmp_path: Path):
        """TSV writer truncates at 100 rows."""
        filepath = tmp_path / "test.tsv"
        results = [(str(i), float(200 - i)) for i in range(150)]
        snippets = [("", "") for _ in range(150)]

        write_tsv(filepath, results, snippets, max_rows=100)

        content = filepath.read_text(encoding="utf-8")
        data_lines = [l for l in content.split("\n") if l]
        assert len(data_lines) == 100

    def test_write_tsv_unix_newlines(self, tmp_path: Path):
        """TSV uses Unix newlines (no \\r\\n)."""
        filepath = tmp_path / "test.tsv"
        results = [("1", 1.0)]
        snippets = [("text", "1")]

        write_tsv(filepath, results, snippets)

        raw = filepath.read_bytes()
        assert b"\r\n" not in raw
        assert b"\n" in raw


class TestTSVValidator:
    """Test the TSV validation logic."""

    def test_validator_detects_missing_file(self, tmp_path: Path):
        """Validator reports missing files."""
        tsv_dir = tmp_path / "tsv"
        tsv_dir.mkdir()

        # Create only 17 files (missing 1_structured.tsv)
        for qn in range(1, 7):
            for mn in MODEL_NAMES:
                if qn == 1 and mn == "structured":
                    continue
                filepath = tsv_dir / f"{qn}_{mn}.tsv"
                filepath.write_text("1\t42\t1.000000\tsnippet\t1\n")

        all_pass, failures = validate_tsv(str(tsv_dir))
        assert not all_pass
        assert any("1_structured.tsv" in f for f in failures)

    def test_validator_detects_wrong_column_count(self, tmp_path: Path):
        """Validator catches rows with != 5 columns."""
        tsv_dir = tmp_path / "tsv"
        tsv_dir.mkdir()

        for qn in range(1, 7):
            for mn in MODEL_NAMES:
                filepath = tsv_dir / f"{qn}_{mn}.tsv"
                if qn == 1 and mn == "bm25":
                    # Wrong: only 3 columns
                    filepath.write_text("1\t42\t1.000000\n")
                else:
                    filepath.write_text("1\t42\t1.000000\tsnippet\t1\n")

        all_pass, failures = validate_tsv(str(tsv_dir))
        assert not all_pass
        assert any("expected 5 columns" in f for f in failures)

    def test_validator_detects_non_sequential_rank(self, tmp_path: Path):
        """Validator catches non-sequential ranks."""
        tsv_dir = tmp_path / "tsv"
        tsv_dir.mkdir()

        for qn in range(1, 7):
            for mn in MODEL_NAMES:
                filepath = tsv_dir / f"{qn}_{mn}.tsv"
                if qn == 2 and mn == "vsm":
                    # Non-sequential: 1, 3 instead of 1, 2
                    filepath.write_text(
                        "1\t42\t2.000000\tsnippet\t1\n"
                        "3\t7\t1.000000\tsnippet\t2\n"
                    )
                else:
                    filepath.write_text("1\t42\t1.000000\tsnippet\t1\n")

        all_pass, failures = validate_tsv(str(tsv_dir))
        assert not all_pass
        assert any("expected rank 2" in f for f in failures)

    def test_validator_detects_increasing_scores(self, tmp_path: Path):
        """Validator catches scores that increase."""
        tsv_dir = tmp_path / "tsv"
        tsv_dir.mkdir()

        for qn in range(1, 7):
            for mn in MODEL_NAMES:
                filepath = tsv_dir / f"{qn}_{mn}.tsv"
                if qn == 3 and mn == "bm25":
                    # Scores go UP: 1.0 then 2.0
                    filepath.write_text(
                        "1\t42\t1.000000\tsnippet\t1\n"
                        "2\t7\t2.000000\tsnippet\t2\n"
                    )
                else:
                    filepath.write_text("1\t42\t1.000000\tsnippet\t1\n")

        all_pass, failures = validate_tsv(str(tsv_dir))
        assert not all_pass
        assert any("not non-increasing" in f for f in failures)

    def test_validator_passes_correct_files(self, tmp_path: Path):
        """Validator passes well-formed TSV files."""
        tsv_dir = tmp_path / "tsv"
        tsv_dir.mkdir()

        for qn in range(1, 7):
            for mn in MODEL_NAMES:
                filepath = tsv_dir / f"{qn}_{mn}.tsv"
                filepath.write_text(
                    "1\t42\t2.000000\tsnippet a\t1\n"
                    "2\t100\t1.500000\tsnippet b\t3\n"
                    "3\t7\t1.500000\tsnippet c\t5\n"
                )

        all_pass, failures = validate_tsv(str(tsv_dir))
        assert all_pass
        assert failures == []

    def test_validator_detects_tie_break_violation(self, tmp_path: Path):
        """Validator catches tied scores with wrong gutenberg_id order."""
        tsv_dir = tmp_path / "tsv"
        tsv_dir.mkdir()

        for qn in range(1, 7):
            for mn in MODEL_NAMES:
                filepath = tsv_dir / f"{qn}_{mn}.tsv"
                if qn == 1 and mn == "structured":
                    # Tied score, but gutid goes backwards: "B" then "A"
                    filepath.write_text(
                        "1\tB\t1.000000\t\t\n"
                        "2\tA\t1.000000\t\t\n"
                    )
                else:
                    filepath.write_text("1\t42\t1.000000\t\t1\n")

        all_pass, failures = validate_tsv(str(tsv_dir))
        assert not all_pass
        assert any("tie-break" in f for f in failures)


# ---------------------------------------------------------------------------
# Eval-level integration test (tiny index)
# ---------------------------------------------------------------------------


class TestEvalRunner:
    """Integration tests for run_eval_queries on tiny index."""

    def test_run_eval_queries_writes_18_files(self, eval_index: Path):
        """Batch runner creates all 18 TSV files."""
        index_dir = str(eval_index / "index")
        tsv_dir = str(eval_index / "tsv_out")
        text_dir = str(eval_index / "texts")

        run_eval_queries(
            index_dir=index_dir,
            tsv_dir=tsv_dir,
            allow_full_scan=False,
            text_dir=text_dir,
        )

        tsv_path = Path(tsv_dir)
        assert tsv_path.exists()

        expected_files = [
            f"{qn}_{mn}.tsv" for qn in range(1, 7) for mn in MODEL_NAMES
        ]
        for filename in expected_files:
            assert (tsv_path / filename).exists(), f"Missing: {filename}"

    def test_run_eval_queries_tsv_valid(self, eval_index: Path):
        """TSV files pass validation."""
        index_dir = str(eval_index / "index")
        tsv_dir = str(eval_index / "tsv_out2")
        text_dir = str(eval_index / "texts")

        run_eval_queries(
            index_dir=index_dir,
            tsv_dir=tsv_dir,
            allow_full_scan=False,
            text_dir=text_dir,
        )

        all_pass, failures = validate_tsv(tsv_dir)
        assert all_pass, f"Validation failures: {failures}"

    def test_determinism_byte_identical(self, eval_index: Path):
        """Two consecutive runs produce byte-identical TSVs."""
        index_dir = str(eval_index / "index")
        tsv_dir = str(eval_index / "tsv_det")
        text_dir = str(eval_index / "texts")

        # Run 1
        run_eval_queries(
            index_dir=index_dir,
            tsv_dir=tsv_dir,
            allow_full_scan=False,
            text_dir=text_dir,
        )

        tsv_path = Path(tsv_dir)
        expected_files = [
            f"{qn}_{mn}.tsv" for qn in range(1, 7) for mn in MODEL_NAMES
        ]

        # Save run 1 contents
        run1_contents = {}
        for filename in expected_files:
            run1_contents[filename] = (tsv_path / filename).read_bytes()

        # Run 2
        run_eval_queries(
            index_dir=index_dir,
            tsv_dir=tsv_dir,
            allow_full_scan=False,
            text_dir=text_dir,
        )

        # Compare
        for filename in expected_files:
            run2_bytes = (tsv_path / filename).read_bytes()
            assert run1_contents[filename] == run2_bytes, (
                f"Non-deterministic output: {filename}"
            )

    def test_eval_queries_are_exact_strings(self):
        """Verify hardcoded eval queries match the spec."""
        assert EVAL_QUERIES[1] == "to be, or not to be"
        assert EVAL_QUERIES[2] == "English Grammar"
        assert EVAL_QUERIES[3] == "Philip K Dick"
        assert EVAL_QUERIES[4] == "Jabberwocky"
        assert EVAL_QUERIES[5] == "Gutenberg"
        assert EVAL_QUERIES[6] == "Dornröschen"
        assert len(EVAL_QUERIES) == 6
        assert MODEL_NAMES == ["structured", "vsm", "bm25"]
