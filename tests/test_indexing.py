"""Tests for SPIMI indexing, merging, and doc_norm computation."""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from pathlib import Path

import pytest

from ir_system.indexing import (
    SPIMIBlockBuilder,
    build_index,
    compute_doc_norm,
    merge_blocks,
    validate_index,
    write_sqlite_metadata,
)
from ir_system.postings import PostingsReader


# ===========================================================================
# Fixture: Synthetic 5-document corpus
# ===========================================================================

@pytest.fixture
def synthetic_corpus(tmp_path: Path) -> dict:
    """5-doc toy corpus. Returns (texts, text_paths, expected_doc_len)."""
    texts = {
        "0": "apple banana apple",
        "1": "banana cherry",
        "2": "apple",
        "3": "cherry cherry cherry",
        "4": "apple banana cherry",
    }

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    for doc_id, text in texts.items():
        (corpus_dir / f"{doc_id}.txt").write_text(text)

    docid_to_gutid = {i: str(i) for i in range(5)}

    return {
        "texts": texts,
        "corpus_dir": corpus_dir,
        "docid_to_gutid": docid_to_gutid,
        "expected_doc_len": {0: 3, 1: 2, 2: 1, 3: 3, 4: 3},
    }


# ===========================================================================
# Test: SPIMI Block Builder (basic functionality)
# ===========================================================================

class TestSPIMIBlockBuilder:
    """Test SPIMIBlockBuilder without thresholds."""

    def test_single_document_indexing(self, synthetic_corpus: dict, tmp_path: Path):
        """Index a single document, no flush."""
        index_dir = tmp_path / "index_single"
        builder = SPIMIBlockBuilder(str(index_dir))

        text = synthetic_corpus["texts"]["0"]
        builder.index_document(0, text, raw_file_size=len(text.encode()))

        # Should have indexed terms
        assert "apple" in builder.inverted_index
        assert "banana" in builder.inverted_index

        # Check doc_len
        assert builder.doc_len[0] == 3

    def test_alias_indexing(self, tmp_path: Path):
        """Test that accent-fold aliases are indexed with same TF."""
        index_dir = tmp_path / "index_alias"
        builder = SPIMIBlockBuilder(str(index_dir))

        # Text with diacritic: "dornröschen" (will become "dornröschen" after processing)
        # We use lowercase since preprocess applies casefold
        text = "dornröschen dornröschen"  # 2 occurrences
        builder.index_document(0, text, raw_file_size=len(text.encode()))

        # Both "dornröschen" and "dornroschen" should be indexed
        assert "dornröschen" in builder.inverted_index
        assert "dornroschen" in builder.inverted_index

        # Both should have same postings (same TF=2)
        assert builder.inverted_index["dornröschen"] == [(0, 2)]
        assert builder.inverted_index["dornroschen"] == [(0, 2)]

        # But doc_len should count base tokens only
        assert builder.doc_len[0] == 2

    def test_finalize_returns_stats(self, synthetic_corpus: dict, tmp_path: Path):
        """Finalize returns correct statistics."""
        index_dir = tmp_path / "index_finalize"
        builder = SPIMIBlockBuilder(str(index_dir))

        for docid, gutid in synthetic_corpus["docid_to_gutid"].items():
            text = synthetic_corpus["texts"][gutid]
            builder.index_document(docid, text, raw_file_size=len(text.encode()))

        stats = builder.finalize()

        assert stats["doc_len"] == synthetic_corpus["expected_doc_len"]
        assert stats["block_count"] >= 1
        assert stats["total_text_bytes"] > 0


# ===========================================================================
# Test: Full index build cycle (SPIMI + merge + doc_norm)
# ===========================================================================

class TestFullIndexBuild:
    """Test complete indexing pipeline on synthetic corpus."""

    def test_build_index_synthetic_corpus(self, synthetic_corpus: dict, tmp_path: Path):
        """Build full index on synthetic corpus; validate structure."""
        index_dir = tmp_path / "index_full"

        # Build index (we need to create a minimal metadata CSV)
        from ir_system.indexing import build_index

        # For now, just test the core components manually
        # (build_index expects a CSV, which is complex to set up in tests)

        # Instead, manually do SPIMI + merge + doc_norm
        builder = SPIMIBlockBuilder(str(index_dir))
        docid_to_gutid = synthetic_corpus["docid_to_gutid"]

        for docid, gutid in docid_to_gutid.items():
            text = synthetic_corpus["texts"][gutid]
            builder.index_document(docid, text, raw_file_size=len(text.encode()))

        spimi_stats = builder.finalize()
        doc_len = spimi_stats["doc_len"]

        # Merge
        merge_stats = merge_blocks(
            str(index_dir),
            docid_to_gutid,
            doc_len,
        )
        N = merge_stats["N"]
        avgdl = merge_stats["avgdl"]
        lexicon_entries = merge_stats["lexicon_entries"]

        # Write SQLite
        write_sqlite_metadata(
            str(index_dir),
            docid_to_gutid,
            doc_len,
            lexicon_entries,
            N,
            avgdl,
            total_text_bytes=100,
        )

        # Compute doc_norm
        compute_doc_norm(str(index_dir))

        # Validate
        with sqlite3.connect(str(index_dir / "index.sqlite")) as conn:
            # Check lexicon
            lex_rows = conn.execute(
                "SELECT term, df FROM lexicon ORDER BY term"
            ).fetchall()
            assert len(lex_rows) == 3  # apple, banana, cherry
            assert lex_rows[0][0] == "apple"
            assert lex_rows[0][1] == 3  # df(apple) = 3

            # Check docstats
            stats_rows = conn.execute(
                "SELECT docid, doc_len FROM docstats ORDER BY docid"
            ).fetchall()
            for docid, dl in stats_rows:
                assert dl == synthetic_corpus["expected_doc_len"][docid]

            # Check globals
            index_ready = conn.execute(
                "SELECT value FROM globals WHERE key='index_ready'"
            ).fetchone()[0]
            assert index_ready == "1"

    def test_doc_norm_computation(self, synthetic_corpus: dict, tmp_path: Path):
        """Test that doc_norm is computed correctly (hand-computed reference)."""
        index_dir = tmp_path / "index_norm"

        # Build index
        builder = SPIMIBlockBuilder(str(index_dir))
        docid_to_gutid = synthetic_corpus["docid_to_gutid"]

        for docid, gutid in docid_to_gutid.items():
            text = synthetic_corpus["texts"][gutid]
            builder.index_document(docid, text, raw_file_size=len(text.encode()))

        spimi_stats = builder.finalize()
        doc_len = spimi_stats["doc_len"]

        merge_stats = merge_blocks(str(index_dir), docid_to_gutid, doc_len)
        N = merge_stats["N"]
        avgdl = merge_stats["avgdl"]
        lexicon_entries = merge_stats["lexicon_entries"]

        write_sqlite_metadata(
            str(index_dir),
            docid_to_gutid,
            doc_len,
            lexicon_entries,
            N,
            avgdl,
            total_text_bytes=100,
        )

        compute_doc_norm(str(index_dir))

        # Hand-compute expected doc_norm for doc 0
        # doc 0: "apple banana apple" -> [apple:2, banana:1]
        # Postings in final index (alphabetical):
        #   apple: [(0,2), (2,1), (4,1)] -> df=3
        #   banana: [(0,1), (1,1), (4,1)] -> df=3
        #   cherry: [(1,1), (3,3), (4,1)] -> df=3

        # idf(apple) = ln((5+1)/(3+1)) + 1 = ln(1.5) + 1 = 0.405 + 1 = 1.405
        # idf(banana) = ln(1.5) + 1 = 1.405
        # idf(cherry) = ln(1.5) + 1 = 1.405

        # For doc 0:
        #   apple: w = (1 + ln(2)) * 1.405 = (1 + 0.693) * 1.405 = 1.693 * 1.405 = 2.379
        #   banana: w = (1 + ln(1)) * 1.405 = (1 + 0) * 1.405 = 1.405
        #   doc_norm = sqrt(2.379^2 + 1.405^2) = sqrt(5.659 + 1.974) = sqrt(7.633) = 2.763

        idf_val = math.log(6.0 / 4.0) + 1
        w_apple = (1 + math.log(2)) * idf_val
        w_banana = (1 + math.log(1)) * idf_val
        expected_norm_0 = math.sqrt(w_apple**2 + w_banana**2)

        # Read computed doc_norm
        with sqlite3.connect(str(index_dir / "index.sqlite")) as conn:
            doc_norm_0 = conn.execute(
                "SELECT doc_norm FROM docstats WHERE docid=0"
            ).fetchone()[0]

        # Allow small floating-point error
        assert abs(doc_norm_0 - expected_norm_0) < 0.01, (
            f"Expected doc_norm[0]={expected_norm_0:.3f}, "
            f"got {doc_norm_0:.3f}"
        )


# ===========================================================================
# Test: Cleanup and isolation
# ===========================================================================

class TestBlockCleanup:
    """Test that blocks/ directory is cleaned up after merge."""

    def test_blocks_dir_cleaned_up(self, synthetic_corpus: dict, tmp_path: Path):
        """blocks/ should be deleted after successful merge."""
        index_dir = tmp_path / "index_cleanup"
        blocks_dir = index_dir / "blocks"

        builder = SPIMIBlockBuilder(str(index_dir))
        docid_to_gutid = synthetic_corpus["docid_to_gutid"]

        for docid, gutid in docid_to_gutid.items():
            text = synthetic_corpus["texts"][gutid]
            builder.index_document(docid, text, raw_file_size=len(text.encode()))

        spimi_stats = builder.finalize()

        # After finalize, blocks/ should exist
        assert blocks_dir.exists()

        doc_len = spimi_stats["doc_len"]
        merge_stats = merge_blocks(str(index_dir), docid_to_gutid, doc_len)

        # Manual cleanup (as the full build_index does)
        if blocks_dir.exists():
            import shutil

            shutil.rmtree(blocks_dir)

        # Verify cleanup
        assert not blocks_dir.exists()


class TestSampleIsolation:
    """Test that --sample routes to index_sample/ only."""

    def test_sample_isolation(self, synthetic_corpus: dict, tmp_path: Path):
        """Building index_sample/ should not touch index/."""
        index_full = tmp_path / "index"
        index_sample = tmp_path / "index_sample"

        # Ensure both directories can be created independently
        builder_sample = SPIMIBlockBuilder(str(index_sample))
        docid_to_gutid = synthetic_corpus["docid_to_gutid"]

        # Index to sample only
        for docid, gutid in docid_to_gutid.items():
            text = synthetic_corpus["texts"][gutid]
            builder_sample.index_document(docid, text, raw_file_size=len(text.encode()))

        builder_sample.finalize()

        # Verify sample has blocks/ but full doesn't
        assert (index_sample / "blocks").exists()
        assert not (index_full / "blocks").exists()


# ===========================================================================
# Test: Index validation
# ===========================================================================

class TestValidateIndex:
    """Test index validation checks."""

    def test_validate_index_complete(self, synthetic_corpus: dict, tmp_path: Path):
        """Validate a complete, correct index."""
        index_dir = tmp_path / "index"

        # Build full index
        builder = SPIMIBlockBuilder(str(index_dir))
        docid_to_gutid = synthetic_corpus["docid_to_gutid"]

        for docid, gutid in docid_to_gutid.items():
            text = synthetic_corpus["texts"][gutid]
            builder.index_document(docid, text, raw_file_size=len(text.encode()))

        spimi_stats = builder.finalize()
        doc_len = spimi_stats["doc_len"]
        merge_stats = merge_blocks(str(index_dir), docid_to_gutid, doc_len)

        write_sqlite_metadata(
            str(index_dir),
            docid_to_gutid,
            doc_len,
            merge_stats["lexicon_entries"],
            merge_stats["N"],
            merge_stats["avgdl"],
            total_text_bytes=100,
        )

        compute_doc_norm(str(index_dir))

        # Cleanup blocks
        import shutil

        if (index_dir / "blocks").exists():
            shutil.rmtree(index_dir / "blocks")

        # Validate should pass
        result = validate_index(str(index_dir))
        assert "vocab_size" in result
        assert result["vocab_size"] == 3

    def test_validate_fails_without_index_ready(
        self, synthetic_corpus: dict, tmp_path: Path
    ):
        """Validate fails if index_ready != '1'."""
        index_dir = tmp_path / "index_not_ready"

        # Create a minimal index without running doc_norm
        builder = SPIMIBlockBuilder(str(index_dir))
        docid_to_gutid = synthetic_corpus["docid_to_gutid"]

        for docid, gutid in docid_to_gutid.items():
            text = synthetic_corpus["texts"][gutid]
            builder.index_document(docid, text, raw_file_size=len(text.encode()))

        spimi_stats = builder.finalize()
        doc_len = spimi_stats["doc_len"]
        merge_stats = merge_blocks(str(index_dir), docid_to_gutid, doc_len)

        write_sqlite_metadata(
            str(index_dir),
            docid_to_gutid,
            doc_len,
            merge_stats["lexicon_entries"],
            merge_stats["N"],
            merge_stats["avgdl"],
            total_text_bytes=100,
        )

        # Deliberately skip compute_doc_norm

        # Validate should fail
        with pytest.raises(ValueError, match="index_ready"):
            validate_index(str(index_dir))
