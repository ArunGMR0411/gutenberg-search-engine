"""Tests for dataset loading, deduplication, and SQLite metadata."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from ir_system.dataset import (
    build_metadata_table,
    deduplicate_metadata,
    load_metadata,
    log_missing_ids,
    validate_dataset,
)

# ---------------------------------------------------------------------------
# Paths — adjust if the repo root differs
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "datasets" / "metadata.csv"
TEXT_DIR = REPO_ROOT / "datasets" / "texts"


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    """Create a minimal CSV with known duplicates for unit testing."""
    import csv as csv_mod

    csv_path = tmp_path / "metadata.csv"
    rows = [
        ["gutenberg_id", "title", "author", "gutenberg_author_id",
         "language", "gutenberg_bookshelf", "rights", "has_text"],
        [1, "Book Alpha", "Smith, John", 100, "en", "Science",
         "Public domain in the USA.", "true"],
        [1, "Book Alpha", "Doe, Jane", 101, "en", "Science",
         "Public domain in the USA.", "true"],
        [2, "Book Beta", "Smith, John", 100, "en", "",
         "Public domain in the USA.", "true"],
        [3, "Book Gamma", "", "", "", "fr", "",
         "Public domain in the USA."],
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.writer(f)
        writer.writerows(rows)
    return csv_path


@pytest.fixture
def tiny_text_dir(tmp_path: Path) -> Path:
    """Create a text dir with files for IDs 1 and 2 but NOT 3."""
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    (text_dir / "1.txt").write_text("hello world")
    (text_dir / "2.txt").write_text("foo bar")
    return text_dir


# ===========================================================================
# Unit tests — tiny synthetic corpus
# ===========================================================================

class TestLoadMetadata:
    def test_columns_renamed(self, tiny_csv: Path):
        df = load_metadata(tiny_csv)
        assert "authors" in df.columns, "CSV 'author' column should be renamed to 'authors'"
        assert "bookshelf" in df.columns, "'gutenberg_bookshelf' should be renamed to 'bookshelf'"
        assert "gutenberg_author_id" not in df.columns

    def test_raw_row_count(self, tiny_csv: Path):
        df = load_metadata(tiny_csv)
        assert len(df) == 4, "Raw CSV should have 4 rows (including duplicates)"


class TestDeduplication:
    def test_dedup_reduces_rows(self, tiny_csv: Path, tiny_text_dir: Path):
        raw = load_metadata(tiny_csv)
        canon = deduplicate_metadata(raw, tiny_text_dir)
        assert len(canon) < len(raw), "Deduplication must reduce row count"
        assert len(canon) == 3, "Should have 3 unique gutenberg_ids"

    def test_authors_merged(self, tiny_csv: Path, tiny_text_dir: Path):
        raw = load_metadata(tiny_csv)
        canon = deduplicate_metadata(raw, tiny_text_dir)
        row1 = canon.loc[canon["gutenberg_id"] == "1"].iloc[0]
        authors = row1["authors"]
        # Both authors should be present, semicolon-separated, sorted
        assert "Doe, Jane" in authors
        assert "Smith, John" in authors
        assert ";" in authors

    def test_has_text_flag(self, tiny_csv: Path, tiny_text_dir: Path):
        raw = load_metadata(tiny_csv)
        canon = deduplicate_metadata(raw, tiny_text_dir)
        # IDs 1 and 2 have text files, ID 3 does not
        assert int(canon.loc[canon["gutenberg_id"] == "1", "has_text"].iloc[0]) == 1
        assert int(canon.loc[canon["gutenberg_id"] == "2", "has_text"].iloc[0]) == 1
        assert int(canon.loc[canon["gutenberg_id"] == "3", "has_text"].iloc[0]) == 0

    def test_has_text_sum(self, tiny_csv: Path, tiny_text_dir: Path):
        raw = load_metadata(tiny_csv)
        canon = deduplicate_metadata(raw, tiny_text_dir)
        # 2 text files exist → has_text sum should be 2
        assert int(canon["has_text"].sum()) == 2

    def test_missing_field_merged_as_empty(self, tiny_csv: Path, tiny_text_dir: Path):
        raw = load_metadata(tiny_csv)
        canon = deduplicate_metadata(raw, tiny_text_dir)
        row3 = canon.loc[canon["gutenberg_id"] == "3"].iloc[0]
        # author was NaN → should be empty string
        assert row3["authors"] == ""


class TestSQLiteMetadata:
    def test_table_exists(self, tiny_csv: Path, tiny_text_dir: Path, tmp_path: Path):
        raw = load_metadata(tiny_csv)
        canon = deduplicate_metadata(raw, tiny_text_dir)
        index_dir = tmp_path / "index_test"
        db_path = build_metadata_table(canon, index_dir)

        with sqlite3.connect(str(db_path)) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'"
            ).fetchall()
            assert len(tables) == 1, "metadata table must exist"

    def test_column_names(self, tiny_csv: Path, tiny_text_dir: Path, tmp_path: Path):
        raw = load_metadata(tiny_csv)
        canon = deduplicate_metadata(raw, tiny_text_dir)
        index_dir = tmp_path / "index_test"
        db_path = build_metadata_table(canon, index_dir)

        expected_cols = {"gutenberg_id", "title", "authors", "language",
                         "bookshelf", "rights", "has_text"}
        with sqlite3.connect(str(db_path)) as conn:
            info = conn.execute("PRAGMA table_info(metadata)").fetchall()
            actual_cols = {row[1] for row in info}
        assert actual_cols == expected_cols

    def test_row_count(self, tiny_csv: Path, tiny_text_dir: Path, tmp_path: Path):
        raw = load_metadata(tiny_csv)
        canon = deduplicate_metadata(raw, tiny_text_dir)
        index_dir = tmp_path / "index_test"
        db_path = build_metadata_table(canon, index_dir)

        with sqlite3.connect(str(db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
        assert count == 3


class TestLogging:
    def test_missing_log_content(self, tiny_csv: Path, tiny_text_dir: Path, tmp_path: Path):
        raw = load_metadata(tiny_csv)
        canon = deduplicate_metadata(raw, tiny_text_dir)
        log_dir = tmp_path / "logs"
        log_path, report_path = log_missing_ids(canon, log_dir)

        lines = log_path.read_text().strip().splitlines()
        assert "3" in lines, "ID 3 (no text file) should be in missing log"
        assert len(lines) == 1, "Only 1 missing ID expected"

    def test_report_json(self, tiny_csv: Path, tiny_text_dir: Path, tmp_path: Path):
        import json
        raw = load_metadata(tiny_csv)
        canon = deduplicate_metadata(raw, tiny_text_dir)
        log_dir = tmp_path / "logs"
        _, report_path = log_missing_ids(canon, log_dir)

        report = json.loads(report_path.read_text())
        assert report["missing_text_count"] == 1
        assert report["has_text_count"] == 2


# ===========================================================================
# Integration tests — real full dataset (skipped if files absent)
# ===========================================================================

_REAL_DATA_AVAILABLE = CSV_PATH.exists() and TEXT_DIR.exists()


@pytest.mark.skipif(not _REAL_DATA_AVAILABLE, reason="Full dataset not present")
class TestRealDataset:
    """Smoke tests against the actual Gutenberg dataset."""

    def test_dedup_reduces_count(self):
        raw = load_metadata(CSV_PATH)
        canon = deduplicate_metadata(raw, TEXT_DIR)
        assert len(canon) < len(raw), "Deduplicated count must be < raw CSV row count"

    def test_has_text_matches_file_count(self):
        raw = load_metadata(CSV_PATH)
        canon = deduplicate_metadata(raw, TEXT_DIR)
        actual_file_count = sum(1 for f in os.listdir(TEXT_DIR) if f.endswith(".txt"))
        assert int(canon["has_text"].sum()) == actual_file_count, (
            f"has_text sum ({int(canon['has_text'].sum())}) must equal "
            f"actual .txt file count ({actual_file_count})"
        )

    def test_sqlite_table_exists_and_columns(self, tmp_path: Path):
        raw = load_metadata(CSV_PATH)
        canon = deduplicate_metadata(raw, TEXT_DIR)
        db_path = build_metadata_table(canon, tmp_path / "index")

        expected_cols = {"gutenberg_id", "title", "authors", "language",
                         "bookshelf", "rights", "has_text"}
        with sqlite3.connect(str(db_path)) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'"
            ).fetchall()
            assert len(tables) == 1
            info = conn.execute("PRAGMA table_info(metadata)").fetchall()
            actual_cols = {row[1] for row in info}
            assert actual_cols == expected_cols
