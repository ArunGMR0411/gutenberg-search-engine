"""Dataset loading, deduplication, and SQLite metadata writer."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CSV column name → SQLite column name mapping
_CSV_COL_MAP = {
    "gutenberg_id": "gutenberg_id",
    "title": "title",
    "author": "authors",              # CSV has singular "author"
    "language": "language",
    "gutenberg_bookshelf": "bookshelf",  # CSV has "gutenberg_bookshelf"
    "rights": "rights",
}

# Fields that are merged across duplicate rows (everything except gutenberg_id)
_MERGE_FIELDS = ["title", "authors", "language", "bookshelf", "rights"]

# SQLite schema for the metadata table
_METADATA_DDL = """\
CREATE TABLE IF NOT EXISTS metadata (
    gutenberg_id TEXT PRIMARY KEY,
    title        TEXT,
    authors      TEXT,
    language     TEXT,
    bookshelf    TEXT,
    rights       TEXT,
    has_text     INTEGER NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Load raw CSV
# ---------------------------------------------------------------------------

def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Read the Gutenberg metadata CSV and rename columns to canonical names."""
    df = pd.read_csv(csv_path, dtype={"gutenberg_id": int})

    # Rename per mapping
    df = df.rename(columns=_CSV_COL_MAP)

    # Keep only the columns we need
    keep = list(_CSV_COL_MAP.values())
    df = df[[c for c in keep if c in df.columns]]

    return df


# ---------------------------------------------------------------------------
# Deduplicate and canonicalize
# ---------------------------------------------------------------------------

def _merge_field_values(series: pd.Series) -> str:
    """Merge unique non-empty values across duplicate rows, joined with ';'."""
    vals: list[str] = []
    for v in series:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            vals.append(s)
    unique = sorted(set(vals))
    return ";".join(unique)


def deduplicate_metadata(df: pd.DataFrame, text_dir: str | Path) -> pd.DataFrame:
    """Merge duplicate gutenberg_id rows and add the has_text flag."""
    text_dir = Path(text_dir)

    # Build set of existing text file IDs for O(1) lookup
    existing_ids: set[str] = set()
    if text_dir.is_dir():
        for fname in os.listdir(text_dir):
            if fname.endswith(".txt"):
                existing_ids.add(fname[:-4])  # strip ".txt"

    # Group by gutenberg_id and merge each field
    grouped = df.groupby("gutenberg_id", sort=True)
    records: list[dict] = []
    for gid, group in grouped:
        row: dict = {"gutenberg_id": str(gid)}
        for field in _MERGE_FIELDS:
            if field in group.columns:
                row[field] = _merge_field_values(group[field])
            else:
                row[field] = ""
        # has_text: 1 if <gid>.txt exists, else 0
        row["has_text"] = 1 if str(gid) in existing_ids else 0
        records.append(row)

    canon = pd.DataFrame(records)
    canon = canon.sort_values("gutenberg_id", key=lambda s: s.astype(int))
    canon = canon.reset_index(drop=True)
    return canon


# ---------------------------------------------------------------------------
# Write to SQLite
# ---------------------------------------------------------------------------

def build_metadata_table(
    canon_df: pd.DataFrame,
    index_dir: str | Path,
) -> Path:
    """Write canonical metadata to index.sqlite, returns the db path."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "index.sqlite"

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("DROP TABLE IF EXISTS metadata;")
        conn.execute(_METADATA_DDL)

        rows = [
            (
                str(r["gutenberg_id"]),
                r.get("title", ""),
                r.get("authors", ""),
                r.get("language", ""),
                r.get("bookshelf", ""),
                r.get("rights", ""),
                int(r["has_text"]),
            )
            for _, r in canon_df.iterrows()
        ]
        conn.executemany(
            "INSERT INTO metadata "
            "(gutenberg_id, title, authors, language, bookshelf, rights, has_text) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()

    return db_path


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_missing_ids(
    canon_df: pd.DataFrame,
    log_dir: str | Path = "outputs/logs",
) -> tuple[Path, Path]:
    """Write missing-text IDs and a JSON summary report."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    missing = canon_df.loc[canon_df["has_text"] == 0, "gutenberg_id"].tolist()

    # One ID per line
    log_path = log_dir / "missing_text_ids.log"
    with open(log_path, "w", encoding="utf-8") as f:
        for gid in missing:
            f.write(f"{gid}\n")

    # JSON summary
    report = {
        "total_metadata_rows_raw": None,  # filled by caller
        "unique_gutenberg_ids": len(canon_df),
        "has_text_count": int(canon_df["has_text"].sum()),
        "missing_text_count": len(missing),
        "missing_text_ids": missing,
    }
    report_path = log_dir / "dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return log_path, report_path


# ---------------------------------------------------------------------------
# Top-level validate_dataset orchestrator
# ---------------------------------------------------------------------------

def validate_dataset(
    csv_path: str | Path = "datasets/metadata.csv",
    text_dir: str | Path = "datasets/texts",
    index_dir: str | Path = "index",
) -> dict:
    """Load CSV, deduplicate, write SQLite metadata, log missing IDs."""
    print(f"[validate_dataset] CSV        : {csv_path}")
    print(f"[validate_dataset] Text dir   : {text_dir}")
    print(f"[validate_dataset] Index dir  : {index_dir}")

    # 1. Load
    raw = load_metadata(csv_path)
    raw_count = len(raw)
    print(f"[validate_dataset] Raw rows   : {raw_count}")

    # 2. Deduplicate
    canon = deduplicate_metadata(raw, text_dir)
    dedup_count = len(canon)
    has_text_count = int(canon["has_text"].sum())
    missing_count = dedup_count - has_text_count
    print(f"[validate_dataset] Unique IDs : {dedup_count}")
    print(f"[validate_dataset] has_text=1 : {has_text_count}")
    print(f"[validate_dataset] has_text=0 : {missing_count}")

    # 3. Write SQLite metadata table
    db_path = build_metadata_table(canon, index_dir)
    print(f"[validate_dataset] SQLite DB  : {db_path}")

    # 4. Log missing
    log_path, report_path = log_missing_ids(canon)
    # Patch raw count into the report
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    report["total_metadata_rows_raw"] = raw_count
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[validate_dataset] Missing log: {log_path}")
    print(f"[validate_dataset] Report     : {report_path}")
    print("[validate_dataset] Done.")

    return {
        "raw_count": raw_count,
        "dedup_count": dedup_count,
        "has_text_count": has_text_count,
        "missing_count": missing_count,
        "db_path": str(db_path),
    }
