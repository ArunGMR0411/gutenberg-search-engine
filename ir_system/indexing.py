"""SPIMI block-based indexing, merging, and doc_norm computation."""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import sqlite3
import tracemalloc
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from ir_system.postings import PostingsReader, gap_encode, varint_encode
from ir_system.preprocessing import preprocess, preprocess_query, strip_boilerplate


# ---------------------------------------------------------------------------
# SPIMI Block Builder
# ---------------------------------------------------------------------------

class SPIMIBlockBuilder:
    """In-memory SPIMI block builder that flushes to disk when thresholds are hit."""

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.blocks_dir = self.index_dir / "blocks"
        self.blocks_dir.mkdir(parents=True, exist_ok=True)

        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        self.doc_len: dict[int, int] = defaultdict(int)
        self.block_count = 0
        self.total_text_bytes = 0
        self.peak_tracemalloc_mb = 0.0

        tracemalloc.start()

    def index_document(
        self,
        docid: int,
        text: str,
        raw_file_size: int,
    ) -> None:
        """Index a single document, flushing if threshold reached."""
        self.total_text_bytes += raw_file_size

        text_stripped = strip_boilerplate(text)
        base_tokens = preprocess(text_stripped)

        tf_counter = Counter(base_tokens)

        # doc_len counts base tokens only, not aliases
        self.doc_len[docid] = len(base_tokens)

        for token, tf in tf_counter.items():
            if tf <= 0:
                continue

            if token not in self.inverted_index:
                self.inverted_index[token] = []

            self.inverted_index[token].append((docid, tf))

            # accent-fold aliases get same TF but don't increment doc_len
            alias = self._compute_accent_fold_alias(token)
            if alias != token:
                if alias not in self.inverted_index:
                    self.inverted_index[alias] = []
                self.inverted_index[alias].append((docid, tf))

        self._check_and_flush_if_needed()

    @staticmethod
    def _compute_accent_fold_alias(token: str) -> str:
        import unicodedata

        nfd = unicodedata.normalize("NFD", token)
        alias = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
        return alias

    def _check_and_flush_if_needed(self) -> None:
        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / 1_000_000
        peak_mb = peak / 1_000_000
        if peak_mb > self.peak_tracemalloc_mb:
            self.peak_tracemalloc_mb = peak_mb

        unique_terms = len(self.inverted_index)

        if current_mb >= 350 or unique_terms >= 1_000_000:
            self.flush_block()

    def flush_block(self) -> None:
        """Flush current in-memory block to disk."""
        if not self.inverted_index:
            return

        block_num = self.block_count
        block_num_str = str(block_num).zfill(6)

        sorted_terms = sorted(self.inverted_index.keys())

        postings_path = self.blocks_dir / f"block_{block_num_str}.postings.bin"
        lexicon_data: dict[str, list] = {}

        with open(postings_path, "wb") as f:
            for term in sorted_terms:
                postings = self.inverted_index[term]
                postings_sorted = sorted(postings, key=lambda x: x[0])

                docids = [docid for docid, _ in postings_sorted]
                tfs = [tf for _, tf in postings_sorted]

                gaps = gap_encode(docids)

                byte_offset = f.tell()
                df = len(postings_sorted)

                f.write(varint_encode(df))

                for gap, tf in zip(gaps, tfs):
                    f.write(varint_encode(gap))
                    f.write(varint_encode(tf))

                byte_length = f.tell() - byte_offset

                lexicon_data[term] = [df, byte_offset, byte_length]

        lexicon_path = self.blocks_dir / f"block_{block_num_str}.lexicon.json"
        with open(lexicon_path, "w", encoding="utf-8") as f:
            json.dump(lexicon_data, f)

        self.inverted_index.clear()

        gc.collect()
        tracemalloc.clear_traces()

        self.block_count += 1
        print(f"[SPIMI] Flushed block {block_num} ({len(lexicon_data)} terms)")

    def finalize(self) -> dict:
        """Flush any remaining in-memory data and return statistics."""
        self.flush_block()
        _, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / 1_000_000
        if peak_mb > self.peak_tracemalloc_mb:
            self.peak_tracemalloc_mb = peak_mb
        tracemalloc.stop()
        return {
            "doc_len": dict(self.doc_len),
            "block_count": self.block_count,
            "total_text_bytes": self.total_text_bytes,
            "peak_tracemalloc_mb": self.peak_tracemalloc_mb,
        }


# ---------------------------------------------------------------------------
# Block Merging
# ---------------------------------------------------------------------------

def merge_blocks(
    index_dir: str,
    docid_to_gutid: dict[int, str],
    doc_len: dict[int, int],
    k1: float = 1.2,
    b: float = 0.75,
) -> dict:
    """Merge all sorted block lexicons into final postings.bin and SQLite tables.

    Returns dict with: {N, avgdl, lexicon_entries}.
    """
    index_dir = Path(index_dir)
    blocks_dir = index_dir / "blocks"

    lexicon_files = sorted(blocks_dir.glob("block_*.lexicon.json"))

    postings_files: dict[int, Path] = {}
    for f in blocks_dir.glob("block_*.postings.bin"):
        parts = f.name.replace("block_", "").replace(".postings.bin", "")
        block_idx = int(parts)
        postings_files[block_idx] = f

    if not lexicon_files:
        raise RuntimeError(f"No block lexicon files found in {blocks_dir}")

    block_lexicons: dict[int, dict] = {}
    for lex_file in lexicon_files:
        parts = lex_file.name.replace("block_", "").replace(".lexicon.json", "")
        block_idx = int(parts)
        with open(lex_file, "r", encoding="utf-8") as f:
            block_lexicons[block_idx] = json.load(f)

    all_terms = set()
    for lex in block_lexicons.values():
        all_terms.update(lex.keys())
    sorted_terms = sorted(all_terms)

    merged_postings_path = index_dir / "postings.bin"
    lexicon_entries: dict[str, list] = {}

    # import once
    from ir_system.postings import varint_decode, gap_decode

    # keep block files open for efficiency
    block_file_handles: dict[int, object] = {}
    for block_idx, pf in postings_files.items():
        block_file_handles[block_idx] = open(pf, "rb")

    try:
      with open(merged_postings_path, "wb") as f_out:
        for term in sorted_terms:
            all_postings: list[tuple[int, int]] = []

            for block_idx, lex in block_lexicons.items():
                if term not in lex:
                    continue

                df_block, offset_block, length_block = lex[term]
                f_block = block_file_handles[block_idx]

                f_block.seek(offset_block)
                buf = f_block.read(length_block)

                df_decoded, pos = varint_decode(buf, 0)
                gaps = []
                tfs = []
                for _ in range(df_decoded):
                    gap, pos = varint_decode(buf, pos)
                    tf, pos = varint_decode(buf, pos)
                    gaps.append(gap)
                    tfs.append(tf)

                docids = gap_decode(gaps)
                all_postings.extend(zip(docids, tfs))

            all_postings.sort(key=lambda x: x[0])

            docids_merged = [docid for docid, _ in all_postings]
            tfs_merged = [tf for _, tf in all_postings]
            gaps_merged = gap_encode(docids_merged)

            byte_offset = f_out.tell()
            f_out.write(varint_encode(len(all_postings)))
            for gap, tf in zip(gaps_merged, tfs_merged):
                f_out.write(varint_encode(gap))
                f_out.write(varint_encode(tf))
            byte_length = f_out.tell() - byte_offset

            lexicon_entries[term] = [len(all_postings), byte_offset, byte_length]

    finally:
        for fh in block_file_handles.values():
            fh.close()

    # free lexicons before doc_norm pass
    vocab_count = len(sorted_terms)
    del block_lexicons
    del all_terms
    del sorted_terms
    gc.collect()

    N = len(doc_len)
    total_tokens = sum(doc_len.values())
    avgdl = total_tokens / N if N > 0 else 0.0

    print(f"[Merge] Merged {vocab_count} terms into postings.bin")
    print(f"[Merge] N={N}, avgdl={avgdl:.2f}")

    return {
        "N": N,
        "avgdl": avgdl,
        "lexicon_entries": lexicon_entries,
    }


# ---------------------------------------------------------------------------
# Write SQLite Metadata
# ---------------------------------------------------------------------------

def write_sqlite_metadata(
    index_dir: str,
    docid_to_gutid: dict[int, str],
    doc_len: dict[int, int],
    lexicon_entries: dict[str, list],
    N: int,
    avgdl: float,
    total_text_bytes: int,
    k1: float = 1.2,
    b: float = 0.75,
) -> None:
    """Write lexicon, docmap, docstats (doc_norm empty), and globals to SQLite."""
    index_dir = Path(index_dir)
    db_path = index_dir / "index.sqlite"

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("DROP TABLE IF EXISTS lexicon;")
        conn.execute("DROP TABLE IF EXISTS docmap;")
        conn.execute("DROP TABLE IF EXISTS docstats;")
        conn.execute("DROP TABLE IF EXISTS globals;")

        conn.execute(
            """
            CREATE TABLE lexicon (
                term    TEXT    PRIMARY KEY,
                df      INTEGER NOT NULL,
                offset  INTEGER NOT NULL,
                length  INTEGER NOT NULL
            );
            """
        )
        for term, (df, offset, length) in lexicon_entries.items():
            conn.execute(
                "INSERT INTO lexicon (term, df, offset, length) VALUES (?,?,?,?)",
                (term, df, offset, length),
            )

        conn.execute(
            """
            CREATE TABLE docmap (
                docid        INTEGER PRIMARY KEY,
                gutenberg_id TEXT UNIQUE NOT NULL
            );
            """
        )
        for docid, gutid in docid_to_gutid.items():
            conn.execute(
                "INSERT INTO docmap (docid, gutenberg_id) VALUES (?,?)",
                (docid, gutid),
            )

        # doc_norm is filled in later by compute_doc_norm
        conn.execute(
            """
            CREATE TABLE docstats (
                docid    INTEGER PRIMARY KEY,
                doc_len  INTEGER NOT NULL,
                doc_norm REAL    NOT NULL DEFAULT 0.0
            );
            """
        )
        for docid, dl in doc_len.items():
            conn.execute(
                "INSERT INTO docstats (docid, doc_len, doc_norm) VALUES (?,?,?)",
                (docid, dl, 0.0),
            )

        conn.execute(
            """
            CREATE TABLE globals (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        globals_data = {
            "N": str(N),
            "avgdl": str(avgdl),
            "build_date": datetime.now().isoformat(),
            "k1": str(k1),
            "b": str(b),
            "total_text_bytes": str(total_text_bytes),
            "index_ready": "0",  # Will be set to "1" after doc_norm pass
        }
        for key, value in globals_data.items():
            conn.execute(
                "INSERT INTO globals (key, value) VALUES (?,?)",
                (key, value),
            )

        conn.commit()

    print(f"[SQLite] Wrote {len(lexicon_entries)} lexicon entries")


# ---------------------------------------------------------------------------
# Compute doc_norm
# ---------------------------------------------------------------------------

def compute_doc_norm(index_dir: str) -> None:
    """Compute and store L2 doc_norm for every document."""
    index_dir = Path(index_dir)
    db_path = index_dir / "index.sqlite"
    postings_path = index_dir / "postings.bin"

    if not postings_path.exists():
        raise FileNotFoundError(f"postings.bin not found at {postings_path}")

    with sqlite3.connect(str(db_path)) as conn:
        N = int(
            conn.execute(
                "SELECT value FROM globals WHERE key='N'"
            ).fetchone()[0]
        )

    norm_acc: dict[int, float] = defaultdict(float)

    with PostingsReader(str(postings_path)) as reader:
        # sequential IO order
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT term, df, offset, length FROM lexicon ORDER BY offset ASC"
            ).fetchall()

            for term, df, offset, length in rows:
                idf = math.log((N + 1) / (df + 1)) + 1

                postings = reader.decode_postings(offset, length)

                for docid, tf in postings:
                    if tf <= 0:
                        continue
                    w = (1 + math.log(tf)) * idf
                    norm_acc[docid] += w * w

    with sqlite3.connect(str(db_path)) as conn:
        for docid, acc in norm_acc.items():
            doc_norm = math.sqrt(acc)
            conn.execute(
                "UPDATE docstats SET doc_norm = ? WHERE docid = ?",
                (doc_norm, docid),
            )

        conn.execute(
            "UPDATE globals SET value='1' WHERE key='index_ready'"
        )
        conn.commit()

    print(f"[doc_norm] Computed norms for {len(norm_acc)} documents")


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def build_index(
    csv_path: str = "datasets/metadata.csv",
    text_dir: str = "datasets/texts",
    index_dir: str = "index",
    k1: float = 1.2,
    b: float = 0.75,
) -> dict:
    """End-to-end indexing: SPIMI blocks → merge → doc_norm.

    Returns dict with build statistics.
    """
    import time

    text_dir = Path(text_dir)
    start_time = time.time()

    print(f"[build_index] CSV       : {csv_path}")
    print(f"[build_index] Text dir  : {text_dir}")
    print(f"[build_index] Index dir : {index_dir}")
    print(f"[build_index] k1={k1}, b={b}")

    # Build docid -> gutenberg_id mapping (sorted by gutenberg_id for determinism)
    text_files = sorted([f[:-4] for f in os.listdir(text_dir) if f.endswith(".txt")])
    docid_to_gutid = {i: gid for i, gid in enumerate(text_files)}

    print(f"[build_index] Indexed documents: {len(docid_to_gutid)}")

    builder = SPIMIBlockBuilder(index_dir)

    for docid, gutid in docid_to_gutid.items():
        text_path = text_dir / f"{gutid}.txt"
        raw_size = text_path.stat().st_size

        with open(text_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        builder.index_document(docid, text, raw_size)

        if (docid + 1) % 10000 == 0:
            print(f"[build_index] Indexed {docid + 1} documents...")

    spimi_stats = builder.finalize()
    doc_len = spimi_stats["doc_len"]
    block_count = spimi_stats["block_count"]
    total_text_bytes = spimi_stats["total_text_bytes"]
    peak_tracemalloc_mb = spimi_stats.get("peak_tracemalloc_mb", 0.0)

    print(f"[build_index] SPIMI complete: {block_count} blocks")

    merge_stats = merge_blocks(index_dir, docid_to_gutid, doc_len, k1, b)
    N = merge_stats["N"]
    avgdl = merge_stats["avgdl"]
    lexicon_entries = merge_stats["lexicon_entries"]

    write_sqlite_metadata(
        index_dir,
        docid_to_gutid,
        doc_len,
        lexicon_entries,
        N,
        avgdl,
        total_text_bytes,
        k1,
        b,
    )

    from ir_system.dataset import load_metadata, deduplicate_metadata, build_metadata_table
    raw_df = load_metadata(csv_path)
    canon_df = deduplicate_metadata(raw_df, str(text_dir))
    build_metadata_table(canon_df, index_dir)
    print(f"[build_index] Metadata table written ({len(canon_df)} rows)")

    compute_doc_norm(index_dir)

    blocks_dir = Path(index_dir) / "blocks"
    if blocks_dir.exists():
        shutil.rmtree(blocks_dir)
        print(f"[build_index] Cleaned up {blocks_dir}")

    elapsed = time.time() - start_time
    docs_per_sec = len(docid_to_gutid) / elapsed if elapsed > 0 else 0
    mb_per_sec = total_text_bytes / elapsed / 1_000_000 if elapsed > 0 else 0

    print(f"[build_index] Build time: {elapsed:.2f}s")
    print(f"[build_index] {docs_per_sec:.1f} docs/sec, {mb_per_sec:.1f} MB/sec")

    return {
        "elapsed_seconds": elapsed,
        "N": N,
        "avgdl": avgdl,
        "vocab_size": len(lexicon_entries),
        "block_count": block_count,
        "total_text_bytes": total_text_bytes,
        "docs_per_second": docs_per_sec,
        "mb_per_second": mb_per_sec,
        "peak_tracemalloc_mb": peak_tracemalloc_mb,
    }


# ---------------------------------------------------------------------------
# Index validation
# ---------------------------------------------------------------------------

def validate_index(index_dir: str = "index") -> dict:
    """Validate that the index is complete and consistent."""
    index_dir = Path(index_dir)
    db_path = index_dir / "index.sqlite"
    postings_path = index_dir / "postings.bin"

    print(f"[validate_index] Checking {index_dir}...")

    if not db_path.exists():
        raise FileNotFoundError(f"Missing {db_path}")
    if not postings_path.exists():
        raise FileNotFoundError(f"Missing {postings_path}")

    postings_size = postings_path.stat().st_size

    with sqlite3.connect(str(db_path)) as conn:
        # Check index_ready
        index_ready = conn.execute(
            "SELECT value FROM globals WHERE key='index_ready'"
        ).fetchone()
        if not index_ready or index_ready[0] != "1":
            raise ValueError("globals.index_ready is not '1'")

        # Check all lexicon entries fit within postings.bin
        lex_rows = conn.execute(
            "SELECT term, offset, length FROM lexicon"
        ).fetchall()

        for term, offset, length in lex_rows:
            if offset + length > postings_size:
                raise ValueError(
                    f"Term '{term}': offset={offset}, length={length} "
                    f"exceeds file size {postings_size}"
                )

        # Sample decode: first 50 terms
        sample_rows = conn.execute(
            "SELECT term, df, offset, length FROM lexicon "
            "ORDER BY offset ASC LIMIT 50"
        ).fetchall()

    with PostingsReader(str(postings_path)) as reader:
        for term, df, offset, length in sample_rows:
            postings = reader.decode_postings(offset, length)

            # Verify: docids strictly increasing
            prev_docid = -1
            for docid, tf in postings:
                if docid <= prev_docid:
                    raise ValueError(
                        f"Term '{term}': docids not strictly increasing "
                        f"({prev_docid} >= {docid})"
                    )
                if tf <= 0:
                    raise ValueError(
                        f"Term '{term}', docid {docid}: tf={tf} (should be > 0)"
                    )
                prev_docid = docid

            if len(postings) != df:
                raise ValueError(
                    f"Term '{term}': decoded {len(postings)} postings, "
                    f"expected df={df}"
                )

    print(f"[validate_index] ✓ Index is valid and ready")
    return {"index_dir": str(index_dir), "vocab_size": len(lex_rows)}
