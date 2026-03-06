r"""Postings codec and canonical reader API."""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Varint encoding
# ---------------------------------------------------------------------------

def varint_encode(n: int) -> bytes:
    """Encode non-negative int as a 7-bit varint (MSB = continuation bit)."""
    if n < 0:
        raise ValueError(f"varint must be non-negative, got {n}")

    result = bytearray()
    while n >= 128:
        # Emit lowest 7 bits with MSB=1 (continuation)
        result.append((n & 0x7F) | 0x80)
        n >>= 7
    # Emit remaining bits with MSB=0 (end of number)
    result.append(n & 0x7F)
    return bytes(result)


# ---------------------------------------------------------------------------
# Varint decoding
# ---------------------------------------------------------------------------

def varint_decode(buf: bytes, offset: int) -> tuple[int, int]:
    """Decode a varint starting at *offset*. Returns (value, new_offset)."""
    result = 0
    shift = 0
    pos = offset
    max_bytes = 5  # Prevent unbounded reads

    for i in range(max_bytes):
        if pos >= len(buf):
            raise IndexError(
                f"varint_decode: buffer exhausted at offset {offset}, "
                f"byte {i}"
            )
        byte = buf[pos]
        pos += 1

        # Extract 7 bits of data
        result |= (byte & 0x7F) << shift

        # Check if this is the last byte (MSB=0)
        if (byte & 0x80) == 0:
            return result, pos

        shift += 7

    raise ValueError(
        f"varint_decode: varint at offset {offset} exceeds {max_bytes} bytes"
    )


# ---------------------------------------------------------------------------
# Gap encoding
# ---------------------------------------------------------------------------

def gap_encode(docids: list[int]) -> list[int]:
    """Delta-encode a sorted docid list. First element kept as-is."""
    if not docids:
        return []

    gaps = [docids[0]]
    for i in range(1, len(docids)):
        gaps.append(docids[i] - docids[i - 1])
    return gaps


# ---------------------------------------------------------------------------
# Gap decoding
# ---------------------------------------------------------------------------

def gap_decode(gaps: list[int]) -> list[int]:
    """Recover docids from delta-encoded gaps via cumulative sum."""
    if not gaps:
        return []

    docids = [gaps[0]]
    for i in range(1, len(gaps)):
        docids.append(docids[i - 1] + gaps[i])
    return docids


# ---------------------------------------------------------------------------
# PostingsReader
# ---------------------------------------------------------------------------

class PostingsReader:
    r"""Reads encoded postings from postings.bin.

    Usage:
        with PostingsReader("index/postings.bin") as reader:
            postings = reader.decode_postings(offset=1024, length=512)
    """

    def __init__(self, postings_path: str):
        self.postings_path = postings_path
        self._file_handle: Optional[object] = None

    def _ensure_open(self) -> object:
        """Lazily open file handle if not already open."""
        if self._file_handle is None:
            self._file_handle = open(self.postings_path, "rb")
        return self._file_handle

    def decode_postings(self, offset: int, length: int) -> list[tuple[int, int]]:
        """Decode postings at (offset, length). Returns [(docid, tf), ...]."""
        f = self._ensure_open()
        f.seek(offset)
        buf = f.read(length)

        if len(buf) < length:
            raise IOError(
                f"decode_postings: expected {length} bytes at offset {offset}, "
                f"got {len(buf)} (file might be truncated)"
            )

        df, pos = varint_decode(buf, 0)

        docid_gaps = []
        tfs = []
        for _ in range(df):
            docid_gap, pos = varint_decode(buf, pos)
            tf, pos = varint_decode(buf, pos)
            docid_gaps.append(docid_gap)
            tfs.append(tf)

        docids = gap_decode(docid_gaps)

        postings = list(zip(docids, tfs))
        return postings

    def decode_postings_cached(
        self, offset: int, length: int, cache: dict
    ) -> list[tuple[int, int]]:
        """Like decode_postings() but with a session-scoped cache dict."""
        key = (offset, length)
        if key not in cache:
            cache[key] = self.decode_postings(offset, length)
        return cache[key]

    def close(self) -> None:
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
