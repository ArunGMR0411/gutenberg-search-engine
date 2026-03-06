"""Tests for postings codec (varint, gap encoding, PostingsReader)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ir_system.postings import (
    PostingsReader,
    gap_decode,
    gap_encode,
    varint_decode,
    varint_encode,
)


# ===========================================================================
# Varint encoding/decoding tests
# ===========================================================================

class TestVarintEncode:
    """Test varint_encode() with various integer magnitudes."""

    def test_zero(self):
        """Encode 0."""
        result = varint_encode(0)
        assert result == b"\x00"

    def test_small_values(self):
        """Values 0-127 encode to single byte."""
        for i in range(128):
            result = varint_encode(i)
            assert len(result) == 1
            assert result[0] == i

    def test_127(self):
        """127 = 0x7F = single byte."""
        result = varint_encode(127)
        assert result == b"\x7f"

    def test_128(self):
        """128 = 0x80 requires two bytes (0x80, 0x01)."""
        result = varint_encode(128)
        assert result == b"\x80\x01"
        assert len(result) == 2

    def test_255(self):
        """255 encodes to 0xFF, 0x01."""
        result = varint_encode(255)
        assert result == b"\xff\x01"

    def test_16383(self):
        """16383 = 0x3FFF = 127 * 128 + 127 (two bytes)."""
        result = varint_encode(16383)
        assert len(result) == 2
        assert result == b"\xff\x7f"

    def test_16384(self):
        """16384 requires three bytes."""
        result = varint_encode(16384)
        assert len(result) == 3
        assert result == b"\x80\x80\x01"

    def test_large_value(self):
        """2**21 - 1 = 2097151 (requires 3 bytes)."""
        result = varint_encode(2**21 - 1)
        assert len(result) == 3

    def test_very_large_value(self):
        """2**21 and beyond (4+ bytes)."""
        result = varint_encode(2**21)
        assert len(result) == 4


class TestVarintDecode:
    """Test varint_decode() roundtrip and edge cases."""

    def test_decode_zero(self):
        """Decode 0."""
        buf = b"\x00"
        val, next_offset = varint_decode(buf, 0)
        assert val == 0
        assert next_offset == 1

    def test_decode_single_byte(self):
        """Decode values 0-127."""
        for i in range(128):
            buf = bytes([i])
            val, next_offset = varint_decode(buf, 0)
            assert val == i
            assert next_offset == 1

    def test_decode_128(self):
        """Decode 128."""
        buf = b"\x80\x01"
        val, next_offset = varint_decode(buf, 0)
        assert val == 128
        assert next_offset == 2

    def test_decode_16383(self):
        """Decode 16383."""
        buf = b"\xff\x7f"
        val, next_offset = varint_decode(buf, 0)
        assert val == 16383
        assert next_offset == 2

    def test_decode_16384(self):
        """Decode 16384."""
        buf = b"\x80\x80\x01"
        val, next_offset = varint_decode(buf, 0)
        assert val == 16384
        assert next_offset == 3

    def test_decode_with_offset(self):
        """Decode from non-zero offset in buffer."""
        # Buffer: [0x10, 0x7f, 0x80, 0x01]
        # At offset 1: 0x7f = 127
        # At offset 2: 0x80, 0x01 = 128
        buf = b"\x10\x7f\x80\x01"
        val, next_offset = varint_decode(buf, 1)
        assert val == 127
        assert next_offset == 2

        val, next_offset = varint_decode(buf, 2)
        assert val == 128
        assert next_offset == 4

    def test_decode_multiple_sequential(self):
        """Decode multiple varints in sequence from one buffer."""
        # Encode [10, 200, 3] consecutively
        buf = varint_encode(10) + varint_encode(200) + varint_encode(3)
        val1, pos1 = varint_decode(buf, 0)
        val2, pos2 = varint_decode(buf, pos1)
        val3, pos3 = varint_decode(buf, pos2)
        assert val1 == 10
        assert val2 == 200
        assert val3 == 3

    def test_decode_incomplete_buffer(self):
        """Decode fails gracefully if buffer ends mid-varint."""
        # 0x80 alone means continuation, but no next byte
        buf = b"\x80"
        with pytest.raises(IndexError):
            varint_decode(buf, 0)

    def test_decode_too_long(self):
        """Decode fails if varint is suspiciously long (>5 bytes)."""
        # Construct a malformed varint with too many continuation bytes
        buf = b"\x80" * 6  # 6 continuation bytes = invalid
        with pytest.raises(ValueError):
            varint_decode(buf, 0)


# ===========================================================================
# Roundtrip tests: encode → decode
# ===========================================================================

class TestVarintRoundtrip:
    """Test that encode/decode are true inverses."""

    @pytest.mark.parametrize(
        "value",
        [0, 1, 10, 127, 128, 255, 256, 16383, 16384, 65535, 65536, 2**21 - 1, 2**21],
    )
    def test_roundtrip(self, value):
        """Encode then decode recovers original value."""
        buf = varint_encode(value)
        decoded, _ = varint_decode(buf, 0)
        assert decoded == value


# ===========================================================================
# Gap encoding/decoding tests
# ===========================================================================

class TestGapEncode:
    """Test gap_encode() delta encoding."""

    def test_empty_list(self):
        """Empty docid list → empty gaps."""
        result = gap_encode([])
        assert result == []

    def test_single_element(self):
        """Single docid returns itself."""
        result = gap_encode([0])
        assert result == [0]

    def test_simple_sequence(self):
        """[0, 1, 5, 100] → [0, 1, 4, 95]."""
        docids = [0, 1, 5, 100]
        result = gap_encode(docids)
        assert result == [0, 1, 4, 95]

    def test_non_zero_start(self):
        """[10, 20, 25] → [10, 10, 5]."""
        docids = [10, 20, 25]
        result = gap_encode(docids)
        assert result == [10, 10, 5]

    def test_dense_sequence(self):
        """Sequential IDs: [0, 1, 2, 3] → [0, 1, 1, 1]."""
        docids = [0, 1, 2, 3]
        result = gap_encode(docids)
        assert result == [0, 1, 1, 1]

    def test_sparse_sequence(self):
        """Large gaps: [0, 1000, 70772] → [0, 1000, 69772]."""
        docids = [0, 1000, 70772]
        result = gap_encode(docids)
        assert result == [0, 1000, 69772]


class TestGapDecode:
    """Test gap_decode() delta decoding."""

    def test_empty_list(self):
        """Empty gaps → empty docids."""
        result = gap_decode([])
        assert result == []

    def test_single_gap(self):
        """Single gap returns itself as docid."""
        result = gap_decode([5])
        assert result == [5]

    def test_simple_gaps(self):
        """[0, 1, 4, 95] → [0, 1, 5, 100]."""
        gaps = [0, 1, 4, 95]
        result = gap_decode(gaps)
        assert result == [0, 1, 5, 100]

    def test_non_zero_start(self):
        """[10, 10, 5] → [10, 20, 25]."""
        gaps = [10, 10, 5]
        result = gap_decode(gaps)
        assert result == [10, 20, 25]


class TestGapRoundtrip:
    """Test that gap_encode/gap_decode are inverses."""

    @pytest.mark.parametrize(
        "docids",
        [
            [],
            [0],
            [1, 2, 3],
            [0, 1, 5, 100],
            [0, 1, 5, 100, 70772],
            [0, 100, 200, 300, 10000],
        ],
    )
    def test_roundtrip(self, docids):
        """Encode then decode recovers original."""
        gaps = gap_encode(docids)
        recovered = gap_decode(gaps)
        assert recovered == docids


# ===========================================================================
# PostingsReader tests with binary fixtures
# ===========================================================================

class TestPostingsReaderBinaryRoundtrip:
    """Test PostingsReader on constructed binary data."""

    @staticmethod
    def _build_postings_binary(posting_lists: list[list[tuple[int, int]]]) -> bytes:
        """Build a concatenated postings.bin fragment for testing."""
        result = bytearray()
        for postings in posting_lists:
            if postings:
                docids, tfs = zip(*postings)
            else:
                docids, tfs = [], []

            result.extend(varint_encode(len(postings)))

            gaps = gap_encode(list(docids))

            for gap, tf in zip(gaps, tfs):
                result.extend(varint_encode(gap))
                result.extend(varint_encode(tf))

        return bytes(result)

    def test_single_term_single_posting(self):
        """One term with one posting: (docid=0, tf=5)."""
        posting_lists = [[(0, 5)]]
        buf = self._build_postings_binary(posting_lists)

        with tempfile.NamedTemporaryFile() as f:
            f.write(buf)
            f.flush()

            with PostingsReader(f.name) as reader:
                postings = reader.decode_postings(offset=0, length=len(buf))
                assert postings == [(0, 5)]

    def test_single_term_multiple_postings(self):
        """One term with 3 postings: (0,5), (10,3), (20,2)."""
        posting_lists = [[(0, 5), (10, 3), (20, 2)]]
        buf = self._build_postings_binary(posting_lists)

        with tempfile.NamedTemporaryFile() as f:
            f.write(buf)
            f.flush()

            with PostingsReader(f.name) as reader:
                postings = reader.decode_postings(offset=0, length=len(buf))
                assert postings == [(0, 5), (10, 3), (20, 2)]

    def test_multiple_terms_sequential(self):
        """Three terms with different postings; read them sequentially."""
        posting_lists = [
            [(0, 5), (10, 3)],
            [(1, 2), (5, 1), (100, 4)],
            [(0, 7)],
        ]
        buf = self._build_postings_binary(posting_lists)

        with tempfile.NamedTemporaryFile() as f:
            f.write(buf)
            f.flush()

            # Manually find offsets and lengths
            with PostingsReader(f.name) as reader:
                # Term 0: offset 0
                offsets_lengths = []
                pos = 0
                for postings in posting_lists:
                    # Compute length by encoding
                    term_buf = self._build_postings_binary([postings])
                    offsets_lengths.append((pos, len(term_buf)))
                    pos += len(term_buf)

                # Now read each term
                result0 = reader.decode_postings(offsets_lengths[0][0], offsets_lengths[0][1])
                assert result0 == [(0, 5), (10, 3)]

                result1 = reader.decode_postings(offsets_lengths[1][0], offsets_lengths[1][1])
                assert result1 == [(1, 2), (5, 1), (100, 4)]

                result2 = reader.decode_postings(offsets_lengths[2][0], offsets_lengths[2][1])
                assert result2 == [(0, 7)]

    def test_large_docids(self):
        """Test with large docid values (near max expected)."""
        posting_lists = [[(0, 1), (1000, 2), (70771, 3)]]
        buf = self._build_postings_binary(posting_lists)

        with tempfile.NamedTemporaryFile() as f:
            f.write(buf)
            f.flush()

            with PostingsReader(f.name) as reader:
                postings = reader.decode_postings(offset=0, length=len(buf))
                assert postings == [(0, 1), (1000, 2), (70771, 3)]


class TestPostingsReaderCaching:
    """Test decode_postings_cached() behavior."""

    @staticmethod
    def _build_postings_binary(posting_lists: list[list[tuple[int, int]]]) -> bytes:
        """Same helper as above."""
        result = bytearray()
        for postings in posting_lists:
            if postings:
                docids, tfs = zip(*postings)
            else:
                docids, tfs = [], []
            result.extend(varint_encode(len(postings)))
            gaps = gap_encode(list(docids))
            for gap, tf in zip(gaps, tfs):
                result.extend(varint_encode(gap))
                result.extend(varint_encode(tf))
        return bytes(result)

    def test_cache_hit_on_repeat(self):
        """Second call with same (offset, length) returns cached result."""
        posting_lists = [[(0, 5), (10, 3)]]
        buf = self._build_postings_binary(posting_lists)

        with tempfile.NamedTemporaryFile() as f:
            f.write(buf)
            f.flush()

            cache = {}
            with PostingsReader(f.name) as reader:
                result1 = reader.decode_postings_cached(0, len(buf), cache)
                assert result1 == [(0, 5), (10, 3)]
                assert (0, len(buf)) in cache

                # Second call should hit cache (not re-read file)
                result2 = reader.decode_postings_cached(0, len(buf), cache)
                assert result2 == [(0, 5), (10, 3)]
                # Cache should only have one entry
                assert len(cache) == 1

    def test_cache_misses_different_offsets(self):
        """Different offsets create separate cache entries."""
        posting_lists = [
            [(0, 5), (10, 3)],
            [(1, 2), (100, 4)],
        ]
        buf = self._build_postings_binary(posting_lists)

        with tempfile.NamedTemporaryFile() as f:
            f.write(buf)
            f.flush()

            cache = {}
            with PostingsReader(f.name) as reader:
                # Get offsets
                term0_buf = self._build_postings_binary([posting_lists[0]])
                term1_buf = self._build_postings_binary([posting_lists[1]])
                term0_len = len(term0_buf)
                term1_offset = term0_len

                result0 = reader.decode_postings_cached(0, term0_len, cache)
                assert result0 == [(0, 5), (10, 3)]

                result1 = reader.decode_postings_cached(
                    term1_offset, len(term1_buf), cache
                )
                assert result1 == [(1, 2), (100, 4)]

                # Cache should have two entries
                assert len(cache) == 2


class TestPostingsReaderContextManager:
    """Test __enter__ and __exit__ behavior."""

    @staticmethod
    def _build_postings_binary(posting_lists: list[list[tuple[int, int]]]) -> bytes:
        """Same helper as above."""
        result = bytearray()
        for postings in posting_lists:
            if postings:
                docids, tfs = zip(*postings)
            else:
                docids, tfs = [], []
            result.extend(varint_encode(len(postings)))
            gaps = gap_encode(list(docids))
            for gap, tf in zip(gaps, tfs):
                result.extend(varint_encode(gap))
                result.extend(varint_encode(tf))
        return bytes(result)

    def test_context_manager_closes_file(self):
        """File handle is closed after exiting context."""
        # Create valid postings data
        posting_lists = [[(0, 5)]]
        buf = self._build_postings_binary(posting_lists)

        with tempfile.NamedTemporaryFile() as tf:
            tf.write(buf)
            tf.flush()

            reader = PostingsReader(tf.name)
            assert reader._file_handle is None  # Not opened yet

            with reader as r:
                # Force open by reading
                _ = r.decode_postings(0, len(buf))
                assert r._file_handle is not None

            # After context exit, should be closed
            assert reader._file_handle is None

    def test_explicit_close(self):
        """Explicit close() closes the file handle."""
        # Create valid postings data
        posting_lists = [[(0, 5)]]
        buf = self._build_postings_binary(posting_lists)

        with tempfile.NamedTemporaryFile() as tf:
            tf.write(buf)
            tf.flush()

            reader = PostingsReader(tf.name)
            _ = reader.decode_postings(0, len(buf))
            assert reader._file_handle is not None

            reader.close()
            assert reader._file_handle is None


class TestPostingsReaderErrors:
    """Test error handling in PostingsReader."""

    def test_truncated_file(self):
        """Reading past end of file raises IOError."""
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"\x05")  # Only 1 byte
            f.flush()

            with PostingsReader(f.name) as reader:
                # Try to read 100 bytes starting at 0
                with pytest.raises(IOError):
                    reader.decode_postings(0, 100)

    def test_malformed_postings(self):
        """Incomplete varint in postings raises error during decode."""
        with tempfile.NamedTemporaryFile() as f:
            # df=2, but only one postings entry (incomplete)
            # varint(2) = 0x02, then incomplete gaps/tfs
            f.write(b"\x02\x80")  # df=2, then incomplete varint 0x80 (needs continuation)
            f.flush()

            with PostingsReader(f.name) as reader:
                with pytest.raises(IndexError):
                    reader.decode_postings(0, 2)
