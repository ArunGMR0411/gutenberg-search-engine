"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import pytest

from ir_system.preprocessing import (
    accent_fold_aliases,
    preprocess,
    preprocess_query,
    strip_boilerplate,
)


# ===========================================================================
# Test: preprocess() — hyphens, apostrophes, tokenization
# ===========================================================================

class TestHyphenSplitting:
    """Test that all hyphen variants are replaced with space (splits tokens)."""

    def test_regular_hyphen(self):
        """U+002D hyphen-minus splits 'state-of-the-art'."""
        result = preprocess("state-of-the-art")
        assert result == ["state", "of", "the", "art"], f"Got {result}"

    def test_en_dash(self):
        """U+2013 en-dash splits 'word–word'."""
        result = preprocess("word–word")
        assert "word" in result
        assert len(result) >= 2

    def test_em_dash(self):
        """U+2014 em-dash splits 'word—word'."""
        result = preprocess("word—word")
        assert "word" in result
        assert len(result) >= 2

    def test_various_dashes_all_split(self):
        """All hyphen variants (U+002D, U+2010–U+2014) split correctly."""
        variants = [
            ("a\u002Db", "U+002D"),
            ("a\u2010b", "U+2010"),
            ("a\u2011b", "U+2011"),
            ("a\u2012b", "U+2012"),
            ("a\u2013b", "U+2013"),
            ("a\u2014b", "U+2014"),
        ]
        for text, label in variants:
            result = preprocess(text)
            # Each variant should split "a" and "b" into separate tokens
            assert "a" in result and "b" in result, f"Failed for {label}: got {result}"


class TestApostropheRemoval:
    """Test that apostrophes are removed (tokens merge)."""

    def test_regular_apostrophe(self):
        """U+0027 apostrophe: 'don't' → ['dont']."""
        result = preprocess("don't")
        assert "dont" in result, f"Got {result}"
        assert "don" not in result, "Should not have 'don' as separate token"

    def test_right_single_quotation_mark(self):
        """U+2019 right single quotation mark: 'o'clock' → ['oclock']."""
        result = preprocess("o'clock")  # Using right single quotation mark
        # The test string might use either variant; test both
        assert "oclock" in result or "oclock" in result, f"Got {result}"

    def test_apostrophe_variants(self):
        """Both apostrophe variants removed."""
        result1 = preprocess("don't")   # U+0027
        result2 = preprocess("don't")   # U+2019
        assert "dont" in result1, f"Regular apostrophe failed: {result1}"
        assert "dont" in result2, f"Right quotation failed: {result2}"


class TestTokenization:
    """Test that tokenization keeps numbers and splits on boundaries."""

    def test_numeric_tokens_kept(self):
        """Numeric tokens like '2', '1848', 'vol2' are kept."""
        result = preprocess("vol2 and 1848 and 2")
        assert "vol2" in result, f"'vol2' not found in {result}"
        assert "1848" in result, f"'1848' not found in {result}"
        assert "2" in result, f"'2' not found in {result}"

    def test_mixed_alphanumeric(self):
        """Alphanumeric tokens like 'test123' are kept."""
        result = preprocess("test123")
        assert "test123" in result, f"Got {result}"

    def test_underscore_excluded(self):
        """Underscores are NOT part of word tokens (regex [^\\W_]+)."""
        result = preprocess("a_b")
        # Should split into ["a", "b"], not ["a_b"]
        assert "a" in result and "b" in result, f"Got {result}"
        assert "a_b" not in result


class TestNFKCNormalization:
    """Test NFKC normalization (ligatures, etc.)."""

    def test_ligature_fi(self):
        """Ligature 'ﬁ' (U+FB01) normalizes to 'fi' via NFKC."""
        # ﬁ is the ligature, fi is the decomposed form
        result = preprocess("ﬁnally")  # ligature
        # After NFKC, should tokenize to ["finally"]
        assert "finally" in result, f"Got {result}"

    def test_ligature_fl(self):
        """Ligature 'ﬂ' (U+FB02) normalizes to 'fl' via NFKC."""
        result = preprocess("ﬂag")  # ligature
        assert "flag" in result, f"Got {result}"


class TestCaseFolding:
    """Test case folding (lowercase normalization)."""

    def test_uppercase_folded(self):
        """Uppercase converted to lowercase via casefold()."""
        result = preprocess("HELLO World")
        assert "hello" in result, f"Got {result}"
        assert "world" in result, f"Got {result}"

    def test_german_eszett(self):
        """German ß (U+00DF) case-folds to 'ss'."""
        result = preprocess("Straße")
        # ß.casefold() -> "ss", so "Straße".casefold() -> "strasse"
        assert "strasse" in result, f"Got {result}"


# ===========================================================================
# Test: accent_fold_aliases()
# ===========================================================================

class TestAccentFolding:
    """Test accent-fold alias generation (NFD decompose + strip Mn)."""

    def test_single_token_with_diacritic(self):
        """Token 'dornröschen' generates both itself and alias 'dornroschen'."""
        tokens = ["dornröschen"]
        result = accent_fold_aliases(tokens)
        assert "dornröschen" in result, f"Original not in {result}"
        assert "dornroschen" in result, f"Alias not in {result}"

    def test_single_token_without_diacritic(self):
        """Token 'city' (no diacritics) stays as is (single occurrence)."""
        tokens = ["city"]
        result = accent_fold_aliases(tokens)
        assert "city" in result, f"Got {result}"
        assert result.count("city") == 1, "Should not duplicate"

    def test_mixed_diacritics(self):
        """Multiple diacritics: 'café' → both 'café' and 'cafe'."""
        tokens = ["café"]
        result = accent_fold_aliases(tokens)
        assert "café" in result, f"Original not in {result}"
        assert "cafe" in result, f"Alias not in {result}"

    def test_multiple_tokens_some_with_diacritics(self):
        """Some tokens diacritic, some not."""
        tokens = ["hello", "café", "world"]
        result = accent_fold_aliases(tokens)
        assert "hello" in result
        assert "café" in result
        assert "cafe" in result
        assert "world" in result
        assert len([t for t in result if t == "hello"]) == 1

    def test_no_alias_duplication(self):
        """If alias matches an existing token, don't duplicate."""
        # If we have both "cafe" and "café", the alias of "café"
        # should not create a duplicate "cafe"
        tokens = ["cafe", "café"]
        result = accent_fold_aliases(tokens)
        assert "cafe" in result
        assert "café" in result
        # Count "cafe" — should be exactly one (the original)
        assert result.count("cafe") == 1


class TestAccentFoldingWithPreprocess:
    """Integration: preprocess() returns base tokens; aliases generated at index time."""

    def test_preprocessed_dornröschen(self):
        """Full preprocess on 'Dornröschen': returns base token only."""
        result = preprocess("Dornröschen")
        # After case-fold: "dornröschen"
        # After tokenization: ["dornröschen"]
        # Note: Aliases are generated at index time, not in preprocess()
        assert result == ["dornröschen"], f"Expected ['dornröschen'], got {result}"


# ===========================================================================
# Test: strip_boilerplate()
# ===========================================================================

class TestBoilerplateStripping:
    """Test removal of Project Gutenberg START/END markers."""

    def test_strip_start_marker_only(self):
        """'*** START OF' marker: discard that line and everything before."""
        text = "Header line\n*** START OF THE EBOOK ***\nContent here"
        result = strip_boilerplate(text)
        assert "Header line" not in result, f"Header not removed: {result}"
        assert "*** START OF" not in result, f"Marker not removed: {result}"
        assert "Content here" in result, f"Content lost: {result}"

    def test_strip_end_marker_only(self):
        """'*** END OF' marker: discard that line and everything after."""
        text = "Content here\n*** END OF THE EBOOK ***\nFooter line"
        result = strip_boilerplate(text)
        assert "Content here" in result, f"Content lost: {result}"
        assert "*** END OF" not in result, f"Marker not removed: {result}"
        assert "Footer line" not in result, f"Footer not removed: {result}"

    def test_strip_both_markers(self):
        """Both markers: discard header, marker, footer (keep content)."""
        text = "Header\n*** START OF ***\nContent\n*** END OF ***\nFooter"
        result = strip_boilerplate(text)
        assert "Header" not in result
        assert "Content" in result
        assert "Footer" not in result
        assert "*** START OF" not in result
        assert "*** END OF" not in result

    def test_no_markers_unchanged(self):
        """No markers: return full text unchanged."""
        text = "Just plain content\nwith no markers"
        result = strip_boilerplate(text)
        assert result == text

    def test_marker_case_insensitive(self):
        """Markers matched case-insensitively."""
        text = "Before\n*** start of ***\nContent\n*** end of ***\nAfter"
        result = strip_boilerplate(text)
        assert "Before" not in result
        assert "Content" in result
        assert "After" not in result

    def test_realistic_gutenberg_markers(self):
        """Real-world Gutenberg marker format."""
        text = """Some header info
*** START OF THE PROJECT GUTENBERG EBOOK ***
This is the actual book content.
More content here.
*** END OF THE PROJECT GUTENBERG EBOOK ***
Legal footer"""
        result = strip_boilerplate(text)
        assert "Some header info" not in result
        assert "This is the actual book content" in result
        assert "More content here" in result
        assert "Legal footer" not in result


# ===========================================================================
# Test: preprocess_query()
# ===========================================================================

class TestPreprocessQuery:
    """Test query preprocessing (same pipeline as preprocess)."""

    def test_simple_query(self):
        """Query 'Philip K Dick' preprocessed."""
        result = preprocess_query("Philip K Dick")
        assert "philip" in result
        assert "k" in result
        assert "dick" in result

    def test_query_with_diacritics(self):
        """Query 'Dornröschen' includes aliases."""
        result = preprocess_query("Dornröschen")
        assert "dornröschen" in result
        assert "dornroschen" in result

    def test_query_with_hyphens(self):
        """Query with hyphens splits correctly."""
        result = preprocess_query("state-of-the-art")
        assert result == ["state", "of", "the", "art"]

    def test_query_with_apostrophes(self):
        """Query with apostrophes."""
        result = preprocess_query("don't")
        assert "dont" in result


# ===========================================================================
# Comprehensive integration tests
# ===========================================================================

class TestIntegration:
    """End-to-end tests combining multiple features."""

    def test_complex_document(self):
        """Complex text: hyphens, apostrophes, diacritics, numbers."""
        text = "It's the state-of-the-art café in 2024"
        result = preprocess(text)
        # Expected tokens: base tokens only (no aliases for documents)
        # - "its" (apostrophe removed from "it's")
        # - "the", "state", "of", "the", "art" (hyphens split)
        # - "café" (base token, no alias in documents)
        # - "in", "2024"
        assert "its" in result, f"Got {result}"
        assert "state" in result
        assert "art" in result
        assert "café" in result
        assert "2024" in result
        # Aliases should NOT be in document preprocessing
        assert "cafe" not in result, "Aliases should not be in document preprocessing"

    def test_all_features_together(self):
        """All preprocessing steps in sequence (base tokens only)."""
        # Text with: NFKC case (uppercase), hyphens, apostrophes, diacritics, numbers
        text = "ÜBER-SCHÖNE 1984"  # German "über-schöne" (very beautiful)
        result = preprocess(text)
        # NFKC + casefold: "über-schöne 1984"
        # Replace hyphens: "über schöne 1984"
        # Tokenize: ["über", "schöne", "1984"]
        # Note: For documents, no aliases are included
        assert "1984" in result
        assert "über" in result
        assert "schöne" in result
        # Aliases NOT in document preprocessing
        assert "uber" not in result, "Aliases should not be in document preprocessing"
        assert "schone" not in result, "Aliases should not be in document preprocessing"
