r"""Preprocessing pipeline for documents and queries."""

from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def preprocess(text: str, include_aliases: bool = False) -> list[str]:
    r"""Normalize and tokenize a text string."""
    # NFKC
    text = unicodedata.normalize("NFKC", text)

    # casefold
    text = text.casefold()

    # replace hyphen variants with space
    #    U+002D = hyphen-minus, U+2010–U+2014 = various dashes
    hyphen_variants = [
        "\u002D",  # HYPHEN-MINUS
        "\u2010",  # HYPHEN
        "\u2011",  # NON-BREAKING HYPHEN
        "\u2012",  # FIGURE DASH
        "\u2013",  # EN DASH
        "\u2014",  # EM DASH
    ]
    for h in hyphen_variants:
        text = text.replace(h, " ")

    # remove apostrophes
    #    U+0027 = APOSTROPHE, U+2019 = RIGHT SINGLE QUOTATION MARK
    apostrophe_variants = ["\u0027", "\u2019"]
    for apos in apostrophe_variants:
        text = text.replace(apos, "")

    # tokenize
    tokens = re.findall(r"[^\W_]+", text)

    # accent-fold aliases if requested
    if include_aliases:
        tokens = accent_fold_aliases(tokens)

    return tokens


# ---------------------------------------------------------------------------
# Accent-fold aliases
# ---------------------------------------------------------------------------

def accent_fold_aliases(tokens: list[str]) -> list[str]:
    """Add accent-stripped duplicates for tokens with diacritics."""
    result = list(tokens)  # Keep originals
    seen = set(tokens)     # Track what we've already added

    for token in tokens:
        # NFD decompose: separate base chars from combining marks
        nfd = unicodedata.normalize("NFD", token)
        # Strip combining diacritical marks (Unicode category Mn)
        alias = "".join(c for c in nfd if unicodedata.category(c) != "Mn")

        # If alias differs and not already in result, add it
        if alias != token and alias not in seen:
            result.append(alias)
            seen.add(alias)

    return result


# ---------------------------------------------------------------------------
# Boilerplate stripping
# ---------------------------------------------------------------------------

def strip_boilerplate(text: str) -> str:
    """Strip text outside Gutenberg *** START/END OF *** markers.

    If neither marker is found, returns the full text unchanged.
    """
    lines = text.split("\n")

    start_idx = None
    for i, line in enumerate(lines):
        if re.search(r"\*\*\* START OF", line, re.IGNORECASE):
            start_idx = i
            break

    end_idx = None
    for i, line in enumerate(lines):
        if re.search(r"\*\*\* END OF", line, re.IGNORECASE):
            end_idx = i
            break

    if start_idx is not None and end_idx is not None:
        # Both markers found: text between them
        content_lines = lines[start_idx + 1 : end_idx]
    elif start_idx is not None:
        # Only START found: text after it
        content_lines = lines[start_idx + 1 :]
    elif end_idx is not None:
        # Only END found: text before it
        content_lines = lines[:end_idx]
    else:
        # No markers: return full text
        content_lines = lines

    return "\n".join(content_lines)


# ---------------------------------------------------------------------------
# Query preprocessing
# ---------------------------------------------------------------------------

def preprocess_query(raw_query: str) -> list[str]:
    """preprocess() with aliases enabled, used for query expansion."""
    return preprocess(raw_query, include_aliases=True)
