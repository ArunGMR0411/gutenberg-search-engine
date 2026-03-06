# Gutenberg Search Engine

Full-text search engine for the Project Gutenberg corpus (~70,000 books). Implements three retrieval models from scratch (structured metadata search, TF-IDF VSM, and BM25) without external IR libraries.

## Setup

```bash
# Clone the repository
git clone git@github.com:ArunGMR0411/gutenberg-search-engine.git
cd gutenberg-search-engine

# Set up a virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Verify installation
python -m ir_system --help
```

## Data Setup

The corpus is not included in this repository (~27.8 GB). To obtain it:

1. Download the Project Gutenberg UTF-8 text archive from a [Gutenberg mirror](https://www.gutenberg.org/robot/harvest).
2. Place individual `.txt` files in `datasets/texts/` (one file per book, named `<gutenberg_id>.txt`).
3. Place the metadata CSV in `datasets/metadata.csv`.

Once the data is in place:

```bash
# Validate the dataset
python -m ir_system validate_dataset

# Build the inverted index (takes ~10.5 hours on an Intel i7-14650HX, 16 GB RAM)
python -m ir_system index

# Validate the built index
python -m ir_system validate_index
```

## Usage

### Run All Evaluation Queries

Generates 18 TSV files (6 queries x 3 models) in `outputs/tsv/`:

```bash
python -m ir_system run_eval_queries
```

### Single Query

```bash
# BM25 search
python -m ir_system search --model bm25 --query "Jabberwocky" --top-k 10

# VSM search
python -m ir_system search --model vsm --query "English Grammar"

# Structured metadata search
python -m ir_system search --model structured --query "Philip K Dick"
```

### Interactive Mode

```bash
python -m ir_system interactive
```

Launches a REPL where you can pick a model, type queries, and see ranked results with scores and snippets.

### Validate Output

```bash
python -m ir_system validate_tsv
```

Checks all 18 TSVs for correct format: 5 tab-separated columns, sequential ranks, monotonically non-increasing scores, and valid Gutenberg IDs.

## Testing

```bash
pytest
```

Runs 154 tests covering:
- Preprocessing (Unicode normalization, hyphen splitting, apostrophe removal, accent folding)
- Indexing (SPIMI block creation, varint encoding/decoding, K-way merge, delta encoding)
- Retrieval (structured search, VSM scoring, BM25 scoring, phrase fallback)
- Evaluation (snippet generation, TSV format, batch runner determinism)

## Project Structure

```
gutenberg-search-engine/
    ir_system/              # Core search engine package
        __main__.py         # Entry point: python -m ir_system
        cli.py              # Argument parsing and command dispatch
        preprocessing.py    # Text normalization pipeline (6 steps)
        indexing.py         # SPIMI block builder and K-way merge
        postings.py         # Varint encoding/decoding, posting list I/O
        retrieval_structured.py  # Metadata field-matching model
        retrieval_vsm.py    # TF-IDF cosine similarity model
        retrieval_bm25.py   # Okapi BM25 model
        phrase_fallback.py  # Full-corpus phrase scan for stopword queries
        snippets.py         # Query-biased snippet extraction
        evaluation.py       # Batch query runner and TSV generation
        dataset.py          # Metadata CSV loading and deduplication
        judge.py            # Manual relevance judgment interface
        utils.py            # Shared helpers
    tests/                  # 154 unit tests (pytest)
    tools/                  # Analysis scripts (metrics, overlap, LaTeX tables)
    outputs/
        tsv/                # 18 result files (6 queries x 3 models)
        analysis/           # Evaluation metrics, tables, summaries
        judgments/          # Manual relevance labels
    datasets/
        metadata.csv        # Gutenberg metadata (titles, authors, subjects)
        texts/              # Raw text files (not in repo, ~27.8 GB)
    report/                 # ACM sigconf LaTeX source and compiled PDF
    requirements.txt        # Pinned dependency versions
    pyproject.toml          # Project metadata and dependencies
```

## Index Format

The final index consists of two files:

| File | Size | Contents |
|------|------|----------|
| `index/index.sqlite` | 1.28 GB | Lexicon (22.5M terms), document map, document statistics (lengths + L2 norms), global parameters, metadata |
| `index/postings.bin` | 1.15 GB | Varint-encoded, delta-compressed posting lists |

## Evaluation Summary

Six benchmark queries evaluated with Precision@10, AP@10, and NDCG@10:

| Model | Mean P@10 | Mean AP@10 | Mean NDCG@10 |
|-------|-----------|------------|--------------|
| Structured | 0.525 | 0.750 | 0.750 |
| VSM | 0.240 | 0.358 | 0.775 |
| BM25 | 0.267 | 0.440 | 0.693 |

No single model dominates: structured search excels for metadata queries, BM25 is the most versatile full-text model, and VSM correctly abstains on stopword-heavy queries where its scores would be non-discriminative.

## Hardware

Developed and benchmarked on:
- Intel Core i7-14650HX, 16 GB RAM, 1 TB SSD
- Ubuntu 24.04, Python 3.12.3

## License

This project is part of academic coursework at Dublin City University. See the report in `report/` for full details.
