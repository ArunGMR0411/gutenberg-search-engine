# Per-Query Narrative Scaffolding

Source material for the ACM report Evaluation section. For each query, observations tied back to model assumptions and design choices.

---

## Q1: "to be, or not to be"

### Structured (P@10=0.00)
- Matched on title fragments containing "to be" / "not to be" — returned unrelated books (e.g., "How Not to Be Them", "Dare to Be Healthy")
- Structured search operates on metadata fields only (title, author, bookshelf); it cannot find content phrases
- **Design implication:** Structured is fundamentally unsuited for in-text phrase queries

### VSM (empty — 0 results)
- All query tokens ("to", "be", "or", "not") have df/N > 0.8 → phrase fallback trigger fires
- VSM early-aborts when phrase fallback activates because TF-IDF weights are near-zero for such tokens (no discriminative signal)
- **Design choice:** This is intentional — returning results with all-zero IDF would produce random rankings

### BM25 (P@10=0.20, NDCG@10=0.89)
- Same zero-IDF issue as VSM, but BM25 activates full-corpus phrase scan
- Scanned 70,772 files for substring "to be or not to be" (punctuation-normalized), found 200 matches (cap)
- The Complete Works of Shakespeare ranked #1 (relevant). Literary biographies and criticism also matched (partial)
- **Trade-off:** Full scan took ~459 seconds but ensured non-empty results for this challenging query

---

## Q2: "English Grammar"

### Structured (P@10=1.00 — perfect)
- All 10 results are actual English grammar textbooks: "The Grammar of English Grammars", "An English Grammar", etc.
- Metadata title field matching is exact and precise — both words appear in titles
- **Strength:** For topic queries matching metadata fields, structured search provides near-perfect precision

### BM25 (P@10=0.20, NDCG@10=0.65)
- "Grammar School Boys" fiction ranked above actual grammar textbooks
- BM25 rewards high term frequency of "grammar" and "english" regardless of context
- **Weakness:** Lacks semantic understanding — fiction set in grammar schools scores as high as grammar textbooks

### VSM (P@10=0.20, NDCG@10=0.60)
- Short catalog/sketch-book documents with high TF-IDF for "grammar" dominate
- VSM's lack of length normalization causes short irrelevant documents to rank highly
- **Weakness:** Very short documents where "grammar" appears once have disproportionately high TF-IDF

---

## Q3: "Philip K Dick"

### Structured (P@10=1.00 — perfect)
- All 10 are Philip K. Dick stories from the author field
- Author metadata matching is exact: "Philip K. Dick" appears in the author field
- **Strength:** Author name queries are structured search's ideal use case

### BM25 (P@10=0.00 — complete failure)
- "Dick" matches dime novels, copyright catalogs, motion picture credits
- "Philip" and "K" provide weak discriminative power
- BM25 treats the query as a bag of words — it cannot connect "Philip", "K", and "Dick" as a proper name
- **Weakness:** For multi-word proper nouns where individual words are common, bag-of-words models fail catastrophically

### VSM (P@10=0.20)
- "Dick and His Cat" ranked #1 (false positive from high TF-IDF for "Dick" in short document)
- Only 2 of 10 results are actual PKD stories (Beyond the Door, The Eyes Have It)
- **Weakness:** Same bag-of-words limitation as BM25, compounded by VSM's bias toward short documents

---

## Q4: "Jabberwocky"

### BM25 (P@10=0.60, NDCG@10=1.00)
- Strong results: A Nonsense Anthology, Through the Looking-Glass, Lewis Carroll biography
- "Jabberwocky" is a highly distinctive term (low df) → excellent IDF discriminator
- BM25's length normalization correctly ranks books that discuss Jabberwocky extensively
- **Strength:** For distinctive single-term queries, BM25 excels

### VSM (P@10=0.70, NDCG@10=0.99)
- Very similar top-10 to BM25 (Jaccard=0.82), slightly different ordering
- "Songs From Alice in Wonderland" ranked #1 (short doc, high TF for "Jabberwocky")
- **Observation:** VSM and BM25 converge on the same candidate set for single distinctive terms

### Structured (empty — 0 results)
- "Jabberwocky" is a poem title within "Through the Looking-Glass", not a book title
- Metadata fields don't contain poem-level titles
- **Limitation:** Structured search can only find works where the query appears in metadata fields

---

## Q5: "Gutenberg"

### Structured (P@10=0.10, P@10_partial=0.55)
- Found PG Encyclopedia (#200) and 9 PG quotation/index compilations (partial)
- "Gutenberg" appears in titles of PG meta-content — these are about PG, not Gutenberg the inventor
- **Ambiguity:** "Gutenberg" is confounded between inventor and publisher name

### BM25 (P@10=0.40, P@10_partial=0.50)
- Found "Gutenberg, pièce historique" (French play about the inventor), PG history docs
- Also found false positives: plays where boilerplate stripping failed (no standard markers)
- **Weakness:** Residual PG header text inflates BM25 scores for short plays

### VSM (P@10=0.00, P@10_partial=0.25)
- PG index pages dominate (very short docs with high TF-IDF for "Gutenberg" from the remaining header)
- No results directly about Gutenberg the inventor
- **Weakness:** VSM is most vulnerable to the boilerplate contamination issue in very short documents

---

## Q6: "Dornröschen"

### BM25 (P@10=0.20, NDCG@10=0.84)
- Found Bechstein's Märchenbuch (#63465) and Gänsemütterchens Märchen (#42900) — both contain the fairy tale
- German literary texts discussing fairy tales scored partially (Madonna: Novellen, Heilige Zeiten)
- **Strength:** BM25's accent-fold aliasing (Dornröschen → dornroschen) found both exact and normalized matches

### VSM (P@10=0.10, NDCG@10=0.83)
- Similar results to BM25 but with slightly different ranking
- Gänsemütterchens Märchen is ranked lower (#7 vs #6 in BM25)
- **Observation:** For non-English queries, BM25 and VSM produce comparable results (Jaccard=0.67)

### Structured (empty — 0 results)
- "Dornröschen" doesn't appear in metadata title/author/bookshelf fields in English-cataloged metadata
- German fairy tales are typically listed under their collection titles, not individual tale names
- **Limitation:** Metadata coverage is language-biased; non-English content queries fail

---

## Cross-Query Observations

1. **Structured search excels for metadata-matching queries** (Q2, Q3: P@10=1.0) but fails completely for content-only queries (Q1, Q4, Q6: P@10=0.0). It serves a fundamentally different purpose than full-text models.

2. **BM25 slightly outperforms VSM** on mean metrics (MAP 0.44 vs 0.36), primarily due to length normalization preventing short-document bias.

3. **VSM and BM25 have high overlap for distinctive terms** (Q4 Jaccard=0.82, Q6 Jaccard=0.67) but diverge for common terms where length normalization matters.

4. **The phrase fallback mechanism** saved Q1 for BM25 (from 0 to 100 results), demonstrating the value of hybrid bag-of-words + phrase matching.

5. **Stopword-heavy queries** (Q1) remain the hardest challenge — VSM correctly abstains rather than returning noise, while BM25's phrase fallback provides a meaningful (if expensive) fallback.
