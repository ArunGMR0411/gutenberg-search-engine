# Judging Notes

## Overview
- 150 results judged across 15 non-empty TSVs (6 queries × 3 models - 3 empty)
- 3 empty TSVs with justification: `1_vsm.tsv` (VSM early-abort on stopword-heavy Q1), `4_structured.tsv` ("Jabberwocky" not in any metadata field), `6_structured.tsv` ("Dornröschen" not in any metadata field)
- Rubric: 1=relevant, 2=partially relevant, 0=not relevant

## Per-Query Notes

### Q1: "to be, or not to be"
- **Structured** (0/10 relevant): Matched titles containing fragments "to be" / "not to be" (e.g., "How Not to Be Them", "Dare to Be Healthy"). None are about Shakespeare's soliloquy. All scored 0.
- **BM25** (1/10 relevant, 5/10 partial): Full-corpus phrase scan matched texts containing the exact phrase. The Complete Works of Shakespeare (#100) ranked first — clearly relevant. Several literary works (Lamb, Johnson biographies) quote the phrase in literary context (partial). Some incidental appearances scored 0.
- **VSM** (empty): Correctly early-aborted — all query terms are high-df stopwords, no useful TF-IDF signal possible.
- **Key observation**: Structured search fails completely for content-only phrases not in metadata. BM25's phrase fallback successfully finds the Shakespeare quote.

### Q2: "English Grammar"
- **Structured** (10/10 relevant): Perfect precision — all 10 results are actual English Grammar textbooks with both words in their title. Best performing model for this query.
- **BM25** (2/10 relevant, 5/10 partial): "Grammar School Boys" fiction ranked higher than actual grammar textbooks. BM25 rewards documents where "grammar" and "english" appear frequently but not necessarily together as a topic.
- **VSM** (2/10 relevant, 2/10 partial): Similar issue — small documents and catalog entries with high TF-IDF for these terms but not actually about English grammar.
- **Key observation**: Structured metadata search excels for topic queries matching title/subject fields.

### Q3: "Philip K Dick"
- **Structured** (10/10 relevant): Perfect — all 10 are PKD stories. Author field matching is precise.
- **BM25** (0/10 relevant): Complete failure — "Dick" matches dime novels, copyright catalogs, movie credits. The name "Dick" is common outside PKD's works. "Philip" and "K" have weak discriminative power.
- **VSM** (2/10 relevant): Slightly better — "Beyond the Door" and "The Eyes Have It" appear. But "Dick and His Cat" and many false positives dominate because "Dick" appears with high TF in short, unrelated texts.
- **Key observation**: Author name queries strongly favor structured (metadata) search. Bag-of-words models break down when name components are common words.

### Q4: "Jabberwocky"
- **BM25** (6/10 relevant, 4/10 partial): Strong results — Nonsense Anthology, Through the Looking-Glass, Lewis Carroll biography all rank highly. A distinctive term like "Jabberwocky" has excellent IDF.
- **VSM** (6/10 relevant, 2/10 partial): Very similar top-10 to BM25 with different ranking. Same core set of relevant texts.
- **Structured** (empty): Expected — "Jabberwocky" doesn't appear in metadata title/author/bookshelf fields. It's a poem title within a book, not a book title.
- **Key observation**: For distinctive terms, BM25 and VSM both work well. BM25 slightly better at ranking thanks to length normalization.

### Q5: "Gutenberg"
- **Structured** (1/10 relevant, 9/10 partial): Found PG Encyclopedia and PG quotation/index pages. These have "Gutenberg" in their titles but are PG meta-content rather than about Gutenberg the inventor.
- **BM25** (4/10 relevant, 2/10 partial): Mixed — found the French play about Gutenberg and PG history docs, but also false positives from texts where boilerplate stripping failed (plays without standard markers).
- **VSM** (0/10 relevant, 5/10 partial): PG index pages dominate (high TF-IDF in short docs). No results directly about Gutenberg the inventor.
- **Key observation**: "Gutenberg" is ambiguous (inventor vs. Project Gutenberg header). Boilerplate stripping removes most PG headers, but residual mentions inflate scores.

### Q6: "Dornröschen"
- **BM25** (2/10 relevant, 6/10 partial): Found Bechstein's Märchenbuch and Gänsemütterchens Märchen (fairy tale collections). German literary texts discussing fairy tales scored partially.
- **VSM** (1/10 relevant, 6/10 partial): Similar set with different ranking. Gänsemütterchens Märchen relevant; literary criticism partially relevant.
- **Structured** (empty): Expected — "Dornröschen" doesn't appear in metadata fields.
- **Key observation**: For non-English queries, both BM25 and VSM work comparably. The accent-fold aliasing (dornroschen) helps find matches. BM25's length normalization produced slightly better ranking.

## Tricky Judgment Decisions
1. **Q1 BM25 phrase matches**: All results contain "to be or not to be" as a substring. Rated literary works that quote/discuss the phrase as partial (2), not fully relevant, since they aren't primarily about Hamlet's soliloquy.
2. **Q5 PG meta-content**: PG quotation collections and index pages have "Gutenberg" in title but are compilations from other authors — rated as partial (2).
3. **Q6 German literary texts**: "Dornröschen" appears in German literary criticism/essays — rated partial since they discuss but aren't the fairy tale itself.
