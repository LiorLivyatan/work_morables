# Experiment 21: Lexical Controls

Reviewer-facing controls for the clustered MoraLink benchmark.

This experiment evaluates shallow lexical retrieval methods on the official
clustered multi-positive qrels:

- TF-IDF cosine similarity over raw fable text.
- BM25 over raw fable text.

The goal is not to maximize performance, but to test whether lexical overlap
alone can explain the embedding-model results.

Run through `run.sh`:

```bash
./run.sh review_response/21_lexical_controls/eval.py
```

Outputs are written to `review_response/21_lexical_controls/results/`:

- `<timestamp>_lexical_controls.json` with aggregate metrics.
- `<timestamp>_metrics.csv` with one row per lexical method.
- `rankings/<method>.json` with full 709-fable rankings for every query.

## Lexical-Overlap Correlation

Use the saved clustered LINQ rankings to test whether token overlap predicts
dense-model scores or ranks:

```bash
./run.sh review_response/21_lexical_controls/overlap_analysis.py
```

This writes `overlap_summary.json` and `per_query_overlap.csv` under a
timestamped result folder.

Latest clustered LINQ raw finding:

- Pair-level content-token IoU vs dense score: Spearman `0.060`.
- Pair-level content-token IoU vs dense reciprocal rank: Spearman `0.065`.
- Query-level max relevant content-token IoU vs best relevant reciprocal rank: Spearman `0.220`.
- The highest lexical-overlap fable is relevant for only `3.1%` of queries.

## Surface-Only Controls

Run title-only, first-sentence-only, and truncated-fable variants:

```bash
./run.sh review_response/21_lexical_controls/eval.py --doc-variants raw title first_sentence first_50
```

Latest results:

| Method | Doc variant | MRR | MRR@10 | MAP@10 | Hit@10 | Recall@100 |
|---|---|---:|---:|---:|---:|---:|
| TF-IDF | raw | 0.091 | 0.080 | 0.063 | 15.9% | 31.2% |
| BM25 | raw | 0.084 | 0.073 | 0.057 | 15.9% | 31.5% |
| TF-IDF | title | 0.033 | 0.026 | 0.021 | 4.6% | 16.1% |
| BM25 | title | 0.030 | 0.022 | 0.019 | 3.9% | 17.0% |
| TF-IDF | first sentence | 0.038 | 0.030 | 0.020 | 7.0% | 18.9% |
| BM25 | first sentence | 0.033 | 0.024 | 0.017 | 5.8% | 18.9% |
| TF-IDF | first 50 words | 0.055 | 0.046 | 0.034 | 11.4% | 24.5% |
| BM25 | first 50 words | 0.051 | 0.041 | 0.031 | 8.5% | 24.2% |
