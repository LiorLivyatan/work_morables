# Experiment 22: Adversarial Robustness

Reviewer-facing robustness checks using MORABLES adversarial story variants.

The raw MORABLES files under `data/raw/adversarial_*.json` alter fable text
while preserving the original moral label. This experiment evaluates whether
retrieval metrics change when the corpus documents are replaced by those
perturbed story variants, while keeping the official clustered moral queries and
multi-positive qrels.

Run through `run.sh`:

```bash
./run.sh review_response/22_adversarial_robustness/eval_lexical.py
```

By default this evaluates the `*_not_shuffled.json` adversarial files with
TF-IDF and BM25. Shuffled files mostly affect MCQA choice order and are less
relevant for retrieval.

Outputs are written to `review_response/22_adversarial_robustness/results/`.

Latest lexical robustness result:

- Original TF-IDF MRR: `0.091`.
- Character-swap variants reduce TF-IDF MRR to `0.078` for `char_swap` and
  `0.070` for `pre_post_char`.
- The strongest lexical degradation is `pre_post_char_adj`, with TF-IDF MRR
  `0.069` (delta `-0.022` from original).
- Pure injection variants barely change lexical retrieval: `pre_inj`, `post_inj`,
  and `adj_inj` stay near MRR `0.090`.

Latest output:

- `results/2026-05-19_18-11-44/metrics.csv`
- `results/2026-05-19_18-11-44/adversarial_lexical_results.json`
