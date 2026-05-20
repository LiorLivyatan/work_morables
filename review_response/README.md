# MoraLink Review-Response Workspace

This directory tracks work motivated by the IdioLink rejection feedback and
applied to the MoraLink paper.

## Tracking

- `TODO.md` — prioritized checklist of paper, evaluation, analysis, and
  reproducibility tasks.

## Completed So Far

- Paper framing cleanup:
  - use `MoraLink` consistently in the paper draft and planning docs
  - avoid strong unsupported "understanding" claims
  - rename the oracle framing to a gold-moral appended upper bound
  - make clustered multi-positive qrels the official benchmark definition
- Lexical and surface controls:
  - BM25 and TF-IDF over raw fables
  - title-only, first-sentence-only, and first-50-word controls
  - MRR, MRR@10, MAP, MAP@10, Hit@k, Recall@k, NDCG@k, and R-Precision
  - full 709-fable rankings for every query
- Lexical-overlap shortcut analysis:
  - pair-level content-token IoU vs dense score/rank
  - query-level overlap vs best relevant rank
- Adversarial lexical robustness:
  - original, character-swap, adjective-injection, pre/post injection, and
    combined not-shuffled MORABLES variants
  - TF-IDF and BM25 clustered retrieval metrics with deltas from original corpus

## Directory Map

- `21_lexical_controls/` — lexical baselines, surface-only controls, overlap
  analysis, and generated outputs.
- `22_adversarial_robustness/` — MORABLES adversarial-variant robustness
  checks and generated outputs.

## Next Open Items

- Evaluate dense retriever robustness on MORABLES adversarial variants.
- Analyze error correlations with available metadata or derived features.
- Refresh all headline results on clustered multi-positive metrics.
- Add reproducibility appendix material: prompts, exact checkpoints, splits,
  commands, qrels, summaries, rankings, and evaluation code.
