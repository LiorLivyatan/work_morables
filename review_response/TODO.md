# MoraLink Review-Response TODO

This checklist translates the IdioLink rejection feedback into concrete work for the MoraLink paper.

## P0 — Paper Framing And Validity

- [x] Use one benchmark name consistently: `MoraLink`.
- [x] Remove IdioLink as a central related-work dependency.
- [x] Replace strong "understanding" claims with retrieval/alignment claims.
- [x] Rename the "oracle ceiling" to a gold-moral appended upper bound.
- [x] Fill all basic MORABLES statistics in the introduction.
- [x] Make clustered multi-positive qrels the official benchmark definition.
- [x] Explain why single-label scoring is insufficient for fables with equivalent or near-equivalent morals.

## P1 — Shortcut And Control Analyses

- [x] Add lexical baselines: BM25 and TF-IDF.
- [x] Measure whether lexical overlap predicts rank or score.
- [x] Add surface-only baselines: title-only, first-sentence-only, and truncated fable.
- [x] Evaluate lexical robustness on MORABLES adversarial variants, especially character swaps and adjective injections.
- [ ] Evaluate dense retriever robustness on MORABLES adversarial variants.
- [ ] Analyze whether errors correlate with character overlap, theme overlap, fable length, or attractor fables.

## P1 — Main Results Refresh

- [ ] Recompute all headline results on the clustered benchmark.
- [ ] Use multi-positive metrics: MRR, MAP, Hit@k, Recall@k, NDCG@k, and R-Precision.
- [ ] Keep full ranked lists for every official result.
- [ ] Separate benchmark results on raw fables from LLM-summary interventions.
- [ ] Use only full-corpus results in the main paper; move pilot results to appendix or omit.

## P2 — Diagnostic Analysis Section

- [ ] Add a rank-distribution figure/table.
- [ ] Add a score-gap analysis for near misses vs confident errors.
- [ ] Add qualitative examples of ambiguity corrected by clustered qrels.
- [ ] Add qualitative examples of remaining genuine retrieval failures.
- [ ] Report performance by cluster type: singleton, exact duplicate, near-paraphrase.
- [ ] Report performance by source or fable family if metadata supports it.

## P2 — Reproducibility And Artifacts

- [ ] Add appendix with all LLM prompts for summaries and query expansions.
- [ ] List exact model checkpoints, licenses, and inference settings.
- [ ] Document split construction and cluster construction.
- [ ] Release processed qrels, generated summaries, rankings, and evaluation code.
- [ ] Include exact `run.sh` commands for main experiments.

## P3 — Optional Method Extensions

- [ ] Add cluster-aware fine-tuning as the main supervised method if results are stable.
- [ ] Evaluate hard negatives from MORABLES MCQA distractors by distractor type.
- [ ] Consider cross-encoder reranking only if it becomes part of the main story.
- [ ] Consider LLM-as-retriever results only as a separate comparison, not as the core benchmark claim.
