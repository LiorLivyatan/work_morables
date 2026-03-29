# Experiment 06: Sentence-Level Chunking Retrieval

## What it tests
Whether splitting fables into individual sentences and matching morals against sentence-level embeddings improves retrieval. Inspired by RAG chunking strategies.

## Method
Instead of one embedding per fable, split each fable into sentences and embed each separately. For each moral query, compute similarity to all sentence embeddings, then aggregate per-fable to produce a fable-level score.

**Chunking strategies:** sentence, last_N, sliding window
**Aggregation methods:** max, top-k mean, weighted (favor later sentences)

## How to run
```bash
python experiments/06_sentence_chunking/run.py
python experiments/06_sentence_chunking/run.py --strategies sentence__max last_1__max
```

## Key results

| Strategy | Chunks | MRR | R@1 | vs Baseline |
|----------|--------|-----|-----|-------------|
| full_fable (baseline) | 709 | **0.210** | **14.1%** | — |
| sliding_3_2__max | 1,965 | 0.180 | 10.7% | -14% |
| sentence__max | 3,574 | 0.151 | 8.5% | -28% |
| last_1__max | 709 | 0.130 | 7.8% | -38% |
| sentence__top_3_mean | 3,574 | 0.108 | 6.3% | -48% |

## Key findings
- **Every chunking strategy is worse than full-fable baseline**
- More chunks = more noise from false positive matches
- The model needs the entire narrative context to capture moral signal
- Last-sentence-only (`last_1__max`) is particularly bad — most fables don't explicitly state the moral in a single separable sentence
- Sliding window (3 sentences) was closest to baseline, preserving some context
