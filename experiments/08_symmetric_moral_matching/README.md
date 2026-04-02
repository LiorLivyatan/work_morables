# Experiment 08: Symmetric Moral Matching (PAR-11)

## Hypothesis
Experiment 07 enriched fable corpus documents with Gemini summaries but left the moral
queries unchanged. The query (a ~10-word moral) is still matched against paragraph-length
documents. If we convert the corpus to short moral-style sentences (same length/style as
queries) and expand queries with paraphrases, the embedding model matches apples-to-apples
and R@1 should improve significantly.

## Method

### Step 0 — Baseline
Re-run exp 07's R@1-best config (`conceptual_abstract__summary_only`) restricted to the
10-fable pilot subset to establish the formal comparison threshold.

### New Corpus Variants (generate_corpus_summaries.py)
| Variant | Prompt goal |
|---------|-------------|
| `ground_truth_style` | 5–15 word aphorism with few-shot grounding from dataset morals |
| `declarative_universal` | 5–15 word declarative universal truth, few-shot grounded |

### Query Expansion (generate_query_expansions.py)
For each of the 10 moral queries, generate 3 paraphrases (rephrase / elaborate / abstract).
At retrieval, take max cosine score across all 4 query vectors.

### Fusion
Reciprocal Rank Fusion (k=60) over ranked lists from all 5 configs.

## How to run
```bash
# Step 1: Generate corpus summaries for 10 fables
python experiments/08_symmetric_moral_matching/generate_corpus_summaries.py --sample 10

# Step 2: Generate query paraphrases for 10 morals
python experiments/08_symmetric_moral_matching/generate_query_expansions.py --sample 10

# Step 3: Run retrieval + RRF
python experiments/08_symmetric_moral_matching/run.py
```

## Baselines
- exp 07 conceptual_abstract__summary_only: R@1=26.52%, MRR=0.355 (full 709)
- exp 07 conceptual_abstract__summary_only on 10-fable subset: computed in Step 0

## Success Criteria
R@1 on the 10-fable subset (RRF-all config) beats Step 0 baseline on the same subset.

## Key Results
*Pending — run experiment to populate.*
