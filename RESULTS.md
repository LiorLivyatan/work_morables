# MORABLES Retrieval — Results Summary

**Task:** Given a fable (~117 words), retrieve its correct moral (~10 words) from a pool of 709 candidates using embedding-based cosine similarity.

**Why it's hard:** Lexical overlap between fable and moral is near-zero (mean word IoU = 0.011, median = 0.000). The connection is purely abstract — surface matching fails entirely.

**Oracle ceiling:** 82.7% R@1 (Experiment C — encode morals against themselves).
**Random baseline:** 0.14% R@1 (1/709).

---

## Results by Experiment

| Exp | Approach | Corpus | R@1 | MRR | Notes |
|-----|----------|--------|-----|-----|-------|
| 01 | Off-the-shelf SBERT (MiniLM, MPNet) | full 709 | 3.4% | 0.063 | Barely above random |
| 01 | Improved embedders (e5-large-v2) | full 709 | 5.6% | 0.102 | moral→fable direction |
| 02 | Model sweep — 15 models | full 709 | 5.5% | 0.103 | Qwen3-Embed-0.6B best; all cluster 3–6% |
| 04 | LLM reranking (Gemini 2.5 Pro CoT) | 5-query pilot | 20.0% | 0.201 | Rerank top-10 candidates; not scalable |
| 07 | Gemini corpus summarization (conceptual_abstract) | full 709 | 26.5% | 0.356 | Replace raw fable with moral-style summary |
| **08** | **Symmetric matching + query expansion (B_expand)** | **50-fable pilot** | **70.0%** | **0.782** | **Best result to date** |

---

## Experiment 08 Detail — Symmetric Moral Matching

**Setup:** 50-fable pilot. Baseline = exp07 `conceptual_abstract__summary_only` on same subset (62.0% R@1).

Model: `Linq-AI-Research/Linq-Embed-Mistral`

| Config | Description | R@1 | MRR | Δ vs baseline |
|--------|-------------|-----|-----|---------------|
| **B_expand** | declarative_universal corpus + query expansion | **70.0%** | 0.782 | **+8.0%** |
| RRF_all | Reciprocal Rank Fusion over all 4 configs | 68.0% | **0.789** | +6.0% |
| B | declarative_universal corpus only | 64.0% | 0.743 | +2.0% |
| A_expand | ground_truth_style corpus + query expansion | 64.0% | 0.720 | +2.0% |
| step0_baseline_exp07 | conceptual_abstract (exp07 best) | 62.0% | 0.725 | — |
| A | ground_truth_style corpus only | 58.0% | 0.685 | **−4.0%** |

**Corpus variants:**
- `ground_truth_style` — few-shot guided 5–15 word aphorism matching dataset moral style
- `declarative_universal` — 5–15 word declarative universal truth (no few-shot anchoring)

**Query expansion:** 3 paraphrases per moral (rephrase / elaborate / abstract) generated offline via Gemini. At retrieval: max cosine score across original + 3 paraphrases.

**Fusion:** Reciprocal Rank Fusion (k=60) over ranked lists of all 4 configs.

---

## Key Takeaways

### 1. The bottleneck is the query representation, not the encoder

Swapping raw fable text for a short moral-style summary (exp07) pushed R@1 from ~3% to 26% — a 7× gain — using the same embedding model. The encoder is not broken; it just cannot extract the moral signal from unstructured narrative text unaided.

### 2. Symmetric representation matters most

Exp 08's core insight: if both corpus documents (summaries) and queries (morals) are in the same register — short, declarative, universal — the embedding model matches apples-to-apples. `declarative_universal` summaries (+2% over baseline alone) reflect this better than `ground_truth_style`, which over-anchors to the training distribution.

### 3. Query expansion adds meaningful signal

Paraphrasing the moral query offline (3 variants) and taking max-score at retrieval adds +6% R@1 on top of the best single-query config. This does not violate the no-LLM-at-inference constraint — expansions are generated once and cached.

### 4. RRF smooths ranks but doesn't always win at R@1

RRF-all achieves the best MRR (0.789) — meaning it consistently ranks the correct moral higher across queries — but B_expand leads on R@1 (70% vs 68%). For thesis evaluation, B_expand is the stronger result; RRF is useful when top-5 precision matters.

### 5. ground_truth_style hurts relative to the baseline

Config A (`ground_truth_style`) underperforms the exp07 baseline by 4%. The few-shot examples may cause the model to produce summaries that mimic known morals too closely, introducing false neighbours in embedding space.

### 6. 70% R@1 vs 82.7% oracle ceiling — gap is closing

Starting from 3.4%, the combination of LLM-generated short summaries + query expansion reaches 70% on a 50-fable pilot. The remaining 12.7% gap to the oracle is the target for future work (fine-tuning, full-scale runs).

---

## Open Questions

- Does the 70% result hold on the full 709 fables? (50-fable pilot may not generalise)
- Why does `ground_truth_style` underperform? Inspect generated summaries for over-fitting to known moral phrasings.
- Can dropping Config A from RRF improve R@1 (RRF of B + A_expand + B_expand only)?
- Would fine-tuning `Linq-Embed-Mistral` on MORABLES push past the oracle ceiling?

---

## Progression Chart

```
R@1 (fable→moral, clean corpus)

 3.4%  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Exp01 (SBERT off-the-shelf)
 5.6%  █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Exp01/02 (best embedder)
20.0%  ████████████████░░░░░░░░░░░░░░░░░  Exp04 (LLM reranking, 5-query pilot)
26.5%  ██████████████████████░░░░░░░░░░░  Exp07 (Gemini summarization, full 709)
70.0%  ████████████████████████████████████████████████████████░░░░░░  Exp08 (symmetric + QE, 50-fable)
82.7%  ████████████████████████████████████████████████████████████████████  Oracle ceiling
```
