# ParabeLink — Paper Structure

> ARR May 2026 → ACL 2026 | 8 pages main body + unlimited references

---

## Abstract (~150 words) — *write last*

---

## §1 Introduction (~1.5 columns)

- **Para 1** — Broad: moral reasoning and narrative understanding pose a fundamental challenge...
- **Para 2** — Narrow: retrieval across divergent surface forms (moral ≈ 12 words, fable ≈ 133 words, lexical IoU = 0.011)
- **Para 3** — Gap: no benchmark exists for this; strong off-the-shelf models plateau at MRR 0.210
- **Para 4** — "We introduce **ParabeLink**, a benchmark designed to..."
- **Para 5** — Contributions:
  1. We introduce the benchmark and formalize the task
  2. We evaluate off-the-shelf and fine-tuned embedding models under a rigorous 5-fold CV protocol
  3. We show that hard negatives and false-negative masking together close the gap significantly — yet a large oracle shortfall remains

---

## §2 Related Work (~1.5 columns)

### 2.1 Moral Reasoning and Figurative Language in NLP
Story understanding, moral inference, fable comprehension, MORABLES (Marcuzzo et al., 2025).

### 2.2 Semantic Retrieval and Embedding Models
Dense retrieval (SBERT, BGE, Linq-Embed-Mistral), instruction-following embedders, summarization-augmented retrieval.

### 2.3 Contrastive Fine-Tuning and Data Augmentation
InfoNCE / MultipleNegativesRankingLoss, STORAL (Guan et al., NAACL 2022), hard negative mining, false-negative handling in dense retrieval.

---

## §3 The ParabeLink Benchmark (~2 columns)

### 3.1 Problem Formulation
Formal notation: query `q` = moral statement, corpus `C` = 709 fables, retrieve `f*`.
Why it's hard: near-zero lexical overlap (IoU = 0.011), abstraction gap, 709-way retrieval.

### 3.2 The MORABLES Dataset
Source: Marcuzzo et al. (2025). Stats: 709 pairs, avg 133 / 12 words, genre diversity.

> **Figure 1:** Illustrative moral ↔ fable pair showing the lexical divergence.

### 3.3 Measuring Task Difficulty
Random baseline R@1 = 0.14%. Oracle ceiling R@1 = 82.7% (Gemini re-ranker on gold moral text).
This oracle gap is the paper's central motivation.

---

## §4 Experiments (~1.5 columns)

### 4.1 Evaluation Protocol
5-fold GroupKFold CV; metrics: MRR (primary), R@1, R@5, NDCG@10, ± std across folds.

### 4.2 Off-the-Shelf Retrieval Baselines
Zero-shot embedders: BGE-base (≈ 0.15 MRR), **Linq-Embed-Mistral = 0.210 MRR** (best).
Summarization pipeline: Gemini CoT-proverb summary → embed → retrieve (R@1 = 26.5%).
Shows that document representation matters before any fine-tuning.

### 4.3 Contrastive Fine-Tuning
| Model | Experiment | MRR | Note |
|---|---|---|---|
| BGE-base (110M), full FT | ft_01 | 0.122 | Worse than Linq zero-shot → capacity bottleneck |
| Linq + LoRA (7B) | ft_02 | 0.318 | LoRA r=64 on q/k/v/o, InfoNCE, 5-fold CV |

Both use identical splits for direct comparison.

### 4.4 STORAL Data Augmentation
`ft_04 / ft_05` — mixing STORAL (moral, story) pairs into training.
Ratio sweep: 0 / 200 / 500 / 1000 / 1675 pairs.
Best gain is marginal (+0.010 MRR at ratio=200 for BGE).
`ft_07` — STORAL pre-training then MORABLES fine-tune: Linq-s500 = 0.357 MRR.
Shows out-of-domain augmentation does not substitute for task-specific signal.

### 4.5 Hard Negatives
`ft_03` — Type-2 moral negatives: fables paired with semantically similar (but wrong) morals, mined via adjective injection.
**MRR 0.432** — best variant: injected-adjective negatives.

### 4.6 False-Negative Masking
`ft_09` — Similarity matrix reveals ~15% of moral pairs are near-duplicates (threshold 0.85).
These are masked to −∞ in the InfoNCE denominator rather than treated as true negatives.
**MRR 0.435 ± 0.055** — modest absolute gain, improved training stability.

### 4.7 Transfer from Concept-Level Tasks *(ft_10 — in progress)*
Motivation: IdioLink shares the same *align-meaning-across-form* inductive prior.
Four strategies:
- **Option A** — Zero-shot transfer (fine-tune on IdioLink only, eval on MORABLES)
- **Option B** — Sequential pre-training (IdioLink → MORABLES)
- **Option C** — Data mixing (IdioLink triplets mixed into MORABLES batches)
- **Option D** — Hard-negative curriculum transfer (IdioLink curriculum design applied to MORABLES)

> Present whichever options are completed by submission. §4.7 can be cut cleanly if ft_10 is not ready.

---

## §5 Results and Discussion (~2 columns)

### 5.1 Main Results Table

| Model / Method | MRR | R@1 | R@5 | NDCG@10 |
|---|---|---|---|---|
| *Oracle ceiling* | — | 82.7% | — | — |
| Random baseline | — | 0.14% | — | — |
| BGE-base, zero-shot | ~0.15 | — | — | — |
| Linq-Embed-Mistral, zero-shot | 0.210 | — | — | — |
| Gemini summarization pipeline | — | 26.5% | — | — |
| BGE-base, fine-tuned (ft_01) | 0.122 | — | — | — |
| Linq + LoRA (ft_02) | 0.318 | — | — | — |
| Linq + LoRA + Hard Neg (ft_03) | 0.432 | — | — | — |
| Linq + LoRA + Hard Neg + FN Mask (ft_09) | **0.435** | — | — | — |
| IdioLink transfer (ft_10) | TBD | — | — | — |

### 5.2 Findings

**First, model capacity is a prerequisite for moral abstraction.**
BGE fine-tuned (0.122) underperforms Linq zero-shot (0.210). Scaling from 110M to 7B parameters yields a larger gain than fine-tuning alone on the smaller model.

**Second, hard negatives are the single largest lever.**
ft_03 (+0.114 MRR over ft_02) shows that exposing the model to morally adjacent fables as explicit negatives forces it to learn moral specificity rather than thematic proximity.

**Third, false-negative masking stabilizes training on a semantically dense corpus.**
~15% of MORABLES moral pairs share cosine similarity > 0.85, making standard InfoNCE noisy. Masking these improves fold consistency.

**Fourth, data augmentation from related corpora has limited and asymmetric returns.**
STORAL mixing offers marginal gains at low ratios and degrades at high ratios. STORAL pre-training (ft_07, Linq-s500 = 0.357) underperforms directly fine-tuned Linq (ft_02 = 0.318) — domain shift from story-moral to fable-moral narrows the transfer benefit.

**Fifth, a persistent oracle gap remains.**
Best fine-tuned MRR = 0.435, yet oracle R@1 = 82.7%. This gap is the benchmark's core challenge for future work.

**Summary of key observations.**
- *(bulleted list — write after all sections are drafted)*

---

## §6 Conclusions (~0.5 column)

"We introduce **ParabeLink**, a benchmark..."
One sentence per contribution.
Future directions: better oracle elicitation, multilingual extension, LLM-as-retriever approaches.

---

## Limitations *(unnumbered)*
English-only, 709-fable corpus, closed-set retrieval, oracle uses proprietary LLM (Gemini).

## Ethics Statement *(unnumbered)*
MORABLES is a public dataset; no PII; research purpose only.

## References

---

## Appendix A — Hyperparameters
Full training configuration for all experiments (LoRA rank, LR, batch size, early stopping patience).

## Appendix B — Full Per-Fold Results
Per-fold MRR breakdown for all fine-tuning experiments.

## Appendix C — LLM Prompts
Gemini CoT-proverb summarization prompt used in the baseline pipeline.

## Appendix D — STORAL Pre-Training Details
Full ft_07 results table (BGE / Linq / Qwen3 × s500 / s1000 / sfull).

---

## Experiment → Section Map

| Experiment | Description | Result | Maps to |
|---|---|---|---|
| Off-the-shelf zero-shot | BGE, Linq, Gemini summ. | Linq MRR 0.210 / Gemini R@1 26.5% | §4.2 + §5 Finding 1 |
| ft_01 | BGE full fine-tune | MRR 0.122 | §4.3 + §5 Finding 1 |
| ft_02 | Linq + LoRA | MRR 0.318 | §4.3 + §5 Main Table |
| ft_03 | Linq + LoRA + Hard Neg | MRR 0.432 | §4.5 + §5 Finding 2 |
| ft_04 / ft_05 | STORAL data mixing (ratio sweep) | +0.010 MRR at best | §4.4 + §5 Finding 4 |
| ft_06 | Hyperparameter random search | — | Appendix A |
| ft_07 | STORAL pre-training | Linq-s500 MRR 0.357 | §4.4 + §5 Finding 4 |
| ft_08 | Clean Linq+LoRA sanity check | — | Folded into ft_03 or Appendix |
| ft_09 | FN masking via sim matrix | MRR 0.435 ± 0.055 | §4.6 + §5 Finding 3 |
| ft_10 | IdioLink transfer (4 options) | TBD | §4.7 + §5 (if done) |

---

## Writing Order

1. §1 Introduction
2. §2 Related Work
3. §3 Benchmark (Problem Formulation → Dataset → Oracle)
4. §4 Experiments
5. §5 Results and Discussion
6. §6 Conclusions
7. Limitations + Ethics Statement
8. Abstract (last — written after everything else)
9. Appendices
