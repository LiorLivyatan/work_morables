# Next Steps — MORABLES Retrieval Paper
_Updated: April 2026 | Deadline: May 27, 2026_

---

## Where We Stand

| Experiment | Method | Best MRR | Status |
|-----------|--------|----------|--------|
| 01–02 | Baseline embedding models | 0.210 | ✅ Done |
| 03 | LLM summarization (Qwen) | 0.215 | ✅ Done |
| 04 | Gemini reranking | ~0.257 | ⚠️ Small sample only (100 queries) |
| 05 | Instruction-steered embeddings | 0.183 | ✅ Done (worse) |
| 06 | Sentence chunking | 0.151 | ✅ Done (worse) |
| 07 | Gemini "oracle" summaries | 0.360 | ✅ Done |
| 08 | Autoresearch loop | — | 🔄 Barely started |
| 09 | Gemma 4 local (E2B/E4B/31B) | 0.237 | 🔄 In progress |
| Oracle | Fable + ground-truth moral | 0.893 | ✅ (theoretical ceiling) |

---

## Priority 1 — Finish In-Progress

### Exp 09: Gemma 4 (all 3 models)
- E2B: done (Config B / direct_moral MRR=0.237)
- E4B: generated, eval pending
- 31B: generating now (~35 min remaining)
- Key question: does scaling local models approach Gemini quality?

---

## Priority 2 — Quick Wins (1 day each)

### Corpus-Side Instructions
**What:** Apply instructions to fables too, not just morals. Currently only queries get an instruction prefix; fables are embedded raw.
**Try:** `CORPUS_INSTRUCTION = "passage: a fable that teaches a moral lesson"` in Exp 08 autoresearch.
**Why:** E5-instruct was designed for asymmetric instructions; may help the model understand the retrieval context on both sides.
**Caveat:** Linq-Embed-Mistral is query-instruction-only by design — may not help or could hurt.

### Adversarial Robustness (MORABLES built-in)
**What:** Run Linq-Embed-Mistral on the adversarial MORABLES split (18 variants: character swaps, trait injections, tautological morals).
**Why:** Reveals what the model is *actually* matching. If swapping "Fox" → "Dog" destabilizes rankings, the model is matching character names, not moral content. Essential for the error analysis section.
**Implementation:** Load `adversarial` config from the MORABLES dataset, run retrieval per variant type, compare MRR to standard split.

### Distractor Analysis (MORABLES MCQA split)
**What:** For each fable in the MCQA split, embed all 5 candidate morals and rank them. Compute "confusion rate" per distractor type.
**Distractor types:** `similar_characters`, `based_on_adjectives`, `injected_adjectives`, `partial_story`
**Output:** Bar chart — which distractor type fools the model most?
**Why:** This is your mechanistic error analysis. Shows *how* and *why* MRR is stuck at 0.21.

---

## Priority 3 — High Impact (1–2 days each)

### HyDE — Hypothetical Document Embeddings
**What:** Instead of embedding the short moral (12 words) as the query, use an LLM to generate a *hypothetical fable* (~100–150 words) that teaches that moral. Embed the hypothetical fable against the real fable corpus.
**Why powerful:** Directly fixes the vocabulary mismatch — the query becomes the same register as the documents. Matching fable-length text to fable-length text is a fundamentally easier problem for the embedding model.
**Implementation:** Reuse the `generate_summaries.py` infrastructure from Exp 09. One new prompt: "Write a short fable that illustrates this moral: {moral}". Run Gemma 4 31B or Gemini API.
**Expected gain:** Potentially the strongest result after fine-tuning.

### Cross-Encoder Reranking at Full Scale
**What:** Two-stage pipeline — Linq-Embed-Mistral retrieves top-50, then a cross-encoder re-scores each (moral, fable) pair jointly.
**Why:** Exp 04 only tested on 5–100 queries. Full-scale (709) results are needed for the paper. Cross-encoders see both texts simultaneously (like reading comprehension), capturing interactions bi-encoders miss.
**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, ~2 min total at 709 queries)

---

## Priority 4 — Major Contribution (3–5 days)

### Contrastive Fine-Tuning
**Goal:** Explicitly teach an embedding model what "moral similarity" means by training on MORABLES pairs.

#### Training Data Sources

**1. MORABLES itself (709 pairs)**
- Positives: (moral, fable) pairs — 709 examples
- Hard negatives: the MCQA distractors are *perfect* hard negatives — semantically similar but wrong
- ⚠️ Requires 5-fold cross-validation to avoid data leakage (train on 80%, test on 20%)

**2. Other moral/narrative datasets**
| Dataset | Size | Format | Fit |
|---------|------|--------|-----|
| STORAL (ACL 2022) | ~3K | story–moral pairs | High — same format |
| Moral Stories (EMNLP 2021) | 12K | short narrative + moral | High |
| ePiC | ~5K | proverb–narrative pairs | Medium |
| Aesop's Fables (Project Gutenberg) | ~300 | fable–moral pairs | High |

Training on external datasets and evaluating on MORABLES = no leakage, generalizable.

**3. Synthetic data (LLM-generated)**
- Use Gemini/Claude to generate (moral, fable) pairs:
  - Take a real moral → generate 3–5 synthetic fables that teach it (positive pairs)
  - Take a real moral → generate fables with a *different* moral (hard negatives)
- Scale: 709 morals × 5 synthetic fables = ~3,500 training pairs, zero leakage
- This is a strong paper contribution — synthetic data augmentation for moral reasoning

**Architecture:**
- Base: `Linq-Embed-Mistral` or lighter `BAAI/bge-large-en-v1.5`
- Loss: `MultipleNegativesRankingLoss` (in-batch negatives + hard negatives)
- Framework: `sentence-transformers` `SentenceTransformerTrainer`
- Evaluation: MRR on held-out MORABLES fold

---

## Suggested Paper Structure

1. **Introduction** — the 4x gap (0.21 vs 0.89), why this is hard
2. **Related Work** — STS, moral reasoning NLP, retrieval (BEIR, MTEB)
3. **Dataset** — MORABLES, stats, the lexical overlap problem
4. **Methods** — all experiments as a structured ablation
5. **Results** — full table + Gemma 4 scaling curve
6. **Analysis** — adversarial robustness, distractor analysis, error cases
7. **Fine-tuning** — contrastive approach + synthetic data
8. **Conclusion** — what's hard, what works, open questions

---

## Timeline (May 27 Deadline)

| Date | Task |
|------|------|
| Apr 5–6 | Exp 09 complete (E4B + 31B evaluation) |
| Apr 7 | Corpus instructions + adversarial robustness + distractor analysis |
| Apr 8–9 | HyDE implementation + full run |
| Apr 10–11 | Cross-encoder reranking (full 709 scale) |
| Apr 12–17 | Contrastive fine-tuning (MORABLES CV + external/synthetic data) |
| Apr 18–20 | Full results table, all figures, analysis |
| Apr 21 – May 4 | Paper writing (2 weeks) |
| May 5–15 | Advisor review + revisions |
| May 16–26 | Final polish |
| **May 27** | **Submit** |
