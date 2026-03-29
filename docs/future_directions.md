# Future Directions for Moral-to-Fable Retrieval

**Context:** Best real method is Linq-Embed-Mistral (MRR=0.210, R@1=14.1%). Oracle upper bound is MRR=0.893 when ground-truth moral is concatenated with fable. All instruction-steering and LLM summarisation approaches have failed to beat the baseline.

---

## Direction 1: Contrastive Fine-Tuning

Fine-tune an embedding model on moral-fable pairs using contrastive loss (e.g., `MultipleNegativesRankingLoss`). Push moral embeddings closer to their paired fable embeddings and away from other fables.

**Important:** Must use k-fold cross-validation (train on 80%, test on 20%) or train on external datasets (STORAL, Moral Stories) and evaluate on MORABLES as unseen test set. Training and testing on the same 709 pairs would be data leakage.

**Status:** Not started

---

## Direction 2: Cross-Encoder Reranking (Pipeline)

Two-stage pipeline:
1. Stage 1: Linq-Embed-Mistral retrieves top-50 candidates (fast bi-encoder)
2. Stage 2: Cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-12-v2`) scores each (moral, fable) pair jointly and reranks

Cross-encoders see query+document together as a single input, capturing token-level interactions that bi-encoders miss. Small BERT-sized models, not LLMs.

**Status:** Not started

---

## Direction 3: ColBERT / Late-Interaction Retrieval

Use per-token embeddings with MaxSim matching instead of single-vector representations. Models like `colbert-ir/colbertv2.0` keep token-level granularity, so a moral keyword like "honesty" matches against every token in the fable.

**Status:** Not started

---

## Direction 4: Projection Layer

Train a lightweight MLP (2-3 linear layers) on frozen Linq-Embed-Mistral embeddings to transform moral embeddings into the fable embedding space. Same train/test leakage concern as Direction 1 — needs cross-validation or external training data.

**Status:** Not started

---

## Direction 5: Hybrid BM25 + Dense Retrieval

Combine BM25 keyword matching with dense retrieval using rank fusion. **Likely low impact** — vocabulary overlap between abstract morals and narrative fables is near zero.

**Status:** Deprioritised

---

## Direction 6: Sentence-Level Chunking (RAG-style)

Instead of one embedding per fable, embed each sentence separately. Many Aesop fables state the moral explicitly in the last line. A moral query matches against individual sentences rather than the full narrative.

No training needed — just a different indexing strategy with the same embedding model.

**Status:** Starting next

---

## Direction 7: Reasoning Model Query Expansion

Use a reasoning model (Claude, Gemini) to expand a short moral into a richer description or hypothetical fable, then embed the expansion as the query. Related to HyDE (Hypothetical Document Embeddings).

**Status:** Not started (discussed, deferred)

---

## Direction 8: Prompt Repetition (Leviathan et al., 2025)

Repeat the query prompt (×2, ×3) so that later tokens in the causal decoder can attend to earlier ones. Based on "Prompt Repetition Improves Non-Reasoning LLMs" paper. Applied to Qwen3-Embedding-8B with baseline instruction.

**Status:** In progress (running)
