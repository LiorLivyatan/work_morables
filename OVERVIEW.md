# Moral Retrieval Benchmark — Project Overview

This is a Master's thesis project investigating whether embedding models can perform abstract moral reasoning in an information retrieval setting. The core idea: given a short moral lesson (e.g., *"Gratitude is the sign of noble souls"*), can a retrieval system find the fable that teaches it from a corpus of hundreds of candidates?

This turns out to be surprisingly hard. Morals and fables share almost no vocabulary, so success requires bridging the gap between abstract principles and narrative content — a form of analogical reasoning that current embedding models largely fail at.

## The Research Question

Can embedding-based retrieval systems capture the abstract semantic relationship between a moral lesson and the narrative that teaches it?

So far: barely. The best off-the-shelf model achieves an MRR of 0.210, while an oracle upper bound reaches 0.893 — a 4x gap that none of our interventions have meaningfully closed. This gap is itself a key finding: the information *exists* in the embeddings, but current models can't bridge the abstraction.

## Dataset

We use **MORABLES** (Marcuzzo et al., EMNLP 2025), a curated benchmark of 709 fable-moral pairs drawn from the Western literary tradition (Aesop, Perry, Gibbs, Abstemius). Each entry pairs a narrative fable (~133 words on average) with its attributed moral (~12 words). The lexical overlap between morals and fables is near zero (IoU = 0.011), confirming that keyword matching is useless here.

The dataset is available on [HuggingFace](https://huggingface.co/datasets/cardiffnlp/Morables). The original paper: Marcuzzo, M., Zangari, A., Albarelli, A., Camacho-Collados, J., & Pilehvar, M. T. (2025). *MORABLES: A Benchmark for Assessing Abstract Moral Reasoning in LLMs with Fables*. EMNLP 2025.

## What We've Explored

We frame the task as **moral-to-fable retrieval**: given a moral (query), retrieve the fable that teaches it (document) from all 709 candidates. We evaluate with standard IR metrics (MRR, Recall@k, NDCG).

Our experiments are organized as a series of numbered investigations, each isolating a specific hypothesis:

**Exp 01–02 — Embedding Model Comparison.** We tested 20+ off-the-shelf embedding models. Linq-Embed-Mistral emerged as the strongest (MRR 0.210, R@1 14.1%). Larger models don't necessarily perform better on this task.

**Exp 03 — LLM Summarisation.** We used LLMs (Qwen, Gemini) to generate moral summaries and appended them to fables before embedding. Marginal gains at best (MRR 0.215), suggesting the bottleneck is the embedding space itself, not input quality.

**Exp 04 — LLM Reranking.** Gemini-based chain-of-thought reranking of top retrieval candidates. Uses LLM reasoning to re-score (moral, fable) pairs.

**Exp 05 — Instruction-Steered Embeddings.** Tested Qwen3-Embedding with task-specific instructions and prompt repetition. Counter-intuitively, telling the model what to do made it *worse*.

**Exp 06 — Sentence Chunking.** RAG-style sentence-level retrieval. Morals require holistic fable understanding — chunking fables into sentences hurts performance.

**Exp 07 — SOTA Summarisation Oracle.** Using Gemini 3.1 Pro to generate "perfect" moral summaries, testing whether summary quality or the embedding space is the true bottleneck.

### Key Takeaways

The results point to a fundamental limitation: current embedding models struggle with the abstraction gap between morals and narratives. Lexical methods fail completely, instruction-steering and prompt engineering don't help, and even high-quality LLM summaries barely move the needle. The oracle experiment (MRR 0.893) shows the ceiling is high — the challenge is getting there.

## Planned Directions

**Contrastive fine-tuning** — Train an embedding model on moral-fable pairs with contrastive loss, pushing morals closer to their paired fables in embedding space. Requires careful cross-validation to avoid data leakage.

**Cross-encoder reranking** — A two-stage pipeline: bi-encoder retrieval (top-50) followed by a cross-encoder that jointly scores each (moral, fable) pair, capturing token-level interactions that bi-encoders miss.

**ColBERT / late-interaction models** — Per-token embeddings with MaxSim matching instead of single-vector representations, preserving more fine-grained semantic information.

**Projection layers** — A lightweight MLP to transform moral embeddings into the fable embedding space on top of frozen Linq-Embed-Mistral representations.

**Dataset expansion** — Incorporating additional moral-narrative datasets (STORAL, Moral Stories, ePiC, UniMoral, Project Gutenberg) into a unified format with clustered morals.

**Cross-lingual moral retrieval** — Extending to Hebrew and other languages, testing whether multilingual embeddings can match morals in one language to fables in another.

## Related Work

This project sits at the intersection of several research areas: semantic textual similarity (STS/SemEval benchmarks), description-based similarity (searching by abstract descriptions rather than surface cues), moral reasoning in NLP (MORABLES, Moral Stories, STORAL, UniMoral), and representation learning for retrieval (MTEB, BEIR). A related benchmark, IdioLink, tests idiom retrieval — another setting where literal meaning diverges from abstract meaning.

## Links

| Resource | URL |
|----------|-----|
| Git repository | [github.com/LiorLivyatan/work_morables](https://github.com/LiorLivyatan/work_morables) |
| Shared Google Drive | [M.Sc. Thesis folder](https://drive.google.com/drive/u/0/folders/1cPOxW7FKecQ9gNk6sPYG8Q48kazKsldJ) |
| MORABLES dataset | [HuggingFace](https://huggingface.co/datasets/cardiffnlp/Morables) |
| MORABLES paper | Marcuzzo et al., EMNLP 2025 |
