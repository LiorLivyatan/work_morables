# MORABLES Retrieval — Project Summary

## What Is This Project?

This is a Master's thesis project investigating whether embedding models can perform **abstract moral reasoning** in a retrieval setting. We use the MORABLES dataset — 709 fable-moral pairs from the Western literary tradition — and frame it as an information retrieval problem: given a short moral (e.g., *"Gratitude is the sign of noble souls"*), retrieve the fable that teaches it from a corpus of 709 candidates.

This is hard because morals and fables share almost no vocabulary (lexical overlap IoU = 0.011). Success requires a model to bridge the gap between abstract principles and narrative content — a form of analogical reasoning that current embedding models largely fail at.

## The Core Question

**Can embedding-based retrieval systems capture the abstract semantic relationship between a moral lesson and the narrative that teaches it?**

The answer, so far, is: barely. The best off-the-shelf embedding model (Linq-Embed-Mistral) achieves MRR 0.210, while an oracle that concatenates the ground-truth moral with the fable reaches MRR 0.893 — a 4x gap that none of our interventions have meaningfully closed.

## The Dataset

**MORABLES** (Marcuzzo et al., EMNLP 2025) is a curated benchmark of 709 fable-moral pairs drawn from historical sources (Aesop, Perry, Gibbs, Abstemius). Each entry contains a fable (avg. 133 words) and its attributed moral (avg. 12 words). The dataset also includes adversarial variants and an MCQA formulation, though our work focuses exclusively on the retrieval task.

| Statistic | Value |
|-----------|-------|
| Fable-moral pairs | 709 |
| Unique morals | 678 (27 shared across fables) |
| Avg. fable length | 133 words |
| Avg. moral length | 12 words |
| Lexical overlap (IoU) | 0.011 |

- Dataset: [cardiffnlp/Morables on HuggingFace](https://huggingface.co/datasets/cardiffnlp/Morables)
- Paper: Marcuzzo, M., Zangari, A., Albarelli, A., Camacho-Collados, J., & Pilehvar, M. T. (2025). *MORABLES: A Benchmark for Assessing Abstract Moral Reasoning in LLMs with Fables*. EMNLP 2025.

## What We've Explored

We structure our work as a series of numbered experiments, each isolating a specific hypothesis:

| # | Experiment | Method | Best MRR | Finding |
|---|-----------|--------|----------|---------|
| 01 | Baselines | Small embedding models | — | Established lower bounds |
| 02 | Model comparison | 20+ embedding models | **0.210** | Linq-Embed-Mistral is the strongest |
| 03 | LLM summarisation | LLM-generated moral summaries appended to fables | 0.215 | Marginal gain; summary quality is not the bottleneck |
| 04 | LLM reranking | Gemini CoT reranking | — | LLM reasoning on candidates |
| 05 | Qwen3-Embedding | Instruction-steered embeddings + prompt repetition | 0.183 | Task-specific instructions *hurt* performance |
| 06 | Sentence chunking | RAG-style sentence-level retrieval | 0.151 | Morals need full fable context; chunking hurts |
| 07 | SOTA summarisation oracle | Gemini 3.1 Pro "perfect" summaries | — | Tests whether bottleneck is summary quality vs. embedding space |
| 08 | Autoresearch loop | Automated agent-driven experiment loop | — | In progress |

### Key Findings So Far

- **Lexical methods fail completely** — near-zero vocabulary overlap between morals and fables.
- **Off-the-shelf embeddings plateau around MRR 0.21** — Linq-Embed-Mistral is the best of 20+ models tested.
- **Task-specific instructions hurt** — counter-intuitively, telling the model what to do makes it worse (Exp 05).
- **Sentence chunking hurts** — morals require holistic fable understanding, not fragment matching (Exp 06).
- **LLM summarisation gives marginal gains** — the bottleneck appears to be the embedding space, not input quality.
- **The oracle gap is massive** — MRR 0.21 vs. 0.89 suggests the information *exists* in embeddings but current models can't bridge the abstraction gap.

## Planned Directions

- **Contrastive fine-tuning** — train an embedding model on moral-fable pairs with cross-validation
- **Cross-encoder reranking** — two-stage pipeline with bi-encoder retrieval + cross-encoder reranking
- **ColBERT / late-interaction models** — per-token matching instead of single-vector representations
- **Projection layers** — lightweight MLP to map moral embeddings into fable embedding space

See `docs/future_directions.md` for detailed descriptions.

## Links

| Resource | Link |
|----------|------|
| Code repository | [github.com/LiorLivyatan/work_morables](https://github.com/LiorLivyatan/work_morables) |
| Shared Drive | [Google Drive folder](https://drive.google.com/drive/u/0/folders/1cPOxW7FKecQ9gNk6sPYG8Q48kazKsldJ) |
| MORABLES dataset | [HuggingFace](https://huggingface.co/datasets/cardiffnlp/Morables) |
| MORABLES paper | Marcuzzo et al., EMNLP 2025 |

## Repo Structure

```
work_morables/
├── data/raw/              # Original MORABLES dataset (709 fables + adversarial variants)
├── data/processed/        # Retrieval-formatted corpus and relevance judgments (qrels)
├── experiments/           # Numbered experiment directories (each with run.py + results/)
├── lib/                   # Shared utilities (metrics, data loading, embedding cache)
├── docs/                  # Walkthroughs and future directions
├── papers/                # Reference PDFs
├── meeting_materials/     # Advisor meeting prep
└── scripts/               # One-off utility scripts
```
