---
title: "Qwen3-Embedding with Instruction-Steered Reasoning Layer"
ticket: PAR-5
date: 2026-03-28
status: draft
---

# Qwen3-Embedding with Instruction-Steered Reasoning Layer

## Motivation

Approach B showed that explicitly generating moral summaries with Qwen3.5 and then embedding them with Linq-Embed-Mistral does not beat the baseline (MRR 0.215 vs 0.210). The bottleneck is not the text representation — it's the embedding model's ability to surface abstract moral meaning from narrative text.

Approach C (oracle) showed that when the ground-truth moral is concatenated with the fable, MRR jumps to 0.893 (R@1 = 82.7%). The embedding space *can* represent moral meaning — the model just can't extract it from fable text alone.

**Hypothesis:** Qwen3-Embedding — a decoder-only LLM (Qwen3) fine-tuned for embedding with contrastive learning — can outperform Linq-Embed-Mistral because:
1. It inherits the reasoning capabilities of the Qwen3 foundation model
2. It is instruction-aware: the instruction field acts as a "reasoning layer" that steers the model's hidden representation toward moral meaning
3. It is SOTA on MTEB benchmarks, outperforming Linq-Embed-Mistral's class of models

This follows the professor's framing: "add a reasoning layer above the text — either mechanically, or by prompting the model." Here we test the prompting path.

**Reference paper:** Zhang et al. (2025), "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models" (arXiv:2506.05176v1).

## Task

**Moral → Fable retrieval** (same as Approach B/C): Given a moral statement as query, retrieve the correct fable from a corpus of 709 fables.

## Models

All from the Qwen3-Embedding series, loaded via `transformers` with last-token pooling:

| Model | HuggingFace ID | Size | Embedding Dim |
|-------|----------------|------|---------------|
| Qwen3-Embedding-0.6B | `Qwen/Qwen3-Embedding-0.6B` | 0.6B | 1024 |
| Qwen3-Embedding-4B | `Qwen/Qwen3-Embedding-4B` | 4B | 2560 |
| Qwen3-Embedding-8B | `Qwen/Qwen3-Embedding-8B` | 8B | 4096 |

Compute: M4 Pro (20-core GPU, 64GB unified memory) via MPS backend.

## Instruction Variants

Qwen3-Embedding query format: `Instruct: {instruction}\nQuery:{text}`. Documents are encoded as plain text (no instruction). This follows the paper: "we concatenate the instruction and the query into a single input context, while leaving the document unchanged."

The instruction steers the moral (query) representation. We test 4 variants:

**1. baseline:**
> Given a text, retrieve the most relevant passage that matches this text

**2. moral_focused:**
> Given a moral principle or life lesson, retrieve the fable or parable that teaches this exact lesson through its characters, conflict, and resolution. Focus on the underlying meaning, not surface-level details like character names or settings.

**3. analytical:**
> Given a moral statement about human nature, retrieve the fable that illustrates this principle. Consider: the fable will dramatize this lesson through a narrative arc where characters' choices lead to consequences that reveal the truth of this moral. The fable may use animals, people, or objects as allegories. Look for the structural match between the abstract lesson and the narrative pattern.

**4. abstract:**
> Given a concise moral truth — a statement about virtue, vice, or the human condition — retrieve the fable that serves as its narrative embodiment. The fable will not state this moral explicitly; instead, the moral emerges from the interplay of characters' actions and their consequences. Look past literal content to find the fable whose deeper meaning aligns with this principle.

## Experiment Matrix

3 models x 4 instructions = **12 runs**.

Each run:
1. Load model via `transformers` AutoModel (last-token pooling)
2. Encode 709 fables as plain text → fable embeddings
3. Encode 709 morals with instruction prefix → moral embeddings
4. Cosine similarity matrix → rankings
5. Compute metrics via existing `retrieval_utils.compute_metrics()`

## Metrics

Same as all previous experiments:
- **MRR** (Mean Reciprocal Rank) — primary metric
- **Recall@1, Recall@5, Recall@10**
- **NDCG@10**
- **Mean Rank, Median Rank**

## Comparison Baselines (from existing results, not re-run)

| Approach | Model | MRR | R@1 | R@5 | R@10 |
|----------|-------|-----|-----|-----|------|
| Best embedding | Linq-Embed-Mistral | 0.210 | 14.1% | 28.2% | 36.4% |
| Approach B best | Qwen3.5-9b + CoT summary | 0.215 | 14.1% | 27.9% | 36.1% |
| Approach C oracle | Fable + true moral | 0.893 | 82.7% | — | — |

## Script

**New file:** `scripts/08_qwen3_embedding_retrieval.py`

Follows the same patterns as `07_llm_experiments.py`:
- CLI args: `--models`, `--instructions`, `--sample` (for quick test on N fables)
- Loads data from `data/processed/` (same corpus files)
- Uses `retrieval_utils.compute_metrics()` for evaluation
- Saves results to timestamped run directory
- Generates comparison bar chart

## Output Structure

```
results/qwen3_embedding_runs/
  {YYYY-MM-DD_HH-MM-SS}/
    metadata.json           # model IDs, instructions, git hash, runtime, device
    results.json            # metrics for all model x instruction combos
    comparison_chart.png    # bar chart vs baselines
    predictions/
      {model}_{instruction}_moral_to_fable.json   # per-query rankings
```

## Success Criteria

- Any Qwen3-Embedding configuration beats Linq-Embed-Mistral (MRR > 0.210)
- Detailed instructions outperform baseline instruction (demonstrating the "reasoning layer" hypothesis)
- Larger models benefit more from detailed instructions than smaller ones

## Future Extensions (not in scope)

- Approach 2 ("mechanical" path): generate reasoning chain with base Qwen3, then embed
- Approach 3: Qwen3-Reranker on top of Qwen3-Embedding results
- Fine-tuning Qwen3-Embedding on MORABLES fable-moral pairs
- Other retrieval tasks (fable → moral, augmented corpus)
