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

All from the Qwen3-Embedding series, loaded via `sentence-transformers`:

| Model | HuggingFace ID | Size | Embedding Dim |
|-------|----------------|------|---------------|
| Qwen3-Embedding-0.6B | `Qwen/Qwen3-Embedding-0.6B` | 0.6B | 1024 |
| Qwen3-Embedding-4B | `Qwen/Qwen3-Embedding-4B` | 4B | 2560 |
| Qwen3-Embedding-8B | `Qwen/Qwen3-Embedding-8B` | 8B | 4096 |

Compute: M4 Pro (20-core GPU, 64GB unified memory) via MPS backend.

## Instruction Variants

Qwen3-Embedding input format: `{Instruction} {Text}<|endoftext|>`. The instruction steers what the embedding represents. We test 4 instruction variants, applied asymmetrically (different instructions for fable-side and moral-side encoding).

### Fable-side instructions (encoding fables as corpus documents):

**1. baseline:**
> Retrieve the most relevant passage that matches this text

**2. moral_focused:**
> Read this fable carefully. Identify the central moral lesson or life principle it teaches — the underlying truth about human nature, virtue, or vice that the narrative illustrates. Represent the moral meaning, not the surface story.

**3. analytical:**
> Analyze this fable step by step: What conflict or situation does the story present? What choices do the characters make, and what consequences follow? What general principle about human behavior or wisdom does this pattern reveal? Focus your representation on that abstract principle.

**4. abstract:**
> This is a fable — a short narrative that uses characters and events as vehicles for a deeper truth. Look past the literal characters and plot. What universal principle, virtue, or vice is being illustrated? What would a wise reader take away as the enduring lesson? Represent that abstract meaning.

### Moral-side instructions (encoding morals as queries):

**1. baseline:**
> Retrieve the most relevant passage that matches this text

**2. moral_focused:**
> This is an abstract moral principle or life lesson. Find the narrative — a fable or parable — that teaches this exact lesson through its characters and plot.

**3. analytical:**
> This moral statement expresses a truth about human nature. Find the fable whose characters, conflict, and resolution illustrate this principle. The fable may use animals, people, or objects as allegories — look for the structural match between the lesson and the narrative pattern.

**4. abstract:**
> This is a concise moral truth. A fable exists that dramatizes this principle through a specific story. The fable will not state this moral explicitly — instead, the moral emerges from the characters' actions and their consequences. Represent this moral for matching against such narratives.

## Experiment Matrix

3 models x 4 instructions = **12 runs**.

Each run:
1. Load model via `sentence-transformers`
2. Encode 709 fables with fable-side instruction → fable embeddings
3. Encode 709 morals with moral-side instruction → moral embeddings
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
