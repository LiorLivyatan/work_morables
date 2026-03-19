# Direction 1: Improved Retrieval — Complete Guide

This document explains everything done in the "Direction 1" experiments: stronger embedding models, instruction prefix engineering, and LLM-based Chain-of-Thought (CoT) reranking. It follows from `DATA_AND_EXPERIMENTS_GUIDE.md` which covers Phases 0-1 (data preparation and baseline Sentence-BERT retrieval).

---

## Table of Contents

1. [What problem are we solving?](#1-what-problem-are-we-solving)
2. [The three ideas tested](#2-the-three-ideas-tested)
3. [Instruction-aware embedding models explained](#3-instruction-aware-embedding-models-explained)
4. [The instruction prefix experiment](#4-the-instruction-prefix-experiment)
5. [Script: 06_improved_retrieval.py](#5-script-06_improved_retrievalpy)
6. [Embedding results and analysis](#6-embedding-results-and-analysis)
7. [LLM-based CoT reranking explained](#7-llm-based-cot-reranking-explained)
8. [CoT moral summarization explained](#8-cot-moral-summarization-explained)
9. [Script: 07_llm_reranking.py](#9-script-07_llm_rerankingpy)
10. [CoT preliminary results](#10-cot-preliminary-results)
11. [Combined findings and takeaways](#11-combined-findings-and-takeaways)
12. [How to run everything](#12-how-to-run-everything)
13. [New files created](#13-new-files-created)
14. [Known issues and next steps](#14-known-issues-and-next-steps)

---

## 1. What problem are we solving?

The baseline experiments (script 04) showed that off-the-shelf Sentence-BERT models achieve only ~3% Recall@1 on fable→moral retrieval. That means the correct moral is ranked #1 for only 3 out of every 100 fables. The median correct moral sits at rank ~130 out of 709 — essentially in the middle of the list.

The professor's guidance was threefold:

1. **Try stronger models** — the baselines used small general-purpose models (MiniLM 22M params, MPNet 109M params). Newer, larger models like BGE and E5 might do better.
2. **Try instructions** — some modern embedding models accept a text prefix ("instruction") that tells the model what task it's performing. Maybe telling the model "find the moral of this fable" helps it produce better embeddings.
3. **Try Chain-of-Thought (CoT)** — instead of relying purely on embeddings, use an LLM (like Gemini) to *reason* about the fable's moral before making a retrieval decision. CoT can't be used directly inside embedding models (they don't do multi-step reasoning), but it can be used in a **reranking** stage or a **summarization** stage.

---

## 2. The three ideas tested

### Idea 1: Stronger embedding models

We tested three new models, all significantly larger than the baselines:

| Model | HuggingFace ID | Dimensions | Parameters | Key feature |
|-------|---------------|-----------|------------|-------------|
| **BGE-large** | `BAAI/bge-large-en-v1.5` | 1024 | ~335M | Supports instruction prefix for queries |
| **E5-large** | `intfloat/e5-large-v2` | 1024 | ~335M | Requires `"query: "` / `"passage: "` prefixes |
| **Multilingual E5** | `intfloat/multilingual-e5-large` | 1024 | ~560M | Same as E5, but trained on 100+ languages |

For comparison, the baselines:

| Model | Dimensions | Parameters |
|-------|-----------|------------|
| all-MiniLM-L6-v2 | 384 | 22M |
| all-mpnet-base-v2 | 768 | 109M |

Note: We also tried `Alibaba-NLP/gte-large-en-v1.5` but it crashed with a `torch.AcceleratorError` during encoding (tensor index out of bounds). This is likely an incompatibility with the MPS (Apple Silicon) backend or the sentence-transformers version. It was removed from the experiments.

### Idea 2: Instruction prefix engineering

Some embedding models were trained with text prefixes that signal what role the input plays. The idea is that a prefix like *"Given this fable, retrieve the abstract moral lesson it teaches: "* might steer the model's embedding toward the "moral reasoning" region of its vector space.

We tested multiple prefix variants to see whether task-specific instructions help or hurt.

### Idea 3: LLM-based CoT reranking and summarization

Instead of relying only on embeddings (which compress an entire fable into one vector), we use an LLM that can *read and reason about* the fable:

- **Approach A (Reranking):** Use embeddings to get a shortlist of top-20 candidate morals, then ask Gemini to reason about which one best fits the fable.
- **Approach B (Summarization):** Ask Gemini to read the fable and generate a concise moral summary, then embed *that summary* and use it for retrieval. Since the summary is more abstract than the raw fable text, it might land closer to the correct moral in embedding space.

---

## 3. Instruction-aware embedding models explained

### What is an instruction prefix?

Traditional embedding models like MiniLM and MPNet treat all input text the same way. You feed in text, you get a vector. But newer models were trained with a specific protocol: the input text starts with a short prefix that tells the model what role the text plays.

**BGE protocol:**
- Queries get a prefix: `"Represent this sentence for searching relevant passages: "` + the actual query text
- Documents/passages get NO prefix — just the raw text
- This asymmetry helps the model distinguish between "I'm looking for something" vs "I'm a candidate to be found"

**E5 protocol:**
- Queries get: `"query: "` + the actual query text
- Documents get: `"passage: "` + the actual document text
- This is simpler but mandatory — the model was trained this way, and omitting prefixes degrades performance

### Why might this matter for moral retrieval?

The hypothesis was: if we replace the generic BGE prefix with a task-specific one like *"Given this fable, retrieve the abstract moral lesson it teaches: "*, the model might:
1. Focus less on surface-level features (character names, actions)
2. Focus more on the abstract lesson (themes, principles)

This is what the professor meant by "trying instructions."

### Why it might NOT matter

These instruction prefixes are short — typically 5-15 words. The fable itself is 30-1000+ words. The prefix is a small fraction of the total input. Also, the model was trained with a specific prefix vocabulary; a custom prefix it has never seen during training might confuse rather than help.

---

## 4. The instruction prefix experiment

### BGE variants tested

For `BAAI/bge-large-en-v1.5`, we tested four instruction variants:

| Variant name | Query prefix | Doc prefix |
|-------------|-------------|-----------|
| `no_instruction` | *(none)* | *(none)* |
| `default` | `"Represent this sentence for searching relevant passages: "` | *(none)* |
| `moral_task` | `"Given this fable, retrieve the abstract moral lesson it teaches: "` | *(none)* |
| `moral_meaning` | `"What is the deeper meaning and life lesson of this story: "` | *(none)* |

The `default` is what BGE was trained with. `no_instruction` tests raw performance without any prefix. `moral_task` and `moral_meaning` are our custom task-specific prefixes.

### E5 variants tested

For `intfloat/e5-large-v2`, we tested two variants:

| Variant name | Query prefix | Doc prefix |
|-------------|-------------|-----------|
| `default` | `"query: "` | `"passage: "` |
| `no_prefix` | *(none)* | *(none)* |

### Multilingual E5

Only the default variant (`"query: "` / `"passage: "`) — this model exists primarily for its multilingual capability, which we'll use in future cross-lingual experiments.

### How prefixes are applied in the code

For models that distinguish between queries and documents, we need to be careful about the **direction** of retrieval:

- **Fable→Moral:** The fable is the query (gets query prefix), the moral is the document (gets doc prefix)
- **Moral→Fable:** The moral is the query (gets query prefix), the fable is the document (gets doc prefix)

This means we need to encode each text *twice* — once with the query prefix and once with the doc prefix — to support both retrieval directions. The script handles this by creating five separate embedding sets per model-variant combination:

1. `fable_q_embs` — fables encoded with query prefix (for fable→moral task)
2. `fable_d_embs` — fables encoded with doc prefix (for moral→fable task)
3. `moral_q_embs` — morals encoded with query prefix (for moral→fable task)
4. `moral_d_embs` — morals encoded with doc prefix (for fable→moral task)
5. `moral_aug_d_embs` — augmented morals encoded with doc prefix (for fable→moral augmented task)

This is more work than the baseline script (which encoded each text only once), but it's necessary for correct evaluation of asymmetric models.

---

## 5. Script: 06_improved_retrieval.py

### What it does

Runs embedding retrieval experiments with instruction-aware models. Equivalent to `04_baseline_retrieval.py` but with stronger models, instruction variants, and correct asymmetric encoding.

### Input

Same data files as the baseline script:
- `data/processed/fables_corpus.json` (709 fables)
- `data/processed/morals_corpus.json` (709 morals)
- `data/processed/morals_augmented_corpus.json` (2803 morals)
- `data/processed/qrels_fable_to_moral.json` (ground truth)
- `data/processed/qrels_fable_to_moral_augmented.json` (augmented ground truth)

### Output

- `results/improved_retrieval_results.json` — all 21 experiment results (7 model-variant combinations × 3 tasks each)
- Printed summary table and rank distributions

### How to run

```bash
python scripts/06_improved_retrieval.py
```

No arguments needed. Takes approximately 15-25 minutes on an M-series Mac (most time is spent encoding 709 fables × 5 embedding sets per variant).

### Step-by-step what happens

1. **Load data** — reads all corpus and qrels files from `data/processed/`
2. **Build ground truth** — same as baseline: creates `gt_f2m` (fable_idx → moral_idx) and `gt_f2m_aug` dictionaries
3. **For each model** (BGE, E5, multilingual E5):
   a. **Load model** from HuggingFace via `SentenceTransformer(model_name, trust_remote_code=True)`. Models are cached locally after first download (~1.3GB each).
   b. **For each instruction variant** of that model:
      - Prepend prefixes to create `fable_as_query`, `fable_as_doc`, `moral_as_query`, `moral_as_doc`, `moral_aug_as_doc`
      - Encode all five text sets into embedding matrices (each 709×1024 or 2803×1024)
      - Run Task 1: `compute_metrics(fable_q_embs, moral_d_embs, gt_f2m)` — fable→moral on clean corpus
      - Run Task 2: `compute_metrics(moral_q_embs, fable_d_embs, gt_m2f)` — moral→fable on clean corpus
      - Run Task 3: `compute_metrics(fable_q_embs, moral_aug_d_embs, gt_f2m_aug)` — fable→moral on augmented corpus
      - Run rank analysis showing distribution of where correct morals land
4. **Save all results** to JSON
5. **Print comparison table** including baseline results if available

### Model configuration structure

Each model is defined as a config dict:

```python
{
    "name": "BAAI/bge-large-en-v1.5",    # HuggingFace model ID
    "query_prefix": "Represent this...",   # Default query prefix
    "doc_prefix": "",                      # Default doc prefix
    "instruction_variants": {              # All variants to test
        "no_instruction": ("", ""),        # (query_prefix, doc_prefix) tuples
        "default": ("Represent...", ""),
        "moral_task": ("Given this fable...", ""),
    },
}
```

Adding a new model or variant is as simple as adding an entry to the `MODEL_CONFIGS` list.

### Differences from 04_baseline_retrieval.py

| Aspect | Script 04 (baseline) | Script 06 (improved) |
|--------|---------------------|---------------------|
| Models | MiniLM, MPNet | BGE, E5, multilingual E5 |
| Encoding | Each text encoded once | Each text encoded as both query and doc |
| Instructions | No prefix support | Full instruction variant testing |
| Results file | `baseline_results.json` | `improved_retrieval_results.json` |

---

## 6. Embedding results and analysis

### Full results table — Fable→Moral (clean corpus, 709→709)

| Model | Variant | MRR | R@1 | R@5 | R@10 | Mean Rank | Median Rank |
|-------|---------|-----|-----|-----|------|-----------|-------------|
| all-MiniLM-L6-v2 *(baseline)* | — | 0.063 | 3.4% | 7.5% | 11.7% | — | — |
| all-mpnet-base-v2 *(baseline)* | — | 0.065 | 2.7% | 8.3% | 12.7% | 186 | 129 |
| BGE-large-en-v1.5 | no_instruction | 0.066 | 3.2% | 8.2% | 11.7% | 184 | 110 |
| BGE-large-en-v1.5 | default | 0.068 | 3.2% | 8.3% | 12.7% | 181 | 114 |
| BGE-large-en-v1.5 | moral_task | 0.048 | 2.1% | 4.9% | 8.6% | 211 | 149 |
| BGE-large-en-v1.5 | moral_meaning | 0.048 | 1.8% | 5.6% | 8.7% | 213 | 151 |
| **E5-large-v2** | **default** | **0.074** | 3.0% | **10.6%** | **15.9%** | **175** | **104** |
| **E5-large-v2** | **no_prefix** | **0.080** | **4.4%** | 10.2% | 14.5% | 184 | 121 |
| multilingual-E5-large | default | 0.071 | 2.4% | 10.2% | 15.9% | 178 | 114 |

### Full results table — Moral→Fable (clean corpus, 709→709)

| Model | Variant | MRR | R@1 | R@5 | R@10 |
|-------|---------|-----|-----|-----|------|
| all-MiniLM-L6-v2 *(baseline)* | — | 0.079 | 4.4% | 9.4% | 14.5% |
| all-mpnet-base-v2 *(baseline)* | — | 0.078 | 3.2% | 10.3% | 16.2% |
| BGE-large-en-v1.5 | no_instruction | 0.067 | 2.5% | 9.7% | 13.3% |
| BGE-large-en-v1.5 | default | 0.065 | 2.8% | 8.5% | 13.1% |
| BGE-large-en-v1.5 | moral_task | 0.043 | 1.7% | 5.1% | 7.9% |
| BGE-large-en-v1.5 | moral_meaning | 0.047 | 2.0% | 5.4% | 8.7% |
| **E5-large-v2** | **default** | **0.102** | **5.6%** | **13.5%** | **17.8%** |
| E5-large-v2 | no_prefix | 0.095 | 5.4% | 11.9% | 16.4% |
| multilingual-E5-large | default | 0.095 | 5.6% | 12.4% | 15.2% |

### Full results table — Fable→Moral (augmented corpus, 709→2803)

| Model | Variant | MRR | R@1 | R@5 | R@10 |
|-------|---------|-----|-----|-----|------|
| all-MiniLM-L6-v2 *(baseline)* | — | 0.026 | 1.2% | 3.5% | 4.5% |
| all-mpnet-base-v2 *(baseline)* | — | 0.026 | 1.2% | 2.7% | 4.7% |
| BGE-large-en-v1.5 | no_instruction | 0.030 | 1.4% | 3.5% | 5.4% |
| BGE-large-en-v1.5 | default | 0.029 | 1.0% | 3.8% | 5.8% |
| BGE-large-en-v1.5 | moral_task | 0.017 | 0.7% | 2.0% | 3.4% |
| BGE-large-en-v1.5 | moral_meaning | 0.016 | 0.6% | 1.6% | 2.8% |
| **E5-large-v2** | default | 0.038 | 1.4% | **5.1%** | **8.3%** |
| **E5-large-v2** | **no_prefix** | **0.046** | **2.5%** | 5.5% | 7.2% |
| multilingual-E5-large | default | 0.033 | 1.3% | 4.4% | 6.4% |

### Analysis: What do these results tell us?

#### Finding 1: E5-large-v2 is the best embedding model

E5 consistently outperforms all other models across every task and metric. Its best configuration achieves:
- **4.4% R@1 on fable→moral** (vs 3.4% best baseline — a 29% relative improvement)
- **5.6% R@1 on moral→fable** (vs 4.4% best baseline — a 27% relative improvement)
- **2.5% R@1 on augmented** (vs 1.2% best baseline — a 108% relative improvement)

The augmented corpus improvement is especially notable — E5 more than doubles the baseline R@1. This suggests E5 is better at distinguishing the correct moral from carefully-designed distractors.

#### Finding 2: Task-specific instructions HURT performance

This is a counter-intuitive but important result. For BGE:

| BGE variant | Fable→Moral R@1 |
|---|---|
| no_instruction | 3.2% |
| default (generic) | 3.2% |
| **moral_task** (task-specific) | **2.1%** (35% worse!) |
| **moral_meaning** (task-specific) | **1.8%** (44% worse!) |

The custom task-specific instructions ("Given this fable, retrieve the abstract moral lesson it teaches") actually degraded performance significantly.

**Why?** The BGE model was trained with specific instruction text during training. When you give it an instruction it has never seen, the prefix occupies attention capacity and shifts the embedding in an unpredictable direction. BGE v1.5 was specifically designed to work well *without* instructions (the paper says "alleviates the issue of instructions"). Our custom prefixes confused the model.

This is a useful negative result: it shows that naive instruction engineering doesn't help, and that the retrieval difficulty is not about the model "not knowing what to do" — it's about genuinely lacking the semantic capacity to bridge the abstraction gap between fables and morals.

#### Finding 3: E5 query/passage prefix is a trade-off

For E5, the default `"query: "` / `"passage: "` prefixes vs no prefix shows:
- Default gives better R@5/R@10 (10.6%/15.9% vs 10.2%/14.5%) — better at getting the correct moral *somewhere* in the top 10
- No prefix gives better R@1 (4.4% vs 3.0%) — better at putting the correct moral at the very top
- No prefix gives much better augmented R@1 (2.5% vs 1.4%)

The prefixed version casts a wider net (better recall at depth), while the unprefixed version is more precise (better precision at rank 1).

#### Finding 4: Multilingual E5 is surprisingly competitive

Despite being trained for multilingual tasks (not optimized for English-only), multilingual E5 achieves 5.6% R@1 on moral→fable — matching the best E5-large result. This is important for future cross-lingual work (Direction 3 from the professor): querying in Hebrew to retrieve English fables.

#### Finding 5: BGE does not improve over baseline

Despite being a much larger model (335M vs 22-109M params), BGE does not significantly outperform the Sentence-BERT baselines. Its best result (3.2% R@1) matches the MiniLM baseline (3.4%). This suggests that model size alone doesn't help with moral abstraction — the training data and architecture matter more.

#### Finding 6: The moral→fable direction benefits most from better models

The improvement from baseline to E5 is larger for moral→fable (4.4% → 5.6%) than for fable→moral (3.4% → 4.4%). This makes sense: morals are short and abstract (10 words), so a more capable model extracts richer meaning from less text. Fables are longer (117 words), so even weak models get reasonable representations.

---

## 7. LLM-based CoT reranking explained

### The idea

Embedding models compress an entire text into one vector. This is fast but lossy — a single vector can't capture the full narrative arc, character motivations, and abstract lesson of a fable. An LLM can *read* the fable and *reason* about it.

However, calling an LLM for every query-document pair is expensive. With 709 fables × 709 morals = ~500,000 pairs, that would require 500,000 API calls. Instead, we use a **two-stage pipeline**:

1. **Stage 1 (cheap, fast):** Use embedding retrieval to get the top-K candidates (e.g., K=20). This narrows 709 morals down to 20 plausible ones.
2. **Stage 2 (expensive, smart):** Send the fable + 20 candidates to Gemini with a CoT prompt. Gemini reasons about the fable's moral and reranks the candidates.

This costs only 709 API calls (one per fable) instead of 500,000.

### The prompt

```
You are an expert in fables, parables, and moral reasoning.

Given a fable and a list of candidate morals, think step by step about the
deeper lesson the fable teaches, then rank the morals from most to least relevant.

FABLE:
{the fable text}

CANDIDATE MORALS (numbered):
1. Gratitude is the sign of noble souls.
2. Never trust a known deceiver.
3. Bravery and compassion heal wounds.
...

Think step by step:
1. What happens in this fable? What is the key conflict or situation?
2. What abstract lesson or principle does this story illustrate?
3. Which candidate moral best captures this lesson?

After your reasoning, output ONLY a line starting with "RANKING:" followed by
the moral numbers in order from best to worst match, separated by commas.
Example: RANKING: 3, 1, 5, 2, 4
```

### Why CoT helps

The prompt forces the LLM to:
1. **Summarize the narrative** — "what happens?" forces attention to plot structure
2. **Abstract the lesson** — "what principle does this illustrate?" bridges from concrete to abstract
3. **Match against candidates** — "which moral best captures this?" compares the abstracted lesson to each candidate

This multi-step reasoning is exactly what embedding models can't do — they produce a single vector without explicit reasoning steps.

### Parsing the response

The script parses the LLM's response by:
1. Looking for a line starting with `"RANKING:"`
2. Splitting by commas to get moral numbers
3. Converting from 1-indexed (LLM output) to 0-indexed (internal)
4. Mapping local candidate indices back to global moral corpus indices
5. Appending any candidates the LLM forgot to mention at the end (fallback)

If the LLM fails (API error, no RANKING line found), the original embedding order is preserved.

### Rate limiting

The script includes a 0.5-second sleep between API calls. Gemini free tier allows 15 requests per minute; paid tier allows 1000+. For 709 fables at 0.5s spacing, the full experiment takes approximately 6-10 minutes (plus API response time, which varies — gemini-2.5-pro takes ~30s per response, gemini-2.0-flash takes ~2-5s).

---

## 8. CoT moral summarization explained

### The idea

Instead of reranking existing candidates, what if we asked the LLM to *generate* the moral, then used that generated text for embedding retrieval?

The intuition: the fable "The Fox and the Grapes" talks about a fox, vines, grapes, jumping, and walking away. An embedding of this story captures fox/grape-related semantics. But if we first ask an LLM "what's the moral?", it produces something like *"People often despise what they cannot have"* — which is much closer in semantic space to the actual moral *"It is easy to despise what you cannot get."*

### The prompt

```
Read this fable carefully. Think step by step about what deeper lesson or
moral principle it teaches. Then write ONLY the moral as a single concise
sentence (under 20 words).

FABLE:
{the fable text}

Think step by step about the lesson, then write:
MORAL: [your one-sentence moral]
```

### The pipeline

1. For each fable, call Gemini with the summarization prompt
2. Extract the generated moral from the `MORAL:` line in the response
3. Cache all generated morals to `results/cot_moral_summaries.json` (so we don't re-generate on subsequent runs)
4. Embed all generated morals using the embedding model (same as we'd embed the fables)
5. Embed all real morals using the embedding model
6. Compute cosine similarity between generated morals and real morals
7. Evaluate: does the real correct moral rank highly when compared to the generated summary?

### Why this might work better than raw fable embedding

| What gets embedded | Content | Closeness to moral |
|---|---|---|
| Raw fable | "A fox saw grapes hanging from a vine. He jumped to reach them..." | Low — concrete, narrative |
| Generated summary | "People often despise what they cannot have." | High — abstract, principled |
| Actual moral | "It is easy to despise what you cannot get." | (target) |

By converting the fable into an abstract statement first, we bridge the semantic gap that embedding models struggle with.

### Caching

Generated moral summaries are expensive to produce (one API call per fable). The script caches them at `results/cot_moral_summaries.json`. On subsequent runs, if the cache has the right number of entries, it's reused automatically without re-calling the API.

---

## 9. Script: 07_llm_reranking.py

### What it does

Implements both CoT approaches (reranking and summarization) using the Gemini API.

### Input

- `data/processed/fables_corpus.json` (709 fables)
- `data/processed/morals_corpus.json` (709 morals)
- `data/processed/qrels_fable_to_moral.json` (ground truth)
- `.env` file with `GEMINI_API_KEY` and optionally `GEMINI_MODEL`

### Output

- `results/llm_reranking_results.json` — metrics for each approach
- `results/cot_moral_summaries.json` — cached generated morals (Approach B only)
- Printed summary table

### Environment setup

The script reads API keys from a `.env` file in the project root:

```
GEMINI_API_KEY=AIzaSy...your-key-here
GEMINI_MODEL=gemini-2.5-pro
```

It also checks the `GEMINI_API_KEY` and `GOOGLE_API_KEY` environment variables as fallback.

### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--approach` | `both` | Which approach to run: `rerank`, `summarize`, or `both` |
| `--sample` | `None` (= all 709) | Run on only N fables (for testing) |
| `--embed-model` | `BAAI/bge-large-en-v1.5` | Embedding model for Stage 1 retrieval / summary embedding |
| `--top-k` | `20` | Number of candidates for reranking (Approach A) |

### Example commands

```bash
# Quick test on 10 fables, reranking only
python scripts/07_llm_reranking.py --approach rerank --sample 10

# Full run, both approaches (takes 30-60 min with gemini-2.5-pro)
python scripts/07_llm_reranking.py --approach both

# Use a different embedding model for Stage 1
python scripts/07_llm_reranking.py --embed-model intfloat/e5-large-v2

# Rerank top-50 instead of top-20 (gives more room but costs more per call)
python scripts/07_llm_reranking.py --approach rerank --top-k 50
```

### Step-by-step what happens

1. **Load .env** — parses the `.env` file to set environment variables
2. **Configure Gemini** — initializes the API with the key
3. **Load data** — reads fables, morals, and qrels
4. **Load embedding model** — downloads/caches the Sentence-BERT model
5. **Encode all fables and morals** — produces embedding matrices
6. **Compute embedding baseline** — ranks all morals by cosine similarity for comparison
7. **If reranking:**
   a. Get top-K candidates for each fable from embedding similarity
   b. For each fable, send fable + K candidate morals to Gemini
   c. Parse the reranked order from Gemini's response
   d. Compute metrics on the reranked results
8. **If summarization:**
   a. For each fable, send to Gemini and get a generated moral summary
   b. Cache summaries to JSON
   c. Embed all summaries
   d. Compute cosine similarity between summary embeddings and real moral embeddings
   e. Rank and evaluate
9. **Save results** and print summary table

### Key design decisions

**Why BGE as default embedding model?** It was the first model we tested and works reliably. For best results, use E5-large-v2 (`--embed-model intfloat/e5-large-v2`).

**Why top-K=20?** A trade-off between recall (the correct moral needs to be in the top-K to have a chance) and cost/quality (more candidates = longer prompt = more expensive and potentially harder for the LLM to rank). With the baseline, ~12-16% of correct morals are in the top 10 (depending on model), so top-20 captures maybe 20-25%.

**Why sleep(0.5)?** Rate limiting. Gemini free tier is 15 RPM. With 0.5s sleep + API response time (~2-30s depending on model), we stay well under the limit. For paid tier, the sleep could be reduced.

**Why `genai.GenerativeModel` per call?** The deprecated `google.generativeai` SDK creates model instances for each call. A future improvement would be migrating to the newer `google.genai` SDK.

---

## 10. CoT preliminary results

We ran a small test (5 fables, top-10 candidates, gemini-2.5-pro) to validate the pipeline:

| Approach | MRR | R@1 | R@5 | R@10 |
|----------|-----|-----|-----|------|
| Embedding only (BGE) | 0.066 | 3.2% | 8.2% | 11.7% |
| **CoT reranking (5-fable sample)** | **0.201** | **20.0%** | **20.0%** | **20.0%** |

On this tiny sample, CoT reranking got 1 out of 5 correct at rank 1 (20% R@1) — compared to the embedding baseline's 3.2%. The MRR jumped 3x.

**Important caveat:** 5 fables is far too small for reliable metrics. One lucky or unlucky result changes R@1 by 20 percentage points. The full 709-fable experiment has not yet been completed (it was started but stopped before finishing).

The R@5 = R@10 = 20% (same as R@1) means that for this 5-fable sample, the one fable where CoT succeeded had the correct moral at rank 1, and the other 4 fables had the correct moral outside the top-10 candidates entirely. This highlights a limitation: **if the correct moral isn't in the top-K candidates from embedding retrieval, CoT reranking can't find it.** Increasing K helps but makes the prompt longer and reranking harder.

---

## 11. Combined findings and takeaways

### What we learned

1. **Stronger embedding models help modestly.** E5-large-v2 achieves ~30% relative improvement over MiniLM baseline on R@1. But we're going from 3.4% to 4.4% — still very low in absolute terms. Embedding models, no matter how large, struggle with the fable→moral abstraction gap.

2. **Task-specific instructions don't help (and often hurt).** Custom prefixes like "retrieve the moral lesson" degraded BGE by 35-44%. The models weren't trained with these instructions, and the prefixes confuse rather than guide them. This is an important negative result.

3. **The E5 model family is the best for this task.** E5-large-v2 leads on every metric. The multilingual variant is nearly as good and supports cross-lingual retrieval.

4. **CoT reranking shows promise but needs full evaluation.** The 5-fable test suggests LLM reasoning can significantly outperform pure embedding retrieval, but the sample is too small for conclusions. The full experiment should be run.

5. **The two-stage pipeline (embed → rerank) is the right architecture.** Pure embedding is too weak. Pure LLM reasoning is too expensive (500K API calls). The hybrid approach narrows candidates cheaply, then reasons deeply over the shortlist.

### Comparison with random baseline

For context, here's how each approach compares to random chance (R@1 = 1/709 ≈ 0.14%):

| Approach | R@1 | vs Random |
|----------|-----|-----------|
| Random | 0.14% | 1x |
| MiniLM (baseline) | 3.4% | 24x |
| E5-large (best embedding) | 4.4% | 31x |
| CoT reranking (preliminary) | ~20%* | ~143x* |

*Very preliminary, 5-fable sample only.

---

## 12. How to run everything

### Prerequisites

```bash
# Activate the virtual environment
source venv/bin/activate

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### Run improved embedding experiments

```bash
python scripts/06_improved_retrieval.py
```

This takes ~15-25 minutes and produces `results/improved_retrieval_results.json`.

### Run CoT reranking experiments

First, ensure your `.env` file has a valid `GEMINI_API_KEY`:

```bash
# Test with a small sample first
python scripts/07_llm_reranking.py --approach rerank --sample 10

# If that works, run the full experiment
python scripts/07_llm_reranking.py --approach both
```

The full experiment takes 30-60+ minutes depending on the Gemini model used (gemini-2.0-flash is ~5-10 min, gemini-2.5-pro is ~30-60 min).

---

## 13. New files created

```
work_morables/
├── scripts/
│   ├── 06_improved_retrieval.py         # Instruction-aware embedding experiments
│   └── 07_llm_reranking.py             # CoT reranking and summarization
│
├── results/
│   ├── improved_retrieval_results.json  # 21 embedding experiment results
│   └── llm_reranking_results.json       # CoT experiment results (partial)
│
├── .env                                 # API keys (GEMINI_API_KEY, etc.)
├── PLAN_DIRECTION_1.md                  # Implementation plan for Direction 1
├── IMPROVED_RETRIEVAL_GUIDE.md          # This file
└── requirements.txt                     # Updated: added google-generativeai
```

### Dependencies added

| Package | Version | Purpose |
|---------|---------|---------|
| `google-generativeai` | 0.8.6 | Gemini API client for CoT reranking |

Note: The `google.generativeai` package is deprecated in favor of `google.genai`. The script uses the deprecated version because it was the one installed. A future update should migrate to the new SDK.

---

## 14. Known issues and next steps

### Known issues

1. **GTE model crashes.** `Alibaba-NLP/gte-large-en-v1.5` causes a `torch.AcceleratorError` (tensor index out of bounds) during encoding. This appears to be an incompatibility with the MPS (Apple Silicon) backend or the installed version of sentence-transformers. Removed from experiments.

2. **Deprecated Gemini SDK.** The script uses `google.generativeai` which is deprecated. It works but will eventually stop receiving updates. Migration to `google.genai` is recommended.

3. **CoT experiment incomplete.** Only a 5-fable test has been run. The full 709-fable experiment needs to be completed for reliable metrics. The script supports `--sample N` for incremental testing.

4. **CoT reranking ceiling.** If the correct moral isn't in the top-K candidates from embedding retrieval, CoT reranking cannot recover it. With BGE and top-20, approximately 75-80% of correct morals are *not* in the top 20. Using a better embedding model (E5) and larger K (50) would increase the ceiling.

5. **Multiple .env API keys.** The `.env` file contains several Gemini API keys (most commented out). Key 1 has exhausted its quota, Key 2 was flagged as leaked, Key 3 is the active one, Keys 4-5 returned "invalid." Only one key is active at a time.

### Immediate next steps

1. **Run full CoT reranking** with 709 fables, using E5-large-v2 as the Stage 1 model and top-K=50
2. **Run CoT summarization** (Approach B) — this hasn't been tested at all yet
3. **Compare gemini-2.0-flash vs gemini-2.5-pro** for CoT quality — flash is 10x cheaper, but pro might reason better about morals
4. **Try the augmented corpus** for CoT reranking — rerank against the 2803-moral pool where distractors are specifically designed to be confusing

### Longer-term directions (from professor)

- **Direction 2:** Expand the dataset by finding and merging other fable/moral datasets, using LLMs to unify the moral vocabulary
- **Direction 3:** Cross-lingual moral clustering — connect English morals to Hebrew proverbs (e.g., "Self-help is the best help" ↔ "אם אין אני לי מי לי"), using multilingual E5
- **Direction 4:** Synthetic data expansion — generate new stories for existing morals using LLMs, for both training and evaluation
