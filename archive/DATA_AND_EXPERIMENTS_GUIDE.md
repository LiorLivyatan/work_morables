# Complete Guide: Data, Scripts, and Experiments

This document explains everything in the `work_morables` project from the ground up. No prior knowledge assumed.

---

## Table of Contents

1. [What is this project about?](#1-what-is-this-project-about)
2. [The MORABLES dataset (original)](#2-the-morables-dataset-original)
3. [Our raw data files](#3-our-raw-data-files)
4. [Our retrieval-formatted data files](#4-our-retrieval-formatted-data-files)
5. [The retrieval task explained](#5-the-retrieval-task-explained)
6. [The scripts](#6-the-scripts)
7. [The metrics explained](#7-the-metrics-explained)
8. [The baseline results and what they mean](#8-the-baseline-results-and-what-they-mean)
9. [Known issues and caveats](#9-known-issues-and-caveats)

---

## 1. What is this project about?

**The big question:** Can AI models understand the *abstract moral meaning* of stories?

For example, take the fable of "The Ass and the Lapdog": an Ass sees a Lapdog getting pampered by its master, tries to imitate the Lapdog's tricks, causes chaos, and gets beaten. The moral is: *"To be satisfied with one's lot is better than to desire something which one is not fitted to receive."*

Notice that the moral shares almost no words with the story. You can't find the moral by searching for "Ass" or "Lapdog" or "master". You need to *understand* that the story is about envy, imitation beyond one's nature, and contentment. This is abstract moral reasoning.

**Our approach:** We frame this as a **retrieval task**. We take 709 fables and 709 morals, encode them into numerical vectors (embeddings) using AI models, and ask: can the model match each fable to its correct moral by placing them "close together" in embedding space?

If standard embedding models fail (spoiler: they do), it means their embeddings don't capture abstract moral meaning — and that's a meaningful research finding, because it reveals a gap that better models or training could fill.

---

## 2. The MORABLES dataset (original)

**Source:** Published at EMNLP 2025 by Marcuzzo et al. Available at [cardiffnlp/Morables](https://huggingface.co/datasets/cardiffnlp/Morables) on HuggingFace.

**What it contains:** 709 fable-moral pairs from Western literary tradition. Each fable is a short story (avg. 117 words) with talking animals or metaphorical characters. Each moral is a concise lesson (avg. 10 words).

**Original purpose:** The paper uses these as a **multiple-choice question answering (MCQA)** benchmark — give an LLM a fable and 5 candidate morals, ask it to pick the right one. They found that even large LLMs often rely on surface-level cues rather than genuine moral understanding.

**Our purpose:** We repurpose it as a **retrieval benchmark** — instead of picking from 5 choices, we ask: given a fable, can you find its moral in a pool of 709 (or 2803) candidates?

The dataset has multiple "configs" (subsets) on HuggingFace:

| Config | What it contains |
|--------|-----------------|
| `fables_only` | The 709 fable-moral pairs (our main data) |
| `mcqa` | MCQA format: fable + 5 choices with labels |
| `binary` | True/False format: is this moral correct for this fable? |
| `adversarial` | 18 variants with modified stories (character swaps, injected tautologies, etc.) |
| `extracted_info` | Supporting metadata (has a schema bug on HuggingFace, could not load) |
| `supporting_info` | Additional info (same schema bug) |

---

## 3. Our raw data files

These are downloaded from HuggingFace and saved as JSON by `scripts/01_load_data.py`. They live in `data/raw/`.

### `data/raw/fables.json` — 709 entries

The core dataset. Each entry is one fable-moral pair.

**Example entry:**
```json
{
  "alias": "aesop_section_1_21",
  "title": "The Ass and the Lapdog",
  "story": "A Man had an Ass, and a Maltese Lapdog, a very great beauty...",
  "moral": "To be satisfied with one's lot is better than to desire something which one is not fitted to receive.",
  "note": null,
  "alternative_moral": null
}
```

**Every field explained:**

| Field | Type | Meaning |
|-------|------|---------|
| `alias` | string | A unique identifier from the original dataset. Format: `{source}_{section/book}_{number}`. Examples: `aesop_section_1_5`, `gibbs_fable_102`, `perry_fable_45`, `abstemius_fable_12`. The source prefix tells you where the fable comes from. |
| `title` | string | Human-readable name of the fable (e.g., "Androcles", "The Fox and the Grapes"). |
| `story` | string | The full text of the fable. Ranges from 28 to 1132 words (mean: 117, median: 99). |
| `moral` | string | The correct moral/lesson of the fable. A short statement, 2-30 words (mean: 10, median: 9). |
| `note` | string or null | Occasional editorial notes from the original source. Almost always null. |
| `alternative_moral` | string or null | Some fables have an alternate valid moral from a different source/translator. Almost always null. |

**Source distribution (derived from `alias` prefix):**

| Source | Count | % | Who they are |
|--------|-------|---|-------------|
| gibbs | 360 | 50.8% | Laura Gibbs — modern English translator of Aesop and other fabulists |
| aesop | 119 | 16.8% | Direct Aesop collections |
| perry | 89 | 12.6% | Ben Edwin Perry — compiled the Perry Index, a standard numbering of Aesop's fables |
| abstemius | 83 | 11.7% | Laurentius Abstemius — 15th century Italian fabulist |
| gibbs_noted | 34 | 4.8% | Gibbs translations with editorial notes |
| phaedrus | 7 | 1.0% | Phaedrus — Roman fabulist who wrote Latin verse fables |
| others | ~17 | ~2.4% | La Fontaine, Australian folklore, etc. |

### `data/raw/mcqa.json` — 709 entries

Same fables but with 5 multiple-choice options. This is how the MORABLES paper evaluates LLMs.

**Example entry (Androcles, index 0):**
```json
{
  "alias": "aesop_section_1_5",
  "story_title": "Androcles",
  "story": "A slave named Androcles once escaped from his master...",
  "moral": "Gratitude is the sign of noble souls.",
  "is_altered": false,
  "correct_moral_label": 0,
  "classes": [
    "ground_truth",
    "similar_characters",
    "based_on_adjectives",
    "injected_adjectives",
    "partial_story"
  ],
  "choices": [
    "Gratitude is the sign of noble souls.",
    "Never trust a known deceiver.",
    "Bravery and compassion heal wounds.",
    "The true leader proves himself by his brave qualities.",
    "Compassion can bridge the gap between the strongest and the weakest."
  ]
}
```

**Every field explained:**

| Field | Type | Meaning |
|-------|------|---------|
| `alias` | string | Same unique ID as in `fables.json`. |
| `story_title` | string | Same as `title` in `fables.json`. |
| `story` | string | Same fable text. |
| `moral` | string | The correct moral (same as in `fables.json`, with rare minor text differences). |
| `is_altered` | boolean | Always `false` in the core MCQA. Set to `true` in adversarial variants. |
| `correct_moral_label` | integer | Always `0` in the `mcqa_not_shuffled` split (because the correct answer is always placed first). In the `mcqa_shuffled` split, this varies. |
| `classes` | list of 5 strings | The *type* of each choice option (see distractor types below). |
| `choices` | list of 5 strings | The actual text of each choice option. Index 0 = ground truth in the not-shuffled version. |

**The 5 distractor types explained:**

Each fable has 1 correct moral + 4 distractors. The distractors are designed to be plausible but wrong:

| Class | How it was created | Why it's tricky |
|-------|-------------------|-----------------|
| `ground_truth` | The original moral from the fable's author/translator. | This is the correct answer. |
| `similar_characters` | A real moral from a *different* fable that features similar characters. E.g., if the story has a fox, this moral comes from another fox story. | Tests whether the model is just matching characters/entities rather than understanding the story's message. |
| `based_on_adjectives` | An LLM-generated moral based ONLY on the character traits (e.g., "brave lion", "cunning fox"), without seeing the actual story. | Sounds thematically related but misses the actual narrative arc. |
| `injected_adjectives` | A real moral from another fable, but with character traits from *this* fable injected into it (e.g., adding "brave" or "cunning" to the wrong moral). | Has both real moral structure AND matching character traits, but the moral itself is wrong for this story. |
| `partial_story` | An LLM-generated moral based only on the first ~10% of the story. | Captures the *setup* but misses the resolution, twist, or actual lesson. |

**Concrete example for "Androcles" (a slave who helps a lion, and the lion later spares him in the arena):**

- `ground_truth`: *"Gratitude is the sign of noble souls."* — correct
- `similar_characters`: *"Never trust a known deceiver."* — from another fable with similar characters (people/animals in power dynamics), but wrong message
- `based_on_adjectives`: *"Bravery and compassion heal wounds."* — sounds right based on traits ("brave slave", "wounded lion") but misses the actual point about gratitude
- `injected_adjectives`: *"The true leader proves himself by his brave qualities."* — real-sounding moral with "brave" injected, but not what this fable teaches
- `partial_story`: *"Compassion can bridge the gap between the strongest and the weakest."* — based only on the opening (slave helps lion), misses that the lion later repays the favor

### `data/raw/adversarial_*.json` — 18 files, 709 entries each

Modified versions of the stories designed to test robustness. Each file has the same schema as `mcqa.json` but with `is_altered: true` and modified story text.

**Types of adversarial modifications:**

| Modification | What changes | Example |
|-------------|-------------|---------|
| `char_swap` | Characters replaced with similar ones | "Androcles" → "Marcus", "Lion" → "Tiger" |
| `adj_inj` | Adjectives from other fables injected into the story | Add "cunning" or "proud" where they don't belong |
| `pre_inj` / `post_inj` | Tautologies prepended/appended to the story | "A fact is a fact." added before the story |
| Combinations | Multiple modifications combined | `pre_post_char_adj` = all of the above |

These are downloaded but **not yet used** in our experiments.

---

## 4. Our retrieval-formatted data files

Created by `scripts/03_prepare_retrieval.py`. These restructure the raw data into a standard retrieval format with separate corpus files, query files, and relevance judgments (qrels). They live in `data/processed/`.

### `data/processed/fables_corpus.json` — 709 entries

Each fable as a "document" in the retrieval corpus.

```json
{
  "doc_id": "fable_0005",
  "title": "The Ass and the Lapdog",
  "text": "A Man had an Ass, and a Maltese Lapdog, a very great beauty...",
  "alias": "aesop_section_1_21"
}
```

| Field | Meaning |
|-------|---------|
| `doc_id` | Our assigned ID: `fable_NNNN` where NNNN is the index (0000-0708). Used to link to qrels. |
| `title` | Fable title (from original data). |
| `text` | The fable story text. This is what gets embedded for retrieval. |
| `alias` | Original dataset alias (for traceability back to MORABLES). |

### `data/processed/morals_corpus.json` — 709 entries

Each moral as a "document" in the retrieval corpus.

```json
{
  "doc_id": "moral_0005",
  "text": "To be satisfied with one's lot is better than to desire something which one is not fitted to receive.",
  "fable_id": "fable_0005"
}
```

| Field | Meaning |
|-------|---------|
| `doc_id` | Our assigned ID: `moral_NNNN`. The numbering mirrors the fables (moral_0005 belongs to fable_0005). |
| `text` | The moral text. This is what gets embedded. |
| `fable_id` | Which fable this moral belongs to. |

**Key property:** There's a perfect 1-to-1 mapping: `fable_0000` ↔ `moral_0000`, `fable_0001` ↔ `moral_0001`, etc. This is the "clean" evaluation setting.

### `data/processed/morals_augmented_corpus.json` — 2803 entries

A larger moral pool that includes both correct morals AND distractors from the MCQA data, **deduplicated** by exact text match (case-insensitive). Each entry tracks ALL roles the moral plays across the dataset.

**Example 1 — a moral that is only correct (for one fable):**
```json
{
  "doc_id": "moral_aug_0000",
  "text": "Gratitude is the sign of noble souls.",
  "is_correct": true,
  "classes": ["ground_truth"],
  "correct_for_fables": ["fable_0000"]
}
```

**Example 2 — a moral that is correct for one fable AND a distractor for others:**
```json
{
  "doc_id": "moral_aug_0001",
  "text": "Never trust a known deceiver.",
  "is_correct": true,
  "classes": ["ground_truth", "similar_characters"],
  "correct_for_fables": ["fable_0594"]
}
```
This moral is the correct answer for fable_0594, but it also appears as a `similar_characters` distractor for other fables (e.g., fable_0000 "Androcles").

**Example 3 — a pure distractor (never the correct answer):**
```json
{
  "doc_id": "moral_aug_0002",
  "text": "Bravery and compassion heal wounds.",
  "is_correct": false,
  "classes": ["based_on_adjectives"],
  "correct_for_fables": []
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `doc_id` | string | Sequential ID: `moral_aug_NNNN`. |
| `text` | string | The moral text. |
| `is_correct` | boolean | `true` if this moral is the correct answer for **at least one** fable. |
| `classes` | list of strings | **All** roles this moral plays across the dataset. A moral can be `["ground_truth"]` only, `["based_on_adjectives"]` only, or `["ground_truth", "similar_characters"]` if it's correct for one fable but used as a distractor for another. |
| `correct_for_fables` | list of strings | Which fable(s) this moral is the correct answer for. Empty list `[]` for pure distractors. Contains multiple fable IDs for the 27 morals shared by multiple fables (e.g., "Appearances are deceptive." is correct for both `fable_0001` and `fable_0117`). |

**Why 2803?** Each of the 709 fables has 5 choices = 3545 total. But many are duplicates (e.g., a moral that is the correct answer for one fable might also appear as a `similar_characters` distractor for another). After deduplication: 2803 unique moral texts.

**Composition:** 678 correct morals + 2125 pure distractors = 2803 total. Of the 678 correct morals, 299 also serve as distractors for other fables, and 27 are correct for multiple fables.

### `data/processed/qrels_fable_to_moral.json` — 709 entries

"Qrels" (query relevance judgments) = the ground truth for evaluation. This file says which moral is correct for each fable.

```json
{"query_id": "fable_0005", "doc_id": "moral_0005", "relevance": 1}
```

| Field | Meaning |
|-------|---------|
| `query_id` | The fable being used as a query. |
| `doc_id` | The correct moral for that fable. |
| `relevance` | Always `1` (= relevant). Non-listed pairs are implicitly `0` (= not relevant). |

**How it's used:** When evaluating fable→moral retrieval, the system encodes `fable_0005` as a query, searches through all 709 morals, and we check whether `moral_0005` appears in the top results.

### `data/processed/qrels_moral_to_fable.json` — 709 entries

Same thing but reversed: which fable is correct for each moral.

```json
{"query_id": "moral_0005", "doc_id": "fable_0005", "relevance": 1}
```

Used for evaluating moral→fable retrieval (given a moral, find its fable).

### `data/processed/qrels_fable_to_moral_augmented.json` — 709 entries

Ground truth for the harder setting: fable→moral retrieval against the augmented corpus (2803 morals).

```json
{"query_id": "fable_0005", "doc_id": "moral_aug_0025", "relevance": 1}
```

Every fable maps to exactly one entry in the augmented corpus. When multiple fables share the same moral (e.g., "Appearances are deceptive."), they all point to the same `moral_aug_NNNN` document.

---

## 5. The retrieval task explained

### What is retrieval?

Retrieval means: given a **query** (a piece of text), find the most relevant **document** from a large collection (the **corpus**).

Think of it like a search engine: you type a query, and it returns ranked results. We measure how good the system is by checking whether the *correct* document appears near the top.

### How embedding-based retrieval works

1. **Encode everything:** Feed every fable and every moral through an embedding model. The model converts each piece of text into a dense numerical vector (e.g., a list of 384 or 768 numbers). Texts with similar meaning should get similar vectors.

2. **Compute similarity:** For each query, compute the **cosine similarity** between the query's vector and every document's vector. Cosine similarity ranges from -1 (opposite) to +1 (identical direction).

3. **Rank:** Sort documents by similarity score (highest = most similar = rank 1).

4. **Evaluate:** Check where the correct document ended up in the ranking.

### Our three evaluation settings

| Setting | Query (what we search with) | Corpus (what we search through) | Size | Purpose |
|---------|---------------------------|-------------------------------|------|---------|
| **Fable→Moral (clean)** | 709 fables | 709 morals | 709×709 | Core task: can the model match a story to its lesson? |
| **Moral→Fable (clean)** | 709 morals | 709 fables | 709×709 | Reverse: can the model match a lesson to its story? |
| **Fable→Moral (augmented)** | 709 fables | 2803 morals (correct + distractors) | 709×2803 | Harder version: the corpus is polluted with plausible-sounding wrong morals |

### Why is this hard?

The key insight from the EDA: **lexical overlap between fables and their morals is near-zero**. When we compute word-level Intersection-over-Union (IoU) on content words (excluding stopwords like "the", "is", "a"):

- **Mean IoU: 0.011**
- **Median IoU: 0.000** (more than half of pairs share ZERO content words)

This means a system can't just look for word matches. The story says "Ass", "Lapdog", "master", "kicked"; the moral says "satisfied", "lot", "desire", "fitted". Understanding requires bridging this abstraction gap.

---

## 6. The scripts

### `scripts/01_load_data.py`

**What it does:** Downloads the MORABLES dataset from HuggingFace and saves it as local JSON files.

**Input:** HuggingFace dataset `cardiffnlp/Morables`
**Output:** `data/raw/fables.json`, `data/raw/mcqa.json`, `data/raw/adversarial_*.json`

**How to run:** `python scripts/01_load_data.py`

### `scripts/02_eda.py`

**What it does:** Computes statistics about the dataset and produces visualizations.

**Input:** `data/raw/fables.json`, `data/raw/mcqa.json`
**Output:** Printed statistics + `results/eda_overview.png`, `results/distractor_similarity.png`

**What it computes:**

1. **Basic text stats** — story/moral lengths in words, sentence counts
2. **Source distribution** — parses the `alias` field to extract the source (gibbs, aesop, perry, etc.)
3. **Lexical overlap** — for each fable-moral pair, computes word IoU (with and without stopwords)
4. **Moral uniqueness** — counts unique morals, finds morals shared by multiple fables
5. **Distractor similarity** — for each distractor class, how much word overlap it has with the correct moral

**Visualizations produced:**

- `results/eda_overview.png` — 4-panel chart:
  - Top-left: Pie chart of source distribution
  - Top-right: Histogram of story lengths
  - Bottom-left: Histogram of moral lengths
  - Bottom-right: Histogram of lexical overlap scores
- `results/distractor_similarity.png` — Bar chart showing each distractor type's lexical similarity to the correct moral

### `scripts/03_prepare_retrieval.py`

**What it does:** Reformats the raw data into standard retrieval format (corpus + qrels).

**Input:** `data/raw/fables.json`, `data/raw/mcqa.json`
**Output:** 6 files in `data/processed/` (see Section 4 above)

**Key logic:**

1. Assigns sequential IDs (`fable_0000`, `moral_0000`, etc.)
2. Creates 1-to-1 fable↔moral mappings for the clean evaluation
3. Builds the augmented moral corpus using a **two-pass approach**:
   - Pass 1: Iterates through all MCQA choices across all fables, collecting every role each unique moral plays (ground_truth for which fables, distractor of which type for which fables)
   - Pass 2: Builds the corpus entries with complete role information — a moral is marked `is_correct: true` if it's ground_truth for ANY fable, even if it also appears as a distractor elsewhere
4. Creates qrels (ground truth mappings) for all settings — all 709 fables get a mapping in the augmented qrels

### `scripts/04_baseline_retrieval.py`

**What it does:** Runs the actual retrieval experiments.

**Input:** All retrieval-formatted files from `data/processed/`
**Output:** `results/baseline_results.json` + printed tables

**Step by step:**

1. **Load data** — reads corpus files and qrels
2. **Build ground truth mappings** — converts qrels into dictionaries (fable_idx → moral_idx) for fast lookup
3. **For each model** (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`):
   a. **Encode** all fables, morals, and augmented morals into embedding vectors
   b. **Compute similarity** — cosine similarity matrix between queries and corpus
   c. **Rank** — sort by similarity (highest first)
   d. **Evaluate** — compute metrics (see Section 7)
   e. **Rank analysis** — count how many correct morals end up at each rank position

**Models used:**

| Model | Dimensions | Parameters | Notes |
|-------|-----------|------------|-------|
| `all-MiniLM-L6-v2` | 384 | 22M | Fast, lightweight baseline. Trained on 1B+ sentence pairs. |
| `all-mpnet-base-v2` | 768 | 109M | Stronger, 5x larger. Generally the best general-purpose Sentence-BERT model. |

Both are **bi-encoder** models: they encode query and document independently, then compare via cosine similarity. This is standard for first-stage retrieval.

---

## 7. The metrics explained

Since each query has exactly **1 correct document**, the metrics simplify:

### R-Precision (= Recall@1 = Precision@1)

> What fraction of queries have the correct document ranked #1?

If the model puts the right moral at the very top for 19 out of 709 fables, R-Precision = 19/709 = 2.7%.

### Recall@k

> What fraction of queries have the correct document somewhere in the top k results?

- **Recall@1:** correct doc is rank #1 (same as R-Precision for single-relevant-doc)
- **Recall@5:** correct doc is in top 5
- **Recall@10:** correct doc is in top 10

### MRR (Mean Reciprocal Rank)

> Average of 1/rank across all queries.

If correct doc is at rank 1 → score is 1/1 = 1.0
If correct doc is at rank 2 → score is 1/2 = 0.5
If correct doc is at rank 10 → score is 1/10 = 0.1
If correct doc is at rank 100 → score is 1/100 = 0.01

Then MRR = average of all these scores. Higher = better. MRR rewards getting the right answer early.

### What would random performance look like?

With 709 documents, random ranking gives:
- R@1 = 1/709 ≈ 0.14%
- R@5 = 5/709 ≈ 0.71%
- R@10 = 10/709 ≈ 1.41%
- MRR ≈ (1/709) × sum(1/k for k=1..709) ≈ 0.009

---

## 8. The baseline results and what they mean

### Full results table

| Model | Task | MRR | R@1 | R@5 | R@10 |
|-------|------|-----|-----|-----|------|
| all-MiniLM-L6-v2 | fable→moral (clean) | 0.063 | 3.4% | 7.5% | 11.7% |
| all-MiniLM-L6-v2 | moral→fable (clean) | 0.079 | 4.4% | 9.4% | 14.5% |
| all-MiniLM-L6-v2 | fable→moral (augmented) | 0.026 | 1.2% | 3.5% | 4.5% |
| all-mpnet-base-v2 | fable→moral (clean) | 0.065 | 2.7% | 8.3% | 12.7% |
| all-mpnet-base-v2 | moral→fable (clean) | 0.078 | 3.2% | 10.3% | 16.2% |
| all-mpnet-base-v2 | fable→moral (augmented) | 0.026 | 1.2% | 2.7% | 4.7% |

### Rank distribution (all-mpnet-base-v2, fable→moral clean)

| Where the correct moral ends up | Count | % |
|--------------------------------|-------|---|
| Rank #1 (perfect) | 19 | 2.7% |
| Rank #2–5 | 40 | 5.6% |
| Rank #6–10 | 31 | 4.4% |
| Rank #11–50 | 136 | 19.2% |
| Rank #51+ (bottom 93%) | 483 | **68.1%** |
| Mean rank | 186 | — |
| Median rank | 129 | — |

### Interpretation

1. **Models beat random, but barely.** R@1 of ~3% vs random's 0.14% means the model has *some* signal, but it's weak. The median correct moral is at rank 129 — roughly in the middle of the 709-item list.

2. **Moral→fable is slightly easier than fable→moral.** Morals are short abstract statements (10 words). Fables are longer and more distinctive (117 words). When you use a moral as the query, the richer fable text gives a "bigger target" to match against. When you use a fable as query, all the morals look similar to each other (short, abstract), making discrimination harder.

3. **The augmented corpus is much harder.** Going from 709 morals to 2803 (adding distractors) cuts R@1 roughly in half. The distractors are designed to be plausible alternatives, and the embedding model can't tell them apart from the correct moral.

4. **all-mpnet-base-v2 is slightly better on Recall@5/10 but worse on R@1 than MiniLM.** Both are in the same performance range — neither model captures moral abstraction.

5. **This validates the benchmark.** If off-the-shelf models scored 90%, there'd be no interesting research. The fact that they score ~3% means the task is genuinely hard, and improvements (fine-tuning, better models, rerankers) have a large headroom to explore.

---

## 9. Known issues and minor caveats

### Minor text difference between fables.json and mcqa.json

One moral has a slight wording difference between the two source files:
- `fables.json`: *"beware of swindlers, as they often exploit the gullibility of others."*
- `mcqa.json`: *"beware of swindlers that exploit the foolishness of others."*

This affects 1 out of 709 entries. Since the clean retrieval corpus is built from `fables.json` and the augmented corpus from `mcqa.json`, this one fable's correct moral text differs slightly between the two evaluation settings. No practical impact on metrics.

### The `distractor_analysis()` function in `04_baseline_retrieval.py` is a stub

The function at line 122-131 is unfinished (just has `pass`). It was intended to analyze which distractor types the model confuses most, but this analysis hasn't been implemented yet.

### Baseline results were run before the augmented corpus bug fix

The results in `results/baseline_results.json` were generated with the old augmented corpus (490 correct morals, 510 qrels). The clean evaluation results (fable→moral and moral→fable on the 709-item corpus) are unaffected. The augmented evaluation results should be re-run with the fixed data for accurate numbers.

---

## File tree summary

```
work_morables/
├── .git/                                    # Git repo (initialized, nothing committed)
├── .gitignore
├── requirements.txt                         # Python dependencies
├── venv/                                    # Python 3.13 virtual environment
├── TASKS.md                                 # Master task tracker
├── DATA_AND_EXPERIMENTS_GUIDE.md            # This file
├── morables.pdf                             # The MORABLES paper (EMNLP 2025)
├── Thesis Proposal_template_example.pdf     # Thesis proposal template
│
├── scripts/
│   ├── 01_load_data.py                      # HuggingFace → data/raw/
│   ├── 02_eda.py                            # Statistics + charts → results/
│   ├── 03_prepare_retrieval.py              # data/raw/ → data/processed/
│   └── 04_baseline_retrieval.py             # data/processed/ → results/
│
├── data/
│   ├── raw/                                 # Downloaded from HuggingFace (untouched)
│   │   ├── fables.json                      #   709 raw fable-moral pairs
│   │   ├── mcqa.json                        #   709 MCQA entries (5 choices each)
│   │   └── adversarial_*.json               #   18 adversarial variant files (709 each)
│   │
│   └── processed/                           # Created by us for retrieval evaluation
│       ├── fables_corpus.json               #   709 fables (retrieval format)
│       ├── morals_corpus.json               #   709 morals (retrieval format)
│       ├── morals_augmented_corpus.json     #   2803 morals (678 correct + 2125 distractors)
│       ├── qrels_fable_to_moral.json        #   709 ground truth (fable→moral)
│       ├── qrels_moral_to_fable.json        #   709 ground truth (moral→fable)
│       └── qrels_fable_to_moral_augmented.json  #   709 ground truth (fable→augmented moral)
│
└── results/
    ├── eda_overview.png                     # 4-panel EDA visualization
    ├── distractor_similarity.png            # Distractor lexical overlap chart
    └── baseline_results.json                # All experiment metrics
```
