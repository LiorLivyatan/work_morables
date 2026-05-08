# MORABLES — Team Meeting Briefing

**Date:** 2026-05-06  
**Model under analysis:** ft_07 — Linq-Embed-Mistral, fine-tuned on STORAL s500, fable+summary doc format  
**Dataset:** 709 Aesop fables × 709 morals

---

## What Is This Project?

We built a retrieval system that, given a moral lesson (like *"look before you leap"*), finds the correct Aesop fable that teaches that lesson. We have **709 fables**, each paired with a moral. The model must rank the correct fable at position 1 out of 709.

**Key metric: MRR@10** — Mean Reciprocal Rank. If the correct fable is always at rank 1, MRR = 1.0. If it's never found, MRR = 0. Higher is always better. We also report:
- **R@1** — how often the correct fable is the very first result
- **R@5** / **R@10** — how often it appears in the top 5 or top 10

---

## The Story in One Table

> Full data: [`docs/results_all_experiments.csv`](results_all_experiments.csv)

| Stage | What We Tried | Best MRR@10 | R@1 |
|---|---|---|---|
| Off-the-shelf embedding models | No training, just encode and search | 0.095 | 5.4% |
| LLM-generated summaries + embeddings | Ask an LLM to describe the fable's moral, then embed | 0.363 | 25.5% |
| Fine-tuning on MORABLES directly | Train the embedding model on our own data | 0.438 | 31.9% |
| **Fine-tuning on external STORAL data** | **Train on a different dataset, test on ours** | **0.450** | **33.1%** |
| Oracle — actual moral appended to fable | Upper bound: attach the correct moral from the answer key directly to each fable | **0.893** | **82.7%** |

---

## Phase 1 — Baseline Models: No Training (Exp 01, 02, 11, 12)

We first tested how well embedding models perform out-of-the-box with no training at all. The idea is simple: encode the moral as a query, encode each fable as a document, and rank by cosine similarity. No learning happens.

**Key finding: the document format matters more than the model.**

Adding an AI-generated moral summary to each fable document — format: `{fable text}\n\nMoral summary: {summary}` — roughly **doubled MRR** for most models compared to raw fable text alone.

**Best zero-shot results (Exp 12 — 8 models × 6 corpus formats × 2 instruction variants):**

> Full data: [`experiments/12_zero_shot_comprehensive/results/zero_shot_comprehensive.csv`](../experiments/12_zero_shot_comprehensive/results/zero_shot_comprehensive.csv)

| Model | Corpus Format | Instruction | MRR@10 | R@1 | R@10 |
|---|---|---|---|---|---|
| Qwen3-Embedding-8B | Fable + CoT summary | Generic | **0.363** | 25.5% | 58.1% |
| Linq-Embed-Mistral | Fable + CoT summary | Generic | 0.360 | 25.1% | 57.8% |
| BGE-en-ICL | Fable + direct moral summary | Generic | 0.321 | 22.1% | 51.3% |
| Qwen3-Embedding-0.6B | Fable + direct moral summary | Task-specific | 0.256 | 16.5% | 42.6% |
| Nomic-Embed-v2-MoE | Fable + CoT summary | Default | 0.252 | 17.2% | 38.6% |
| Linq-Embed-Mistral | Raw fable text only | Default | 0.095 | 5.4% | 16.8% |

**Note on instructions:** "Generic" instruction = *"Given a text, retrieve the most relevant passage that answers the query"*. Task-specific = *"Given a moral statement, retrieve the fable that best conveys this moral."* Counterintuitively, the generic instruction wins for most models — including Linq, where it's ~70% better zero-shot.

---

## Phase 2 — LLM Summarization (Exp 03, 07, 08, 13)

Instead of encoding the raw fable, we ask an LLM to first write a one-line moral description for each fable. That summary gets appended to the fable before encoding. The quality of the summary directly determines how good retrieval is.

### Exp 07 — Oracle Upper Bound

The oracle has nothing to do with Gemini. It is the simplest possible cheat: take each fable and append its own **actual annotated moral** from the dataset directly — `{fable text}\n\nMoral summary: {actual moral}`. At query time, the moral query and the fable document now say essentially the same thing, so any embedding model trivially matches them.

This is called an oracle because it uses the answer key. In a real system you would never have the correct moral pre-attached to each fable. It tells us the theoretical ceiling: the best MRR we could ever reach if we had a perfect description of each fable's moral already written.

**Result: MRR = 0.893, R@1 = 82.7%** — the remaining ~10% that still fail are mostly annotation duplicates where two fables share the exact same moral text.

**What Gemini actually did in Exp 07:** After establishing the oracle, we used Gemini Flash to *generate* moral summaries for all 709 fables — since you obviously can't attach the answer key in production. The best Gemini variant (`cot_proverb`, Chain-of-Thought prompting) reached **MRR ≈ 0.36 zero-shot**. These Gemini-generated summaries then became the `fable_plus_summary` document format used in all subsequent fine-tuning experiments (ft_02, ft_03, ft_07, ft_08).

### Exp 13 — Gemma 4 GPU Evaluation (run today)

We tested whether Google's open-source **Gemma 4** models can replace Gemini for generating summaries. Models tested: Gemma4-E2B (2B), E4B (4B), 26B-A4B (MoE), 31B — plus Gemini as a reference. All evaluated with Qwen3-Embedding-8B and Linq-Embed-Mistral as embedders.

> Full data: [`experiments/13_gemma4_gpu_summarization/results/2026-05-06_11-54-49_eval_all.csv`](../experiments/13_gemma4_gpu_summarization/results/2026-05-06_11-54-49_eval_all.csv)

**Best result per generator (Qwen3-8B embedder, fable+summary corpus):**

| Generator | Best Prompt | MRR@10 | R@1 | R@10 |
|---|---|---|---|---|
| Gemini (reference) | conceptual_abstract, summary only | **0.350** | 24.3% | 55.4% |
| Gemma4-31B | conceptual_abstract | 0.304 | 19.9% | 50.4% |
| Gemma4-E4B (4B) | thinking + direct_moral | 0.296 | 20.2% | 47.5% |
| Gemma4-E2B (2B) | thinking + cot_proverb | 0.247 | 15.2% | 43.0% |
| Gemma4-26B-A4B (MoE) | — | **0.010** | 0.1% | 1.4% |

**Notable failure:** Gemma4-26B-A4B (the large MoE model) produced identical output for every single fable, resulting in near-zero MRR. This is a known issue with quantized MoE models producing degenerate text under certain sampling conditions.

**Bottom line:** Gemma4-31B reaches ~85% of Gemini's quality (MRR 0.304 vs 0.350). The 4B model with thinking mode is surprisingly competitive. But none of the Gemma 4 models match our fine-tuned results (0.450).

---

## Phase 3 — Fine-Tuning

### What is fine-tuning, in simple terms?

The embedding model was pre-trained on billions of general text pairs from the internet. Fine-tuning means we continue training it — but now specifically on (moral → fable) pairs. We're teaching the model: "when you see this moral, recognize that *this fable* is the matching document."

### What is LoRA?

Updating all 7 billion parameters of a large model is very expensive. **LoRA** (Low-Rank Adaptation) adds small trainable "adapter" layers on top of the frozen model weights. For Linq-7B, this means only 54 million parameters are actually trained (0.76% of the model), making it fast and memory-efficient while keeping the full model capacity.

Key LoRA parameters: **r=64** (adapter size), **alpha=128** (scaling), **dropout=0.05** (regularization).

### How the Training Works (InfoNCE Loss)

During training, we show the model a batch of 16 (moral, fable) pairs at once. For each moral in the batch, the model must score its correct fable higher than all other fables in the batch. Temperature τ=0.05 makes this strict — there has to be a clear margin, not just a slight preference.

> All fine-tuning hyperparameters: [`docs/finetuning_configs.csv`](finetuning_configs.csv)

---

### ft_01 — BGE-base Baseline

| Parameter | Value |
|---|---|
| Model | BGE-base-en-v1.5 (110M params) |
| Training data | 709 MORABLES pairs, 5-fold CV |
| Doc format | Raw fable text |
| Epochs | 30 |
| Batch size | 32 |
| **MRR@10** | **0.122 ± 0.022** |

BGE-base is too small. Even after training directly on our data, it barely improves over random.

---

### ft_02 — Linq-Embed-Mistral + LoRA (First Large Model)

| Parameter | Value |
|---|---|
| Model | Linq-Embed-Mistral (7B — 60× bigger than BGE-base) |
| Training data | 709 MORABLES pairs, 5-fold CV |
| LoRA | r=64, alpha=128, target: q/k/v/o projections |
| Epochs | 10, early stopping (patience=3) |
| Effective batch | 16 (batch=4, grad accumulation=4) |
| Learning rate | 1e-4 |
| Temperature τ | 0.05 |
| **MRR@10 (raw)** | **0.318** |
| **MRR@10 (+ summary at eval)** | **0.416** |

Switching to the 7B model gives a massive jump. Importantly: even though training used raw fables, adding the AI summary at evaluation time still helps (+9.8pp), because the model learned the underlying semantics.

---

### ft_03 — Hard Negative Mining

Same model and hyperparameters as ft_02, but with a smarter training signal.

**Standard training:** For each moral, the 15 "negative" fables it must push away are random.

**Hard negative mining:** Instead of random negatives, we use fables that share themes or adjectives with the correct fable — the ones the model is most likely to confuse. This directly targets the failure mode.

> Full results: [`docs/results_ft07_comprehensive.csv`](results_ft07_comprehensive.csv)

| Parameter | Value |
|---|---|
| Model | Linq-Embed-Mistral + LoRA |
| Hard negatives | Fables sharing adjectives/themes with the correct fable |
| **MRR@10** | **0.438** |

Modest gain (+2.2pp over ft_02 with summary). The hard negatives help, but we're still limited by training on MORABLES which has annotation noise.

---

### ft_07 — STORAL Transfer Learning (Best Result ⭐)

**The key insight from earlier experiments:** What if we *don't* train on MORABLES at all, and instead train on a completely different dataset?

**STORAL** is an external dataset of 1,675 (moral, story) pairs from modern English stories — a different domain from Aesop fables. We trained on STORAL and evaluated directly on MORABLES, with no in-domain training.

We swept three things: **3 models** × **3 dataset sizes** × **2 doc formats** = 18 configurations.

STORAL subsets:
- **s500**: 500 stories length-matched to fables (56–195 words) — most similar to our domain
- **s1000**: 1000 stories (30–300 words)
- **sfull**: all 1675 clean pairs

> Full data: [`docs/results_ft07_comprehensive.csv`](results_ft07_comprehensive.csv)

**All ft_07 results (fable+summary corpus):**

| Model | Training Set | MRR@10 | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|
| BGE-base | STORAL s500 | 0.255 | 17.1% | 34.6% | 43.3% |
| BGE-base | STORAL s1000 | 0.256 | 16.6% | 35.4% | 43.2% |
| BGE-base | STORAL sfull | 0.260 | 17.2% | 35.1% | 41.8% |
| Qwen3-8B | STORAL s500 | 0.425 | 30.9% | 56.0% | 64.9% |
| Qwen3-8B | STORAL s1000 | 0.430 | 31.2% | 55.9% | 66.2% |
| Qwen3-8B | STORAL sfull | 0.435 | 31.6% | 56.3% | 68.8% |
| **Linq** | **STORAL s500** | **0.450** | **33.1%** | **59.0%** | **67.8%** |
| Linq | STORAL s1000 | 0.422 | 30.3% | 55.4% | 64.7% |
| Linq | STORAL sfull | 0.433 | 31.3% | 57.1% | 64.5% |

**Key observations:**
1. **Linq on s500 is the winner.** Training on only 500 external stories beats training on 1,675.
2. **More data ≠ better.** The 500 length-matched stories are more similar to fables than the full 1,675, so quality and domain match matters more than quantity.
3. **Transfer learning works surprisingly well.** Training on a completely different domain (modern stories) still produces the best fine-tuned model on fables.

---

### ft_08 — Annotation Noise Sanity Check (In Progress)

> Status: Running now on GPU 0. Results pending.

**Motivation:** 13.3% of the dataset has annotation problems (same moral assigned to multiple fables). The model gets penalized for retrieving a valid fable that just happens to not be the one that was annotated. This suppresses our reported MRR.

**Design:** Run the exact same ft_07 best setup (Linq + LoRA, STORAL s500 transfer) twice:
- **Noisy variant:** all 709 queries — standard setup, apples-to-apples with ft_07
- **Clean variant:** only 615 queries — 94 duplicate/ambiguous morals removed

The difference in MRR tells us exactly how much annotation noise is holding back the reported numbers.

**Expected gain:** +3–6pp on clean variant if the noise hypothesis is correct.

---

## Analysis of Best Model (ft_07 Linq s500)

After identifying the best model, we ran a detailed analysis on all 709 queries to understand where and why it fails.

### Overall Performance

> Source: [`analysis/01_rank_distribution/results/ft07-linq-s500-fable+summary/rank_distribution.csv`](../analysis/01_rank_distribution/results/ft07-linq-s500-fable+summary/rank_distribution.csv)

| Metric | Value |
|---|---|
| MRR@10 | **0.440** |
| R@1 | 33.1% (235 / 709 queries) |
| R@5 | 58.9% |
| R@10 | 67.7% |
| Median ground-truth rank | 3 |
| Mean ground-truth rank | 31.5 |

The model gets the right fable at rank 1 for 1 in 3 queries. On average, the correct fable appears around rank 2–3. 67.7% of correct answers are somewhere in the top 10.

### Where Do Failures Land?

Of the 709 queries, **474 are misranked (66.9%)** — the correct fable does not appear at rank 1.

| Rank bucket | Count | % of all queries | Cumulative |
|---|---|---|---|
| Rank 1 (correct) | 235 | 33.1% | 33.1% |
| Rank 2 | 79 | 11.1% | 44.3% |
| Rank 3 | 46 | 6.5% | 50.8% |
| Rank 4 | 38 | 5.4% | 56.1% |
| Rank 5 | 20 | 2.8% | 59.0% |
| Rank 6–10 | 62 | 8.7% | 67.7% |
| Rank 11–50 | 131 | 18.5% | 86.2% |
| Rank 51+ | 98 | 13.8% | 100% |

Most failures are close — 50.8% of all queries have the correct fable in the top 3. But a long tail of 98 queries (13.8%) have the correct fable ranked below 50.

### How Confident Are the Errors?

For each misranked query, we measured the **score gap**: how much higher the wrong rank-1 fable scored versus the correct fable.

> Source: [`analysis/03_score_gap_distribution/results/ft07-linq-s500-fable+summary/score_gaps.csv`](../analysis/03_score_gap_distribution/results/ft07-linq-s500-fable+summary/score_gaps.csv)

| Score gap | Count | % of misranked |
|---|---|---|
| < 0.02 (near-miss) | 37 | 7.8% |
| 0.02–0.10 | 174 | 36.7% |
| **> 0.10 (confident wrong)** | **263** | **55.5%** |

- Mean gap: **0.135** — Median gap: **0.114** — Max gap: **0.486**

Over half of errors are **confident mistakes**, not near-misses. The model is not slightly confused — it is strongly pulling toward the wrong fable.

### Dataset Annotation Problems (Most Important Finding)

**The dataset has structural issues that directly cause apparent model failures.**

> Source: [`docs/annotation_issues.csv`](annotation_issues.csv) | [`analysis/07_soft_mrr/results/ft07-linq-s500-fable+summary/ambiguous_queries.csv`](../analysis/07_soft_mrr/results/ft07-linq-s500-fable+summary/ambiguous_queries.csv)

#### Exact Duplicate Morals

**27 unique moral texts are each assigned as the ground truth for 2 or more different fables.**  
These affect **58 queries (8.2% of the dataset).**

When the model retrieves fable A for a moral that is also the ground truth of fable B, it is scored as wrong — even though it retrieved a perfectly valid answer.

| Moral | Fables | Count |
|---|---|---|
| "Nature reveals itself." | The Cat-Maiden, The Cat and Venus, The Kingdom of The Lion, Orpheus and the Dogs | 4 |
| "Look before you leap." | The Fox and the Goat, The Fox And The Goat In The Well, The Two Frogs At The Well | 3 |
| "Self-help is the best help." | Hercules and the Wagoner, The Lark and Her Young Ones, The Crested Lark And The Farmer | 3 |
| "Be careful what you wish for." | The Poor Man And Death, The Eyes and the Honey | 2 |
| "Greed often overreaches itself." | The Goose With the Golden Eggs, The Mouse and the Oyster | 2 |
| "Appearances are deceptive." | The Ant and the Chrysalis, The Wolf in Sheep's Clothing | 2 |

#### Near-Duplicate Morals

Beyond exact duplicates, **18 additional moral pairs** (36 queries) have nearly identical texts with cosine similarity > 0.90:

| Moral A | Moral B | Similarity |
|---|---|---|
| "Enemies' promises are made to be broken." | "Enemies' promises were made to be broken." | 0.991 |
| "Wit has always an answer ready." | "Wit always has an answer ready." | 0.991 |
| "Nothing escapes the master's eye" | "Nothing escapes the master's eye." | 0.977 |

**Total queries affected by annotation problems: 94 / 709 (13.3%)**

### Thematic Ambiguity (Model Confusion)

For every misranked query, we checked whether the wrong rank-1 fable belongs to the same moral theme as the correct fable.

> Source: [`analysis/04_thematic_overlap/results/ft07-linq-s500-fable+summary/thematic_overlap.csv`](../analysis/04_thematic_overlap/results/ft07-linq-s500-fable+summary/thematic_overlap.csv)

**158 / 474 confused pairs (33.3%) share the same moral category.**  
This means a third of all failures happen inside the same theme cluster — the model is choosing between two fables that genuinely teach the same type of lesson.

| Moral-moral similarity threshold | Confused pairs above threshold | % of confused |
|---|---|---|
| > 0.70 | 79 | 16.7% |
| > 0.80 | 46 | 9.7% |
| > 0.85 | 32 | 6.8% |
| > 0.90 | 26 | 5.5% |

32 confused pairs have morals so semantically similar (similarity > 0.85) that even a human might struggle to distinguish them.

### The Model Is Often Right When It Looks Wrong

Looking at the 4 hardest errors (largest score gap):

> Source: [`analysis/02_nearest_neighbor_confusion/results/ft07-linq-s500-fable+summary/confusion_cases.csv`](../analysis/02_nearest_neighbor_confusion/results/ft07-linq-s500-fable+summary/confusion_cases.csv)

**Case 1 — moral: "Be careful what you wish for."**
- Model retrieves: *The Shepherd And The Lion* (score 0.812) — a shepherd makes a vow wishing to find a thief, and wishes he hadn't when he finds it's a lion. **Directly on-theme.**
- Ground truth: *The Poor Man And Death* (score 0.325, rank **76**) — a man calls for Death and then backtracks.
- Score gap: **0.486** — the model is very confident, and arguably correct.

**Case 2 — moral: "Those who seek to please everybody please nobody."**
- Model retrieves: *The Miller, His Son, and Their Ass* (score 0.886) — literally a story about trying to please everyone and ending up pleasing no one. **Perfect match.**
- Ground truth: *The Bald Man And His Two Mistresses* (score 0.466, rank **20**) — about vanity and rivalry.
- Score gap: **0.420**

**Case 3 — moral: "Greed for more can lead to losing what you already have."**
- Model retrieves: *Zeus And The Camel* (score 0.726) — a camel asks Zeus for horns out of greed, and gets its ears cropped instead. **On-theme.**
- Ground truth: *The Jackdaw And The Doves* (score 0.288, rank **235**) — categorised as *deception*, not greed.
- Score gap: **0.437**

**Case 4 — moral: "Understand what you are doing before you do it."**
- Model retrieves: *The Fox And The Hare In The Well* (score 0.688) — a hare jumps into a well without planning how to get out. **On-theme.**
- Ground truth: *The Shepherd and the Sea* (score 0.254, rank **270**) — a shepherd sells his flock to trade at sea without understanding it.
- Score gap: **0.434**

In all four cases, a reasonable person would argue the model's answer is at least as correct as the annotated ground truth — and in cases 2 and 3, arguably **more** correct.

### Ambiguity-Corrected Performance

> Source: [`analysis/07_soft_mrr/results/ft07-linq-s500-fable+summary/soft_mrr_summary.txt`](../analysis/07_soft_mrr/results/ft07-linq-s500-fable+summary/soft_mrr_summary.txt)

| Evaluation | MRR@10 | n queries | vs. standard |
|---|---|---|---|
| Standard | 0.440 | 709 | — |
| Soft MRR (forgive retrievals with semantically equivalent moral) | **0.468** | 709 | +2.8% |
| Clean subset (exclude 94 ambiguous queries) | **0.448** | 615 | +0.8% |

42 queries were "rescued" by the soft correction — the model retrieved a fable whose own moral is semantically equivalent to the query moral. The clean subset gain (+0.8%) is modest because the model was still *trained* on the noisy data. ft_08 will show what happens when training is also on the clean set.

### Attractor Fables (Not a Major Issue)

We checked whether a few specific fables were "absorbing" too many wrong retrievals — i.e., the model always retrieves them regardless of the query.

The effect is weak. The worst offender, *The Doves And The Kite*, steals only **7 queries** (1.0%). The top 5 attractors account for 26 false positives out of 474. 305 out of 709 fables appear as a false positive at rank 1 at least once — errors are widely distributed, not concentrated on a few fables.

> Source: [`analysis/05_length_richness_bias/results/ft07-linq-s500-fable+summary/false_positive_fables.csv`](../analysis/05_length_richness_bias/results/ft07-linq-s500-fable+summary/false_positive_fables.csv)

---

## What Is Actually Causing the Failures?

| Cause | Estimated scope | Confidence |
|---|---|---|
| Exact duplicate morals (same text, different fable) | 58 queries (8.2%) | **Certain** |
| Near-duplicate morals (same meaning, slightly different text) | ~36 queries (5.1%) | High |
| Thematic ambiguity (same moral category, debatable annotation) | ~158 queries (22.3%) | Medium |
| Genuine model errors (model is wrong, annotation is right) | ~220 queries (31%) | Medium |
| Attractor/geometry effects | Negligible | High |

**The single most important finding:** at least 13.3% of the dataset has annotation problems severe enough that the model cannot be expected to get them right. A significant additional fraction (~22%) involves genuine ambiguity. Only roughly 30% of all queries represent clear-cut cases where the annotation is unambiguous and the model is genuinely wrong.

---

## Next Steps

### Track 1 — Fix the Data

1. **Remove or merge exact duplicates (58 queries):** For morals like "Look before you leap." that map to 3 fables, either pick one canonical fable or treat all 3 as valid answers (multi-label evaluation).
2. **Review near-duplicates (36 queries):** Standardise wording where the moral text differs only trivially.
3. **Sanity check — ft_08 (in progress):** Retrain on the cleaned 615-query dataset. Expected MRR boost: +3–6% from removing noise alone.

### Track 2 — Fix the Model

For the ~220 queries with genuine model errors:

1. **Hard negative mining (targeted):** Use the model's own top false positives as hard negatives in the next training round. This directly targets the 55.5% of errors that are "confident wrong" (gap > 0.10).
2. **Multi-positive training:** For the 27 duplicate-moral groups, instead of discarding them, treat all valid fables as positive targets and use a listwise loss. This turns an annotation problem into a training signal.

### The Gap to Close

| | MRR@10 |
|---|---|
| Current best (ft_07) | 0.450 |
| Annotation-corrected estimate | ~0.468–0.480 |
| Oracle ceiling (actual moral appended to fable) | 0.893 |

The gap between our best result and the oracle (0.443) tells us there is enormous room for improvement — most of it in closing the summarization quality gap, which is what the Gemma 4 experiments are investigating.

---

## All Result Files

| What | File |
|---|---|
| Master results table (all experiments) | [`docs/results_all_experiments.csv`](results_all_experiments.csv) |
| ft_07 full results (all 18 configs) | [`docs/results_ft07_comprehensive.csv`](results_ft07_comprehensive.csv) |
| Fine-tuning hyperparameter table | [`docs/finetuning_configs.csv`](finetuning_configs.csv) |
| Annotation issues (duplicates) | [`docs/annotation_issues.csv`](annotation_issues.csv) |
| Exp 12 — zero-shot comprehensive matrix | [`experiments/12_zero_shot_comprehensive/results/zero_shot_comprehensive.csv`](../experiments/12_zero_shot_comprehensive/results/zero_shot_comprehensive.csv) |
| Exp 13 — Gemma 4 full results | [`experiments/13_gemma4_gpu_summarization/results/2026-05-06_11-54-49_eval_all.csv`](../experiments/13_gemma4_gpu_summarization/results/2026-05-06_11-54-49_eval_all.csv) |
| Analysis — rank distribution | [`analysis/01_rank_distribution/results/ft07-linq-s500-fable+summary/rank_distribution.csv`](../analysis/01_rank_distribution/results/ft07-linq-s500-fable+summary/rank_distribution.csv) |
| Analysis — confusion cases (474 misranked) | [`analysis/02_nearest_neighbor_confusion/results/ft07-linq-s500-fable+summary/confusion_cases.csv`](../analysis/02_nearest_neighbor_confusion/results/ft07-linq-s500-fable+summary/confusion_cases.csv) |
| Analysis — score gap distribution | [`analysis/03_score_gap_distribution/results/ft07-linq-s500-fable+summary/score_gaps.csv`](../analysis/03_score_gap_distribution/results/ft07-linq-s500-fable+summary/score_gaps.csv) |
| Analysis — thematic overlap in failures | [`analysis/04_thematic_overlap/results/ft07-linq-s500-fable+summary/thematic_overlap.csv`](../analysis/04_thematic_overlap/results/ft07-linq-s500-fable+summary/thematic_overlap.csv) |
| Analysis — false positive fables | [`analysis/05_length_richness_bias/results/ft07-linq-s500-fable+summary/false_positive_fables.csv`](../analysis/05_length_richness_bias/results/ft07-linq-s500-fable+summary/false_positive_fables.csv) |
| Analysis — ambiguous queries list | [`analysis/07_soft_mrr/results/ft07-linq-s500-fable+summary/ambiguous_queries.csv`](../analysis/07_soft_mrr/results/ft07-linq-s500-fable+summary/ambiguous_queries.csv) |
