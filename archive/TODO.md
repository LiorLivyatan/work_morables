# Project TODO

Master task list for the moral retrieval benchmark project. Designed to be split between collaborators.

**What's done so far:** Data downloaded, EDA complete, retrieval format ready, baseline embedding experiments done (Sentence-BERT with `all-MiniLM-L6-v2` only). See `TASKS.md` for details on completed work.

**What's ahead:** Everything below.

---

## Legend

- `[ ]` — Not started
- `[~]` — In progress / partially done
- `[x]` — Done
- Tags: `[INDEPENDENT]` = can be done without waiting for other tasks. `[BLOCKED BY X]` = must wait for task X.
- Suggested owner: `A` or `B` (to be assigned between collaborators)

---

## Track 1: Improve Retrieval Methods

**Goal:** Show that smarter approaches beat naive embedding similarity.

### 1.0 — Instruction-aware embedding models `[INDEPENDENT]`
- [ ] Test stronger embedding models beyond baseline Sentence-BERT:
  - **BGE-large-en-v1.5** (`BAAI/bge-large-en-v1.5`) — supports query instruction prefixes
  - **E5-large-v2** (`intfloat/e5-large-v2`) — requires `query: ` / `passage: ` prefixes
  - **Multilingual E5-large** (`intfloat/multilingual-e5-large`) — multilingual, same prefix format as E5
- [ ] Experiment with instruction prefix variants:
  - No prefix (raw text)
  - Default model prefix (as documented by model authors)
  - Task-specific prefix (e.g., "Given this fable, retrieve the abstract moral lesson it teaches: ")
- [ ] Handle asymmetric encoding: for fable→moral and moral→fable tasks, each text must be encoded twice (once as query, once as document)
- [ ] Run all 3 retrieval tasks per model-variant: fable→moral, moral→fable, fable→moral (augmented corpus)
- [ ] Create script `scripts/06_improved_retrieval.py`
- [ ] Save results to `results/improved_retrieval_results.json`

**Notes:** BGE uses query-only prefixes (no doc prefix). E5 requires both query and passage prefixes. Check each model's documentation for correct prefix format.

### 1.1 — Build CoT reranking & summarization script `[INDEPENDENT]`
- [ ] Create script `scripts/07_llm_reranking.py` with two approaches:
  - **Approach A — CoT Reranking:** Take top-K candidates from embedding retrieval, use an LLM (Gemini) with chain-of-thought reasoning to rerank them
  - **Approach B — CoT Moral Summarization:** Use an LLM to generate a moral summary from each fable, then embed the summary for retrieval
- [ ] Set up Gemini API integration (requires `GEMINI_API_KEY` in `.env`)
- [ ] Add CLI args: `--approach`, `--sample` (for quick tests), `--embed-model`, `--top-k`
- [ ] Add caching for generated moral summaries at `results/cot_moral_summaries.json`
- [ ] Add rate limiting for API calls

### 1.2 — Run full CoT reranking experiment `[BLOCKED BY 1.0, 1.1]`
- [ ] Run CoT reranking on all 709 fables (not just a small test)
- [ ] Use best embedding model from Task 1.0 as the Stage 1 retriever
- [ ] Try top-K = 20 and top-K = 50
- [ ] Compare gemini-2.0-flash vs gemini-2.5-pro (speed vs quality)
- [ ] Record results in `results/llm_reranking_results.json`

### 1.3 — Run full CoT moral summarization experiment `[BLOCKED BY 1.0, 1.1]`
- [ ] Run CoT summarization on all 709 fables
- [ ] This generates a moral summary per fable via Gemini, then embeds the summary for retrieval
- [ ] Summaries are cached — can inspect quality manually
- [ ] Compare: does embedding an LLM-generated moral work better than embedding the raw fable?

### 1.4 — Cross-encoder reranking `[INDEPENDENT]`
- [ ] Try a cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) on top of embedding results
- [ ] Cross-encoders score query-document pairs jointly (not independently like bi-encoders) — more powerful but slower
- [ ] Pipeline: embed to get top-50, rerank with cross-encoder
- [ ] Create a new script `scripts/08_cross_encoder_reranking.py`

### 1.5 — Contrastive fine-tuning `[BLOCKED BY 2.2]`
- [ ] Fine-tune an embedding model (e.g., E5 or MiniLM) using contrastive learning
- [ ] Positive pairs: (fable, correct moral)
- [ ] Hard negatives: MORABLES distractors from the MCQA data
- [ ] Loss function: `MultipleNegativesRankingLoss` or `TripletLoss` (sentence-transformers supports both)
- [ ] Need a proper train/eval split first (Task 2.2)

### 1.6 — Compare all methods `[BLOCKED BY 1.0, 1.2, 1.3, 1.4]`
- [ ] Create a single comparison table across all approaches
- [ ] Include: baseline embedding, instruction-aware embeddings, CoT reranking, CoT summarization, cross-encoder
- [ ] Create a visualization (bar chart) for the paper/presentation

---

## Track 2: Expand the Dataset

**Goal:** Grow beyond 709 pairs. More stories, unified moral vocabulary, multiple languages.

### 2.1 — Survey and collect additional fable/moral datasets `[INDEPENDENT]`
- [ ] Search HuggingFace, Papers With Code, GitHub for fable/moral/proverb datasets
- [ ] Candidates to investigate:
  - **STORAL** (ACL 2022) — ~3K story-moral pairs
  - **Moral Stories** (Emelin et al., EMNLP 2021) — 12K short narratives with moral annotations
  - **ePiC** — proverb-narrative pairs
  - **UniMoral** — unified moral taxonomy
  - Project Gutenberg fable collections (Aesop, La Fontaine, Brothers Grimm)
- [ ] For each dataset found: document format, size, language, license, quality
- [ ] Download and inspect promising ones

### 2.2 — Design train/eval split `[INDEPENDENT]`
- [ ] Decide on a split strategy for the existing 709 pairs (e.g., 80/20 or 5-fold CV)
- [ ] Ensure no fable leaks between train and eval
- [ ] Consider grouping fables by moral (27 morals are shared) — shared-moral fables should go in the same split
- [ ] Save split as a JSON file in `data/processed/`

### 2.3 — Moral deduplication/unification pipeline `[BLOCKED BY 2.1]`
- [ ] Collect all unique morals from all datasets
- [ ] Use an LLM (Gemini/Claude) to cluster semantically equivalent morals
  - Example: "Self-help is the best help" ≈ "God helps those who help themselves"
- [ ] Build a canonical moral mapping: each cluster gets one canonical form
- [ ] Create script `scripts/09_moral_unification.py`

### 2.4 — Convert and merge datasets `[BLOCKED BY 2.1, 2.3]`
- [ ] Normalize all datasets to a common schema: `{story, moral, canonical_moral_id, source_dataset, language}`
- [ ] Merge into a single unified dataset
- [ ] Update retrieval format files (corpus + qrels) to support many-to-one moral mapping
- [ ] Create script `scripts/10_dataset_merge.py`

### 2.5 — Synthetic story generation `[BLOCKED BY 2.3]`
- [ ] For each canonical moral, use an LLM to generate 3-5 new stories
- [ ] Constraints: different characters/settings, moral is implicit (not stated), no lexical overlap with moral text
- [ ] Quality filter: use LLM to verify "does this story teach this moral?"
- [ ] Can be used for both training augmentation and held-out evaluation
- [ ] Create script `scripts/11_synthetic_stories.py`

### 2.6 — Generate synthetic distractors for new data `[BLOCKED BY 2.4, 2.5]`
- [ ] For each new story, generate plausible-but-wrong morals (matching MORABLES distractor types)
- [ ] This creates a full augmented corpus for the expanded dataset

---

## Track 3: Cross-Lingual Moral Retrieval

**Goal:** Query in Hebrew → retrieve English fables (and vice versa).

### 3.1 — Build Hebrew moral/proverb mapping `[INDEPENDENT]`
- [ ] Take all morals from the MORABLES dataset
- [ ] Use an LLM to find the closest Hebrew proverb/saying for each
  - Example: "Self-help is the best help" → "אם אין אני לי מי לי"
  - Example: "Appearances are deceptive" → "אל תסתכל בקנקן אלא במה שיש בו"
- [ ] Manually curate/verify (this is a small annotation effort — 678 unique morals)
- [ ] Save as a mapping file: `data/processed/moral_hebrew_mapping.json`

### 3.2 — Search for Hebrew fable/parable datasets `[INDEPENDENT]`
- [ ] Look for digitized Hebrew fable collections:
  - משלי שועלים (Berechiah ha-Nakdan) — medieval Hebrew fables
  - משלי שלמה (Proverbs of Solomon) — biblical proverbs
  - Talmudic parables (משלים)
- [ ] Search for NLP datasets with Hebrew narratives + morals
- [ ] Document what exists and its format

### 3.3 — Cross-lingual retrieval experiment `[BLOCKED BY 3.1]`
- [ ] Query: Hebrew moral/proverb → Retrieve: English fable
- [ ] Use multilingual embedding models (e.g., `intfloat/multilingual-e5-large`)
- [ ] Evaluate: can the model bridge the language gap AND the abstraction gap?
- [ ] Create script `scripts/12_cross_lingual_retrieval.py`

---

## Track 4: Deeper Analysis

**Goal:** Understand *why* models fail and what makes certain pairs harder.

### 4.1 — Qualitative error analysis `[BLOCKED BY 1.0]`
- [ ] For the best embedding model, find the 50 hardest fable-moral pairs (correct moral ranked lowest)
- [ ] Manually categorize failure reasons:
  - Lexical confounders (model distracted by surface words)
  - Cultural knowledge required
  - Multi-step reasoning required
  - Ambiguous moral
- [ ] Document patterns in a short writeup

### 4.2 — Distractor analysis `[INDEPENDENT]`
- [ ] For each fable, check which distractor type the model ranks highest (similar_characters, based_on_adjectives, injected_adjectives, partial_story)
- [ ] This reveals *how* the model is confused — does it match characters? Traits? Story openings?
- [ ] The function `distractor_analysis()` in `04_baseline_retrieval.py` is a stub — implement it
- [ ] Visualize: bar chart of "which distractor type fools the model most"

### 4.3 — Adversarial robustness evaluation `[INDEPENDENT]`
- [ ] Use the 18 adversarial variants in `data/raw/adversarial_*.json`
- [ ] For each variant, run the same retrieval experiment and measure metric drop
- [ ] Example: if we swap character names (char_swap), does the model's ranking change?
- [ ] Create script `scripts/13_adversarial_eval.py`

### 4.4 — Shared-morals analysis `[INDEPENDENT]`
- [ ] 27 morals are shared across multiple fables — can the model find fables with the same moral?
- [ ] This is the "cross-parable analogy" task: given fable A, find fable B that teaches the same lesson
- [ ] Evaluation: for each shared-moral group, check if the model clusters them together
- [ ] Script `05_shared_morals_plot.py` exists with partial analysis — extend it

---

## Track 5: Writing & Presentation

### 5.1 — Write thesis proposal one-pager `[BLOCKED BY 1.6]`
- [ ] Follow the template in `Thesis_Proposal_template_example.pdf`
- [ ] Sections: intro, task definition, data, related work, research questions, timeline
- [ ] Include key results table and the "near-zero lexical overlap" finding

### 5.2 — Get advisor feedback `[BLOCKED BY 5.1]`
- [ ] Send to Kai for review
- [ ] Iterate based on feedback

---

## Suggested Work Split

Tasks tagged `[INDEPENDENT]` can be started right away by either person. Here's one possible split:

| Person A | Person B |
|----------|----------|
| 1.0 — Instruction-aware embeddings | 2.1 — Survey additional datasets |
| 1.1 — Build CoT scripts | 3.1 — Hebrew moral mapping |
| 1.4 — Cross-encoder reranking | 3.2 — Hebrew dataset search |
| 4.2 — Distractor analysis | 4.1 — Error analysis (after 1.0) |
| 4.3 — Adversarial evaluation | 4.4 — Shared-morals analysis |

After these are done, the blocked tasks (1.2, 1.3, 1.5, 1.6, 2.3–2.6, 3.3, 5.1–5.2) open up and can be split further.

---

## Priority Order

If time is limited, focus on these first:

1. **1.0** (instruction-aware embeddings) — find the best embedding model first
2. **1.1** (build CoT scripts) — enables the LLM-based experiments
3. **1.2 + 1.3** (run CoT experiments) — strongest expected results, most novel
4. **2.1** (dataset survey) — enables everything in Track 2
5. **4.1 + 4.2** (error + distractor analysis) — essential for understanding and for the paper
6. **3.1** (Hebrew mapping) — unique contribution, enables cross-lingual track
7. **2.2** (train/eval split) — needed before any fine-tuning
