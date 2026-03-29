# Open Tasks
*March 2026*

This document describes tasks that can be picked up independently without needing to understand the entire codebase first. All tasks are meaningful contributions to the thesis.

For the full project context, see `project_overview.md` in this folder.

---

## Quick Context

We are building a retrieval benchmark that tests whether AI models can match a **moral lesson** to the **fable** that illustrates it — e.g., given *"Appearances are deceptive"*, find the Aesop fable about the wolf in sheep's clothing.

The dataset is **MORABLES** — 709 fable–moral pairs. Our best model (Linq-Embed-Mistral) gets **R@1 = 14.1%**. If we give it the ground-truth moral directly, it gets **82.7%**. That gap is what the thesis is about.

Repo: `/Users/liorlivyatan/LocalProjects/Thesis/work_morables`
Key folders: `data/processed/`, `scripts/`, `results/`

---

## Track A — Cross-Lingual Retrieval (Hebrew) 🇮🇱

These tasks are fully independent and can be done in parallel.

### A1 — Hebrew Moral Mapping
**What:** Take the 678 unique morals in our dataset and find the closest Hebrew proverb or saying for each.

**Why it matters:** If we can query in Hebrew and retrieve English fables (or vice versa), that's a novel and impactful contribution — testing whether models can bridge *both* the language gap and the abstraction gap simultaneously.

**How:**
1. Load the morals from `data/processed/morals_corpus.json`
2. For each moral, find a matching Hebrew proverb (you can use Gemini/Claude to suggest candidates, then manually verify)
3. Examples of what we're looking for:
   - *"Appearances are deceptive"* → *"אל תסתכל בקנקן אלא במה שיש בו"*
   - *"Self-help is the best help"* → *"אם אין אני לי מי לי"*
   - *"United we stand, divided we fall"* → *"כל ישראל ערבים זה בזה"*
4. Save as `data/processed/moral_hebrew_mapping.json`

**Output format:**
```json
{
  "moral_idx": 0,
  "english_moral": "Gratitude is the sign of noble souls.",
  "hebrew_proverb": "...",
  "hebrew_transliteration": "...",
  "confidence": "high/medium/low",
  "notes": "..."
}
```

**Skills needed:** Hebrew language, basic Python/JSON

---

### A2 — Hebrew Fable Dataset Search
**What:** Search for existing digitized Hebrew fable collections that we could use as a retrieval corpus.

**Why it matters:** If we find Hebrew fables, we can build a cross-lingual task: *Hebrew moral → English fable* and *English moral → Hebrew fable*.

**Where to look:**
- **משלי שועלים** (Fox Fables by Berechiah ha-Nakdan) — medieval Hebrew fables translated from Aesop
- **Talmudic parables (משלים)** — rabbinical stories with explicit morals
- **Biblical proverbs** (משלי שלמה)
- HuggingFace datasets: search "Hebrew narrative", "Hebrew fable", "Hebrew moral"
- Project Gutenberg: Hebrew texts

**Deliverable:** A short document (can be a `.md` file) listing what exists, its format, size, license, and whether it's usable.

---

## Track B — Deeper Analysis

These tasks deepen our understanding of *why* models fail, which is essential for the thesis argument.

### B1 — Distractor Analysis
**What:** Figure out which type of distractor "fools" our best embedding model most.

**Why it matters:** MORABLES has 5 distractor types per fable: `similar_characters`, `based_on_adjectives`, `injected_adjectives`, `partial_story`, `ground_truth`. If the model consistently ranks `similar_characters` distractors above the correct moral, it means the model is matching character names — not moral meaning.

**How:**
1. Load the MCQA split: `data/raw/` (look for the mcqa config)
2. For each fable, embed all 5 options and rank them
3. For each distractor type, compute "how often does the model rank this distractor above the correct moral?"
4. Create a bar chart: distractor type vs. confusion rate
5. Save the chart to `results/distractor_analysis.png`

**Starter code:** `scripts/04_baseline_retrieval.py` has a stub function `distractor_analysis()` — implement it.

**Skills needed:** Python, matplotlib, basic understanding of embedding/cosine similarity

---

### B2 — Adversarial Robustness Evaluation
**What:** Test whether our best model stays robust when the fable is modified in adversarial ways.

**Why it matters:** MORABLES includes 18 adversarial variants (character name swaps, trait injections, tautological morals). If the model's ranking changes dramatically when we swap "Fox" for "Dog", it's matching character names, not moral content.

**How:**
1. Load adversarial data from `data/raw/` (look for `adversarial` config)
2. For each original fable + its adversarial variant, run retrieval with the best embedding model
3. Measure: does the ranking change? By how much?
4. Compare MRR on original vs. each adversarial type

**Skills needed:** Python, basic retrieval evaluation concepts

---

### B3 — Shared-Morals / Cross-Parable Analysis
**What:** 27 morals in our dataset are shared across multiple fables (e.g., both "The Tortoise and the Hare" and "The Ant and the Grasshopper" teach variations of the same patience/preparation lesson). Can our model find fables that teach the *same* moral?

**Why it matters:** This is the "cross-parable analogy" task — the hardest and most interesting version of moral retrieval. It tests whether models understand abstract moral equivalence across completely different surface stories.

**How:**
1. Load `results/shared_morals.png` to see the existing analysis
2. Extend `scripts/05_shared_morals_plot.py` to run retrieval: given fable A, can it retrieve fable B (which shares the same moral)?
3. Evaluate: for each shared-moral group, what fraction of the group does the model retrieve?

**Skills needed:** Python, basic evaluation metrics

---

## Track C — Dataset Expansion

### C1 — Survey Additional Fable/Moral Datasets
**What:** Find other existing datasets we could incorporate to grow beyond 709 fable–moral pairs.

**Candidates:**
| Dataset | Description | Size | Link |
|---------|-------------|------|------|
| STORAL (ACL 2022) | Story–moral pairs | ~3K | HuggingFace |
| Moral Stories (EMNLP 2021) | Short narratives with moral annotations | 12K | HuggingFace |
| ePiC | Proverb–narrative pairs | ? | Papers With Code |
| UniMoral | Unified moral taxonomy | ? | Papers With Code |
| Project Gutenberg | La Fontaine, Brothers Grimm fables | large | gutenberg.org |

**Deliverable:** For each dataset: download it, check the format, note the size and license, and write a short assessment of whether it fits our task.

---

## Suggested Split

| Student 1 | Student 2 |
|-----------|-----------|
| A1 — Hebrew moral mapping | A2 — Hebrew fable dataset search |
| B1 — Distractor analysis | B2 — Adversarial robustness |
| C1 — Dataset survey (half) | B3 — Shared-morals analysis |

---

## Getting Started

```bash
cd /Users/liorlivyatan/LocalProjects/Thesis/work_morables
source venv/bin/activate

# Explore the data
python scripts/01_load_data.py

# See what's already done
cat TASKS.md
```

Key files to read first:
- `meeting_materials/project_overview.md` — full project context
- `DATA_AND_EXPERIMENTS_GUIDE.md` — data format documentation
- `data/processed/fables_corpus.json` — the fables
- `data/processed/morals_corpus.json` — the morals
