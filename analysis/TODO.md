# MORABLES — Retrieval Interpretability Analysis

## Background

Our best retrieval models achieve MRR@10 ≈ 0.40–0.45, meaning the correct fable is on
average ranked 2nd–3rd. The model "understands" the moral — it retrieves a semantically
related fable as rank 1 — but consistently has one or two fables above the ground truth.

The key question is: **why is there always one fable that ranks above the correct one?**
Is it ambiguity in the dataset (multiple fables genuinely match the moral), surface-level
lexical confusion, or a geometric problem in embedding space?

Understanding this drives the next experimental step: if it is ambiguity, we need better
negative mining or dataset curation. If it is surface confusion, we need richer document
representations. If it is geometric, we need better fine-tuning objectives.

---

## Goals

1. Identify the dominant failure mode for each experiment configuration.
2. Produce paper-ready charts, tables, and CSV files for each analysis.
3. Make every analysis **plug-and-play**: point it at any experiment's embedding cache
   and it runs — zero re-embedding needed.

---

## What Is Fixed vs Configurable

### Fixed across ALL experiments
- The 709 fables (corpus), 709 morals (queries), and ground-truth qrels — these never
  change between experiments.
- The evaluation task: moral → fable retrieval (MRR@10).
- The embedding format: `moral_embs.npy` (709 × D) and `doc_embs.npy` (709 × D),
  already L2-normalised so cosine similarity = dot product.

### Configurable per experiment run
- **Embedding paths**: `moral_embs.npy` and `doc_embs.npy` — point to any experiment's
  cache directory (ft_01, ft_07/linq_s500, exp_12/Qwen3, etc.)
- **Experiment label**: human-readable name for plot titles and CSV filenames.
- **Corpus config**: which document representation was embedded (`raw`, `fable_plus_summary`,
  `summary_fable_cot`, etc.) — used for labelling only; the .npy files already encode
  whatever format was used.
- **Additional metadata**: result JSON path (if available) for pulling pre-computed metrics
  to cross-reference.

### How to run any analysis
Every script in `analysis/0N_*/analyze.py` accepts at minimum:

```bash
./run.sh analysis/0N_xxx/analyze.py \
    --moral_embs  <path/to/moral_embs.npy> \
    --doc_embs    <path/to/doc_embs.npy> \
    --label       "ft07-linq-s500-fable+summary" \
    --output_dir  analysis/0N_xxx/results/
```

Additional per-script flags are documented in each script's argparse help.

---

## Analysis Approaches

---

### 01 — Rank Distribution
**Directory:** `01_rank_distribution/`
**Script:** `analyze.py`

**Question:** Where does the ground-truth fable actually land in the ranking — is MRR
driven by mostly rank-2 near-misses, or by a long tail of bad cases?

**Outputs:**
- `rank_distribution.png` — histogram of ground-truth ranks (1–10+)
- `rank_distribution.csv` — raw counts per rank bucket
- `cumulative_recall.png` — R@K curve (K = 1..50)

**Fixed:** Rank computation from embeddings + qrels.
**Configurable:** `--top_k` (default 50), `--bins` for histogram bucketing.

**What to look for:**
If ~60%+ of failures are at rank 2, the model is almost-right and the problem is
fine-grained discrimination. If ranks are spread, the model has a broader understanding
gap.

---

### 02 — Nearest-Neighbor Confusion
**Directory:** `02_nearest_neighbor_confusion/`
**Script:** `analyze.py`

**Question:** For each misranked query, what fable beat the ground truth and why?
Side-by-side: [moral] vs [rank-1 wrong fable] vs [ground-truth fable].

**Outputs:**
- `confusion_cases.csv` — for every moral: moral_text, gt_fable, gt_rank, top1_fable,
  top1_score, gt_score, score_gap
- `hardest_cases.md` — top-N hardest misranked cases rendered as human-readable prose
  (directly citable in paper)
- `confusion_heatmap.png` — pairwise confusion between fable "themes" (if themes are
  available from analysis 04)

**Fixed:** The side-by-side comparison logic and rendering.
**Configurable:** `--n_cases` (how many hard cases to render), `--rank_threshold`
(only show cases where gt_rank ≥ 2), `--themes_csv` (optional output from analysis 04).

**What to look for:**
If the wrong fable is thematically very similar to the correct one → ambiguity problem.
If it shares surface keywords with the moral → lexical confusion.
If it is unrelated → embedding space geometry issue.

---

### 03 — Score Gap Distribution
**Directory:** `03_score_gap_distribution/`
**Script:** `analyze.py`

**Question:** How large is the score gap between rank-1 and the ground-truth fable?
Are failures near-misses (gap ≈ 0.01) or confident mistakes (gap ≈ 0.1+)?

**Outputs:**
- `score_gap_histogram.png` — distribution of `score(rank-1) - score(ground-truth)`
  for all misranked queries
- `score_gap_by_rank.png` — box plot: score gap grouped by ground-truth rank
- `score_gaps.csv` — raw data: moral_id, gt_rank, gap, gt_score, rank1_score

**Fixed:** Gap = score(rank-1) - score(gt) for all queries where gt_rank > 1.
**Configurable:** `--bins`, `--compare` (pass multiple experiment label+emb pairs to
overlay distributions on one plot — useful for zero-shot vs ft comparison).

**What to look for:**
Tight gap distribution (mean gap < 0.02) → fine-tuning with harder negatives may help.
Wide gap → the model is genuinely confused, need richer representations.

---

### 04 — Thematic Overlap / Mutual Moral Fitness
**Directory:** `04_thematic_overlap/`
**Script:** `analyze.py`

**Question:** Does the rank-1 wrong fable's OWN ground-truth moral overlap with the
query moral? If so, it is a genuine dataset ambiguity — two fables share the same theme.

**Outputs:**
- `thematic_overlap.csv` — for each confused pair: query_moral, wrong_fable_moral,
  lexical_overlap (Jaccard on keywords), semantic_overlap (moral-moral cosine sim)
- `ambiguity_score_distribution.png` — distribution of moral-moral semantic similarity
  for confused pairs vs random pairs
- `ambiguous_pairs.md` — cases where moral-moral similarity > threshold (potential
  dataset annotation issues worth noting in paper)

**Fixed:** The ambiguity metric (moral-moral cosine sim using the SAME embedding model).
**Configurable:** `--moral_embs` (the moral embeddings, used for moral-moral sim),
`--ambiguity_threshold` (default 0.8), keyword extraction method (`tfidf` or `keybert`).

**What to look for:**
High moral-moral similarity in confused pairs → dataset ambiguity, not model failure.
Low similarity → the model is making a real error.

---

### 05 — Length & Richness Bias
**Directory:** `05_length_richness_bias/`
**Script:** `analyze.py`

**Question:** Do longer or richer fables systematically rank higher, regardless of the
query? A fable with more text captures more semantic space and may "absorb" morals that
belong to other fables.

**Outputs:**
- `length_vs_rank.png` — scatter: fable word count vs mean rank across all queries
- `richness_bias.csv` — per-fable: word_count, mean_rank_as_retrieved, n_times_rank1,
  n_times_false_positive (rank-1 when NOT the gt)
- `false_positive_fables.md` — top-N fables that most often appear at rank-1 for the
  wrong moral

**Fixed:** Word count as the richness proxy; false-positive = rank-1 but not gt.
**Configurable:** `--richness_metric` (`word_count`, `unique_words`, `sentence_count`),
`--n_top_fp` (how many false-positive fables to show).

**What to look for:**
If the same 5–10 fables keep appearing as false positives → "attractor fables" that
dominate embedding space and pull many morals toward them.

---

### 06 — Embedding Space Geometry (Layer Probing)
**Directory:** `06_layer_probing/`
**Script:** `analyze.py`

**Question:** Where in the embedding space do morals and fables sit? Are correct pairs
geometrically close? Do fable clusters map to moral clusters? This requires re-extracting
embeddings at multiple layers — more expensive than other analyses.

**Outputs:**
- `umap_morals_fables.png` — 2D UMAP of moral + fable embeddings, coloured by gt pair
- `intra_inter_distance.png` — distribution of within-pair (moral, gt-fable) vs
  between-pair cosine distances
- `layer_mrr_curve.png` — MRR as a function of transformer layer index (requires
  per-layer extraction, see notes)

**Fixed:** UMAP projection, intra/inter distance metric.
**Configurable:** `--layers` (which layers to probe, default: final), `--n_neighbors`
(UMAP), `--colour_by` (`gt_pair`, `theme`, `fable_length`).

**Notes:**
- Layer probing requires re-running the model with `output_hidden_states=True`. This is
  GPU-intensive. The other 5 analyses only need the final `.npy` embeddings.
- Implement layers after the cheaper analyses have identified whether there is a geometry
  problem worth investigating.

---

## Shared Library (`lib/`)

`analysis/lib/loader.py` — handles all data loading:
- `load_embeddings(moral_embs_path, doc_embs_path)` → `(moral_embs, doc_embs)` as numpy
- `load_dataset()` → fables, morals, qrels (delegates to `lib/data.py`)
- `compute_rankings(moral_embs, doc_embs, qrels)` → per-query ranked list + scores
- `ExperimentConfig` dataclass — bundles paths + label for passing between scripts

`analysis/lib/plotting.py` — shared matplotlib style (consistent fonts, colours, DPI
for all paper figures).

---

## Suggested Run Order

1. `01_rank_distribution` — get the big picture (1 min)
2. `03_score_gap_distribution` — quantify how bad the failures are (1 min)
3. `02_nearest_neighbor_confusion` — read the hard cases with human eyes (2 min)
4. `04_thematic_overlap` — test the ambiguity hypothesis (3 min)
5. `05_length_richness_bias` — test the attractor-fable hypothesis (2 min)
6. `06_layer_probing` — only if geometry seems to be the issue (GPU, 20+ min)
