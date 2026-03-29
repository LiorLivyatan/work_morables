"""
Baseline retrieval experiments using Sentence-BERT models.

Tasks:
  1. Fable → Moral retrieval (709 fables as queries, 709 morals as corpus)
  2. Moral → Fable retrieval (reversed)
  3. Fable → Moral (augmented corpus with distractors)

Models:
  - sentence-transformers/all-MiniLM-L6-v2 (fast baseline)
  - sentence-transformers/all-mpnet-base-v2 (stronger)

Metrics: R-Precision, Recall@1, Recall@5, Recall@10, MRR
TODO: add Recall@50 / MAP
"""
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Load data ──
with open(DATA_DIR / "fables_corpus.json") as f:
    fables_corpus = json.load(f)
with open(DATA_DIR / "morals_corpus.json") as f:
    morals_corpus = json.load(f)
with open(DATA_DIR / "morals_augmented_corpus.json") as f:
    morals_augmented = json.load(f)
with open(DATA_DIR / "qrels_fable_to_moral.json") as f:
    qrels_f2m = json.load(f)
with open(DATA_DIR / "qrels_fable_to_moral_augmented.json") as f:
    qrels_f2m_aug = json.load(f)

fable_texts = [f["text"] for f in fables_corpus]
moral_texts = [m["text"] for m in morals_corpus]
moral_aug_texts = [m["text"] for m in morals_augmented]

# Build ground truth mappings
# For clean retrieval: fable_i -> moral_i (1-to-1)
gt_f2m = {}  # fable_idx -> moral_idx
for qrel in qrels_f2m:
    fable_idx = int(qrel["query_id"].split("_")[1])
    moral_idx = int(qrel["doc_id"].split("_")[1])
    gt_f2m[fable_idx] = moral_idx

# For augmented: fable_idx -> moral_aug_idx
gt_f2m_aug = {}
moral_aug_id_to_idx = {m["doc_id"]: i for i, m in enumerate(morals_augmented)}
for qrel in qrels_f2m_aug:
    fable_idx = int(qrel["query_id"].split("_")[1])
    moral_aug_idx = moral_aug_id_to_idx.get(qrel["doc_id"])
    if moral_aug_idx is not None:
        gt_f2m_aug[fable_idx] = moral_aug_idx


def compute_metrics(query_embeddings, corpus_embeddings, ground_truth, ks=[1, 5, 10]):
    """
    Compute retrieval metrics.

    Args:
        query_embeddings: (N, D) array
        corpus_embeddings: (M, D) array
        ground_truth: dict mapping query_idx -> correct corpus_idx
        ks: list of k values for Recall@k
    """
    sim_matrix = cosine_similarity(query_embeddings, corpus_embeddings)
    rankings = np.argsort(-sim_matrix, axis=1)  # descending

    results = {}
    reciprocal_ranks = []
    recall_at_k = {k: [] for k in ks}
    r_precisions = []

    for q_idx in range(len(query_embeddings)):
        if q_idx not in ground_truth:
            continue
        correct_idx = ground_truth[q_idx]
        ranked = rankings[q_idx]

        # Find rank of correct document (0-indexed)
        rank = np.where(ranked == correct_idx)[0][0]

        # MRR
        reciprocal_ranks.append(1.0 / (rank + 1))

        # Recall@k
        for k in ks:
            recall_at_k[k].append(1.0 if rank < k else 0.0)

        # R-Precision (with R=1 since each query has 1 relevant doc)
        r_precisions.append(1.0 if rank == 0 else 0.0)

    results["MRR"] = np.mean(reciprocal_ranks)
    results["R-Precision"] = np.mean(r_precisions)
    for k in ks:
        results[f"Recall@{k}"] = np.mean(recall_at_k[k])
    results["n_queries"] = len(reciprocal_ranks)

    return results


def rank_analysis(query_embeddings, corpus_embeddings, ground_truth):
    """Return the rank of the correct doc for each query."""
    sim_matrix = cosine_similarity(query_embeddings, corpus_embeddings)
    rankings = np.argsort(-sim_matrix, axis=1)
    ranks = []
    for q_idx in range(len(query_embeddings)):
        if q_idx not in ground_truth:
            continue
        correct_idx = ground_truth[q_idx]
        rank = np.where(rankings[q_idx] == correct_idx)[0][0]
        ranks.append(rank)
    return np.array(ranks)


def distractor_analysis(query_embeddings, corpus_entries, mcqa_data, fables_data):
    """
    For each fable, check which MCQA distractor class the top-1 retrieved moral belongs to.
    Uses the augmented corpus.
    """
    sim_matrix = cosine_similarity(query_embeddings, np.array([
        model.encode(m["text"]) for m in corpus_entries
    ]))
    # This is slow; we'll do it differently
    pass


# ══════════════════════════════════════════
# Run experiments
# ══════════════════════════════════════════

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]

all_results = []

for model_name in MODELS:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    model = SentenceTransformer(model_name)

    # Encode
    print("Encoding fables...")
    t0 = time.time()
    fable_embs = model.encode(fable_texts, show_progress_bar=True, batch_size=64)
    t_fable = time.time() - t0

    print("Encoding morals (clean)...")
    t0 = time.time()
    moral_embs = model.encode(moral_texts, show_progress_bar=True, batch_size=64)
    t_moral = time.time() - t0

    print("Encoding morals (augmented)...")
    t0 = time.time()
    moral_aug_embs = model.encode(moral_aug_texts, show_progress_bar=True, batch_size=64)
    t_moral_aug = time.time() - t0

    print(f"Encoding time: fables={t_fable:.1f}s, morals={t_moral:.1f}s, augmented={t_moral_aug:.1f}s")

    # Task 1: Fable → Moral (clean, 709 → 709)
    print("\n--- Task 1: Fable → Moral (clean corpus) ---")
    metrics_f2m = compute_metrics(fable_embs, moral_embs, gt_f2m)
    for k, v in metrics_f2m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    metrics_f2m["task"] = "fable→moral (clean)"
    metrics_f2m["model"] = model_name

    # Task 2: Moral → Fable (clean, 709 → 709)
    print("\n--- Task 2: Moral → Fable (clean corpus) ---")
    gt_m2f = {v: k for k, v in gt_f2m.items()}  # reverse mapping
    metrics_m2f = compute_metrics(moral_embs, fable_embs, gt_m2f)
    for k, v in metrics_m2f.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    metrics_m2f["task"] = "moral→fable (clean)"
    metrics_m2f["model"] = model_name

    # Task 3: Fable → Moral (augmented, 709 queries → ~2800 corpus)
    print("\n--- Task 3: Fable → Moral (augmented corpus) ---")
    metrics_f2m_aug = compute_metrics(fable_embs, moral_aug_embs, gt_f2m_aug)
    for k, v in metrics_f2m_aug.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    metrics_f2m_aug["task"] = "fable→moral (augmented)"
    metrics_f2m_aug["model"] = model_name

    all_results.extend([metrics_f2m, metrics_m2f, metrics_f2m_aug])

    # Rank distribution for fable→moral (clean)
    ranks = rank_analysis(fable_embs, moral_embs, gt_f2m)
    print(f"\n--- Rank Distribution (Fable→Moral, clean) ---")
    print(f"  Rank 0 (correct at top): {(ranks == 0).sum()} ({(ranks == 0).mean()*100:.1f}%)")
    print(f"  Rank 1-4:   {((ranks >= 1) & (ranks < 5)).sum()} ({((ranks >= 1) & (ranks < 5)).mean()*100:.1f}%)")
    print(f"  Rank 5-9:   {((ranks >= 5) & (ranks < 10)).sum()} ({((ranks >= 5) & (ranks < 10)).mean()*100:.1f}%)")
    print(f"  Rank 10-49: {((ranks >= 10) & (ranks < 50)).sum()} ({((ranks >= 10) & (ranks < 50)).mean()*100:.1f}%)")
    print(f"  Rank 50+:   {(ranks >= 50).sum()} ({(ranks >= 50).mean()*100:.1f}%)")
    print(f"  Mean rank:  {ranks.mean():.1f}")
    print(f"  Median rank: {np.median(ranks):.1f}")

# ── Save results ──
with open(RESULTS_DIR / "baseline_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved all results to results/baseline_results.json")

# ── Print summary table ──
print(f"\n{'='*80}")
print(f"SUMMARY TABLE")
print(f"{'='*80}")
header = f"{'Model':<40} {'Task':<30} {'MRR':>6} {'R-P':>6} {'R@1':>6} {'R@5':>6} {'R@10':>6}"
print(header)
print("-" * len(header))
for r in all_results:
    model_short = r["model"].split("/")[-1]
    print(f"{model_short:<40} {r['task']:<30} {r['MRR']:.4f} {r['R-Precision']:.4f} {r['Recall@1']:.4f} {r['Recall@5']:.4f} {r['Recall@10']:.4f}")
