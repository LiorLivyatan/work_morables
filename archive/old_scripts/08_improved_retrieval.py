"""
Improved retrieval experiments with instruction-aware embedding models.

Builds on 04_baseline_retrieval.py by adding:
  1. Stronger embedding models (BGE, E5, GTE) with proper instruction prefixes
  2. Instruction variation experiments (generic vs task-specific vs moral-focused)
  3. Comparison against baseline Sentence-BERT results

Models:
  - BAAI/bge-large-en-v1.5 (1024-dim, supports query instruction prefix)
  - intfloat/e5-large-v2 (1024-dim, requires "query: " / "passage: " prefixes)
  - intfloat/multilingual-e5-large (1024-dim, same prefix scheme, multilingual)
  - Alibaba-NLP/gte-large-en-v1.5 (1024-dim)

Reference:
  - BGE: https://huggingface.co/BAAI/bge-large-en-v1.5
  - E5: https://huggingface.co/intfloat/e5-large-v2
"""
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
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
gt_f2m = {}
for qrel in qrels_f2m:
    fable_idx = int(qrel["query_id"].split("_")[1])
    moral_idx = int(qrel["doc_id"].split("_")[1])
    gt_f2m[fable_idx] = moral_idx

gt_f2m_aug = {}
moral_aug_id_to_idx = {m["doc_id"]: i for i, m in enumerate(morals_augmented)}
for qrel in qrels_f2m_aug:
    fable_idx = int(qrel["query_id"].split("_")[1])
    moral_aug_idx = moral_aug_id_to_idx.get(qrel["doc_id"])
    if moral_aug_idx is not None:
        gt_f2m_aug[fable_idx] = moral_aug_idx


def compute_metrics(query_embeddings, corpus_embeddings, ground_truth, ks=[1, 5, 10]):
    """Compute retrieval metrics: MRR, R-Precision, Recall@k."""
    sim_matrix = cosine_similarity(query_embeddings, corpus_embeddings)
    rankings = np.argsort(-sim_matrix, axis=1)

    reciprocal_ranks = []
    recall_at_k = {k: [] for k in ks}
    r_precisions = []

    for q_idx in range(len(query_embeddings)):
        if q_idx not in ground_truth:
            continue
        correct_idx = ground_truth[q_idx]
        ranked = rankings[q_idx]
        rank = np.where(ranked == correct_idx)[0][0]

        reciprocal_ranks.append(1.0 / (rank + 1))
        for k in ks:
            recall_at_k[k].append(1.0 if rank < k else 0.0)
        r_precisions.append(1.0 if rank == 0 else 0.0)

    results = {
        "MRR": float(np.mean(reciprocal_ranks)),
        "R-Precision": float(np.mean(r_precisions)),
        "n_queries": len(reciprocal_ranks),
    }
    for k in ks:
        results[f"Recall@{k}"] = float(np.mean(recall_at_k[k]))
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


def add_prefix(texts, prefix):
    """Prepend a prefix string to each text."""
    if not prefix:
        return texts
    return [prefix + t for t in texts]


# ══════════════════════════════════════════════════════════════
# Model configurations
# ══════════════════════════════════════════════════════════════
#
# Each model config specifies:
#   - name: HuggingFace model ID
#   - instruction_variants: dict of {variant_name: (query_prefix, doc_prefix)}
#     for testing different instruction phrasings
MODEL_CONFIGS = [
    {
        "name": "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
        "instruction_variants": {
            "no_instruction": ("", ""),
            "default": (
                "Represent this sentence for searching relevant passages: ",
                "",
            ),
            "moral_task": (
                "Given this fable, retrieve the abstract moral lesson it teaches: ",
                "",
            ),
            "moral_meaning": (
                "What is the deeper meaning and life lesson of this story: ",
                "",
            ),
        },
    },
    {
        "name": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "instruction_variants": {
            "default": ("query: ", "passage: "),
            "no_prefix": ("", ""),
        },
    },
    {
        "name": "intfloat/multilingual-e5-large",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "instruction_variants": {
            "default": ("query: ", "passage: "),
        },
    },
    # NOTE: Alibaba-NLP/gte-large-en-v1.5 removed — crashes with torch
    # AcceleratorError on encode (tensor index out of bounds). Likely
    # incompatible with sentence-transformers version or MPS backend.
]



# ══════════════════════════════════════════════════════════════
# Run all experiments
# ══════════════════════════════════════════════════════════════

all_results = []

for config in MODEL_CONFIGS:
    model_name = config["name"]

    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    model = SentenceTransformer(model_name, trust_remote_code=True)

    for variant_name, (q_prefix, d_prefix) in config["instruction_variants"].items():
        label = f"{model_name} [{variant_name}]"
        print(f"\n  Variant: {variant_name}")
        print(f"    Query prefix: '{q_prefix}'")
        print(f"    Doc prefix:   '{d_prefix}'")

        fable_prefixed = add_prefix(fable_texts, q_prefix)
        moral_prefixed = add_prefix(moral_texts, d_prefix)
        moral_aug_prefixed = add_prefix(moral_aug_texts, d_prefix)

        # For moral→fable task, morals are queries and fables are docs
        # So we need both directions of prefixing
        # The run_experiment function handles this internally,
        # but we need to note that for m2f, the "query" is the moral with q_prefix
        # and the "doc" is the fable with d_prefix.
        # Since we pass the pre-prefixed texts, and m2f uses moral_embs as queries
        # and fable_embs as corpus, the moral_embs already have d_prefix (corpus prefix).
        # This is slightly wrong for m2f — morals should get q_prefix when they're queries.
        # Let's handle this properly by encoding both directions.

        # Encode fables as queries and as documents
        fable_as_query = add_prefix(fable_texts, q_prefix)
        fable_as_doc = add_prefix(fable_texts, d_prefix)
        moral_as_query = add_prefix(moral_texts, q_prefix)
        moral_as_doc = add_prefix(moral_texts, d_prefix)
        moral_aug_as_doc = add_prefix(moral_aug_texts, d_prefix)

        print(f"  Encoding fables (as queries)...")
        fable_q_embs = model.encode(fable_as_query, show_progress_bar=True,
                                    batch_size=32)

        print(f"  Encoding fables (as docs)...")
        fable_d_embs = model.encode(fable_as_doc, show_progress_bar=True,
                                    batch_size=32)

        print(f"  Encoding morals (as queries)...")
        moral_q_embs = model.encode(moral_as_query, show_progress_bar=True,
                                    batch_size=32)

        print(f"  Encoding morals (as docs)...")
        moral_d_embs = model.encode(moral_as_doc, show_progress_bar=True,
                                    batch_size=32)

        print(f"  Encoding morals augmented (as docs)...")
        moral_aug_d_embs = model.encode(moral_aug_as_doc, show_progress_bar=True,
                                        batch_size=32)

        # Task 1: Fable(query) → Moral(doc) (clean)
        print(f"  --- Fable → Moral (clean) ---")
        metrics = compute_metrics(fable_q_embs, moral_d_embs, gt_f2m)
        metrics["task"] = "fable→moral (clean)"
        metrics["model"] = label
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
        all_results.append(metrics)

        # Task 2: Moral(query) → Fable(doc) (clean)
        print(f"  --- Moral → Fable (clean) ---")
        gt_m2f = {v: k for k, v in gt_f2m.items()}
        metrics = compute_metrics(moral_q_embs, fable_d_embs, gt_m2f)
        metrics["task"] = "moral→fable (clean)"
        metrics["model"] = label
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
        all_results.append(metrics)

        # Task 3: Fable(query) → Moral(doc) (augmented)
        print(f"  --- Fable → Moral (augmented) ---")
        metrics = compute_metrics(fable_q_embs, moral_aug_d_embs, gt_f2m_aug)
        metrics["task"] = "fable→moral (augmented)"
        metrics["model"] = label
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
        all_results.append(metrics)

        # Rank analysis for fable→moral (clean)
        ranks = rank_analysis(fable_q_embs, moral_d_embs, gt_f2m)
        print(f"  --- Rank Distribution (Fable→Moral, clean) ---")
        print(f"    Rank 0 (top-1): {(ranks == 0).sum()} "
              f"({(ranks == 0).mean()*100:.1f}%)")
        print(f"    Rank 1-4:  {((ranks >= 1) & (ranks < 5)).sum()} "
              f"({((ranks >= 1) & (ranks < 5)).mean()*100:.1f}%)")
        print(f"    Rank 5-9:  {((ranks >= 5) & (ranks < 10)).sum()} "
              f"({((ranks >= 5) & (ranks < 10)).mean()*100:.1f}%)")
        print(f"    Rank 10-49: {((ranks >= 10) & (ranks < 50)).sum()} "
              f"({((ranks >= 10) & (ranks < 50)).mean()*100:.1f}%)")
        print(f"    Rank 50+:  {(ranks >= 50).sum()} "
              f"({(ranks >= 50).mean()*100:.1f}%)")
        print(f"    Mean rank: {ranks.mean():.1f}, Median rank: {np.median(ranks):.1f}")

# ── Save results ──
with open(RESULTS_DIR / "improved_retrieval_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved results to results/improved_retrieval_results.json")

# ── Print summary table ──
print(f"\n{'='*100}")
print(f"SUMMARY TABLE")
print(f"{'='*100}")
header = (f"{'Model':<55} {'Task':<25} {'MRR':>6} {'R-P':>6} "
          f"{'R@1':>6} {'R@5':>6} {'R@10':>6}")
print(header)
print("-" * len(header))
for r in all_results:
    model_short = r["model"].replace("sentence-transformers/", "").replace(
        "intfloat/", "").replace("BAAI/", "").replace("Alibaba-NLP/", "")
    print(f"{model_short:<55} {r['task']:<25} "
          f"{r['MRR']:.4f} {r['R-Precision']:.4f} "
          f"{r['Recall@1']:.4f} {r['Recall@5']:.4f} {r['Recall@10']:.4f}")

# ── Also load and display baseline for comparison ──
baseline_path = RESULTS_DIR / "baseline_results.json"
if baseline_path.exists():
    with open(baseline_path) as f:
        baseline_results = json.load(f)
    print(f"\n{'='*100}")
    print(f"BASELINE COMPARISON (from 04_baseline_retrieval.py)")
    print(f"{'='*100}")
    print(header)
    print("-" * len(header))
    for r in baseline_results:
        model_short = r["model"].split("/")[-1]
        print(f"{model_short:<55} {r['task']:<25} "
              f"{r['MRR']:.4f} {r['R-Precision']:.4f} "
              f"{r['Recall@1']:.4f} {r['Recall@5']:.4f} {r['Recall@10']:.4f}")
