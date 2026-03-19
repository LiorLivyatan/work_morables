"""
LLM-based reranking with Chain-of-Thought reasoning.

Two approaches:
  A) CoT Reranking: Take top-K candidates from best embedding model,
     use Gemini with CoT to rerank them.
  B) CoT Summarization: Use Gemini to generate a moral summary from each fable,
     then embed the summary and use it for retrieval.

Requires: GEMINI_API_KEY environment variable set (or .env file).
Usage:  python scripts/07_llm_reranking.py
"""
import json
import os
import time
from pathlib import Path

import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Load .env file if present ──
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                val = val.split("#")[0].strip()  # strip inline comments
                os.environ.setdefault(key.strip(), val)

# ── Configure Gemini ──
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not set. Create a .env file or export GEMINI_API_KEY=your-key"
    )
genai.configure(api_key=api_key)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

# ── Load data ──
with open(DATA_DIR / "fables_corpus.json") as f:
    fables_corpus = json.load(f)
with open(DATA_DIR / "morals_corpus.json") as f:
    morals_corpus = json.load(f)
with open(DATA_DIR / "qrels_fable_to_moral.json") as f:
    qrels_f2m = json.load(f)

fable_texts = [f["text"] for f in fables_corpus]
moral_texts = [m["text"] for m in morals_corpus]

gt_f2m = {}
for qrel in qrels_f2m:
    fable_idx = int(qrel["query_id"].split("_")[1])
    moral_idx = int(qrel["doc_id"].split("_")[1])
    gt_f2m[fable_idx] = moral_idx


def compute_metrics(ranked_results, ground_truth, ks=[1, 5, 10]):
    """
    Compute metrics from pre-computed rankings.

    Args:
        ranked_results: dict mapping query_idx -> list of corpus_idx in ranked order
        ground_truth: dict mapping query_idx -> correct corpus_idx
    """
    reciprocal_ranks = []
    recall_at_k = {k: [] for k in ks}
    r_precisions = []

    for q_idx, ranking in ranked_results.items():
        if q_idx not in ground_truth:
            continue
        correct_idx = ground_truth[q_idx]

        if correct_idx in ranking:
            rank = ranking.index(correct_idx)
        else:
            rank = len(moral_texts)  # not found in candidates

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


# ══════════════════════════════════════════════════════════════
# Approach A: CoT Reranking of top-K embedding candidates
# ══════════════════════════════════════════════════════════════

TOP_K = 20  # Number of candidates to rerank
RERANK_BATCH = 10  # How many morals to show per LLM call


def get_top_k_candidates(fable_embs, moral_embs, k):
    """Get top-K moral candidates for each fable using cosine similarity."""
    sim_matrix = cosine_similarity(fable_embs, moral_embs)
    top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
    return top_k_indices


COT_RERANK_PROMPT = """You are an expert in fables, parables, and moral reasoning.

Given a fable and a list of candidate morals, think step by step about the deeper lesson the fable teaches, then rank the morals from most to least relevant.

FABLE:
{fable}

CANDIDATE MORALS (numbered):
{morals_list}

Think step by step:
1. What happens in this fable? What is the key conflict or situation?
2. What abstract lesson or principle does this story illustrate?
3. Which candidate moral best captures this lesson?

After your reasoning, output ONLY a line starting with "RANKING:" followed by the moral numbers in order from best to worst match, separated by commas.
Example: RANKING: 3, 1, 5, 2, 4"""


def rerank_with_cot(fable_text, candidate_morals, candidate_indices):
    """Use Gemini with CoT to rerank candidate morals for a fable."""
    morals_list = "\n".join(
        f"{i+1}. {moral}" for i, moral in enumerate(candidate_morals)
    )

    prompt = COT_RERANK_PROMPT.format(fable=fable_text, morals_list=morals_list)

    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        response = model.generate_content(prompt)
        text = response.text

        # Parse ranking from response
        for line in text.strip().split("\n"):
            if line.strip().upper().startswith("RANKING:"):
                ranking_str = line.split(":", 1)[1].strip()
                ranked_nums = []
                for part in ranking_str.split(","):
                    part = part.strip().rstrip(".")
                    try:
                        num = int(part)
                        if 1 <= num <= len(candidate_morals):
                            ranked_nums.append(num - 1)  # 0-indexed
                    except ValueError:
                        continue

                # Convert local indices back to global moral indices
                ranked_global = [candidate_indices[i] for i in ranked_nums]

                # Add any candidates not mentioned at the end
                for idx in candidate_indices:
                    if idx not in ranked_global:
                        ranked_global.append(idx)

                return ranked_global

        # If no RANKING line found, return original order
        return list(candidate_indices)

    except Exception as e:
        print(f"    Gemini error: {e}")
        return list(candidate_indices)


def run_cot_reranking(fable_embs, moral_embs, top_k=TOP_K, sample_size=None):
    """
    Run CoT reranking on all fables (or a sample).

    Args:
        sample_size: If set, only rerank this many fables (for testing).
    """
    print(f"\n{'='*60}")
    print(f"Approach A: CoT Reranking (top-{top_k} candidates)")
    print(f"{'='*60}")

    top_k_indices = get_top_k_candidates(fable_embs, moral_embs, top_k)

    fable_indices = list(range(len(fable_texts)))
    if sample_size:
        fable_indices = fable_indices[:sample_size]

    ranked_results = {}
    for fable_idx in tqdm(fable_indices, desc="CoT reranking"):
        candidates = top_k_indices[fable_idx]
        candidate_morals = [moral_texts[i] for i in candidates]

        reranked = rerank_with_cot(fable_texts[fable_idx], candidate_morals,
                                   candidates)
        ranked_results[fable_idx] = reranked

        # Rate limiting: Gemini free tier = 15 RPM, paid = 1000+ RPM
        time.sleep(0.5)

    metrics = compute_metrics(ranked_results, gt_f2m)
    print(f"\nCoT Reranking Results (top-{top_k}):")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    metrics["task"] = f"fable→moral (CoT rerank top-{top_k})"
    metrics["model"] = f"Gemini {GEMINI_MODEL} + embedding retrieval"
    return metrics


# ══════════════════════════════════════════════════════════════
# Approach B: CoT Moral Summarization → Embed summary
# ══════════════════════════════════════════════════════════════

COT_SUMMARIZE_PROMPT = """Read this fable carefully. Think step by step about what deeper lesson or moral principle it teaches. Then write ONLY the moral as a single concise sentence (under 20 words).

FABLE:
{fable}

Think step by step about the lesson, then write:
MORAL: [your one-sentence moral]"""


def generate_moral_summary(fable_text):
    """Use Gemini with CoT to generate a moral summary for a fable."""
    prompt = COT_SUMMARIZE_PROMPT.format(fable=fable_text)
    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        response = model.generate_content(prompt)
        text = response.text

        # Extract moral from response
        for line in text.strip().split("\n"):
            if line.strip().upper().startswith("MORAL:"):
                return line.split(":", 1)[1].strip()

        # Fallback: return last non-empty line
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else fable_text[:100]

    except Exception as e:
        print(f"    Gemini error: {e}")
        return ""


def run_cot_summarization(embed_model, sample_size=None):
    """
    Generate moral summaries for all fables, then embed and retrieve.

    Args:
        embed_model: SentenceTransformer model for embedding the summaries.
        sample_size: If set, only process this many fables (for testing).
    """
    print(f"\n{'='*60}")
    print(f"Approach B: CoT Moral Summarization → Embed")
    print(f"{'='*60}")

    cache_path = RESULTS_DIR / "cot_moral_summaries.json"

    # Check cache
    if cache_path.exists():
        print("Loading cached moral summaries...")
        with open(cache_path) as f:
            summaries = json.load(f)
        if len(summaries) == len(fable_texts):
            print(f"  Loaded {len(summaries)} cached summaries.")
        else:
            print(f"  Cache has {len(summaries)} entries but need {len(fable_texts)}. "
                  f"Regenerating...")
            summaries = None
    else:
        summaries = None

    if summaries is None:
        fable_indices = list(range(len(fable_texts)))
        if sample_size:
            fable_indices = fable_indices[:sample_size]

        summaries = {}
        for fable_idx in tqdm(fable_indices, desc="Generating moral summaries"):
            summary = generate_moral_summary(fable_texts[fable_idx])
            summaries[str(fable_idx)] = summary
            time.sleep(0.5)  # Rate limiting

        with open(cache_path, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"  Saved {len(summaries)} summaries to {cache_path}")

    # Embed summaries and morals, then retrieve
    summary_texts = [summaries.get(str(i), "") for i in range(len(fable_texts))]
    non_empty = [i for i, s in enumerate(summary_texts) if s]
    print(f"  Non-empty summaries: {len(non_empty)}/{len(fable_texts)}")

    print("  Embedding summaries...")
    summary_embs = embed_model.encode(summary_texts, show_progress_bar=True,
                                      batch_size=32)
    print("  Embedding morals...")
    moral_embs = embed_model.encode(moral_texts, show_progress_bar=True,
                                    batch_size=32)

    # Retrieve: summary embedding → moral
    sim_matrix = cosine_similarity(summary_embs, moral_embs)
    rankings = np.argsort(-sim_matrix, axis=1)

    ranked_results = {}
    for q_idx in range(len(summary_texts)):
        if q_idx in gt_f2m and summary_texts[q_idx]:
            ranked_results[q_idx] = list(rankings[q_idx])

    metrics = compute_metrics(ranked_results, gt_f2m)
    print(f"\nCoT Summarization + Embed Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    metrics["task"] = "fable→moral (CoT summary + embed)"
    metrics["model"] = f"Gemini {GEMINI_MODEL} summary → {embed_model._model_card_text[:50] if hasattr(embed_model, '_model_card_text') else 'embedding'}"
    return metrics


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based reranking experiments")
    parser.add_argument("--approach", choices=["rerank", "summarize", "both"],
                        default="both", help="Which approach to run")
    parser.add_argument("--sample", type=int, default=None,
                        help="Run on a sample of N fables (for testing)")
    parser.add_argument("--embed-model", type=str,
                        default="BAAI/bge-large-en-v1.5",
                        help="Embedding model for initial retrieval / summary embedding")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="Number of candidates for reranking")
    args = parser.parse_args()

    all_results = []

    # Load embedding model
    print(f"Loading embedding model: {args.embed_model}")
    embed_model = SentenceTransformer(args.embed_model, trust_remote_code=True)

    print("Encoding fables...")
    fable_embs = embed_model.encode(fable_texts, show_progress_bar=True,
                                    batch_size=32)
    print("Encoding morals...")
    moral_embs = embed_model.encode(moral_texts, show_progress_bar=True,
                                    batch_size=32)

    # Baseline (embedding only) for comparison
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    sim_base = cos_sim(fable_embs, moral_embs)
    rankings_base = np.argsort(-sim_base, axis=1)
    baseline_ranked = {i: list(rankings_base[i]) for i in range(len(fable_texts))}
    baseline_metrics = compute_metrics(baseline_ranked, gt_f2m)
    baseline_metrics["task"] = "fable→moral (embedding only)"
    baseline_metrics["model"] = args.embed_model
    print(f"\nBaseline (embedding only):")
    for k, v in baseline_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    all_results.append(baseline_metrics)

    if args.approach in ("rerank", "both"):
        rerank_metrics = run_cot_reranking(fable_embs, moral_embs,
                                           top_k=args.top_k,
                                           sample_size=args.sample)
        all_results.append(rerank_metrics)

    if args.approach in ("summarize", "both"):
        summary_metrics = run_cot_summarization(embed_model,
                                                sample_size=args.sample)
        all_results.append(summary_metrics)

    # Save results
    with open(RESULTS_DIR / "llm_reranking_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to results/llm_reranking_results.json")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'Task':<40} {'MRR':>6} {'R@1':>6} {'R@5':>6} {'R@10':>6}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['task']:<40} {r['MRR']:.4f} {r['Recall@1']:.4f} "
              f"{r['Recall@5']:.4f} {r['Recall@10']:.4f}")
