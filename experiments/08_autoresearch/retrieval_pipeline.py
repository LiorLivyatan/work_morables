"""
retrieval_pipeline.py — Experiment 08: Retrieval Autoresearch
════════════════════════════════════════════════════════════════
AGENT: You may modify ANYTHING in this file.

Contract (do not change):
  • run_pipeline() returns a dict with at minimum key "MRR"
  • Running as __main__ prints:
        mrr: 0.XXXX
        r@1: 0.XXXX
        r@5: 0.XXXX
        r@10: 0.XXXX
    (each on its own line — parsed by runner.py)

Available from lib/ (do not modify lib/):
  from retrieval_utils import compute_metrics
  from embedding_cache  import encode_with_cache
  from data             import load_moral_to_fable_retrieval_data
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))

from sentence_transformers import SentenceTransformer  # noqa: E402
from data import load_moral_to_fable_retrieval_data    # noqa: E402
from embedding_cache import encode_with_cache          # noqa: E402
from retrieval_utils import compute_metrics            # noqa: E402

# ── Configuration (agent modifies these) ─────────────────────────────────────

MODEL_ID = "Linq-AI-Research/Linq-Embed-Mistral"
QUERY_INSTRUCTION = "Given a text, retrieve the most relevant passage that answers the query"
CORPUS_INSTRUCTION = None   # None = no instruction prefix for corpus texts

CHUNKING = "full"           # "full" | "sentences" | "sliding_N_stride_S"
CHUNK_AGG = "max"           # "max" | "mean"  (used only when CHUNKING != "full")
SPARSE_WEIGHT = 0.0         # 0.0 = pure dense; blend BM25 scores when > 0
RERANKER_ID = None          # None | cross-encoder model id/path
RERANK_TOP_K = 50           # candidate pool size fed to reranker

CACHE_DIR = Path(__file__).parent / "results" / "embedding_cache"

# ── Query rewriting (agent modifies this function) ────────────────────────────

def rewrite_query(moral: str) -> str:
    """
    Optional query rewriting. Return moral unchanged to skip.
    Examples the agent may try:
      - f"This moral means: {moral}"
      - f"Fable lesson: {moral}"
      - moral.lower().strip(".")
    """
    return moral


# ── Pipeline (agent may extend run_pipeline) ─────────────────────────────────

def run_pipeline() -> dict:
    """
    Load data, encode, retrieve, evaluate. Returns metrics dict.
    Must include key 'MRR'.
    """
    fable_texts, moral_texts, ground_truth = load_moral_to_fable_retrieval_data()

    model = SentenceTransformer(MODEL_ID, device="mps")

    query_texts = [rewrite_query(m) for m in moral_texts]

    query_embs = encode_with_cache(
        model, query_texts, MODEL_ID, CACHE_DIR,
        query_instruction=QUERY_INSTRUCTION,
        label="morals (queries)",
    )
    corpus_embs = encode_with_cache(
        model, fable_texts, MODEL_ID, CACHE_DIR,
        query_instruction=CORPUS_INSTRUCTION,
        label="fables (corpus)",
    )

    metrics = compute_metrics(query_embs, corpus_embs, ground_truth)
    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    metrics = run_pipeline()
    print(f"mrr: {metrics['MRR']:.4f}")
    print(f"r@1: {metrics['Recall@1']:.4f}")
    print(f"r@5: {metrics['Recall@5']:.4f}")
    print(f"r@10: {metrics['Recall@10']:.4f}")
