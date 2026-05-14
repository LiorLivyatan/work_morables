import math
import statistics
from typing import Optional


def reciprocal_rank(ranked_ids: list[str], relevant_id: str) -> float:
    try:
        return 1.0 / (ranked_ids.index(relevant_id) + 1)
    except ValueError:
        return 0.0


def recall_at_k(ranked_ids: list[str], relevant_id: str, k: int) -> float:
    return 1.0 if relevant_id in ranked_ids[:k] else 0.0


def rank_of(ranked_ids: list[str], relevant_id: str) -> Optional[int]:
    try:
        return ranked_ids.index(relevant_id) + 1
    except ValueError:
        return None


def ndcg_at_k(ranked_ids: list[str], relevant_id: str, k: int) -> float:
    rank = rank_of(ranked_ids[:k], relevant_id)
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)  # ideal DCG = 1/log2(2) = 1.0


def compute_row_metrics(
    moral_id: str,
    moral_text: str,
    relevant_fable: str,
    ranked_ids: list[str],
    latency_s: float,
) -> dict:
    rr = reciprocal_rank(ranked_ids, relevant_fable)
    return {
        "moral_id": moral_id,
        "moral_text": moral_text,
        "relevant_fable": relevant_fable,
        "ranked_ids": "|".join(ranked_ids),
        "reciprocal_rank": rr,
        "r_at_1": recall_at_k(ranked_ids, relevant_fable, 1),
        "r_at_5": recall_at_k(ranked_ids, relevant_fable, 5),
        "r_at_10": recall_at_k(ranked_ids, relevant_fable, 10),
        "ndcg_at_10": ndcg_at_k(ranked_ids, relevant_fable, 10),
        "rank": rank_of(ranked_ids, relevant_fable),
        "latency_s": round(latency_s, 3),
    }


def aggregate_metrics(rows: list[dict]) -> dict:
    found_ranks = [r["rank"] for r in rows if r["rank"] is not None]
    return {
        "MRR@10": statistics.mean(r["reciprocal_rank"] for r in rows),
        "R@1":    statistics.mean(r["r_at_1"]  for r in rows),
        "R@5":    statistics.mean(r["r_at_5"]  for r in rows),
        "R@10":   statistics.mean(r["r_at_10"] for r in rows),
        "NDCG@10": statistics.mean(r["ndcg_at_10"] for r in rows),
        "Mean_Rank":   statistics.mean(found_ranks)   if found_ranks else None,
        "Median_Rank": statistics.median(found_ranks) if found_ranks else None,
        "n_queries": len(rows),
    }
