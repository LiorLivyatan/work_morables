import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_retrieval.lib.eval import (
    reciprocal_rank,
    recall_at_k,
    rank_of,
    compute_row_metrics,
    aggregate_metrics,
)


def test_reciprocal_rank_first():
    assert reciprocal_rank(["fable_0001", "fable_0002"], "fable_0001") == pytest.approx(1.0)


def test_reciprocal_rank_second():
    assert reciprocal_rank(["fable_0002", "fable_0001"], "fable_0001") == pytest.approx(0.5)


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(["fable_0002", "fable_0003"], "fable_0001") == pytest.approx(0.0)


def test_recall_at_k_hit():
    assert recall_at_k(["fable_0001", "fable_0002"], "fable_0001", k=1) == 1.0


def test_recall_at_k_miss():
    assert recall_at_k(["fable_0002", "fable_0003"], "fable_0001", k=2) == 0.0


def test_recall_at_k_outside_window():
    ids = ["fable_0002", "fable_0003", "fable_0001"]
    assert recall_at_k(ids, "fable_0001", k=2) == 0.0
    assert recall_at_k(ids, "fable_0001", k=3) == 1.0


def test_rank_of_found():
    assert rank_of(["fable_0001", "fable_0002"], "fable_0001") == 1
    assert rank_of(["fable_0002", "fable_0001"], "fable_0001") == 2


def test_rank_of_not_found():
    assert rank_of(["fable_0002"], "fable_0001") is None


def test_compute_row_metrics_perfect():
    row = compute_row_metrics(
        moral_id="moral_0001",
        moral_text="Be kind.",
        relevant_fable="fable_0001",
        ranked_ids=["fable_0001", "fable_0002"],
        latency_s=0.5,
    )
    assert row["moral_id"] == "moral_0001"
    assert row["relevant_fable"] == "fable_0001"
    assert row["reciprocal_rank"] == pytest.approx(1.0)
    assert row["r_at_1"] == 1.0
    assert row["r_at_5"] == 1.0
    assert row["r_at_10"] == 1.0
    assert row["rank"] == 1
    assert row["latency_s"] == pytest.approx(0.5)


def test_compute_row_metrics_miss():
    row = compute_row_metrics(
        moral_id="moral_0001",
        moral_text="Be kind.",
        relevant_fable="fable_0001",
        ranked_ids=["fable_0002", "fable_0003"],
        latency_s=1.0,
    )
    assert row["reciprocal_rank"] == pytest.approx(0.0)
    assert row["r_at_1"] == 0.0
    assert row["rank"] is None


def test_aggregate_metrics_perfect():
    rows = [
        {"reciprocal_rank": 1.0, "r_at_1": 1.0, "r_at_5": 1.0, "r_at_10": 1.0, "ndcg_at_10": 1.0, "rank": 1},
        {"reciprocal_rank": 1.0, "r_at_1": 1.0, "r_at_5": 1.0, "r_at_10": 1.0, "ndcg_at_10": 1.0, "rank": 1},
    ]
    agg = aggregate_metrics(rows)
    assert agg["MRR@10"] == pytest.approx(1.0)
    assert agg["R@1"] == pytest.approx(1.0)
    assert agg["R@10"] == pytest.approx(1.0)
    assert agg["Mean_Rank"] == pytest.approx(1.0)
    assert agg["n_queries"] == 2


def test_aggregate_metrics_with_miss():
    rows = [
        {"reciprocal_rank": 1.0, "r_at_1": 1.0, "r_at_5": 1.0, "r_at_10": 1.0, "ndcg_at_10": 1.0, "rank": 1},
        {"reciprocal_rank": 0.0, "r_at_1": 0.0, "r_at_5": 0.0, "r_at_10": 0.0, "ndcg_at_10": 0.0, "rank": None},
    ]
    agg = aggregate_metrics(rows)
    assert agg["MRR@10"] == pytest.approx(0.5)
    assert agg["R@1"] == pytest.approx(0.5)
    assert agg["Mean_Rank"] == pytest.approx(1.0)
    assert agg["Median_Rank"] == pytest.approx(1.0)
    assert agg["n_queries"] == 2
