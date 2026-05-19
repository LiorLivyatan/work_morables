"""Multi-target qrels semantics — spec §10.1 mandates this fixture.

Covers _qrels_to_multimap normalisation, load_corpus end-to-end on multi-target
qrels, mrr_at_k / reciprocal_rank_per_query best-rank semantics, and the
clustered real-data smoke (668 morals, 1085 qrel rows)."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.data import load_corpus, _qrels_to_multimap
from lib.retrieval import mrr_at_k, _gt_as_lists
from lib.eval import reciprocal_rank_per_query, best_rank_per_query


def test_qrels_to_multimap_list_form():
    qrels = [
        {"query_id": "m0", "doc_id": "f0", "relevance": 1},
        {"query_id": "m1", "doc_id": "f1", "relevance": 1},
        {"query_id": "m1", "doc_id": "f2", "relevance": 1},
        {"query_id": "m2", "doc_id": "f3", "relevance": 1},
        {"query_id": "m2", "doc_id": "f4", "relevance": 1},
        {"query_id": "m2", "doc_id": "f5", "relevance": 0},   # filtered
    ]
    mm = _qrels_to_multimap(qrels)
    assert mm == {"m0": ["f0"], "m1": ["f1", "f2"], "m2": ["f3", "f4"]}


def test_qrels_to_multimap_legacy_dict_form():
    """Old single-target dict form must be wrapped to 1-element lists."""
    qrels = {"m0": "f0", "m1": "f1"}
    mm = _qrels_to_multimap(qrels)
    assert mm == {"m0": ["f0"], "m1": ["f1"]}


def test_load_corpus_multi_target(tmp_path):
    morals = [{"doc_id": "m0", "text": "x"},
               {"doc_id": "m1", "text": "y"},
               {"doc_id": "m2", "text": "z"}]
    fables = [{"doc_id": f"f{i}", "text": f"fable {i}"} for i in range(6)]
    qrels = [
        {"query_id": "m0", "doc_id": "f0", "relevance": 1},
        {"query_id": "m1", "doc_id": "f1", "relevance": 1},
        {"query_id": "m1", "doc_id": "f2", "relevance": 1},
        {"query_id": "m2", "doc_id": "f3", "relevance": 1},
        {"query_id": "m2", "doc_id": "f4", "relevance": 1},
        {"query_id": "m2", "doc_id": "f5", "relevance": 1},
    ]
    mp = tmp_path / "m.json"; mp.write_text(json.dumps(morals))
    fp = tmp_path / "f.json"; fp.write_text(json.dumps(fables))
    qp = tmp_path / "q.json"; qp.write_text(json.dumps(qrels))

    corpus = load_corpus(morals_path=mp, fables_path=fp, qrels_path=qp)
    assert corpus.gt_fable_idxs == [[0], [1, 2], [3, 4, 5]]


def test_mrr_at_k_multi_target_best_rank():
    """rankings[q]=[doc indices in similarity order]. MRR = mean(1/best_rank)."""
    # Query 0: single target = doc 2, in 1st position → rr=1
    # Query 1: targets = {3, 5}, doc 5 in 1st position → rr=1
    # Query 2: targets = {0, 1}, both NOT in top-3 → rr=0
    rankings = np.array([
        [2, 0, 1, 3, 4, 5],
        [5, 3, 4, 0, 1, 2],
        [4, 5, 3, 0, 1, 2],
    ])
    gt = [[2], [3, 5], [0, 1]]
    rr = reciprocal_rank_per_query(rankings, gt, k=3)
    assert np.allclose(rr, [1.0, 1.0, 0.0])
    assert mrr_at_k(rankings, gt, k=3) == 2/3


def test_mrr_takes_best_position_when_two_targets_in_topk():
    """If multiple relevant docs are in top-k, use the BEST (smallest) position."""
    rankings = np.array([[7, 3, 9, 1, 5]])  # doc indices in similarity order
    gt = [[1, 9]]  # both relevant; 9 is at pos 2 (rank 3), 1 is at pos 3 (rank 4)
    rr = reciprocal_rank_per_query(rankings, gt, k=5)
    assert rr[0] == 1/3   # best of {1/3, 1/4} = 1/3


def test_best_rank_per_query_returns_one_based():
    rankings = np.array([[2, 0, 1], [1, 0, 2]])
    gt = [[1], [0, 1]]
    br = best_rank_per_query(rankings, gt)
    assert br.tolist() == [3, 1]


def test_gt_as_lists_accepts_legacy_1d_array():
    """Backward compat: a 1D int array becomes list of singletons."""
    arr = np.array([0, 1, 2])
    g = _gt_as_lists(arr)
    assert g == [{0}, {1}, {2}]


def test_clustered_real_data_shape():
    """Pre-launch corpus assertions hold on the clustered data."""
    ROOT = Path(__file__).resolve().parents[3]
    corpus = load_corpus(
        morals_path=ROOT / "data/clustered/morals_unique_corpus.json",
        fables_path=ROOT / "data/clustered/fables_corpus.json",
        qrels_path =ROOT / "data/clustered/qrels_moral_to_fable_clustered.json",
    )
    assert len(corpus.moral_ids)   == 668
    assert len(corpus.fable_doc_ids) == 709
    assert sum(len(g) for g in corpus.gt_fable_idxs) == 1085
    assert all(len(g) >= 1 for g in corpus.gt_fable_idxs)
    # at least one moral has >1 relevant fable (else clustering wasn't applied)
    assert any(len(g) > 1 for g in corpus.gt_fable_idxs)


def test_clustered_baseline_mrr_gte_processed():
    """Multi-target MRR ≥ single-target MRR on the same rankings (same fables).
    Construct a hand fixture where the multi-target relevant set is a superset."""
    rankings = np.array([
        [5, 4, 3, 2, 1, 0],
        [3, 5, 2, 4, 1, 0],
        [2, 1, 0, 3, 4, 5],
    ])
    # Single-target equivalent: pick the FIRST relevant fable each moral was bound to.
    single = [[5], [3], [2]]
    multi  = [[5, 0], [3, 5, 1], [2, 0]]
    mrr_s = mrr_at_k(rankings, single, k=6)
    mrr_m = mrr_at_k(rankings, multi,  k=6)
    assert mrr_m >= mrr_s   # multi-target can only help, never hurt
