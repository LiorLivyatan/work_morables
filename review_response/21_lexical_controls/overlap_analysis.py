"""Lexical-overlap shortcut analysis for clustered MoraLink rankings.

Run only through run.sh:
    ./run.sh review_response/21_lexical_controls/overlap_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify


CLUSTERED_DIR = ROOT / "data" / "clustered"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_RANKINGS_GLOB = (
    ROOT / "finetuning" / "ft_11_clustered" / "results" / "rankings" / "linq__raw__none__fold_*.json"
)
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def load_json(path: Path):
    return json.loads(path.read_text())


def tokenize(text: str, *, content_only: bool) -> set[str]:
    tokens = {tok.lower() for tok in TOKEN_RE.findall(text)}
    if content_only:
        tokens = {tok for tok in tokens if tok not in ENGLISH_STOP_WORDS and len(tok) > 1}
    return tokens


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def build_relevance(qrels: list[dict], query_ids: list[str], doc_ids: list[str]) -> dict[str, set[str]]:
    valid_queries = set(query_ids)
    valid_docs = set(doc_ids)
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        if int(row.get("relevance", 1)) <= 0:
            continue
        if row["query_id"] in valid_queries and row["doc_id"] in valid_docs:
            relevant[row["query_id"]].add(row["doc_id"])
    return dict(relevant)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearson(x_values: list[float], y_values: list[float]) -> float | None:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x_values: list[float], y_values: list[float]) -> float | None:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return None
    return pearson(rankdata(x), rankdata(y))


def correlations(x_values: list[float], y_values: list[float]) -> dict[str, float | None]:
    return {
        "pearson": pearson(x_values, y_values),
        "spearman": spearman(x_values, y_values),
        "n": len(x_values),
    }


def load_rankings(pattern: str) -> list[dict]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No rankings matched: {pattern}")

    rows = []
    seen_queries = set()
    for path in paths:
        for row in load_json(path):
            qid = row["query_id"]
            if qid in seen_queries:
                raise ValueError(f"Duplicate query_id across rankings: {qid}")
            seen_queries.add(qid)
            row["source_path"] = str(path.relative_to(ROOT))
            rows.append(row)
    return rows


def analyze(args: argparse.Namespace) -> dict:
    morals = load_json(CLUSTERED_DIR / "morals_unique_corpus.json")
    fables = load_json(CLUSTERED_DIR / "fables_corpus.json")
    qrels = load_json(CLUSTERED_DIR / "qrels_moral_to_fable_clustered.json")
    rankings = load_rankings(args.rankings_glob)

    query_by_id = {row["doc_id"]: row for row in morals}
    fable_by_id = {row["doc_id"]: row for row in fables}
    doc_ids = [row["doc_id"] for row in fables]
    relevant = build_relevance(qrels, list(query_by_id), doc_ids)

    query_tokens_all = {qid: tokenize(row["text"], content_only=False) for qid, row in query_by_id.items()}
    query_tokens_content = {qid: tokenize(row["text"], content_only=True) for qid, row in query_by_id.items()}
    doc_tokens_all = {did: tokenize(row["text"], content_only=False) for did, row in fable_by_id.items()}
    doc_tokens_content = {did: tokenize(row["text"], content_only=True) for did, row in fable_by_id.items()}

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = RESULTS_DIR / timestamp / "overlap_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    notify.send(
        "lexical overlap analysis starting\n"
        f"rankings: {args.rankings_glob}\n"
        f"queries: {len(rankings)}\n"
        f"docs: {len(doc_ids)}"
    )

    pair_iou_all = []
    pair_iou_content = []
    pair_scores = []
    pair_rr = []
    relevant_iou_all = []
    relevant_iou_content = []
    relevant_scores = []
    relevant_rr = []
    query_rows = []

    for row in rankings:
        qid = row["query_id"]
        ranked_doc_ids = row["ranked_fable_ids"]
        scores = row["scores"]
        relevant_docs = relevant.get(qid, set())
        if not relevant_docs:
            continue

        rank_by_doc = {doc_id: rank_idx for rank_idx, doc_id in enumerate(ranked_doc_ids, start=1)}
        score_by_doc = {doc_id: float(score) for doc_id, score in zip(ranked_doc_ids, scores)}

        ious_all = {
            doc_id: jaccard(query_tokens_all[qid], doc_tokens_all[doc_id])
            for doc_id in ranked_doc_ids
        }
        ious_content = {
            doc_id: jaccard(query_tokens_content[qid], doc_tokens_content[doc_id])
            for doc_id in ranked_doc_ids
        }

        for doc_id in ranked_doc_ids:
            rr = 1.0 / rank_by_doc[doc_id]
            pair_iou_all.append(ious_all[doc_id])
            pair_iou_content.append(ious_content[doc_id])
            pair_scores.append(score_by_doc[doc_id])
            pair_rr.append(rr)
            if doc_id in relevant_docs:
                relevant_iou_all.append(ious_all[doc_id])
                relevant_iou_content.append(ious_content[doc_id])
                relevant_scores.append(score_by_doc[doc_id])
                relevant_rr.append(rr)

        relevant_ranks = [rank_by_doc[doc_id] for doc_id in relevant_docs]
        best_relevant_doc = min(relevant_docs, key=lambda doc_id: rank_by_doc[doc_id])
        top1_doc = ranked_doc_ids[0]
        best_lexical_doc = max(ranked_doc_ids, key=lambda doc_id: ious_content[doc_id])

        query_rows.append(
            {
                "query_id": qid,
                "query_text": query_by_id[qid]["text"],
                "cluster_type": query_by_id[qid].get("cluster_type", ""),
                "n_relevant": len(relevant_docs),
                "best_relevant_rank": min(relevant_ranks),
                "best_relevant_rr": 1.0 / min(relevant_ranks),
                "best_relevant_score": score_by_doc[best_relevant_doc],
                "best_relevant_iou_all": ious_all[best_relevant_doc],
                "best_relevant_iou_content": ious_content[best_relevant_doc],
                "max_relevant_iou_all": max(ious_all[doc_id] for doc_id in relevant_docs),
                "max_relevant_iou_content": max(ious_content[doc_id] for doc_id in relevant_docs),
                "mean_relevant_iou_content": float(np.mean([ious_content[doc_id] for doc_id in relevant_docs])),
                "top1_doc_id": top1_doc,
                "top1_is_relevant": int(top1_doc in relevant_docs),
                "top1_score": score_by_doc[top1_doc],
                "top1_iou_all": ious_all[top1_doc],
                "top1_iou_content": ious_content[top1_doc],
                "best_lexical_doc_id": best_lexical_doc,
                "best_lexical_doc_is_relevant": int(best_lexical_doc in relevant_docs),
                "best_lexical_doc_dense_rank": rank_by_doc[best_lexical_doc],
                "best_lexical_iou_content": ious_content[best_lexical_doc],
                "source_path": row["source_path"],
            }
        )

    summary = {
        "config": {
            "rankings_glob": args.rankings_glob,
            "n_queries": len(query_rows),
            "n_docs": len(doc_ids),
            "n_pair_observations": len(pair_scores),
            "n_relevant_observations": len(relevant_scores),
        },
        "pair_level": {
            "all_token_iou_vs_dense_score": correlations(pair_iou_all, pair_scores),
            "content_iou_vs_dense_score": correlations(pair_iou_content, pair_scores),
            "all_token_iou_vs_dense_reciprocal_rank": correlations(pair_iou_all, pair_rr),
            "content_iou_vs_dense_reciprocal_rank": correlations(pair_iou_content, pair_rr),
        },
        "relevant_pair_level": {
            "all_token_iou_vs_dense_score": correlations(relevant_iou_all, relevant_scores),
            "content_iou_vs_dense_score": correlations(relevant_iou_content, relevant_scores),
            "all_token_iou_vs_dense_reciprocal_rank": correlations(relevant_iou_all, relevant_rr),
            "content_iou_vs_dense_reciprocal_rank": correlations(relevant_iou_content, relevant_rr),
        },
        "query_level": {
            "max_relevant_content_iou_vs_best_relevant_rr": correlations(
                [row["max_relevant_iou_content"] for row in query_rows],
                [row["best_relevant_rr"] for row in query_rows],
            ),
            "max_relevant_content_iou_vs_best_relevant_score": correlations(
                [row["max_relevant_iou_content"] for row in query_rows],
                [row["best_relevant_score"] for row in query_rows],
            ),
            "top1_content_iou_vs_top1_correct": correlations(
                [row["top1_iou_content"] for row in query_rows],
                [row["top1_is_relevant"] for row in query_rows],
            ),
            "best_lexical_doc_is_relevant_rate": float(
                np.mean([row["best_lexical_doc_is_relevant"] for row in query_rows])
            ),
            "median_best_lexical_doc_dense_rank": float(
                np.median([row["best_lexical_doc_dense_rank"] for row in query_rows])
            ),
        },
    }

    summary_path = out_dir / "overlap_summary.json"
    per_query_path = out_dir / "per_query_overlap.csv"
    summary["summary_path"] = str(summary_path.relative_to(ROOT))
    summary["per_query_path"] = str(per_query_path.relative_to(ROOT))
    summary_path.write_text(json.dumps(summary, indent=2))

    fieldnames = list(query_rows[0].keys())
    with per_query_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(query_rows)

    content_score = summary["pair_level"]["content_iou_vs_dense_score"]
    content_rr = summary["query_level"]["max_relevant_content_iou_vs_best_relevant_rr"]
    notify.send(
        "lexical overlap analysis finished\n"
        f"pair content IoU vs score Spearman: {content_score['spearman']:.4f}\n"
        f"query max relevant IoU vs RR Spearman: {content_rr['spearman']:.4f}"
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rankings-glob",
        default=str(DEFAULT_RANKINGS_GLOB),
        help="Glob for full-ranking JSON files to analyze.",
    )
    argv = [arg for arg in sys.argv[1:] if arg]
    return parser.parse_args(argv)


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
