"""Lexical-control baselines for the clustered MoraLink benchmark.

Run only through run.sh:
    ./run.sh review_response/21_lexical_controls/eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from lib.retrieval_utils import compute_multilabel_metrics_from_matrix

CLUSTERED_DIR = ROOT / "data" / "clustered"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def load_json(path: Path):
    return json.loads(path.read_text())


def load_clustered_data() -> tuple[list[dict], list[dict], list[dict]]:
    morals = load_json(CLUSTERED_DIR / "morals_unique_corpus.json")
    fables = load_json(CLUSTERED_DIR / "fables_corpus.json")
    qrels = load_json(CLUSTERED_DIR / "qrels_moral_to_fable_clustered.json")
    return morals, fables, qrels


def build_relevance(qrels: list[dict], query_ids: list[str], doc_ids: list[str]) -> dict[int, set[int]]:
    query_to_idx = {qid: i for i, qid in enumerate(query_ids)}
    doc_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    relevant: dict[int, set[int]] = defaultdict(set)
    for row in qrels:
        if int(row.get("relevance", 1)) <= 0:
            continue
        relevant[query_to_idx[row["query_id"]]].add(doc_to_idx[row["doc_id"]])
    return dict(relevant)


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def score_tfidf(query_texts: list[str], doc_texts: list[str], ngram_max: int) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[A-Za-z0-9'][A-Za-z0-9']+\b",
        ngram_range=(1, ngram_max),
        norm="l2",
    )
    doc_matrix = vectorizer.fit_transform(doc_texts)
    query_matrix = vectorizer.transform(query_texts)
    return (query_matrix @ doc_matrix.T).toarray()


def score_bm25(
    query_texts: list[str],
    doc_texts: list[str],
    k1: float,
    b: float,
) -> np.ndarray:
    tokenized_docs = [tokenize(text) for text in doc_texts]
    doc_lengths = np.array([len(tokens) for tokens in tokenized_docs], dtype=np.float64)
    avg_doc_len = float(np.mean(doc_lengths)) if len(doc_lengths) else 0.0

    doc_term_freqs = [Counter(tokens) for tokens in tokenized_docs]
    doc_freqs: Counter[str] = Counter()
    for tf in doc_term_freqs:
        doc_freqs.update(tf.keys())

    n_docs = len(doc_texts)
    idf = {
        term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        for term, df in doc_freqs.items()
    }

    scores = np.zeros((len(query_texts), n_docs), dtype=np.float64)
    for query_idx, query in enumerate(query_texts):
        query_tf = Counter(tokenize(query))
        for term, query_count in query_tf.items():
            term_idf = idf.get(term)
            if term_idf is None:
                continue
            for doc_idx, tf in enumerate(doc_term_freqs):
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                denom = freq + k1 * (1.0 - b + b * doc_lengths[doc_idx] / avg_doc_len)
                scores[query_idx, doc_idx] += (
                    query_count * term_idf * (freq * (k1 + 1.0)) / denom
                )
    return scores


def first_sentence(text: str) -> str:
    for part in SENTENCE_SPLIT_RE.split(text.strip()):
        sentence = part.strip()
        if sentence:
            return sentence
    return text.strip()


def first_words(text: str, n_words: int) -> str:
    words = text.split()
    return " ".join(words[:n_words])


def build_doc_text(fable: dict, variant: str) -> str:
    if variant == "raw":
        return fable["text"]
    if variant == "title":
        return fable["title"]
    if variant == "first_sentence":
        return first_sentence(fable["text"])
    if variant == "first_50":
        return first_words(fable["text"], 50)
    raise ValueError(f"Unsupported doc variant: {variant}")


def save_rankings(
    score_matrix: np.ndarray,
    query_ids: list[str],
    doc_ids: list[str],
    path: Path,
) -> None:
    rankings = np.argsort(-score_matrix, axis=1)
    rows = []
    for query_idx, ranked in enumerate(rankings):
        rows.append(
            {
                "query_id": query_ids[query_idx],
                "ranked_fable_ids": [doc_ids[int(doc_idx)] for doc_idx in ranked.tolist()],
                "scores": [
                    round(float(score_matrix[query_idx, int(doc_idx)]), 6)
                    for doc_idx in ranked.tolist()
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))


def add_cutoff_metrics(metrics: dict, score_matrix: np.ndarray, relevant: dict[int, set[int]], k: int) -> None:
    rankings = np.argsort(-score_matrix, axis=1)
    reciprocal_ranks = []
    average_precisions = []

    for query_idx, relevant_docs in relevant.items():
        if not relevant_docs:
            continue
        ranked = rankings[query_idx][:k]

        first_rank = None
        hits = 0
        precision_sum = 0.0
        for rank_idx, doc_idx in enumerate(ranked, start=1):
            if int(doc_idx) not in relevant_docs:
                continue
            if first_rank is None:
                first_rank = rank_idx
            hits += 1
            precision_sum += hits / rank_idx

        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        average_precisions.append(precision_sum / len(relevant_docs))

    metrics[f"MRR@{k}"] = float(np.mean(reciprocal_ranks))
    metrics[f"MAP@{k}"] = float(np.mean(average_precisions))


def write_metrics_csv(rows: list[dict], path: Path) -> None:
    metric_keys = [
        "MRR",
        "MRR@10",
        "MAP",
        "MAP@10",
        "R-Precision",
        "Hit@1",
        "Hit@5",
        "Hit@10",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "Recall@50",
        "Recall@100",
        "NDCG@10",
        "Mean Rank",
        "Median Rank",
        "n_queries",
    ]
    fieldnames = ["method", "doc_variant", *metric_keys]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run(args: argparse.Namespace) -> dict:
    morals, fables, qrels = load_clustered_data()
    query_ids = [moral["doc_id"] for moral in morals]
    doc_ids = [fable["doc_id"] for fable in fables]
    query_texts = [moral["text"] for moral in morals]
    relevant = build_relevance(qrels, query_ids, doc_ids)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = RESULTS_DIR / timestamp
    rankings_dir = out_dir / "rankings"
    out_dir.mkdir(parents=True, exist_ok=True)

    notify.send(
        "lexical controls starting\n"
        f"queries: {len(query_texts)}\n"
        f"fables: {len(fables)}\n"
        f"qrels: {len(qrels)}\n"
        f"methods: {', '.join(args.methods)}\n"
        f"doc variants: {', '.join(args.doc_variants)}"
    )

    rows: list[dict] = []
    results: dict[str, dict] = {
        "config": {
            "n_queries": len(query_texts),
            "n_fables": len(fables),
            "n_qrels": len(qrels),
            "methods": args.methods,
            "doc_variants": args.doc_variants,
            "tfidf_ngram_max": args.tfidf_ngram_max,
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
        },
        "metrics": {},
    }

    for doc_variant in args.doc_variants:
        doc_texts = [build_doc_text(fable, doc_variant) for fable in fables]
        for method in args.methods:
            if method == "tfidf":
                score_matrix = score_tfidf(query_texts, doc_texts, args.tfidf_ngram_max)
            elif method == "bm25":
                score_matrix = score_bm25(query_texts, doc_texts, args.bm25_k1, args.bm25_b)
            else:
                raise ValueError(f"Unsupported method: {method}")

            metrics = compute_multilabel_metrics_from_matrix(score_matrix, relevant, ks=(1, 5, 10, 50, 100))
            add_cutoff_metrics(metrics, score_matrix, relevant, k=10)
            run_key = f"{method}__{doc_variant}"
            ranking_path = rankings_dir / f"{run_key}.json"
            save_rankings(score_matrix, query_ids, doc_ids, ranking_path)

            row = {"method": method, "doc_variant": doc_variant, **metrics}
            rows.append(row)
            results["metrics"][run_key] = {
                "method": method,
                "doc_variant": doc_variant,
                **metrics,
                "rankings_path": str(ranking_path.relative_to(ROOT)),
            }

            print(
                f"{run_key}: MRR={metrics['MRR']:.4f} "
                f"Hit@1={metrics['Hit@1']:.4f} "
                f"Hit@10={metrics['Hit@10']:.4f} "
                f"NDCG@10={metrics['NDCG@10']:.4f}"
            )

    metrics_csv = out_dir / "metrics.csv"
    results_json = out_dir / "lexical_controls.json"
    write_metrics_csv(rows, metrics_csv)
    results["metrics_csv"] = str(metrics_csv.relative_to(ROOT))
    results_json.write_text(json.dumps(results, indent=2))

    best = max(rows, key=lambda row: row["MRR"])
    notify.send(
        "lexical controls finished\n"
        f"best: {best['method']} / {best['doc_variant']}\n"
        f"MRR: {best['MRR']:.4f}\n"
        f"Hit@10: {best['Hit@10']:.4f}"
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["tfidf", "bm25"],
        choices=["tfidf", "bm25"],
        help="Lexical baselines to evaluate.",
    )
    parser.add_argument("--tfidf-ngram-max", type=int, default=1)
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument(
        "--doc-variants",
        nargs="+",
        default=["raw"],
        choices=["raw", "title", "first_sentence", "first_50"],
        help="Document surface variants to evaluate.",
    )
    argv = [arg for arg in sys.argv[1:] if arg]
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
