"""Lexical retrieval robustness on MORABLES adversarial story variants.

Run only through run.sh:
    ./run.sh review_response/22_adversarial_robustness/eval_lexical.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from lib.retrieval_utils import compute_multilabel_metrics_from_matrix


CLUSTERED_DIR = ROOT / "data" / "clustered"
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def load_json(path: Path):
    return json.loads(path.read_text())


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
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[A-Za-z0-9'][A-Za-z0-9']+\b",
        ngram_range=(1, ngram_max),
        norm="l2",
    )
    doc_matrix = vectorizer.fit_transform(doc_texts)
    query_matrix = vectorizer.transform(query_texts)
    return (query_matrix @ doc_matrix.T).toarray()


def score_bm25(query_texts: list[str], doc_texts: list[str], k1: float, b: float) -> np.ndarray:
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


def add_cutoff_metrics(metrics: dict, score_matrix: np.ndarray, relevant: dict[int, set[int]], k: int) -> None:
    rankings = np.argsort(-score_matrix, axis=1)
    reciprocal_ranks = []
    average_precisions = []

    for query_idx, relevant_docs in relevant.items():
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


def variant_label(path: Path) -> str:
    label = path.stem
    if label.startswith("adversarial_"):
        label = label.removeprefix("adversarial_")
    return label


def load_variant_docs(path: Path, fables: list[dict]) -> list[str]:
    rows = load_json(path)
    story_by_alias = {row["alias"]: row["story"] for row in rows}
    missing = [fable["alias"] for fable in fables if fable["alias"] not in story_by_alias]
    if missing:
        raise KeyError(f"{path} missing {len(missing)} aliases; first={missing[0]}")
    return [story_by_alias[fable["alias"]] for fable in fables]


def discover_variant_paths(include_shuffled: bool) -> list[Path]:
    paths = [Path(path) for path in sorted(glob.glob(str(RAW_DIR / "adversarial_*.json")))]
    if not include_shuffled:
        paths = [path for path in paths if path.name.endswith("_not_shuffled.json")]
    return paths


def write_metrics_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "variant",
        "method",
        "MRR",
        "MRR@10",
        "MAP",
        "MAP@10",
        "R-Precision",
        "Hit@1",
        "Hit@5",
        "Hit@10",
        "Recall@10",
        "Recall@50",
        "Recall@100",
        "NDCG@10",
        "Mean Rank",
        "Median Rank",
        "n_queries",
        "delta_mrr_vs_original",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def evaluate(args: argparse.Namespace) -> dict:
    morals = load_json(CLUSTERED_DIR / "morals_unique_corpus.json")
    fables = load_json(CLUSTERED_DIR / "fables_corpus.json")
    qrels = load_json(CLUSTERED_DIR / "qrels_moral_to_fable_clustered.json")

    query_ids = [moral["doc_id"] for moral in morals]
    doc_ids = [fable["doc_id"] for fable in fables]
    query_texts = [moral["text"] for moral in morals]
    relevant = build_relevance(qrels, query_ids, doc_ids)

    variants: list[tuple[str, list[str], str]] = [
        ("original", [fable["text"] for fable in fables], "data/clustered/fables_corpus.json")
    ]
    for path in discover_variant_paths(args.include_shuffled):
        variants.append((variant_label(path), load_variant_docs(path, fables), str(path.relative_to(ROOT))))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    notify.send(
        "adversarial lexical robustness starting\n"
        f"variants: {len(variants)}\n"
        f"methods: {', '.join(args.methods)}"
    )

    rows = []
    result = {
        "config": {
            "methods": args.methods,
            "include_shuffled": args.include_shuffled,
            "n_queries": len(query_texts),
            "n_fables": len(fables),
            "n_qrels": len(qrels),
        },
        "metrics": {},
    }

    for label, doc_texts, source in variants:
        for method in args.methods:
            if method == "tfidf":
                score_matrix = score_tfidf(query_texts, doc_texts, args.tfidf_ngram_max)
            elif method == "bm25":
                score_matrix = score_bm25(query_texts, doc_texts, args.bm25_k1, args.bm25_b)
            else:
                raise ValueError(f"Unsupported method: {method}")

            metrics = compute_multilabel_metrics_from_matrix(score_matrix, relevant, ks=(1, 5, 10, 50, 100))
            add_cutoff_metrics(metrics, score_matrix, relevant, k=10)
            row = {"variant": label, "method": method, **metrics}
            rows.append(row)
            result["metrics"][f"{label}__{method}"] = {
                "variant": label,
                "method": method,
                "source": source,
                **metrics,
            }
            print(
                f"{label} / {method}: MRR={metrics['MRR']:.4f} "
                f"MRR@10={metrics['MRR@10']:.4f} Hit@10={metrics['Hit@10']:.4f}"
            )

    baseline_by_method = {
        row["method"]: row
        for row in rows
        if row["variant"] == "original"
    }
    for row in rows:
        baseline = baseline_by_method[row["method"]]
        row["delta_mrr_vs_original"] = row["MRR"] - baseline["MRR"]
        result["metrics"][f"{row['variant']}__{row['method']}"]["delta_mrr_vs_original"] = row[
            "delta_mrr_vs_original"
        ]

    metrics_csv = out_dir / "metrics.csv"
    result_json = out_dir / "adversarial_lexical_results.json"
    write_metrics_csv(rows, metrics_csv)
    result["metrics_csv"] = str(metrics_csv.relative_to(ROOT))
    result_json.write_text(json.dumps(result, indent=2))

    notify.send(
        "adversarial lexical robustness finished\n"
        f"results: {metrics_csv.relative_to(ROOT)}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["tfidf", "bm25"],
        choices=["tfidf", "bm25"],
    )
    parser.add_argument("--include-shuffled", action="store_true")
    parser.add_argument("--tfidf-ngram-max", type=int, default=1)
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    argv = [arg for arg in sys.argv[1:] if arg]
    return parser.parse_args(argv)


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
