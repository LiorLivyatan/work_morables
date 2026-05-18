"""
Recalculate clustered retrieval metrics from saved ranking predictions.

This is for old runs that already saved full ranked fable indices per original
709 moral query. It maps each of the 669 unique clustered moral queries to the
first matching original moral row, then scores the saved rankings with
cluster-based multi-label qrels.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from lib.data import load_morals, load_qrels_moral_to_fable_clustered, load_unique_morals
from lib.retrieval_utils import compute_multilabel_metrics_from_matrix

OUT_DIR = ROOT / "results" / "clustered_recalculation"


def _norm(text: str) -> str:
    return text.strip()


def _load_prediction_rankings(path: Path) -> dict[int, list[int]]:
    data = json.loads(path.read_text())
    rankings = {}
    for row in data["queries"]:
        rankings[int(row["query_idx"])] = [int(i) for i in row["top_k_indices"]]
    return rankings


def _build_unique_to_original_query_map() -> dict[int, int]:
    original_morals = load_morals()
    unique_morals = load_unique_morals()

    first_original_idx_by_text = {}
    for idx, moral in enumerate(original_morals):
        first_original_idx_by_text.setdefault(_norm(moral["text"]), idx)

    mapping = {}
    missing = []
    for unique_idx, moral in enumerate(unique_morals):
        text = _norm(moral["text"])
        if text not in first_original_idx_by_text:
            missing.append(text)
            continue
        mapping[unique_idx] = first_original_idx_by_text[text]

    if missing:
        print(
            f"Warning: skipping {len(missing)} canonical-only unique moral(s) "
            f"not present in old prediction queries: {missing[:5]}"
        )
    return mapping


def _rankings_to_score_matrix(
    rankings: dict[int, list[int]],
    unique_to_original_query: dict[int, int],
    n_docs: int,
) -> np.ndarray:
    n_queries = max(unique_to_original_query) + 1 if unique_to_original_query else 0
    score_matrix = np.full((n_queries, n_docs), -np.inf, dtype=np.float32)
    for unique_idx, original_idx in unique_to_original_query.items():
        if original_idx not in rankings:
            raise ValueError(f"Missing saved ranking for original query_idx={original_idx}")
        ranked = rankings[original_idx]
        for rank_idx, doc_idx in enumerate(ranked):
            score_matrix[unique_idx, doc_idx] = -rank_idx
    return score_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalculate clustered metrics from saved predictions")
    parser.add_argument("prediction_json", type=Path, help="Saved prediction JSON with full top_k_indices")
    parser.add_argument("--output", type=Path, help="Output JSON path")
    args = parser.parse_args()

    pred_path = args.prediction_json
    notify.send(f"clustered recalculation starting\nprediction: {pred_path.name}")

    pred_data = json.loads(pred_path.read_text())
    rankings = _load_prediction_rankings(pred_path)
    unique_to_original = _build_unique_to_original_query_map()
    ground_truth_all = load_qrels_moral_to_fable_clustered()
    ground_truth = {
        q_idx: rels
        for q_idx, rels in ground_truth_all.items()
        if q_idx in unique_to_original
    }
    score_matrix = _rankings_to_score_matrix(rankings, unique_to_original, n_docs=709)
    metrics = compute_multilabel_metrics_from_matrix(score_matrix, ground_truth)

    result = {
        "source_prediction": str(pred_path),
        "run_key": pred_data.get("run_key"),
        "model": pred_data.get("model"),
        "benchmark": "clustered_unique_morals",
        "n_unique_moral_queries": len(ground_truth_all),
        "n_scored_queries": len(unique_to_original),
        "n_fable_docs": 709,
        "n_qrel_rows": sum(len(v) for v in ground_truth_all.values()),
        "n_scored_qrel_rows": sum(len(v) for v in ground_truth.values()),
        "metrics": metrics,
    }

    output_path = args.output
    if output_path is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUT_DIR / f"{pred_path.stem}__clustered_metrics.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    notify.send(
        f"clustered recalculation done\n"
        f"{result['run_key']}\n"
        f"MRR={metrics['MRR']:.4f} Hit@10={metrics['Hit@10']:.4f} "
        f"Recall@10={metrics['Recall@10']:.4f}"
    )


if __name__ == "__main__":
    main()
