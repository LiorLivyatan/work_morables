"""
Prepare 5-fold clustered MORABLES splits.

Run with:
    ./run.sh finetuning/ft_11_clustered/prepare_data.py
"""
import json
import sys
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_PATH = EXP_DIR / "config.yaml"
DATA_DIR = EXP_DIR / "data"
FOLDS_PATH = DATA_DIR / "folds.json"
SUMMARY_PATH = DATA_DIR / "dataset_summary.json"

CLUSTERED_DIR = ROOT / "data" / "clustered"
MORALS_PATH = CLUSTERED_DIR / "morals_unique_corpus.json"
FABLES_PATH = CLUSTERED_DIR / "fables_corpus.json"
QRELS_PATH = CLUSTERED_DIR / "qrels_moral_to_fable_clustered.json"


def load_clustered_data() -> tuple[list[dict], list[dict], list[dict]]:
    return (
        json.loads(MORALS_PATH.read_text()),
        json.loads(FABLES_PATH.read_text()),
        json.loads(QRELS_PATH.read_text()),
    )


def build_folds(morals: list[dict], qrels: list[dict], n_folds: int) -> list[dict]:
    positives_by_query: dict[str, set[str]] = {m["doc_id"]: set() for m in morals}
    for row in qrels:
        if int(row.get("relevance", 1)) > 0:
            positives_by_query.setdefault(row["query_id"], set()).add(row["doc_id"])

    id_to_idx = {m["doc_id"]: i for i, m in enumerate(morals)}
    ordered = sorted(
        morals,
        key=lambda m: (-len(positives_by_query.get(m["doc_id"], set())), m["doc_id"]),
    )

    fold_tests: list[list[int]] = [[] for _ in range(n_folds)]
    fold_qrels = [0 for _ in range(n_folds)]
    fold_counts = [0 for _ in range(n_folds)]

    for moral in ordered:
        qid = moral["doc_id"]
        fold_idx = min(range(n_folds), key=lambda i: (fold_qrels[i], fold_counts[i], i))
        fold_tests[fold_idx].append(id_to_idx[qid])
        fold_qrels[fold_idx] += len(positives_by_query.get(qid, set()))
        fold_counts[fold_idx] += 1

    all_idx = set(range(len(morals)))
    folds = []
    for fold_idx, test_idx in enumerate(fold_tests):
        test = sorted(test_idx)
        train = sorted(all_idx - set(test))
        folds.append(
            {
                "fold": fold_idx,
                "train": train,
                "test": test,
                "n_train_queries": len(train),
                "n_test_queries": len(test),
                "n_test_qrels": sum(len(positives_by_query[morals[i]["doc_id"]]) for i in test),
            }
        )
    return folds


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    morals, fables, qrels = load_clustered_data()
    folds = build_folds(morals, qrels, int(config["n_folds"]))

    positives_by_query: dict[str, set[str]] = {}
    for row in qrels:
        positives_by_query.setdefault(row["query_id"], set()).add(row["doc_id"])

    summary = {
        "n_queries": len(morals),
        "n_fables": len(fables),
        "n_qrels": len(qrels),
        "avg_relevant_fables_per_query": sum(len(v) for v in positives_by_query.values()) / len(morals),
        "folds": [
            {
                "fold": f["fold"],
                "n_train_queries": f["n_train_queries"],
                "n_test_queries": f["n_test_queries"],
                "n_test_qrels": f["n_test_qrels"],
            }
            for f in folds
        ],
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FOLDS_PATH.write_text(json.dumps(folds, indent=2))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"folds -> {FOLDS_PATH}")
    print(f"summary -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

