"""
recover_defaults.py — Rebuild default-variant rows from cached embeddings.

The second eval run (--variants generic) overwrote the CSV with only 60 generic
rows. The default-variant embeddings are still cached. This script reads the
cached .npy files directly (no model loading) and reconstructs the full 120-row CSV.

Usage:
    ./run.sh experiments/12_zero_shot_comprehensive/recover_defaults.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from lib.retrieval_utils import compute_metrics

CONFIG_PATH = EXP_DIR / "config.yaml"
CACHE_DIR   = EXP_DIR / "cache"
CSV_PATH    = EXP_DIR / "results" / "zero_shot_comprehensive.csv"

CSV_FIELDNAMES = [
    "model_alias", "model_hf_id", "model_size",
    "instruction_variant", "query_instruction",
    "corpus_config", "corpus_description", "corpus_template",
    "MRR@10", "R@1", "R@5", "R@10", "NDCG@10",
    "Mean_Rank", "Median_Rank", "n_queries",
    "notes",
]


def load_ground_truth():
    from lib.data import load_morals, load_qrels_moral_to_fable
    morals        = load_morals()
    qrels         = load_qrels_moral_to_fable()
    moral_indices = sorted(qrels.keys())
    ground_truth  = {i: qrels[idx] for i, idx in enumerate(moral_indices)}
    return ground_truth


def metrics_to_row(m: dict) -> dict:
    return {
        "MRR@10":      round(m.get("MRR",        0), 6),
        "R@1":         round(m.get("Recall@1",   0), 6),
        "R@5":         round(m.get("Recall@5",   0), 6),
        "R@10":        round(m.get("Recall@10",  0), 6),
        "NDCG@10":     round(m.get("NDCG@10",    0), 6),
        "Mean_Rank":   round(m.get("Mean Rank",  0), 2),
        "Median_Rank": round(m.get("Median Rank",0), 1),
        "n_queries":   m.get("n_queries", 0),
    }


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Load existing generic rows
    with open(CSV_PATH) as f:
        generic_rows = list(csv.DictReader(f))
    print(f"  Loaded {len(generic_rows)} existing rows from CSV")

    ground_truth = load_ground_truth()

    new_rows = []
    missing  = []

    for model_cfg in config["models"]:
        alias = model_cfg["alias"]
        query_instr = model_cfg.get("query_instruction", "")

        for corpus_cfg in config["corpus_configs"]:
            cfg_id = corpus_cfg["id"]
            cache  = CACHE_DIR / "embeddings" / alias / cfg_id
            moral_path = cache / "moral_embs.npy"
            doc_path   = cache / "doc_embs.npy"

            base_row = {
                "model_alias":        alias,
                "model_hf_id":        model_cfg["id"],
                "model_size":         model_cfg.get("size", ""),
                "instruction_variant": "default",
                "query_instruction":   query_instr,
                "corpus_config":       cfg_id,
                "corpus_description":  corpus_cfg["description"],
                "corpus_template":     corpus_cfg["template"].replace("\n", "\\n"),
                "notes": "",
            }

            if not moral_path.exists() or not doc_path.exists():
                print(f"  ✗ Cache missing: {alias} / {cfg_id}")
                missing.append((alias, cfg_id))
                base_row.update({m: "" for m in ["MRR@10","R@1","R@5","R@10","NDCG@10","Mean_Rank","Median_Rank","n_queries"]})
                base_row["notes"] = "cache_missing"
                new_rows.append(base_row)
                continue

            moral_embs = np.load(str(moral_path))
            doc_embs   = np.load(str(doc_path))
            metrics    = compute_metrics(moral_embs, doc_embs, ground_truth)
            base_row.update(metrics_to_row(metrics))
            new_rows.append(base_row)
            print(f"  ✓ {alias:<30} {cfg_id:<20} MRR={metrics['MRR']:.4f}")

    all_rows = new_rows + generic_rows
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n  Written {len(all_rows)} rows → {CSV_PATH}")
    if missing:
        print(f"  WARNING: {len(missing)} caches missing: {missing}")


if __name__ == "__main__":
    main()
