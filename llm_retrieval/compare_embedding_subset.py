"""
Compare an embedding retrieval run on the exact moral IDs used by an LLM run.

This is intentionally extraction-only: it loads precomputed moral/document
embeddings, recomputes rankings by dot product, filters to the moral_id values
present in an LLM per-query CSV, and writes comparable metrics.

Usage
-----
./run.sh llm_retrieval/compare_embedding_subset.py \\
    --llm-run llm_retrieval/results/runs/2026-05-15_Gemini-3.1-Flash-Lite-Preview_minimal.csv
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.lib.loader import ExperimentConfig, load_dataset, load_embeddings, compute_rankings


DEFAULT_MORAL_EMBS = (
    ROOT
    / "finetuning/ft_07_storal_transfer/cache/embeddings/linq_s500/"
    / "fable_plus_summary/moral_embs.npy"
)
DEFAULT_DOC_EMBS = (
    ROOT
    / "finetuning/ft_07_storal_transfer/cache/embeddings/linq_s500/"
    / "fable_plus_summary/doc_embs.npy"
)
DEFAULT_LABEL = "ft07-linq-s500-fable+summary"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm-run", required=True, help="LLM per-query CSV to match by moral_id")
    p.add_argument("--moral-embs", default=str(DEFAULT_MORAL_EMBS))
    p.add_argument("--doc-embs", default=str(DEFAULT_DOC_EMBS))
    p.add_argument("--model-path", help="Encode embeddings from this SentenceTransformer model if .npy files are unavailable")
    p.add_argument("--doc-mode", choices=["raw", "fable_plus_summary"], default="fable_plus_summary")
    p.add_argument("--query-instruction", default="")
    p.add_argument(
        "--linq-moral-instruction",
        action="store_true",
        help="Use the ft_07 LINQ moral-to-fable query instruction.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--force-encode", action="store_true")
    p.add_argument("--label", default=DEFAULT_LABEL)
    p.add_argument(
        "--output-dir",
        default=str(ROOT / "llm_retrieval/results/comparisons"),
    )
    return p.parse_args()


def ndcg_at_10(rank: int) -> float:
    return 0.0 if rank > 10 else 1.0 / math.log2(rank + 1)


def encode_embeddings(args, target_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    from sentence_transformers import SentenceTransformer
    from finetuning.lib.data import load_pairs

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    moral_texts, doc_texts, _ = load_pairs(args.doc_mode)
    instruction = args.query_instruction
    if args.linq_moral_instruction:
        instruction = "Instruct: Given a moral statement, retrieve the fable that best conveys this moral.\nQuery: "
    query_texts = [f"{instruction}{moral_texts[i]}" for i in target_indices]

    print(f"Loading model: {model_path}")
    model = SentenceTransformer(str(model_path))

    encode_kwargs = {
        "normalize_embeddings": True,
        "show_progress_bar": True,
        "batch_size": args.batch_size,
    }
    moral_embs_subset = model.encode(query_texts, **encode_kwargs).astype(np.float32)
    doc_embs = model.encode(doc_texts, **encode_kwargs).astype(np.float32)
    return moral_embs_subset, doc_embs


def main():
    args = parse_args()
    llm_run = Path(args.llm_run)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with llm_run.open(newline="") as f:
        llm_rows = list(csv.DictReader(f))

    target_moral_ids = [row["moral_id"] for row in llm_rows]
    target_indices = [int(moral_id.removeprefix("moral_")) for moral_id in target_moral_ids]

    fables, morals, qrels = load_dataset()
    moral_embs_path = Path(args.moral_embs)
    doc_embs_path = Path(args.doc_embs)

    if (
        args.model_path
        and args.force_encode
        or args.model_path
        and (not moral_embs_path.exists() or not doc_embs_path.exists())
    ):
        moral_embs_subset, doc_embs = encode_embeddings(args, target_indices)
        sim = moral_embs_subset @ doc_embs.T
        rankings = {}
        for local_idx, q_idx in enumerate(target_indices):
            gt_fable = qrels[q_idx]
            scores_row = sim[local_idx]
            ranked = np.argsort(-scores_row).tolist()
            rankings[q_idx] = {
                "query_idx": q_idx,
                "gt_fable_idx": gt_fable,
                "ranked_indices": ranked,
                "gt_rank": ranked.index(gt_fable) + 1,
            }
    else:
        cfg = ExperimentConfig(args.moral_embs, args.doc_embs, args.label)
        moral_embs, doc_embs = load_embeddings(cfg)
        rankings = {row["query_idx"]: row for row in compute_rankings(moral_embs, doc_embs, qrels)}

    rows = []
    for moral_id, q_idx in zip(target_moral_ids, target_indices):
        ranking = rankings[q_idx]
        rank = ranking["gt_rank"]
        top10 = ranking["ranked_indices"][:10]
        rows.append(
            {
                "moral_id": moral_id,
                "moral_text": morals[q_idx]["text"],
                "relevant_fable": f"fable_{ranking['gt_fable_idx']:04d}",
                "ranked_ids": "|".join(f"fable_{idx:04d}" for idx in top10),
                "reciprocal_rank": 1.0 / rank if rank <= 10 else 0.0,
                "r_at_1": 1.0 if rank <= 1 else 0.0,
                "r_at_5": 1.0 if rank <= 5 else 0.0,
                "r_at_10": 1.0 if rank <= 10 else 0.0,
                "ndcg_at_10": ndcg_at_10(rank),
                "rank": rank if rank <= 10 else "",
                "full_rank": rank,
            }
        )

    found_ranks = [row["full_rank"] for row in rows]
    summary = {
        "label": args.label,
        "matched_llm_run": str(llm_run),
        "n_queries": len(rows),
        "MRR@10": float(np.mean([row["reciprocal_rank"] for row in rows])),
        "R@1": float(np.mean([row["r_at_1"] for row in rows])),
        "R@5": float(np.mean([row["r_at_5"] for row in rows])),
        "R@10": float(np.mean([row["r_at_10"] for row in rows])),
        "NDCG@10": float(np.mean([row["ndcg_at_10"] for row in rows])),
        "Mean_Rank": float(np.mean(found_ranks)),
        "Median_Rank": float(np.median(found_ranks)),
    }

    stem = llm_run.stem
    rows_path = out_dir / f"{stem}__{args.label}.csv"
    summary_path = out_dir / f"{stem}__{args.label}.json"

    with rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{args.label}")
    print(f"matched: {llm_run}")
    print(f"n={summary['n_queries']}")
    print(
        f"MRR@10={summary['MRR@10']:.4f}  "
        f"R@1={summary['R@1']:.4f}  "
        f"R@5={summary['R@5']:.4f}  "
        f"R@10={summary['R@10']:.4f}  "
        f"NDCG@10={summary['NDCG@10']:.4f}"
    )
    print(f"Mean_Rank={summary['Mean_Rank']:.1f}  Median_Rank={summary['Median_Rank']:.1f}")
    print(f"rows: {rows_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
