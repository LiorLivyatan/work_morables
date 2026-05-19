"""
Step 1: Encode all morals + fables with vanilla Linq-Embed-Mistral, save
embeddings, compute moral→fable rankings, write ranks_baseline.json.

Run via: ./run.sh analysis/08_concept_steering/run_baseline.py [--remote --gpu N]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datetime import datetime

EXP_DIR = Path(__file__).resolve().parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR))

import numpy as np

from finetuning.lib import notify
from lib.config import load_config
from lib.data import load_corpus
from lib.io import save_npy, save_json, text_hash
from lib.model import load_model, encode
from lib.retrieval import compute_rankings, mrr_at_k
from lib.eval import best_rank_per_query


def _assert_corpus_shape(cfg: dict, corpus) -> None:
    """Pre-launch sanity checks: dataset-specific shape assertions.
    Triggered only when an `expect` block is present in config; tolerant otherwise."""
    expect = cfg.get("data", {}).get("expect", {})
    if not expect:
        return
    if "n_morals" in expect:
        assert len(corpus.moral_ids) == expect["n_morals"], \
            f"expected {expect['n_morals']} morals, got {len(corpus.moral_ids)}"
    if "n_fables" in expect:
        assert len(corpus.fable_doc_ids) == expect["n_fables"], \
            f"expected {expect['n_fables']} fables, got {len(corpus.fable_doc_ids)}"
    if "n_qrels" in expect:
        actual = sum(len(g) for g in corpus.gt_fable_idxs)
        assert actual == expect["n_qrels"], \
            f"expected {expect['n_qrels']} qrels rows, got {actual}"
    assert all(len(g) >= 1 for g in corpus.gt_fable_idxs), \
        "every moral must have ≥1 relevant fable"


def main(config_path: Path, force: bool = False) -> int:
    cfg = load_config(config_path)
    cache_dir   = ROOT / cfg["output"]["cache_dir"]
    results_dir = ROOT / cfg["output"]["results_dir"]

    qrels_path = ROOT / cfg["data"].get(
        "qrels_path", "data/processed/qrels_moral_to_fable.json"
    )
    corpus = load_corpus(
        morals_path=ROOT / cfg["data"]["morals_path"],
        fables_path=ROOT / cfg["data"]["fables_path"],
        qrels_path =qrels_path,
    )
    _assert_corpus_shape(cfg, corpus)
    notify.send(
        f"🚀 08_concept_steering: run_baseline starting\n"
        f"model: {cfg['model']['hf_id']}\n"
        f"n_morals={len(corpus.moral_texts)} n_fables={len(corpus.fable_texts)}"
    )

    moral_cache = cache_dir / f"moral_embs_{text_hash(corpus.moral_texts)}.npy"
    fable_cache = cache_dir / f"fable_embs_{text_hash(corpus.fable_texts)}.npy"

    if not force and moral_cache.exists() and fable_cache.exists():
        moral_embs = np.load(moral_cache)
        fable_embs = np.load(fable_cache)
        print("[baseline] cache hit: reusing embeddings")
    else:
        handle = load_model(cfg)
        bs = cfg["model"]["batch_size"]
        moral_embs = encode(handle, corpus.moral_texts, batch_size=bs)
        fable_embs = encode(handle, corpus.fable_texts, batch_size=bs)
        save_npy(moral_cache, moral_embs)
        save_npy(fable_cache, fable_embs)

    rankings = compute_rankings(moral_embs, fable_embs)
    gt = corpus.gt_fable_idxs  # list[list[int]]
    mrr10 = mrr_at_k(rankings, gt, k=10)
    best_ranks = best_rank_per_query(rankings, gt)

    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": cfg["model"]["hf_id"],
        "n_morals": len(corpus.moral_texts),
        "n_fables": len(corpus.fable_texts),
        "mrr_at_10": mrr10,
        "qrels_path": str(qrels_path),
        "queries": [
            {
                "moral_id": corpus.moral_ids[i],
                "gt_fable_idxs": [int(x) for x in gt[i]],
                "gt_fable_doc_ids": [corpus.fable_doc_ids[x] for x in gt[i]],
                "top_50_indices": rankings[i, :50].tolist(),
                "gt_rank": int(best_ranks[i]),   # best rank over relevant set
            }
            for i in range(len(corpus.moral_texts))
        ],
        "fable_doc_ids": corpus.fable_doc_ids,
    }
    save_json(results_dir / "ranks_baseline.json", out)

    n_failed = sum(1 for q in out["queries"] if q["gt_rank"] > 1)
    notify.send(
        f"✅ 08_concept_steering: run_baseline done\n"
        f"MRR@10 = {mrr10:.4f}\n"
        f"failures (rank > 1): {n_failed}/{len(out['queries'])}"
    )
    print(f"[baseline] MRR@10 = {mrr10:.4f}  failures = {n_failed}/{len(out['queries'])}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(EXP_DIR / "config.yaml"),
                    help="Path to config.yaml (default: this experiment's config)")
    p.add_argument("--force", action="store_true",
                    help="Re-encode even if cache exists")
    args = p.parse_args()
    sys.exit(main(Path(args.config), force=args.force))
