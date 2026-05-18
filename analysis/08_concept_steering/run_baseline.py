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


def main(config_path: Path, force: bool = False) -> int:
    cfg = load_config(config_path)
    cache_dir   = ROOT / cfg["output"]["cache_dir"]
    results_dir = ROOT / cfg["output"]["results_dir"]

    corpus = load_corpus(
        morals_path=ROOT / cfg["data"]["morals_path"],
        fables_path=ROOT / cfg["data"]["fables_path"],
        qrels_path =ROOT / "data/processed/qrels_moral_to_fable.json",
    )
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
    gt = np.array(corpus.gt_fable_idx)
    mrr10 = mrr_at_k(rankings, gt, k=10)

    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": cfg["model"]["hf_id"],
        "n_morals": len(corpus.moral_texts),
        "n_fables": len(corpus.fable_texts),
        "mrr_at_10": mrr10,
        "queries": [
            {
                "moral_id": corpus.moral_ids[i],
                "gt_fable_idx": int(gt[i]),
                "gt_fable_doc_id": corpus.fable_doc_ids[gt[i]],
                "top_50_indices": rankings[i, :50].tolist(),
                "gt_rank": int(np.where(rankings[i] == gt[i])[0][0]) + 1,
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
