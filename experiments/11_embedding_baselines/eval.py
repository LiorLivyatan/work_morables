"""
exp_11_embedding_baselines — Zero-shot retrieval baseline for new embedding models.

Evaluates each model on the full MORABLES dataset (709 fables, fable+summary corpus)
with no fine-tuning. Results identify which models are worth fine-tuning next.

Usage
-----
    # All models:
    ./run.sh experiments/11_embedding_baselines/eval.py

    # Specific models (by alias):
    ./run.sh experiments/11_embedding_baselines/eval.py --models NV-Embed-v2 GritLM-7B

    # Force re-embed (ignore cache):
    ./run.sh experiments/11_embedding_baselines/eval.py --force

    # Remote GPU (single):
    ./run.sh experiments/11_embedding_baselines/eval.py --remote --gpu 2

    # Remote GPU (multi — needed for KaLM-12B):
    CUDA_VISIBLE_DEVICES=2,3 ./run.sh experiments/11_embedding_baselines/eval.py --models KaLM-Gemma3-12B --remote
"""
import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from finetuning.lib.data import load_pairs
from finetuning.lib.eval import evaluate

CONFIG_PATH = EXP_DIR / "config.yaml"
CACHE_DIR   = EXP_DIR / "cache"
RESULTS_DIR = EXP_DIR / "results"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_cfg: dict, device: str):
    from sentence_transformers import SentenceTransformer

    model_kwargs: dict = {}
    if "torch_dtype" in model_cfg:
        model_kwargs["torch_dtype"] = getattr(torch, model_cfg["torch_dtype"])

    use_device_map = model_cfg.get("device_map") == "auto"
    if use_device_map:
        model_kwargs["device_map"] = "auto"

    st_kwargs: dict = {
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
    }
    if model_kwargs:
        st_kwargs["model_kwargs"] = model_kwargs

    # When device_map=auto is set, SentenceTransformer must not receive device arg
    if use_device_map:
        model = SentenceTransformer(model_cfg["id"], **st_kwargs)
    else:
        model = SentenceTransformer(model_cfg["id"], device=device, **st_kwargs)

    return model


def run_model(
    model_cfg: dict,
    moral_texts: list[str],
    doc_texts: list[str],
    ground_truth: dict[int, int],
    device: str,
    batch_size: int,
    force: bool,
) -> dict:
    alias = model_cfg["alias"]
    query_instruction = model_cfg.get("query_instruction", "")
    doc_prefix        = model_cfg.get("doc_prefix", "")

    emb_cache = CACHE_DIR / "embeddings" / alias.replace(" ", "_")

    # Apply doc prefix if model needs it (e.g. Nomic search_document:)
    effective_docs = (
        [f"{doc_prefix}{d}" for d in doc_texts] if doc_prefix else doc_texts
    )

    print(f"\n  Loading {alias} ({model_cfg['id']}) …")
    model = load_model(model_cfg, device)

    # Override batch_size if model specifies it (e.g. large models need smaller batches)
    effective_batch = model_cfg.get("batch_size", batch_size)

    print(f"  Evaluating {alias} …")
    metrics = evaluate(
        model,
        moral_texts,
        effective_docs,
        ground_truth,
        cache_dir=emb_cache,
        force=force,
        query_prompt=query_instruction or None,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="exp_11 embedding model baseline evaluation")
    parser.add_argument("--models", nargs="+", help="Model aliases to run (default: all)")
    parser.add_argument("--force", action="store_true", help="Re-embed even if cached")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    all_models = config["models"]
    if args.models:
        selected = {m.lower() for m in args.models}
        all_models = [m for m in all_models if m["alias"].lower() in selected]
        if not all_models:
            print(f"No models matched: {args.models}")
            sys.exit(1)

    doc_mode   = config["doc_mode"]
    batch_size = config["batch_size"]
    device     = get_device()

    moral_texts, doc_texts, ground_truth = load_pairs(doc_mode)
    # Full-dataset evaluation: all 709 morals as queries
    ground_truth_full = {i: ground_truth[i] for i in range(len(moral_texts))}

    print(f"\n[exp_11_embedding_baselines]")
    print(f"  doc_mode={doc_mode}  device={device}  n_queries={len(moral_texts)}  n_docs={len(doc_texts)}")
    print(f"  Models to run: {[m['alias'] for m in all_models]}\n")

    notify.send(
        f"🔍 exp_11 embedding baselines starting\n"
        f"device: {device}  doc_mode: {doc_mode}\n"
        f"models: {', '.join(m['alias'] for m in all_models)}"
    )

    results: list[dict] = []

    for model_cfg in all_models:
        alias = model_cfg["alias"]
        print(f"\n{'─' * 55}")
        print(f"  {alias}")
        print(f"{'─' * 55}")

        try:
            metrics = run_model(
                model_cfg, moral_texts, doc_texts, ground_truth_full,
                device, batch_size, args.force,
            )
            results.append({"alias": alias, "model_id": model_cfg["id"], **metrics})

            mrr = metrics["MRR"]
            r1  = metrics.get("R@1", metrics.get("Recall@1", 0))
            r10 = metrics.get("R@10", metrics.get("Recall@10", 0))
            print(f"\n  → MRR={mrr:.4f}  R@1={r1:.3f}  R@10={r10:.3f}")
            notify.send(f"exp_11 ✓ {alias}\nMRR={mrr:.4f}  R@1={r1:.3f}  R@10={r10:.3f}")

        except Exception as e:
            print(f"\n  ✗ {alias} failed: {e}")
            notify.send(f"exp_11 ✗ {alias} failed\n{e}")
            results.append({"alias": alias, "model_id": model_cfg["id"], "error": str(e)})

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

    # ── Summary table ──────────────────────────────────────────────────────────
    ok = [r for r in results if "MRR" in r]
    ok.sort(key=lambda r: r["MRR"], reverse=True)

    print(f"\n{'=' * 62}")
    print(f"  {'Model':<28}  {'MRR':>7}  {'R@1':>6}  {'R@10':>6}")
    print(f"{'─' * 62}")
    linq_mrr = next((r["MRR"] for r in ok if "Linq" in r["alias"]), None)
    for r in ok:
        delta = f" (+{r['MRR'] - linq_mrr:.4f})" if linq_mrr and r["alias"] != "Linq-Embed-Mistral" else ""
        r1  = r.get("R@1", r.get("Recall@1", 0))
        r10 = r.get("R@10", r.get("Recall@10", 0))
        print(f"  {r['alias']:<28}  {r['MRR']:.4f}  {r1:.4f}  {r10:.4f}{delta}")
    for r in results:
        if "error" in r:
            print(f"  {r['alias']:<28}  ERROR: {r['error'][:30]}")
    print(f"{'=' * 62}")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = "_".join(m["alias"] for m in all_models) if len(all_models) <= 3 else "all"
    out = RESULTS_DIR / f"{ts}_{tag}.json"
    with open(out, "w") as f:
        json.dump({"config": config, "device": device, "results": results}, f, indent=2)
    print(f"\n  Results → {out}")

    best = ok[0] if ok else None
    notify.send(
        f"✅ exp_11 done\n"
        f"Best: {best['alias']} MRR={best['MRR']:.4f}\n" if best else "exp_11 done (no results)\n"
        f"Results → {out.name}"
    )


if __name__ == "__main__":
    main()
