"""
exp_13_gemma4_gpu_summarization — Retrieval eval for Gemma 4 generated summaries.

Loads the unified summaries JSON produced by generate.py and evaluates each
(generator model × prompt variant × corpus type) combination using one or more
retrieval embedding models. Writes a CSV for paper reporting.

Corpus types tested for each summary:
  fable_summary  — "{fable}\\n\\nMoral summary: {summary}"  (Config B analogue)
  summary_only   — "{summary}"                               (Config A analogue)

Optionally compares against the Gemini baseline (exp_07 golden_summaries.json).

Usage
-----
    # Basic eval with default retrieval model (Linq):
    ./run.sh experiments/13_gemma4_gpu_summarization/eval.py \\
        --summaries results/2026-xx-xx_summaries.json

    # Specify retrieval model alias (must be in config.yaml retrieval_models):
    ./run.sh experiments/13_gemma4_gpu_summarization/eval.py \\
        --summaries results/2026-xx-xx_summaries.json \\
        --retrieval_model Qwen3-Embedding-8B

    # Include Gemini baseline comparison:
    ./run.sh experiments/13_gemma4_gpu_summarization/eval.py \\
        --summaries results/2026-xx-xx_summaries.json \\
        --gemini_summaries ../07_sota_summarization_oracle/results/generation_runs/full_709/golden_summaries.json

    # Filter to specific generator models or variants:
    ./run.sh experiments/13_gemma4_gpu_summarization/eval.py \\
        --summaries results/2026-xx-xx_summaries.json \\
        --gen_models gemma4-31B --variants cot_proverb thinking_cot_proverb

    # Remote GPU:
    ./run.sh experiments/13_gemma4_gpu_summarization/eval.py \\
        --summaries results/2026-xx-xx_summaries.json --remote --gpu 2
"""
import argparse
import csv
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from finetuning.lib.eval import evaluate

CONFIG_PATH  = EXP_DIR / "config.yaml"
CACHE_DIR    = EXP_DIR / "cache"
RESULTS_DIR  = EXP_DIR / "results"

GEMINI_SUMMARIES_DEFAULT = (
    ROOT / "experiments/07_sota_summarization_oracle"
    / "results/generation_runs/full_709/golden_summaries.json"
)

CSV_FIELDNAMES = [
    "generator_model", "prompt_variant", "corpus_type",
    "retrieval_model", "retrieval_model_id",
    "MRR@10", "R@1", "R@5", "R@10", "NDCG@10",
    "Mean_Rank", "Median_Rank", "n_queries",
    "notes",
]

# The two corpus types that mirror exp-07/09 Config A and B
CORPUS_TEMPLATES = {
    "fable_summary": "{fable}\n\nMoral summary: {summary}",
    "summary_only":  "{summary}",
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_base_data():
    from lib.data import load_fables, load_morals, load_qrels_moral_to_fable
    fables        = load_fables()
    morals        = load_morals()
    qrels         = load_qrels_moral_to_fable()
    moral_indices = sorted(qrels.keys())
    moral_texts   = [morals[i]["text"] for i in moral_indices]
    ground_truth  = {i: qrels[idx] for i, idx in enumerate(moral_indices)}
    return fables, moral_texts, ground_truth


def build_corpus(
    fables: list[dict],
    summary_by_alias: dict[str, str],
    template: str,
    doc_prefix: str = "",
) -> list[str]:
    docs = []
    for fable in fables:
        alias   = fable["alias"]
        summary = summary_by_alias.get(alias, "")
        doc     = template.format(fable=fable["text"], summary=summary)
        if doc_prefix:
            doc = doc_prefix + doc
        docs.append(doc)
    return docs


# ── Model helpers ──────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():    return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_retrieval_model(model_cfg: dict, device: str):
    from sentence_transformers import SentenceTransformer
    dtype_str  = model_cfg.get("torch_dtype", "bfloat16")
    torch_dtype = getattr(torch, dtype_str) if isinstance(dtype_str, str) else dtype_str
    st_kwargs = {
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
        "model_kwargs": {"torch_dtype": torch_dtype},
    }
    if model_cfg.get("device_map") == "auto":
        return SentenceTransformer(model_cfg["id"], **st_kwargs)
    return SentenceTransformer(model_cfg["id"], device=device, **st_kwargs)


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries",        required=True, help="Path to generate.py output JSON")
    parser.add_argument("--retrieval_model",  default=None,  help="Retrieval model alias from config")
    parser.add_argument("--gemini_summaries", default=None,  help="Path to exp-07 golden_summaries.json for baseline")
    parser.add_argument("--gen_models",       nargs="+",     help="Generator model aliases to eval (default: all)")
    parser.add_argument("--variants",         nargs="+",     help="Prompt variant IDs to eval (default: all)")
    parser.add_argument("--force",            action="store_true", help="Re-embed even if cached")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load Gemma 4 summaries
    with open(args.summaries) as f:
        raw = json.load(f)
    summaries_by_fable: dict[str, dict] = {
        item["fable_alias"]: item["summaries"]
        for item in raw
    }
    print(f"  Loaded {len(summaries_by_fable)} fables from {args.summaries}")

    gen_models  = config["models"]
    variants    = config["prompt_variants"]
    if args.gen_models:
        sel = {m.lower() for m in args.gen_models}
        gen_models = [m for m in gen_models if m["alias"].lower() in sel]
    if args.variants:
        sel = {v.lower() for v in args.variants}
        variants = [v for v in variants if v["id"].lower() in sel]

    retrieval_cfgs = config.get("retrieval_models", [])
    if args.retrieval_model:
        retrieval_cfgs = [r for r in retrieval_cfgs
                          if r["alias"].lower() == args.retrieval_model.lower()]
    if not retrieval_cfgs:
        print("No retrieval models configured. Add retrieval_models to config.yaml.")
        sys.exit(1)

    # Build Gemini baseline entries if requested
    gemini_entries: list[tuple[str, str, dict]] = []  # (gen_label, variant_id, summary_by_alias)
    gemini_path = Path(args.gemini_summaries) if args.gemini_summaries else GEMINI_SUMMARIES_DEFAULT
    if gemini_path.exists():
        with open(gemini_path) as f:
            gemini_data = json.load(f)
        gemini_by_alias = {item["original_fable_id"]: item["summaries"] for item in gemini_data}
        for var_id in ("cot_proverb", "direct_moral", "conceptual_abstract", "proverb"):
            s_by_alias = {alias: s.get(var_id, "") for alias, s in gemini_by_alias.items()}
            gemini_entries.append(("gemini", var_id, s_by_alias))
        print(f"  Loaded Gemini baseline ({len(gemini_data)} fables)")
    elif args.gemini_summaries:
        print(f"  WARNING: Gemini summaries not found at {gemini_path}")

    device = get_device()
    fables, moral_texts, ground_truth = load_base_data()

    n_combos = (len(gemini_entries) + len(gen_models) * len(variants)) * len(CORPUS_TEMPLATES)
    print(f"\n[exp_13 eval]  device={device}")
    print(f"  retrieval models={len(retrieval_cfgs)}")
    print(f"  generator combos={len(gen_models) * len(variants)}  + {len(gemini_entries)} gemini baselines")
    print(f"  corpus types={len(CORPUS_TEMPLATES)}  →  {n_combos} eval runs per retrieval model")

    notify.send(
        f"🔍 exp_13 eval starting\n"
        f"retrieval: {', '.join(r['alias'] for r in retrieval_cfgs)}\n"
        f"gen_models: {', '.join(m['alias'] for m in gen_models)}\n"
        f"variants: {', '.join(v['id'] for v in variants)}\n"
        f"total runs: {n_combos * len(retrieval_cfgs)}"
    )

    all_rows: list[dict] = []

    def _empty_row(gen_label, var_id, corpus_type, ret_alias, ret_id, note):
        return {
            "generator_model": gen_label, "prompt_variant": var_id,
            "corpus_type": corpus_type,
            "retrieval_model": ret_alias, "retrieval_model_id": ret_id,
            **{m: "" for m in ["MRR@10","R@1","R@5","R@10","NDCG@10","Mean_Rank","Median_Rank","n_queries"]},
            "notes": note,
        }

    for ret_cfg in retrieval_cfgs:
        ret_alias      = ret_cfg["alias"]
        ret_id         = ret_cfg["id"]
        query_instr    = ret_cfg.get("query_instruction", "") or None
        doc_prefix     = ret_cfg.get("doc_prefix", "")

        print(f"\n{'='*70}")
        print(f"  Retrieval model: {ret_alias} ({ret_id})")
        try:
            ret_model = load_retrieval_model(ret_cfg, device)
        except Exception as e:
            print(f"  ✗ Failed to load {ret_alias}: {e}")
            notify.send(f"exp_13 ✗ load failed: {ret_alias}\n{str(e)[:200]}")
            continue

        # ── Gemini baselines ──────────────────────────────────────────────────
        for gen_label, var_id, s_by_alias in gemini_entries:
            for corpus_type, template in CORPUS_TEMPLATES.items():
                print(f"  [{gen_label}|{var_id}] × [{corpus_type}]", end="  ")
                doc_texts = build_corpus(fables, s_by_alias, template, doc_prefix)
                cache = CACHE_DIR / "embeddings" / ret_alias / gen_label / var_id / corpus_type
                try:
                    metrics = evaluate(
                        ret_model, moral_texts, doc_texts, ground_truth,
                        cache_dir=cache, force=args.force,
                        query_prompt=query_instr,
                    )
                    row = {"generator_model": gen_label, "prompt_variant": var_id,
                           "corpus_type": corpus_type,
                           "retrieval_model": ret_alias, "retrieval_model_id": ret_id,
                           "notes": "", **metrics_to_row(metrics)}
                    print(f"MRR={metrics['MRR']:.4f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    row = _empty_row(gen_label, var_id, corpus_type, ret_alias, ret_id,
                                     f"error: {str(e)[:100]}")
                all_rows.append(row)

        # ── Gemma 4 combos ────────────────────────────────────────────────────
        for gen_cfg in gen_models:
            gen_alias = gen_cfg["alias"]
            for var_cfg in variants:
                var_id = var_cfg["id"]
                for corpus_type, template in CORPUS_TEMPLATES.items():
                    print(f"  [{gen_alias}|{var_id}] × [{corpus_type}]", end="  ")
                    # Extract per-alias summary for this (model, variant)
                    s_by_alias = {
                        alias: sums.get(gen_alias, {}).get(var_id, "")
                        for alias, sums in summaries_by_fable.items()
                    }
                    doc_texts = build_corpus(fables, s_by_alias, template, doc_prefix)
                    cache = CACHE_DIR / "embeddings" / ret_alias / gen_alias / var_id / corpus_type
                    try:
                        metrics = evaluate(
                            ret_model, moral_texts, doc_texts, ground_truth,
                            cache_dir=cache, force=args.force,
                            query_prompt=query_instr,
                        )
                        row = {"generator_model": gen_alias, "prompt_variant": var_id,
                               "corpus_type": corpus_type,
                               "retrieval_model": ret_alias, "retrieval_model_id": ret_id,
                               "notes": "", **metrics_to_row(metrics)}
                        print(f"MRR={metrics['MRR']:.4f}")
                    except Exception as e:
                        print(f"ERROR: {e}")
                        row = _empty_row(gen_alias, var_id, corpus_type, ret_alias, ret_id,
                                         f"error: {str(e)[:100]}")
                    all_rows.append(row)

        # Write incremental CSV after each retrieval model completes
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = RESULTS_DIR / f"{ts}_eval_{ret_alias}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(r for r in all_rows if r.get("retrieval_model") == ret_alias)
        print(f"\n  {ret_alias} results → {csv_path}")

        del ret_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ── Final combined CSV ────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_all = RESULTS_DIR / f"{ts}_eval_all.csv"
    with open(csv_all, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    # ── Summary table ─────────────────────────────────────────────────────────
    done = sorted(
        [r for r in all_rows if r.get("MRR@10")],
        key=lambda r: float(r["MRR@10"]) if r["MRR@10"] else 0,
        reverse=True,
    )
    print(f"\n{'='*80}")
    print(f"  {'Generator':<20}  {'Variant':<24}  {'Corpus':<14}  {'MRR@10':>7}")
    print(f"{'─'*80}")
    for r in done[:25]:
        print(f"  {r['generator_model']:<20}  {r['prompt_variant']:<24}  "
              f"{r['corpus_type']:<14}  {r['MRR@10']:>7}")
    print(f"\n  Full results → {csv_all}")

    best = done[0] if done else None
    notify.send(
        f"✅ exp_13 eval done\n"
        f"Best: {best['generator_model']} | {best['prompt_variant']} × {best['corpus_type']}\n"
        f"MRR={best['MRR@10']}"
        if best else "exp_13 eval done (no results)"
    )


if __name__ == "__main__":
    main()
