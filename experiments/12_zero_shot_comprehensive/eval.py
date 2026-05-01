"""
exp_12_zero_shot_comprehensive — Zero-shot retrieval across all models × corpus configs
                                 × instruction variants.

Evaluates each model on every corpus configuration and instruction variant.
Results written to a CSV for paper reporting.

Usage
-----
    # All models, all corpus configs, all variants:
    ./run.sh experiments/12_zero_shot_comprehensive/eval.py

    # Specific models:
    ./run.sh experiments/12_zero_shot_comprehensive/eval.py --models Linq-Embed-Mistral BGE-en-ICL

    # Specific corpus configs:
    ./run.sh experiments/12_zero_shot_comprehensive/eval.py --configs raw fable_cot

    # Specific instruction variants:
    ./run.sh experiments/12_zero_shot_comprehensive/eval.py --variants generic

    # Force re-embed (ignore cache):
    ./run.sh experiments/12_zero_shot_comprehensive/eval.py --force

    # Remote GPU:
    ./run.sh experiments/12_zero_shot_comprehensive/eval.py --remote --gpu 2
"""
import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import torch
import yaml

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from finetuning.lib.eval import evaluate

CONFIG_PATH   = EXP_DIR / "config.yaml"
CACHE_DIR     = EXP_DIR / "cache"
RESULTS_DIR   = EXP_DIR / "results"
CSV_MAIN      = RESULTS_DIR / "zero_shot_comprehensive.csv"

SUMMARIES_PATH = (
    ROOT
    / "experiments/07_sota_summarization_oracle"
    / "results/generation_runs/full_709/golden_summaries.json"
)

CSV_FIELDNAMES = [
    "model_alias", "model_hf_id", "model_size",
    "instruction_variant", "query_instruction",
    "corpus_config", "corpus_description", "corpus_template",
    "MRR@10", "R@1", "R@5", "R@10", "NDCG@10",
    "Mean_Rank", "Median_Rank", "n_queries",
    "notes",
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_summaries() -> dict[str, dict[str, str]]:
    data = json.loads(SUMMARIES_PATH.read_text())
    return {entry["original_fable_id"]: entry["summaries"] for entry in data}


def build_corpus(fables: list[dict], summaries: dict, template: str) -> list[str]:
    docs = []
    for fable in fables:
        alias = fable["alias"]
        s = summaries.get(alias, {})
        doc = template.format(
            fable               = fable["text"],
            cot_proverb         = s.get("cot_proverb", ""),
            direct_moral        = s.get("direct_moral", ""),
            conceptual_abstract = s.get("conceptual_abstract", ""),
            proverb             = s.get("proverb", ""),
        )
        docs.append(doc)
    return docs


def load_base_data():
    from lib.data import load_fables, load_morals, load_qrels_moral_to_fable
    fables        = load_fables()
    morals        = load_morals()
    qrels         = load_qrels_moral_to_fable()
    moral_indices = sorted(qrels.keys())
    moral_texts   = [morals[i]["text"] for i in moral_indices]
    ground_truth  = {i: qrels[idx] for i, idx in enumerate(moral_indices)}
    return fables, moral_texts, ground_truth


# ── Model loading ──────────────────────────────────────────────────────────────

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
        dtype = model_cfg["torch_dtype"]
        model_kwargs["torch_dtype"] = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    use_device_map = model_cfg.get("device_map") == "auto"
    if use_device_map:
        model_kwargs["device_map"] = "auto"

    st_kwargs: dict = {"trust_remote_code": model_cfg.get("trust_remote_code", False)}
    if model_kwargs:
        st_kwargs["model_kwargs"] = model_kwargs

    if use_device_map:
        return SentenceTransformer(model_cfg["id"], **st_kwargs)
    return SentenceTransformer(model_cfg["id"], device=device, **st_kwargs)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_existing_results() -> dict[tuple, dict]:
    """Load completed rows keyed by (model_alias, corpus_config, instruction_variant).

    Rows written before instruction_variant was added default to 'default'.
    """
    if not CSV_MAIN.exists():
        return {}
    existing = {}
    with open(CSV_MAIN) as f:
        for row in csv.DictReader(f):
            if row.get("MRR@10"):
                variant = row.get("instruction_variant") or "default"
                existing[(row["model_alias"], row["corpus_config"], variant)] = row
    return existing


def write_csv(all_rows: list[dict]) -> None:
    with open(CSV_MAIN, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)


def metrics_to_row(metrics: dict) -> dict:
    return {
        "MRR@10":      round(metrics.get("MRR",        0), 6),
        "R@1":         round(metrics.get("Recall@1",   0), 6),
        "R@5":         round(metrics.get("Recall@5",   0), 6),
        "R@10":        round(metrics.get("Recall@10",  0), 6),
        "NDCG@10":     round(metrics.get("NDCG@10",    0), 6),
        "Mean_Rank":   round(metrics.get("Mean Rank",  0), 2),
        "Median_Rank": round(metrics.get("Median Rank",0), 1),
        "n_queries":   metrics.get("n_queries", 0),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="exp_12 zero-shot comprehensive eval")
    parser.add_argument("--models",   nargs="+", help="Model aliases to run (default: all)")
    parser.add_argument("--configs",  nargs="+", help="Corpus config IDs to run (default: all)")
    parser.add_argument("--variants", nargs="+", help="Instruction variant labels to run (default: all)")
    parser.add_argument("--force",    action="store_true", help="Re-embed even if cached")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_models   = config["models"]
    all_configs  = config["corpus_configs"]
    # "default" variant uses each model's own query_instruction from config
    all_variants = config.get("instruction_variants", [{"label": "default"}])

    if args.models:
        sel = {m.lower() for m in args.models}
        all_models = [m for m in all_models if m["alias"].lower() in sel]
    if args.configs:
        sel = {c.lower() for c in args.configs}
        all_configs = [c for c in all_configs if c["id"].lower() in sel]
    if args.variants:
        sel = {v.lower() for v in args.variants}
        all_variants = [v for v in all_variants if v["label"].lower() in sel]

    device = get_device()
    n_total = len(all_models) * len(all_configs) * len(all_variants)
    print(f"\n[exp_12_zero_shot_comprehensive]")
    print(f"  device={device}  models={len(all_models)}  configs={len(all_configs)}  variants={len(all_variants)}")
    print(f"  total runs: {n_total}")

    notify.send(
        f"🔍 exp_12 starting\n"
        f"device: {device}\n"
        f"models: {', '.join(m['alias'] for m in all_models)}\n"
        f"configs: {', '.join(c['id'] for c in all_configs)}\n"
        f"variants: {', '.join(v['label'] for v in all_variants)}"
    )

    fables, moral_texts, ground_truth = load_base_data()
    summaries = load_summaries()
    ground_truth_full = {i: ground_truth[i] for i in range(len(moral_texts))}

    existing = load_existing_results()
    print(f"  Already done: {len(existing)} rows (will skip)")

    all_rows: list[dict] = []
    pending:  list[tuple] = []

    for model_cfg in all_models:
        for corpus_cfg in all_configs:
            for variant in all_variants:
                variant_label = variant["label"]
                # "default" uses the model's own instruction; others override it
                effective_instruction = (
                    model_cfg.get("query_instruction", "")
                    if variant_label == "default"
                    else variant.get("query_instruction", "")
                )
                key = (model_cfg["alias"], corpus_cfg["id"], variant_label)
                base_row = {
                    "model_alias":        model_cfg["alias"],
                    "model_hf_id":        model_cfg["id"],
                    "model_size":         model_cfg.get("size", ""),
                    "instruction_variant": variant_label,
                    "query_instruction":   effective_instruction,
                    "corpus_config":       corpus_cfg["id"],
                    "corpus_description":  corpus_cfg["description"],
                    "corpus_template":     corpus_cfg["template"].replace("\n", "\\n"),
                    **{m: "" for m in ["MRR@10","R@1","R@5","R@10","NDCG@10","Mean_Rank","Median_Rank","n_queries"]},
                    "notes": "",
                }
                if key in existing and not args.force:
                    base_row.update(existing[key])
                    all_rows.append(base_row)
                else:
                    all_rows.append(base_row)
                    pending.append((model_cfg, corpus_cfg, variant_label, effective_instruction))

    print(f"  Pending: {len(pending)} runs\n")

    row_index = {
        (r["model_alias"], r["corpus_config"], r["instruction_variant"]): i
        for i, r in enumerate(all_rows)
    }

    from collections import defaultdict
    # Group pending by (model_alias, variant_label) so each model is loaded once per variant
    pending_by_model_variant: dict[tuple, list] = defaultdict(list)
    for model_cfg, corpus_cfg, variant_label, effective_instruction in pending:
        pending_by_model_variant[(model_cfg["alias"], variant_label)].append(
            (model_cfg, corpus_cfg, variant_label, effective_instruction)
        )

    for (model_alias, variant_label), runs in pending_by_model_variant.items():
        model_cfg = runs[0][0]
        print(f"\n{'─'*60}")
        print(f"  Loading {model_alias} [{variant_label}] ({model_cfg['id']}) …")

        try:
            model = load_model(model_cfg, device)
        except Exception as e:
            print(f"  ✗ Failed to load {model_alias}: {e}")
            notify.send(f"exp_12 ✗ load failed: {model_alias}\n{str(e)[:200]}")
            for _, corpus_cfg, vl, _ in runs:
                idx = row_index[(model_alias, corpus_cfg["id"], vl)]
                all_rows[idx]["notes"] = f"load_error: {str(e)[:100]}"
            write_csv(all_rows)
            continue

        doc_prefix = model_cfg.get("doc_prefix", "")

        for model_cfg_inner, corpus_cfg, vl, effective_instruction in runs:
            cfg_id   = corpus_cfg["id"]
            template = corpus_cfg["template"]
            print(f"\n  [{model_alias}|{vl}] × [{cfg_id}]")

            doc_texts = build_corpus(fables, summaries, template)
            if doc_prefix:
                doc_texts = [f"{doc_prefix}{d}" for d in doc_texts]

            # Query embeddings are variant-specific; doc embeddings are shared.
            # For non-default variants, point doc_cache_dir at the default variant's
            # cache so we reuse already-computed doc embeddings.
            base_cache    = CACHE_DIR / "embeddings" / model_alias / cfg_id
            if vl == "default":
                emb_cache     = base_cache
                doc_cache_dir = None          # evaluate() stores both in emb_cache
            else:
                emb_cache     = base_cache / vl
                doc_cache_dir = base_cache    # reuse default's doc_embs.npy

            try:
                metrics = evaluate(
                    model,
                    moral_texts,
                    doc_texts,
                    ground_truth_full,
                    cache_dir=emb_cache,
                    doc_cache_dir=doc_cache_dir,
                    force=args.force,
                    query_prompt=effective_instruction or None,
                )
                mrr = metrics["MRR"]
                r1  = metrics.get("Recall@1", 0)
                r10 = metrics.get("Recall@10", 0)
                print(f"  → MRR={mrr:.4f}  R@1={r1:.3f}  R@10={r10:.3f}")
                notify.send(f"exp_12 ✓ {model_alias}|{vl} × {cfg_id}\nMRR={mrr:.4f}")

                idx = row_index[(model_alias, cfg_id, vl)]
                all_rows[idx].update(metrics_to_row(metrics))

            except Exception as e:
                print(f"  ✗ {model_alias}|{vl} × {cfg_id} failed: {e}")
                notify.send(f"exp_12 ✗ {model_alias}|{vl} × {cfg_id}\n{str(e)[:200]}")
                idx = row_index[(model_alias, cfg_id, vl)]
                all_rows[idx]["notes"] = f"error: {str(e)[:100]}"

            write_csv(all_rows)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ── Summary table ──────────────────────────────────────────────────────────
    done_rows = [r for r in all_rows if r.get("MRR@10")]
    done_rows.sort(key=lambda r: float(r["MRR@10"]) if r["MRR@10"] else 0, reverse=True)

    print(f"\n{'='*75}")
    print(f"  {'Model':<28}  {'Variant':<12}  {'Config':<18}  {'MRR@10':>7}")
    print(f"{'─'*75}")
    for r in done_rows[:25]:
        print(f"  {r['model_alias']:<28}  {r['instruction_variant']:<12}  "
              f"{r['corpus_config']:<18}  {r['MRR@10']:>7}")
    print(f"{'='*75}")
    print(f"\n  Results → {CSV_MAIN}")

    best = done_rows[0] if done_rows else None
    notify.send(
        f"✅ exp_12 done\n"
        f"Best: {best['model_alias']} [{best['instruction_variant']}] × {best['corpus_config']}\n"
        f"MRR={best['MRR@10']}"
        if best else "exp_12 done (no results)"
    )


if __name__ == "__main__":
    main()
