"""
llm_retrieval/run.py — LLM oracle baseline for moral-to-fable retrieval.

Given a moral statement and the full 709-fable corpus in context, asks each
configured LLM to rank the top-k most relevant fables. Results are saved
incrementally to per-run CSVs and a unified summary CSV.

Usage
-----
    # Full run, all models, all variants:
    ./run.sh llm_retrieval/run.py

    # Test mode — 100 morals, one model, one variant:
    ./run.sh llm_retrieval/run.py --test 100 --models GPT-4o-mini --variants minimal

    # Skip already-completed runs:
    ./run.sh llm_retrieval/run.py --skip-existing

    # Force re-run (overwrite existing results):
    ./run.sh llm_retrieval/run.py --force
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel

EXP_DIR     = Path(__file__).parent
ROOT        = EXP_DIR.parent
RESULTS_DIR = EXP_DIR / "results"
CONFIG_PATH = EXP_DIR / "config.yaml"

FABLES_PATH = ROOT / "data/processed/fables_corpus.json"
MORALS_PATH = ROOT / "data/processed/morals_corpus.json"

sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from llm_retrieval.lib.corpus import build_corpus_block
from llm_retrieval.lib.eval import compute_row_metrics, aggregate_metrics
from llm_retrieval.lib.prompt import render_prompt
from llm_retrieval.lib.providers import make_agno_model
from llm_retrieval.lib.results import (
    get_run_path, append_query_row, count_completed_rows,
    load_completed_rows, append_summary_row,
)


class RankedFables(BaseModel):
    ids: list[str]


def load_queries(n: int | None) -> list[dict]:
    morals = json.loads(MORALS_PATH.read_text())
    if n is not None:
        morals = morals[:n]
    return [
        {
            "moral_id":       m["doc_id"],
            "moral_text":     m["text"],
            "relevant_fable": m["fable_id"],
        }
        for m in morals
        if m.get("fable_id")
    ]


async def run_one_query(
    agent,
    query: dict,
    variant_cfg: dict,
    corpus_block: str,
    top_k: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    _, user = render_prompt(query["moral_text"], corpus_block, variant_cfg, top_k)
    async with semaphore:
        t0 = time.monotonic()
        try:
            response = await agent.arun(user)
            content = response.content
            if isinstance(content, RankedFables):
                ranked_ids = content.ids[:top_k]
            elif isinstance(content, dict):
                ranked_ids = content.get("ids", [])[:top_k]
            else:
                import json as _json
                parsed = _json.loads(str(content))
                ranked_ids = parsed.get("ids", [])[:top_k]
        except Exception as exc:
            print(f"  [WARN] {query['moral_id']} failed: {exc}")
            ranked_ids = []
        latency = time.monotonic() - t0
    return compute_row_metrics(
        moral_id=query["moral_id"],
        moral_text=query["moral_text"],
        relevant_fable=query["relevant_fable"],
        ranked_ids=ranked_ids,
        latency_s=latency,
    )


async def run_model_variant(
    model_cfg: dict,
    variant_cfg: dict,
    queries: list[dict],
    corpus_block: str,
    top_k: int,
    run_path: Path,
    run_date: str,
) -> list[dict]:
    from agno.agent import Agent

    already_done = count_completed_rows(run_path)
    queries_to_run = queries[already_done:]

    if not queries_to_run:
        print(f"  ✓ Already complete ({already_done} rows)")
        return load_completed_rows(run_path)

    if already_done:
        print(f"  Resuming from row {already_done}/{len(queries)}")

    model = make_agno_model(model_cfg)
    agent = Agent(
        model=model,
        instructions=variant_cfg["system"],
        output_schema=RankedFables,
        stream=False,
    )
    sem = asyncio.Semaphore(model_cfg.get("concurrency", 5))

    tasks = [
        run_one_query(
            agent=agent,
            query=q,
            variant_cfg=variant_cfg,
            corpus_block=corpus_block,
            top_k=top_k,
            semaphore=sem,
        )
        for q in queries_to_run
    ]

    for coro in asyncio.as_completed(tasks):
        row = await coro
        append_query_row(run_path, row)

    return load_completed_rows(run_path)


def main():
    parser = argparse.ArgumentParser(description="LLM oracle retrieval experiment")
    parser.add_argument("--test", type=int, metavar="N", help="Run on first N morals only")
    parser.add_argument("--models", nargs="+", metavar="ALIAS", help="Model aliases to run (default: all)")
    parser.add_argument("--variants", nargs="+", metavar="LABEL", help="Variant labels to run (default: all)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip (model, variant) pairs with a complete run file")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    top_k = cfg["top_k"]
    run_date = date.today().isoformat()

    models = cfg["models"]
    if args.models:
        models = [m for m in models if m["alias"] in args.models]
        missing = set(args.models) - {m["alias"] for m in models}
        if missing:
            print(f"[WARN] Unknown model aliases: {missing}")

    variants = cfg["prompt_variants"]
    if args.variants:
        variants = [v for v in variants if v["label"] in args.variants]

    queries = load_queries(args.test)
    print(f"Loaded {len(queries)} queries")

    import random
    fables = json.loads(FABLES_PATH.read_text())
    random.seed(42)
    random.shuffle(fables)
    corpus_block = build_corpus_block(fables)
    print(f"Corpus block: {len(corpus_block):,} chars (~{len(corpus_block)//4:,} tokens)")

    mode_label = f"test-{args.test}" if args.test else "full"
    notify.send(
        f"🔬 llm_retrieval starting\n"
        f"mode: {mode_label} ({len(queries)} morals)\n"
        f"models: {[m['alias'] for m in models]}\n"
        f"variants: {[v['label'] for v in variants]}"
    )

    unified_path = RESULTS_DIR / "unified.csv"

    for model_cfg in models:
        for variant_cfg in variants:
            alias    = model_cfg["alias"]
            label    = variant_cfg["label"]
            run_path = get_run_path(RESULTS_DIR, alias, label, run_date)

            if args.skip_existing and count_completed_rows(run_path) >= len(queries):
                print(f"[SKIP] {alias} × {label} (already complete)")
                continue

            if args.force and run_path.exists():
                run_path.unlink()

            print(f"\n▶ {alias} × {label}")
            t_start = time.monotonic()

            rows = asyncio.run(run_model_variant(
                model_cfg=model_cfg,
                variant_cfg=variant_cfg,
                queries=queries,
                corpus_block=corpus_block,
                top_k=top_k,
                run_path=run_path,
                run_date=run_date,
            ))

            elapsed = time.monotonic() - t_start
            agg = aggregate_metrics(rows)

            summary = {
                "run_date":      run_date,
                "model_alias":   alias,
                "model_id":      model_cfg["id"],
                "provider":      model_cfg["provider"],
                "variant_label": label,
                "n_queries":     agg["n_queries"],
                "MRR@10":        round(agg["MRR@10"], 4),
                "R@1":           round(agg["R@1"],    4),
                "R@5":           round(agg["R@5"],    4),
                "R@10":          round(agg["R@10"],   4),
                "NDCG@10":       round(agg["NDCG@10"], 4),
                "Mean_Rank":     round(agg["Mean_Rank"],   1) if agg["Mean_Rank"]   is not None else "",
                "Median_Rank":   round(agg["Median_Rank"], 1) if agg["Median_Rank"] is not None else "",
                "avg_latency_s": round(elapsed / max(len(rows), 1), 2),
            }
            append_summary_row(unified_path, summary)

            print(f"  MRR@10={summary['MRR@10']}  R@10={summary['R@10']}  ({elapsed:.0f}s)")
            notify.send(
                f"✅ {alias} × {label} done\n"
                f"MRR@10={summary['MRR@10']}  R@10={summary['R@10']}\n"
                f"n={agg['n_queries']}  {elapsed:.0f}s"
            )

    notify.send("🏁 llm_retrieval complete — check results/unified.csv")
    print(f"\nDone. Results in {unified_path}")


if __name__ == "__main__":
    main()
