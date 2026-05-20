"""Second-stage rerank saved STORAL retrieval rankings.

Run a local no-download smoke test:
    ./run.sh experiments/21_storal_reranking/rerank.py \
      --rerankers mock_lexical --candidate-k 20 --limit-runs 1 --limit-queries 5
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from lib.retrieval_utils import compute_multilabel_metrics_from_matrix

CONFIG_PATH = EXP_DIR / "config.yaml"
DEFAULT_PLAN = EXP_DIR / "run_plans" / "top_recall100_storal.csv"
RESULTS_DIR = EXP_DIR / "results"
RANKINGS_DIR = RESULTS_DIR / "rankings"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_json(path: Path):
    return json.loads(path.read_text())


def load_clustered(config: dict):
    paths = config["paths"]
    morals = load_json(resolve_path(paths["morals"]))
    fables = load_json(resolve_path(paths["fables"]))
    qrels = load_json(resolve_path(paths["qrels"]))
    return morals, fables, qrels


def load_qrels(qrels: list[dict], query_ids: list[str], doc_ids: list[str]) -> dict[int, set[int]]:
    query_to_idx = {qid: i for i, qid in enumerate(query_ids)}
    doc_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    relevant: dict[int, set[int]] = defaultdict(set)
    for row in qrels:
        if int(row.get("relevance", 1)) <= 0:
            continue
        relevant[query_to_idx[row["query_id"]]].add(doc_to_idx[row["doc_id"]])
    return dict(relevant)


def load_summaries(config: dict, generator: str | None) -> dict[str, dict[str, str]]:
    if not generator or generator == "none":
        return {}
    path = resolve_path(config["paths"]["summary_root"]) / generator / "golden_summaries.json"
    records = load_json(path)
    return {rec["original_fable_id"]: rec["summaries"] for rec in records}


DOC_CONFIGS = {
    "raw": ("{fable}", None),
    "direct_moral_only": ("{direct_moral}", "direct_moral"),
    "conceptual_abstract_only": ("{conceptual_abstract}", "conceptual_abstract"),
    "cot_proverb_only": ("{cot_proverb}", "cot_proverb"),
    "proverb_only": ("{proverb}", "proverb"),
    "fable_direct_moral": ("{fable}\n\nMoral summary: {direct_moral}", "direct_moral"),
    "direct_moral_fable": ("Moral summary: {direct_moral}\n\n{fable}", "direct_moral"),
    "fable_conceptual_abstract": ("{fable}\n\nConceptual summary: {conceptual_abstract}", "conceptual_abstract"),
    "conceptual_abstract_fable": ("Conceptual summary: {conceptual_abstract}\n\n{fable}", "conceptual_abstract"),
    "fable_cot_proverb": ("{fable}\n\nMoral summary: {cot_proverb}", "cot_proverb"),
    "cot_proverb_fable": ("Moral summary: {cot_proverb}\n\n{fable}", "cot_proverb"),
    "fable_proverb": ("{fable}\n\nProverb: {proverb}", "proverb"),
    "proverb_fable": ("Proverb: {proverb}\n\n{fable}", "proverb"),
}


def build_doc_texts(config: dict, fables: list[dict], doc_config_name: str, generator: str | None) -> list[str]:
    template, summary_variant = DOC_CONFIGS[doc_config_name]
    summaries_by_alias = load_summaries(config, generator) if summary_variant else {}
    texts = []
    for fable in fables:
        values = {
            "fable": fable["text"],
            "direct_moral": "",
            "conceptual_abstract": "",
            "proverb": "",
            "cot_proverb": "",
        }
        if summary_variant:
            summaries = summaries_by_alias.get(fable["alias"])
            if summaries is None:
                raise KeyError(f"Missing {generator} summaries for {fable['doc_id']} / {fable['alias']}")
            values.update({k: summaries.get(k, "") for k in values if k != "fable"})
        texts.append(template.format(**values).strip())
    return texts


def tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-zA-Z']+", text.lower()) if len(tok) > 2}


class MockLexicalReranker:
    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 64) -> list[float]:
        scores = []
        for query, doc in pairs:
            q = tokenize(query)
            d = tokenize(doc)
            overlap = len(q & d)
            denom = max(1, len(q))
            scores.append(overlap / denom)
        return scores


class QwenCausalLMReranker:
    """Official Qwen3 reranker scoring path: probability of "yes" vs "no"."""

    def __init__(self, reranker_cfg: dict, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = reranker_cfg["model_id"]
        self.instruction = reranker_cfg.get(
            "prompt",
            "Given a web search query, retrieve relevant passages that answer the query",
        )
        self.max_length = int(reranker_cfg.get("max_length", 8192))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = dict(reranker_cfg.get("model_kwargs") or {})
        dtype = model_kwargs.pop("torch_dtype", reranker_cfg.get("torch_dtype", "auto"))
        if dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif dtype == "auto":
            model_kwargs["torch_dtype"] = "auto"
        elif dtype:
            model_kwargs["torch_dtype"] = dtype
        if reranker_cfg.get("trust_remote_code"):
            model_kwargs["trust_remote_code"] = True

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs).eval()
        self.model.to(self.device)

        self.token_false_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        self.token_true_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the "
            'Instruct provided. Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def _format_instruction(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs: list[tuple[str, str]]):
        texts = [self._format_instruction(query, doc) for query, doc in pairs]
        max_pair_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        inputs = self.tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_pair_length,
        )
        for idx, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][idx] = self.prefix_tokens + input_ids + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 4) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            inputs = self._process_inputs(batch)
            with self.torch.no_grad():
                batch_logits = self.model(**inputs).logits[:, -1, :]
                true_vector = batch_logits[:, self.token_true_id]
                false_vector = batch_logits[:, self.token_false_id]
                yes_no_logits = self.torch.stack([false_vector, true_vector], dim=1)
                batch_scores = self.torch.nn.functional.log_softmax(yes_no_logits, dim=1)[:, 1].exp()
            scores.extend(float(score) for score in batch_scores.detach().cpu().tolist())
        return scores


def build_reranker(alias: str, cfg: dict, device: str | None = None):
    reranker_cfg = cfg["rerankers"][alias]
    if reranker_cfg["type"] == "deterministic_dry_run":
        return MockLexicalReranker(), int(reranker_cfg.get("batch_size", 64))
    if reranker_cfg["type"] == "qwen_causal_lm":
        return QwenCausalLMReranker(reranker_cfg, device=device), int(reranker_cfg.get("batch_size", 4))
    if reranker_cfg["type"] != "cross_encoder":
        raise ValueError(f"Unsupported reranker type: {reranker_cfg['type']}")
    from sentence_transformers import CrossEncoder

    kwargs = {}
    supported_kwargs = set(inspect.signature(CrossEncoder).parameters)
    if reranker_cfg.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    if reranker_cfg.get("prompt") and reranker_cfg.get("prompt_name"):
        if {"prompts", "default_prompt_name"}.issubset(supported_kwargs):
            kwargs["prompts"] = {reranker_cfg["prompt_name"]: reranker_cfg["prompt"]}
            kwargs["default_prompt_name"] = reranker_cfg["prompt_name"]
        else:
            print(f"[warn] CrossEncoder does not support prompts; loading {alias} without custom prompt")
    if device:
        kwargs["device"] = device
    model = CrossEncoder(reranker_cfg["model_id"], **kwargs)
    return model, int(reranker_cfg.get("batch_size", 16))


def metric_row(metrics: dict) -> dict:
    return {
        "MAP@10": metrics.get("MAP", ""),
        "MRR@10": metrics.get("MRR", ""),
        "NDCG@10": metrics.get("NDCG@10", ""),
        "Recall@5": metrics.get("Recall@5", ""),
        "Recall@10": metrics.get("Recall@10", ""),
        "Recall@15": metrics.get("Recall@15", ""),
        "Recall@50": metrics.get("Recall@50", ""),
        "Recall@100": metrics.get("Recall@100", ""),
        "Recall@200": metrics.get("Recall@200", ""),
        "Recall@300": metrics.get("Recall@300", ""),
        "Hit@1": metrics.get("Hit@1", ""),
        "Hit@5": metrics.get("Hit@5", ""),
        "Hit@10": metrics.get("Hit@10", ""),
        "Hit@100": metrics.get("Hit@100", ""),
        "Mean_Rank": metrics.get("Mean Rank", ""),
        "Median_Rank": metrics.get("Median Rank", ""),
        "n_queries": metrics.get("n_queries", ""),
    }


def rerank_one(
    reranker,
    batch_size: int,
    plan_row: dict,
    config: dict,
    morals: list[dict],
    fables: list[dict],
    relevant: dict[int, set[int]],
    candidate_k: int,
    limit_queries: int | None,
    reranker_alias: str,
    progress_every: int,
) -> dict:
    query_ids = [m["doc_id"] for m in morals]
    doc_ids = [f["doc_id"] for f in fables]
    query_id_to_idx = {qid: i for i, qid in enumerate(query_ids)}
    doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    ranking_path = resolve_path(plan_row["local_rankings_path"])
    if not ranking_path.exists() and plan_row.get("rankings_path"):
        ranking_path = resolve_path(plan_row["rankings_path"])
    if not ranking_path.exists():
        raise FileNotFoundError(f"Missing first-stage ranking file: {ranking_path}")
    first_stage_rows = load_json(ranking_path)
    if limit_queries:
        first_stage_rows = first_stage_rows[:limit_queries]

    doc_config = plan_row["eval_doc_config"]
    generator = plan_row.get("summary_generator") or None
    if generator == "none":
        generator = None
    doc_texts = build_doc_texts(config, fables, doc_config, generator)

    score_matrix = np.full((len(first_stage_rows), len(fables)), -1_000_000.0, dtype=np.float32)
    reranked_rows = []
    local_relevant = {}

    for local_q_idx, rank_row in enumerate(first_stage_rows):
        if progress_every and (local_q_idx == 0 or (local_q_idx + 1) % progress_every == 0):
            print(
                f"    [progress] {reranker_alias} {plan_row['model_alias']} "
                f"{plan_row['size']} {plan_row['eval_doc_config']}/"
                f"{plan_row.get('summary_generator') or 'none'} "
                f"query {local_q_idx + 1}/{len(first_stage_rows)}"
            )
        q_global_idx = query_id_to_idx[rank_row["query_id"]]
        local_relevant[local_q_idx] = relevant[q_global_idx]
        query_text = morals[q_global_idx]["text"]
        ranked_doc_ids = rank_row["ranked_fable_ids"]
        candidates = ranked_doc_ids[: min(candidate_k, len(ranked_doc_ids))]
        candidate_indices = [doc_id_to_idx[doc_id] for doc_id in candidates]
        pairs = [(query_text, doc_texts[idx]) for idx in candidate_indices]
        rerank_scores = list(reranker.predict(pairs, batch_size=batch_size))

        order = sorted(range(len(candidate_indices)), key=lambda i: rerank_scores[i], reverse=True)
        reranked_candidate_indices = [candidate_indices[i] for i in order]
        reranked_scores = [float(rerank_scores[i]) for i in order]
        candidate_set = set(candidate_indices)
        rest_indices = [doc_id_to_idx[doc_id] for doc_id in ranked_doc_ids if doc_id_to_idx[doc_id] not in candidate_set]
        final_indices = reranked_candidate_indices + rest_indices

        for rank_idx, doc_idx in enumerate(final_indices):
            score_matrix[local_q_idx, doc_idx] = float(len(final_indices) - rank_idx)

        reranked_rows.append(
            {
                "query_id": rank_row["query_id"],
                "candidate_k": candidate_k,
                "ranked_fable_ids": [doc_ids[idx] for idx in final_indices],
                "reranker_scores_top_candidates": [round(s, 6) for s in reranked_scores],
            }
        )

    ks = tuple(int(k) for k in config["ks"])
    metrics = compute_multilabel_metrics_from_matrix(score_matrix, local_relevant, ks=ks)
    tag = "__".join(
        [
            reranker_alias,
            plan_row["model_alias"],
            plan_row["size"],
            plan_row["eval_doc_config"],
            plan_row.get("summary_generator") or "none",
            f"k{candidate_k}",
        ]
    )
    rankings_path = RANKINGS_DIR / f"{tag}.json"
    rankings_path.parent.mkdir(parents=True, exist_ok=True)
    rankings_path.write_text(json.dumps(reranked_rows, indent=2))

    return {
        "tag": tag,
        "rankings_path": str(rankings_path),
        "metrics": metrics,
        "n_queries": len(first_stage_rows),
    }


def parse_aliases(value: str | None, options: Iterable[str]) -> list[str]:
    option_list = list(options)
    if not value or value == "all":
        return [alias for alias in option_list if alias != "mock_lexical"]
    aliases = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [alias for alias in aliases if alias not in option_list]
    if unknown:
        raise KeyError(f"Unknown rerankers {unknown!r}. Options: {option_list}")
    return aliases


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank saved FT12 STORAL rankings")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--rerankers", help="Comma-separated reranker aliases, or all")
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--device", help="Optional CrossEncoder device, e.g. cpu, cuda, mps")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    candidate_k = args.candidate_k or int(config["default_candidate_k"])
    reranker_aliases = parse_aliases(args.rerankers, config["rerankers"].keys())
    plan_path = resolve_path(args.plan)

    with plan_path.open(newline="", encoding="utf-8") as f:
        plan_rows = list(csv.DictReader(f))
    if args.limit_runs:
        plan_rows = plan_rows[: args.limit_runs]

    notify.send(
        f"storal reranking starting\n"
        f"plan: {plan_path.name}\n"
        f"rerankers: {reranker_aliases}\n"
        f"candidate_k: {candidate_k}\n"
        f"runs: {len(plan_rows)}"
    )

    morals, fables, qrels = load_clustered(config)
    relevant = load_qrels(qrels, [m["doc_id"] for m in morals], [f["doc_id"] for f in fables])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = RESULTS_DIR / f"{timestamp}_rerank_metrics.csv"
    json_path = RESULTS_DIR / f"{timestamp}_rerank_results.json"

    output_rows = []
    json_results = []
    for reranker_alias in reranker_aliases:
        try:
            reranker, batch_size = build_reranker(reranker_alias, config, args.device)
        except Exception as exc:  # noqa: BLE001 - keep overnight queue moving.
            print(f"[error] failed to load reranker {reranker_alias}: {type(exc).__name__}: {exc}")
            for row in plan_rows:
                output_rows.append(
                    {
                        "timestamp": timestamp,
                        "reranker_alias": reranker_alias,
                        "reranker_model_id": config["rerankers"][reranker_alias]["model_id"],
                        "candidate_k": candidate_k,
                        "first_stage_rank": row.get("rank", ""),
                        "first_stage_model": row.get("model_alias", ""),
                        "first_stage_size": row.get("size", ""),
                        "first_stage_doc_config": row.get("eval_doc_config", ""),
                        "first_stage_summary_generator": row.get("summary_generator", ""),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue
        for row in plan_rows:
            try:
                result = rerank_one(
                    reranker,
                    batch_size,
                    row,
                    config,
                    morals,
                    fables,
                    relevant,
                    candidate_k,
                    args.limit_queries,
                    reranker_alias,
                    args.progress_every,
                )
                metrics = result["metrics"]
                out_row = {
                    "timestamp": timestamp,
                    "reranker_alias": reranker_alias,
                    "reranker_model_id": config["rerankers"][reranker_alias]["model_id"],
                    "candidate_k": candidate_k,
                    "first_stage_rank": row["rank"],
                    "first_stage_model": row["model_alias"],
                    "first_stage_size": row["size"],
                    "first_stage_doc_config": row["eval_doc_config"],
                    "first_stage_summary_generator": row.get("summary_generator") or "none",
                    "first_stage_MAP@10": row.get("MAP@10", ""),
                    "first_stage_MRR@10": row.get("MRR@10", ""),
                    "first_stage_NDCG@10": row.get("NDCG@10", ""),
                    "first_stage_Recall@100": row.get("Recall@100", ""),
                    "reranked_rankings_path": result["rankings_path"],
                    **metric_row(metrics),
                    "error": "",
                }
                output_rows.append(out_row)
                json_results.append({"plan_row": row, "reranker": reranker_alias, **result})
                print(
                    f"[rerank] {result['tag']} "
                    f"MRR@10={metrics['MRR']:.4f} MAP@10={metrics['MAP']:.4f} "
                    f"NDCG@10={metrics['NDCG@10']:.4f} Hit@10={metrics['Hit@10']:.4f} "
                    f"Recall@100={metrics['Recall@100']:.4f}"
                )
            except Exception as exc:  # noqa: BLE001 - queued reranking should continue.
                output_rows.append(
                    {
                        "timestamp": timestamp,
                        "reranker_alias": reranker_alias,
                        "reranker_model_id": config["rerankers"][reranker_alias]["model_id"],
                        "candidate_k": candidate_k,
                        "first_stage_rank": row.get("rank", ""),
                        "first_stage_model": row.get("model_alias", ""),
                        "first_stage_size": row.get("size", ""),
                        "first_stage_doc_config": row.get("eval_doc_config", ""),
                        "first_stage_summary_generator": row.get("summary_generator", ""),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"[error] {reranker_alias} {row.get('model_alias')} {row.get('size')}: {type(exc).__name__}: {exc}")

    if output_rows:
        fields = list(output_rows[0].keys())
        for row in output_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)

    json_path.write_text(
        json.dumps(
            {
                "experiment": "21_storal_reranking",
                "plan": str(plan_path),
                "candidate_k": candidate_k,
                "rerankers": reranker_aliases,
                "limit_runs": args.limit_runs,
                "limit_queries": args.limit_queries,
                "results": json_results,
            },
            indent=2,
        )
    )

    notify.send(
        f"storal reranking done\n"
        f"rows: {len(output_rows)}\n"
        f"csv: {csv_path.name}"
    )
    print(f"Results -> {csv_path}")
    print(f"Details -> {json_path}")


if __name__ == "__main__":
    main()
