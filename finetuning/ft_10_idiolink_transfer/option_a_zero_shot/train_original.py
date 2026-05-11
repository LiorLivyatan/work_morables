"""
ft_10 Option A — Original IdioLink Loss variant.

Same setup as train.py (IdioLink → MORABLES transfer) but uses the full
InfoNCE loss from the original IdioLink paper:
  - ALL 5 same-PIE hard negatives per sample (train.py uses only neg[0])
  - Same-tower (query×query) negatives prevent embedding collapse
  - Temperature τ = 0.05
  - Early stopping patience = 2 (matching original paper)
  - No task prefix on queries (paper's best config uses no instructions)

Results written to option_a_original_results.csv for side-by-side comparison
with option_a_results.csv produced by train.py.

Usage
-----
    # Single model comparison run (BGE-M3 recommended for first comparison):
    ./run.sh finetuning/ft_10_idiolink_transfer/option_a_zero_shot/train_original.py \\
        --model BGE-M3 --remote --gpu 2

    # Force re-train:
    ./run.sh finetuning/ft_10_idiolink_transfer/option_a_zero_shot/train_original.py \\
        --model BGE-M3 --force --remote --gpu 2

    # All models (after comparison confirms this approach wins):
    ./run.sh finetuning/ft_10_idiolink_transfer/option_a_zero_shot/train_original.py --remote --gpu 2
"""
import argparse
import csv
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
import yaml

EXP_DIR  = Path(__file__).parent
ROOT     = EXP_DIR.parent.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from finetuning.lib.eval import evaluate

CONFIG_PATH          = EXP_DIR / "config.yaml"
RESULTS_DIR          = EXP_DIR / "results"
ORIGINAL_RESULTS_CSV = RESULTS_DIR / "option_a_original_results.csv"
MODELS_ORIGINAL_DIR  = RESULTS_DIR / "models_original"

GEMINI_SUMMARIES_PATH = (
    ROOT / "experiments/07_sota_summarization_oracle"
    / "results/generation_runs/full_709/golden_summaries.json"
)

CSV_FIELDNAMES = [
    "model_alias", "model_hf_id",
    "corpus_config", "summary_generator", "summary_variant", "corpus_type",
    "zero_shot_MRR@10",  "original_ft_MRR@10",  "delta_MRR",
    "zero_shot_R@1",     "original_ft_R@1",
    "zero_shot_R@5",     "original_ft_R@5",
    "zero_shot_NDCG@10", "original_ft_NDCG@10",
]

CORPUS_CONFIGS = [
    ("raw",                                            "none",       "—",                     "raw"),
    ("gemini/cot_proverb/fable+summary",               "gemini",     "cot_proverb",           "fable_summary"),
    ("gemini/direct_moral/fable+summary",              "gemini",     "direct_moral",          "fable_summary"),
    ("gemini/conceptual_abstract/fable+summary",       "gemini",     "conceptual_abstract",   "fable_summary"),
    ("gemini/cot_proverb/summary_only",                "gemini",     "cot_proverb",           "summary_only"),
    ("gemini/cot_proverb/summary+fable",               "gemini",     "cot_proverb",           "summary+fable"),
    ("gemma4-E2B/direct_moral/fable+summary",          "gemma4-E2B", "direct_moral",          "fable_summary"),
    ("gemma4-E2B/direct_moral/summary_only",           "gemma4-E2B", "direct_moral",          "summary_only"),
    ("gemma4-E4B/thinking_direct_moral/fable+summary", "gemma4-E4B", "thinking_direct_moral", "fable_summary"),
    ("gemma4-E4B/thinking_direct_moral/summary_only",  "gemma4-E4B", "thinking_direct_moral", "summary_only"),
    ("gemma4-31B/conceptual_abstract/fable+summary",   "gemma4-31B", "conceptual_abstract",   "fable_summary"),
    ("gemma4-31B/conceptual_abstract/summary_only",    "gemma4-31B", "conceptual_abstract",   "summary_only"),
]

EARLY_STOPPING_PATIENCE = 2  # matches original IdioLink paper


# ── Original IdioLink InfoNCE loss ─────────────────────────────────────────────

class IdioLinkInfoNCELoss(torch.nn.Module):
    """
    Faithful reproduction of the IdioLink paper's three-component InfoNCE loss.

    Given a batch of (anchor, positive, neg0..neg4) embeddings:
      1. In-batch negatives: anchor_i × all positives in batch → (B, B)
      2. Explicit hard negatives: anchor_i × its 5 same-PIE negatives → (+5 cols)
      3. Same-tower negatives: anchor_i × all other anchors → (+B cols, diagonal=-inf)

    Final logit matrix shape: (B, 2B+5). Cross-entropy target = arange(B)
    (the true positive for anchor_i is at column i in the in-batch block).
    Temperature τ=0.05 sharpens the distribution, forcing the model to be precise.
    """

    def __init__(self, model, temperature: float = 0.05):
        super().__init__()
        self.model       = model
        self.temperature = temperature
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, sentence_features, labels=None):
        # sentence_features: list of 7 feature dicts
        # Order: [anchor, positive, neg0, neg1, neg2, neg3, neg4]
        embeddings = [
            F.normalize(self.model(sf)["sentence_embedding"], p=2, dim=1)
            for sf in sentence_features
        ]
        anchors   = embeddings[0]   # (B, D)
        positives = embeddings[1]   # (B, D)
        negs      = embeddings[2:]  # list of 5 × (B, D)

        B   = anchors.shape[0]
        tau = self.temperature

        # 1. In-batch: each anchor vs all positives in the batch
        sim = torch.mm(anchors, positives.T) / tau   # (B, B)

        # 2. Explicit hard negatives: each anchor vs its own 5 negatives
        neg_stack = torch.stack(negs, dim=1)          # (B, 5, D)
        hard_sims = torch.bmm(
            anchors.unsqueeze(1), neg_stack.transpose(1, 2)
        ).squeeze(1) / tau                            # (B, 5)
        sim = torch.cat([sim, hard_sims], dim=1)      # (B, B+5)

        # 3. Same-tower: each anchor vs all other anchors (prevents collapse)
        qq  = torch.mm(anchors, anchors.T) / tau      # (B, B)
        eye = torch.eye(B, dtype=torch.bool, device=anchors.device)
        qq  = qq.masked_fill(eye, float("-inf"))      # self-similarity excluded
        sim = torch.cat([sim, qq], dim=1)             # (B, 2B+5)

        targets = torch.arange(B, device=anchors.device)
        return self.cross_entropy(sim, targets)

    def get_config_dict(self):
        return {"temperature": self.temperature}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_idiolink_train_all_neg(idiolink_dir: Path) -> dict:
    """Returns 7-column dataset dict with all 5 same-PIE hard negatives per sample."""
    anchors, positives = [], []
    all_negs = [[] for _ in range(5)]

    with open(idiolink_dir / "train_triplets.jsonl") as f:
        for line in f:
            t = json.loads(line)
            anchors.append(t["query"])
            positives.append(t["positive"])
            negs = t["negatives"]
            for i in range(5):
                all_negs[i].append(negs[i] if i < len(negs) else negs[-1])

    return {
        "anchor": anchors, "positive": positives,
        "neg0": all_negs[0], "neg1": all_negs[1],
        "neg2": all_negs[2], "neg3": all_negs[3],
        "neg4": all_negs[4],
    }


def build_idiolink_val_evaluator(idiolink_dir: Path, name: str = "idiolink_val"):
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    with open(idiolink_dir / "val_indexes.json") as f:
        index_entries = json.load(f)
    corpus     = {e["id"]: e["sentence"] for e in index_entries}
    text_to_id = {e["sentence"]: e["id"] for e in index_entries}

    queries, relevant_docs = {}, {}
    with open(idiolink_dir / "val_triplets.jsonl") as f:
        for i, line in enumerate(f):
            t      = json.loads(line)
            qid    = f"q_{i}"
            doc_id = text_to_id.get(t["positive"])
            if doc_id is None:
                continue
            queries[qid]       = t["query"]
            relevant_docs[qid] = {doc_id}

    return InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs,
        ndcg_at_k=[10], mrr_at_k=[10], accuracy_at_k=[1, 5, 10],
        name=name,
    )


# ── Model building ─────────────────────────────────────────────────────────────

def build_model(model_cfg: dict):
    from sentence_transformers import SentenceTransformer
    hf_id        = model_cfg["hf_id"]
    model_kwargs = dict(model_cfg.get("model_kwargs") or {})

    if "torch_dtype" in model_kwargs:
        dtype = model_kwargs.pop("torch_dtype")
        model_kwargs["torch_dtype"] = getattr(torch, dtype, dtype) if isinstance(dtype, str) else dtype

    trust_remote = model_kwargs.pop("trust_remote_code", False)
    st_kwargs    = {"model_kwargs": model_kwargs} if model_kwargs else {}
    if trust_remote:
        st_kwargs["trust_remote_code"] = True

    model = SentenceTransformer(hf_id, **st_kwargs)
    model.max_seq_length = model_cfg.get("max_seq_length", 512)

    if model_cfg.get("use_lora"):
        from peft import LoraConfig, TaskType, get_peft_model
        lora_cfg = model_cfg["lora"]
        model[0].auto_model = get_peft_model(
            model[0].auto_model,
            LoraConfig(
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["alpha"],
                target_modules=lora_cfg["target_modules"],
                lora_dropout=lora_cfg.get("dropout", 0.05),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            ),
        )
        model[0].auto_model.print_trainable_parameters()

    return model


# ── Training ───────────────────────────────────────────────────────────────────

def train_on_idiolink_original(
    model_alias:    str,
    model_cfg:      dict,
    global_cfg:     dict,
    idiolink_dir:   Path,
    checkpoint_dir: Path,
    use_wandb:      bool,
    force:          bool,
) -> Path:
    """Fine-tune with the original IdioLink InfoNCE loss. Returns saved model path."""
    import wandb
    from datasets import Dataset
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from transformers import EarlyStoppingCallback
    from transformers.trainer_utils import get_last_checkpoint

    # Saved to a parallel path so it doesn't collide with train.py output
    if model_cfg.get("model_output_dir"):
        base = Path(model_cfg["model_output_dir"])
        model_save_dir = base.parent / (base.name + "_original")
    else:
        model_save_dir = MODELS_ORIGINAL_DIR / model_alias

    if model_save_dir.exists() and not force:
        print(f"    [cache hit] Saved model found ← {model_save_dir}")
        return model_save_dir

    data_dict = load_idiolink_train_all_neg(idiolink_dir)
    n_samples = len(data_dict["anchor"])
    print(f"    IdioLink train: {n_samples} triplets × 5 hard negatives each")

    val_evaluator = build_idiolink_val_evaluator(idiolink_dir, name=f"{model_alias}_val")

    if force and checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)

    checkpoint_to_resume = None
    if checkpoint_dir.exists():
        last = get_last_checkpoint(str(checkpoint_dir))
        if last:
            checkpoint_to_resume = last
            print(f"    [resume] ← {checkpoint_to_resume}")

    model = build_model(model_cfg)
    loss  = IdioLinkInfoNCELoss(model)

    # 7 columns: anchor, positive, neg0..neg4
    dataset = Dataset.from_dict(data_dict)

    if use_wandb:
        wandb.init(
            project=global_cfg["wandb"]["project"],
            name=f"ft_10a_orig_{model_alias}",
            group="ft_10_option_a_original",
            tags=["ft_10", "option_a", "original_loss", model_alias],
            config={
                "model":       model_cfg["hf_id"],
                "loss":        "IdioLinkInfoNCE",
                "temperature": 0.05,
                "n_negatives": 5,
                "same_tower":  True,
                **{k: v for k, v in global_cfg.items() if k not in ("models", "wandb")},
            },
        )

    batch_size   = model_cfg["batch_size"]
    grad_accum   = model_cfg.get("gradient_accumulation_steps", 1)
    steps_per_ep = max(1, n_samples // batch_size)
    metric_key   = f"{model_alias}_val_cosine_ndcg@10"

    callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    if os.getenv("TG_BOT_TOKEN") and os.getenv("TG_CHAT_ID"):
        from finetuning.lib.notify import TelegramCallback
        callbacks.append(TelegramCallback(label=f"ft_10a_orig/{model_alias}"))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=global_cfg["epochs"],
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
        learning_rate=float(model_cfg["learning_rate"]),
        warmup_steps=global_cfg.get("warmup_steps", 100),
        seed=global_cfg.get("seed", 42),
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_key,
        greater_is_better=True,
        dataloader_pin_memory=False,
        logging_steps=max(1, steps_per_ep // 2),
        report_to="wandb" if (use_wandb and wandb.run is not None) else "none",
    )

    SentenceTransformerTrainer(
        model=model, args=trainer_args,
        train_dataset=dataset, evaluator=val_evaluator,
        loss=loss, callbacks=callbacks,
    ).train(resume_from_checkpoint=checkpoint_to_resume)

    if model_cfg.get("use_lora"):
        model[0].auto_model = model[0].auto_model.merge_and_unload()

    model_save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_save_dir))
    print(f"    [saved] → {model_save_dir}")

    if use_wandb and wandb.run is not None:
        wandb.finish()

    del model, loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model_save_dir


# ── MORABLES corpus ────────────────────────────────────────────────────────────

def load_gemini_summaries() -> dict:
    with open(GEMINI_SUMMARIES_PATH) as f:
        data = json.load(f)
    return {e["original_fable_id"]: e["summaries"] for e in data}


def load_gemma4_summaries(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    result = {}
    for entry in data:
        alias = entry.get("fable_alias") or entry.get("original_fable_id", "")
        result[alias] = {
            model: {v: info["text"] for v, info in variants.items()}
            for model, variants in entry.get("summaries", {}).items()
        }
    return result


def build_morables_corpus(corpus_config, fables, gemini_sums, gemma4_sums):
    label, gen, variant, corpus_type = corpus_config
    if gen == "none":
        return [f["text"] for f in fables]
    if gen == "gemini":
        docs = []
        for f in fables:
            summary = gemini_sums.get(f["alias"], {}).get(variant, "")
            if corpus_type == "fable_summary":
                docs.append(f"{f['text']}\n\nMoral summary: {summary}")
            elif corpus_type == "summary_only":
                docs.append(summary)
            elif corpus_type == "summary+fable":
                docs.append(f"Moral summary: {summary}\n\n{f['text']}")
        return docs
    if gen.startswith("gemma4-"):
        if gemma4_sums is None:
            return None
        docs = []
        for f in fables:
            summary = gemma4_sums.get(f["alias"], {}).get(gen, {}).get(variant, "")
            if corpus_type == "fable_summary":
                docs.append(f"{f['text']}\n\nMoral summary: {summary}")
            elif corpus_type == "summary_only":
                docs.append(summary)
        return docs
    raise ValueError(f"Unknown summary generator: {gen!r}")


# ── MORABLES evaluation ────────────────────────────────────────────────────────

def evaluate_morables_all_configs(
    model_alias:    str,
    model_save_dir: Path,
    model_cfg:      dict,
    gemma4_sums,
    force:          bool,
) -> dict:
    from sentence_transformers import SentenceTransformer
    from lib.data import load_fables, load_morals, load_qrels_moral_to_fable

    model_kwargs = {}
    raw_kwargs   = model_cfg.get("model_kwargs") or {}
    if "torch_dtype" in raw_kwargs:
        dtype = raw_kwargs["torch_dtype"]
        model_kwargs["torch_dtype"] = getattr(torch, dtype, dtype) if isinstance(dtype, str) else dtype
    if raw_kwargs.get("trust_remote_code"):
        model_kwargs["trust_remote_code"] = True

    print(f"\n  Loading fine-tuned model from {model_save_dir} …")
    model = SentenceTransformer(str(model_save_dir), model_kwargs=model_kwargs or None)

    fables        = load_fables()
    morals        = load_morals()
    qrels         = load_qrels_moral_to_fable()
    moral_indices = sorted(qrels.keys())
    moral_texts   = [morals[i]["text"] for i in moral_indices]
    ground_truth  = {i: qrels[idx] for i, idx in enumerate(moral_indices)}
    gemini_sums   = load_gemini_summaries()

    results = {}
    # Separate cache namespace so embeddings don't collide with train.py's cache
    emb_cache_base = EXP_DIR / "cache" / "embeddings_original" / model_alias

    for cfg in CORPUS_CONFIGS:
        label = cfg[0]
        print(f"    Evaluating: {label} …")

        doc_texts = build_morables_corpus(cfg, fables, gemini_sums, gemma4_sums)
        if doc_texts is None:
            print(f"    [skip] Gemma4 summaries not available locally")
            results[label] = None
            continue

        query_cache = emb_cache_base / "queries"
        doc_cache   = emb_cache_base / label.replace("/", "_")

        metrics = evaluate(
            model, moral_texts, doc_texts, ground_truth,
            cache_dir=query_cache, doc_cache_dir=doc_cache, force=force,
        )
        results[label] = metrics
        print(
            f"      MRR={metrics['MRR']:.4f}  "
            f"R@1={metrics['Recall@1']:.4f}  "
            f"R@5={metrics['Recall@5']:.4f}  "
            f"NDCG@10={metrics['NDCG@10']:.4f}"
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ── CSV ────────────────────────────────────────────────────────────────────────

def _read_zero_shot_baselines() -> dict:
    """Read zero_shot_* from option_a_results.csv, keyed by (model_alias, corpus_config)."""
    ref_csv = RESULTS_DIR / "option_a_results.csv"
    if not ref_csv.exists():
        return {}
    out = {}
    with open(ref_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["model_alias"], row["corpus_config"])
            out[key] = {
                "zero_shot_MRR@10":  row.get("zero_shot_MRR@10", ""),
                "zero_shot_R@1":     row.get("zero_shot_R@1", ""),
                "zero_shot_R@5":     row.get("zero_shot_R@5", ""),
                "zero_shot_NDCG@10": row.get("zero_shot_NDCG@10", ""),
            }
    return out


def init_original_csv(config: dict) -> None:
    """Create option_a_original_results.csv pre-filled with zero-shot baselines."""
    if ORIGINAL_RESULTS_CSV.exists():
        return
    baselines = _read_zero_shot_baselines()
    rows = []
    for alias, model_cfg in config["models"].items():
        for label, gen, variant, ctype in CORPUS_CONFIGS:
            zs = baselines.get((alias, label), {})
            rows.append({
                "model_alias":         alias,
                "model_hf_id":         model_cfg["hf_id"],
                "corpus_config":       label,
                "summary_generator":   gen,
                "summary_variant":     variant,
                "corpus_type":         ctype,
                "zero_shot_MRR@10":    zs.get("zero_shot_MRR@10", ""),
                "original_ft_MRR@10":  "",
                "delta_MRR":           "",
                "zero_shot_R@1":       zs.get("zero_shot_R@1", ""),
                "original_ft_R@1":     "",
                "zero_shot_R@5":       zs.get("zero_shot_R@5", ""),
                "original_ft_R@5":     "",
                "zero_shot_NDCG@10":   zs.get("zero_shot_NDCG@10", ""),
                "original_ft_NDCG@10": "",
            })
    with open(ORIGINAL_RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"  [init] Created {ORIGINAL_RESULTS_CSV}")


def update_original_csv(model_alias: str, model_hf_id: str, results: dict) -> None:
    with open(ORIGINAL_RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if row["model_alias"] != model_alias:
            continue
        label   = row["corpus_config"]
        metrics = results.get(label)
        if metrics is None:
            row["original_ft_MRR@10"]  = "n/a (summaries not pulled)"
            row["original_ft_R@1"]     = ""
            row["original_ft_R@5"]     = ""
            row["original_ft_NDCG@10"] = ""
            row["delta_MRR"]           = ""
            continue
        ft_mrr = metrics["MRR"]
        row["original_ft_MRR@10"]  = f"{ft_mrr:.4f}"
        row["original_ft_R@1"]     = f"{metrics['Recall@1']:.4f}"
        row["original_ft_R@5"]     = f"{metrics['Recall@5']:.4f}"
        row["original_ft_NDCG@10"] = f"{metrics['NDCG@10']:.4f}"
        zs = row.get("zero_shot_MRR@10", "")
        if zs:
            row["delta_MRR"] = f"{ft_mrr - float(zs):+.4f}"

    with open(ORIGINAL_RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV updated → {ORIGINAL_RESULTS_CSV}")


# ── Single-model entry point ───────────────────────────────────────────────────

def run_one_model(model_alias: str, config: dict, force: bool, use_wandb: bool) -> None:
    model_cfg    = config["models"][model_alias]
    idiolink_dir = ROOT / config["idiolink_data_dir"]

    gemma4_sums = None
    gemma4_path = config.get("gemma4_summaries_path")
    if gemma4_path:
        p = ROOT / gemma4_path
        if p.exists():
            gemma4_sums = load_gemma4_summaries(p)
        else:
            print(f"  [warn] gemma4_summaries_path set but not found: {p}")

    checkpoint_dir = EXP_DIR / "cache" / "checkpoints_original" / model_alias

    notify.send(
        f"🚀 ft_10a_original starting: {model_alias}\n"
        f"model: {model_cfg['hf_id']}\n"
        f"loss: IdioLinkInfoNCE (5 negs + same-tower, τ=0.05)\n"
        f"lora: {model_cfg.get('use_lora', False)}  "
        f"bs: {model_cfg['batch_size']}×{model_cfg.get('gradient_accumulation_steps', 1)}"
    )

    model_save_dir = train_on_idiolink_original(
        model_alias, model_cfg, config,
        idiolink_dir, checkpoint_dir, use_wandb, force,
    )

    results = evaluate_morables_all_configs(
        model_alias, model_save_dir, model_cfg, gemma4_sums, force,
    )
    update_original_csv(model_alias, model_cfg["hf_id"], results)

    evaluated = [k for k, v in results.items() if v is not None]
    skipped   = [k for k, v in results.items() if v is None]
    mrrs      = [f"{results[k]['MRR']:.4f}" for k in evaluated]

    notify.send(
        f"✅ ft_10a_original done: {model_alias}\n"
        f"configs evaluated: {len(evaluated)}/{len(CORPUS_CONFIGS)}\n"
        f"MRRs: {', '.join(mrrs)}"
        + (f"\nskipped (no Gemma4 summaries): {len(skipped)}" if skipped else "")
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ft_10 Option A Original: IdioLink full InfoNCE → MORABLES transfer"
    )
    parser.add_argument("--model",    type=str, default=None, help="Model alias (default: all)")
    parser.add_argument("--force",    action="store_true",    help="Re-train and re-embed even if cached")
    parser.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    use_wandb = config.get("wandb", {}).get("enabled", False) and not args.no_wandb

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    init_original_csv(config)

    model_aliases = list(config["models"].keys())
    if args.model:
        if args.model not in config["models"]:
            print(f"Unknown model: {args.model!r}. Available: {model_aliases}")
            sys.exit(1)
        model_aliases = [args.model]

    if len(model_aliases) == 1:
        run_one_model(model_aliases[0], config, args.force, use_wandb)
        return

    notify.send(
        f"🚀 ft_10 Option A Original starting\n"
        f"models: {', '.join(model_aliases)}\n"
        f"loss: IdioLinkInfoNCE (all 5 negs + same-tower)"
    )

    extra = ["--force"] if args.force else []
    if not use_wandb:
        extra.append("--no-wandb")

    for alias in model_aliases:
        print(f"\n{'='*60}")
        print(f"  [subprocess] {alias}")
        print(f"{'='*60}")
        subprocess.run(
            [sys.executable, __file__, "--model", alias] + extra,
            check=True,
        )

    notify.send(f"✅ ft_10 Option A Original complete — all {len(model_aliases)} models done")


if __name__ == "__main__":
    main()
