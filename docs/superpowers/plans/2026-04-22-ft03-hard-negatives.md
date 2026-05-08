# ft_03 Hard-Negative Fine-Tuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Linq-Embed-Mistral with hard negatives mined from MCQA distractors, replacing the in-batch-only loss from ft_02 with a custom InfoNCE loss that also pushes morals apart from each other.

**Architecture:** Two scripts — `mine_negatives.py` (runs once offline, encodes distractors and retrieves the most confusable fable per distractor) and `train.py` (uses mined triples, custom InfoNCE loss, 5-fold GroupKFold CV). A new `finetuning/lib/losses.py` implements the InfoNCE loss; `train.py` does not use the existing `trainer.py` because the custom loss and triplet dataset format require a different training setup.

**Tech Stack:** SentenceTransformers, PyTorch, Linq-Embed-Mistral (7B), LoRA (PEFT), HuggingFace Trainer, uv, run.sh for local/remote execution.

---

## Files to create or modify

| File | Action | Responsibility |
|---|---|---|
| `finetuning/ft_03_hard_neg/config.yaml` | Create | Hyperparameters for ft_03 |
| `finetuning/ft_03_hard_neg/mine_negatives.py` | Create | Encode distractors → retrieve hard neg fable per distractor |
| `finetuning/lib/losses.py` | Create | InfoNCE loss (types 1+2+3, custom temperature) |
| `finetuning/ft_03_hard_neg/train.py` | Create | Training loop with hard negatives, ablation args |
| `finetuning/ft_03_hard_neg/data/hard_negatives.json` | Generated | Output of mine_negatives.py |

`trainer.py` and `eval.py` in `finetuning/lib/` are **not modified** — train.py calls the SentenceTransformerTrainer directly.

---

## Experiment grid

We run two phases to avoid committing to 75 jobs upfront:

**Phase 1 — distractor type sweep (25 jobs)**
Fix: τ=0.05, doc_mode=fable_plus_summary, αH=1
Sweep: distractor_type ∈ {similar_characters, based_on_adjectives, injected_adjectives, partial_story, all}
Each distractor type × 5 folds = 25 training jobs

**Phase 2 — temperature sweep (15 jobs)**
Fix: best distractor type from Phase 1, same doc_mode
Sweep: τ ∈ {0.05, 0.07, 0.1}
3 temperatures × 5 folds = 15 training jobs

Total: 40 jobs. Baseline from ft_02 (fable_plus_summary): MRR ≈ 0.41.

---

## Negative types used in ft_03

For a batch of B items [(M1,F1,F1-), (M2,F2,F2-), ...]:

- **Type 1 — moral vs. other fables:** M_i far from F_j for j≠i (in-batch)
- **Type 2 — moral vs. other morals:** M_i far from M_j for j≠i (prevents moral embeddings collapsing together)
- **Type 3 — moral vs. confusable fable:** M_i far from F_i- (explicitly mined hard negative)

Types 4–6 (distractors as training signal) are left for ft_04.

Multi-positive masking (for the 27 morals that appear in multiple fables): skipped in ft_03 since batch size 4 makes collisions rare (~1-2% of batches). Will be added in ft_04 if needed.

---

## Task 1: Directory + config

**Files:**
- Create: `finetuning/ft_03_hard_neg/config.yaml`
- Create: `finetuning/ft_03_hard_neg/data/` (empty dir)
- Create: `finetuning/ft_03_hard_neg/results/` (empty dir)

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p finetuning/ft_03_hard_neg/data
mkdir -p finetuning/ft_03_hard_neg/results
```

- [ ] **Step 2: Create config.yaml**

```yaml
model_name: Linq-AI-Research/Linq-Embed-Mistral
doc_mode: fable_plus_summary
epochs: 10
batch_size: 4
gradient_accumulation_steps: 4
gradient_checkpointing: true
max_seq_length: 512
learning_rate: 0.0001
temperature: 0.05
seed: 42
early_stopping_patience: 3

query_instruction: "Instruct: Given a moral statement, retrieve the fable that best conveys this moral.\nQuery: "

model_kwargs:
  torch_dtype: "auto"

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

model_output_dir: /data/lior/ft_03_hard_neg/models

wandb:
  project: morables-finetuning
  enabled: true
```

- [ ] **Step 3: Commit**

```bash
git add finetuning/ft_03_hard_neg/
git commit -m "feat(ft_03): scaffold directory and config"
```

---

## Task 2: Mine hard negatives

**Files:**
- Create: `finetuning/ft_03_hard_neg/mine_negatives.py`
- Output: `finetuning/ft_03_hard_neg/data/hard_negatives.json`

The script:
1. Loads all 709 morals and encodes them with zero-shot Linq
2. Loads MCQA: for each fable, gets its 4 distractors
3. Encodes each distractor and finds the fable whose **true moral** is most similar (cosine similarity)
4. Excludes the fable itself (can't be its own hard negative)
5. Saves one record per (fable, distractor) pair

Output format (one entry per distractor):
```json
{
  "fable_alias": "aesop_section_1_5",
  "moral_idx": 0,
  "moral": "Gratitude is the sign of noble souls.",
  "distractor_type": "injected_adjectives",
  "distractor_text": "The true leader proves himself by his brave qualities.",
  "hard_neg_fable_idx": 42,
  "hard_neg_alias": "aesop_section_2_14",
  "hard_neg_moral": "..."
}
```

- [ ] **Step 1: Write mine_negatives.py**

```python
"""
ft_03 — Hard negative mining.

For each fable and each of its 4 MCQA distractors, finds the fable whose
true moral is most similar to the distractor (using zero-shot Linq).
This fable becomes the hard negative for training.

Usage:
    ./run.sh finetuning/ft_03_hard_neg/mine_negatives.py
    ./run.sh finetuning/ft_03_hard_neg/mine_negatives.py --force
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from lib.data import load_fables, load_morals, load_qrels_moral_to_fable

MCQA_PATH = ROOT / "data/raw/mcqa.json"
OUT_PATH = EXP_DIR / "data/hard_negatives.json"

DISTRACTOR_TYPES = ["similar_characters", "based_on_adjectives", "injected_adjectives", "partial_story"]
MODEL_NAME = "Linq-AI-Research/Linq-Embed-Mistral"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-mine even if output exists")
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.force:
        print(f"[cache hit] Hard negatives already mined → {OUT_PATH}")
        print("  Use --force to re-mine.")
        return

    notify.send(f"🔍 ft_03: mining hard negatives\nmodel: {MODEL_NAME}")

    fables = load_fables()
    morals = load_morals()
    qrels = load_qrels_moral_to_fable()

    # Build alias → fable_idx mapping
    alias_to_fable_idx = {f["alias"]: i for i, f in enumerate(fables)}

    # Build moral_idx → fable_idx mapping (for all 709 pairs)
    moral_indices = sorted(qrels.keys())
    moral_idx_to_fable_idx = {i: qrels[idx] for i, idx in enumerate(moral_indices)}
    moral_texts = [morals[i]["text"] for i in moral_indices]
    fable_alias_to_moral_idx = {fables[fable_idx]["alias"]: moral_idx
                                 for moral_idx, fable_idx in moral_idx_to_fable_idx.items()}

    with open(MCQA_PATH) as f:
        mcqa = json.load(f)

    # Build alias → mcqa entry mapping
    alias_to_mcqa = {entry["alias"]: entry for entry in mcqa}

    print(f"[1/3] Encoding {len(moral_texts)} true morals with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, model_kwargs={"torch_dtype": "auto"})

    # Encode all true morals (no instruction — we're matching distractor text to moral text)
    moral_embs = model.encode(moral_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    moral_embs = torch.tensor(moral_embs)  # (709, D)

    print("[2/3] Mining hard negatives for each distractor...")
    records = []
    skipped = 0

    for entry in mcqa:
        alias = entry["alias"]
        classes = entry["classes"]      # ["ground_truth", "similar_characters", ...]
        choices = entry["choices"]      # [true_moral, dist1, dist2, dist3, dist4]

        moral_idx = fable_alias_to_moral_idx.get(alias)
        if moral_idx is None:
            skipped += 1
            continue

        fable_idx = moral_idx_to_fable_idx[moral_idx]

        # Encode each distractor and find its nearest true moral
        for dtype in DISTRACTOR_TYPES:
            if dtype not in classes:
                continue
            dist_pos = classes.index(dtype)
            distractor_text = choices[dist_pos]

            dist_emb = model.encode([distractor_text], normalize_embeddings=True)
            dist_emb = torch.tensor(dist_emb)  # (1, D)

            sims = (moral_embs @ dist_emb.T).squeeze(1)  # (709,)

            # Exclude the fable itself (can't be its own hard negative)
            sims[moral_idx] = -1.0

            hard_neg_moral_idx = int(sims.argmax())
            hard_neg_fable_idx = moral_idx_to_fable_idx[hard_neg_moral_idx]

            records.append({
                "fable_alias": alias,
                "moral_idx": moral_idx,
                "moral": choices[0],  # ground truth moral = choices[0]
                "distractor_type": dtype,
                "distractor_text": distractor_text,
                "hard_neg_fable_idx": hard_neg_fable_idx,
                "hard_neg_alias": fables[hard_neg_fable_idx]["alias"],
                "hard_neg_moral": moral_texts[hard_neg_moral_idx],
                "similarity": float(sims[hard_neg_moral_idx]),
            })

    print(f"[3/3] Saving {len(records)} records (skipped {skipped} entries without moral mapping)...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  → {OUT_PATH}")

    notify.send(f"✅ ft_03: hard negatives mined\n{len(records)} records → {OUT_PATH.name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run locally to verify output**

```bash
./run.sh finetuning/ft_03_hard_neg/mine_negatives.py
```

Expected: ~2,836 records in `finetuning/ft_03_hard_neg/data/hard_negatives.json` (709 fables × 4 distractor types).

Sanity check: print the first few records and verify the hard negatives look plausible (similar theme, different fable):
```bash
python3 -c "
import json
with open('finetuning/ft_03_hard_neg/data/hard_negatives.json') as f:
    data = json.load(f)
print(f'Total: {len(data)}')
for r in data[:3]:
    print()
    print('Moral:', r['moral'])
    print('Distractor type:', r['distractor_type'])
    print('Distractor:', r['distractor_text'])
    print('Hard neg moral:', r['hard_neg_moral'])
    print('Similarity:', round(r['similarity'], 3))
"
```

- [ ] **Step 3: Commit**

```bash
git add finetuning/ft_03_hard_neg/mine_negatives.py finetuning/ft_03_hard_neg/data/hard_negatives.json
git commit -m "feat(ft_03): mine hard negatives from MCQA distractors"
```

---

## Task 3: Custom InfoNCE loss

**Files:**
- Create: `finetuning/lib/losses.py`

The loss receives three sets of embeddings per batch:
- Moral embeddings M (shape: B × D)
- Positive fable embeddings F+ (shape: B × D)
- Hard negative fable embeddings F- (shape: B × D)

And computes:
```
L = cross_entropy(logits, targets=arange(B))

logits[:, 0:B]   = sim(M_i, F_j+) / τ   (in-batch fables — type 1, diagonal = positive)
logits[:, B]     = sim(M_i, F_i-) / τ   (hard negative — type 3)
logits[:, B+1:]  = sim(M_i, M_j) / τ    (other morals — type 2, diagonal masked to -inf)
```

Cross-entropy targets = [0, 1, 2, ..., B-1] — the positive for sample i is at column i.

- [ ] **Step 1: Write finetuning/lib/losses.py**

```python
"""
Custom InfoNCE loss for ft_03 hard-negative fine-tuning.

Negative types included:
  Type 1 — moral vs. other fables in the batch (in-batch)
  Type 2 — moral vs. other morals in the batch (prevents embedding collapse)
  Type 3 — moral vs. mined hard negative fable (explicit)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for (anchor=moral, positive=fable, hard_neg=fable) triplets.

    Compatible with SentenceTransformerTrainer: expects sentence_features as a
    list of 3 feature dicts [moral_features, pos_fable_features, neg_fable_features].
    """

    def __init__(self, model, temperature: float = 0.05):
        super().__init__()
        self.model = model
        self.temperature = temperature

    def forward(self, sentence_features, labels=None):
        moral_emb = F.normalize(
            self.model(sentence_features[0])["sentence_embedding"], dim=-1
        )
        pos_emb = F.normalize(
            self.model(sentence_features[1])["sentence_embedding"], dim=-1
        )
        neg_emb = F.normalize(
            self.model(sentence_features[2])["sentence_embedding"], dim=-1
        )

        B = moral_emb.size(0)
        device = moral_emb.device
        τ = self.temperature

        # Type 1+3: moral vs in-batch fables and hard neg
        fable_sim = torch.mm(moral_emb, pos_emb.T) / τ          # (B, B)
        hard_neg_sim = (moral_emb * neg_emb).sum(dim=-1, keepdim=True) / τ  # (B, 1)

        # Type 2: moral vs other morals (mask diagonal — moral vs itself is trivial)
        moral_sim = torch.mm(moral_emb, moral_emb.T) / τ         # (B, B)
        moral_sim = moral_sim.masked_fill(
            torch.eye(B, dtype=torch.bool, device=device), float("-inf")
        )

        # Concatenate: positive is at column i in fable_sim
        all_logits = torch.cat([fable_sim, hard_neg_sim, moral_sim], dim=1)  # (B, 2B+1)

        targets = torch.arange(B, device=device)
        return F.cross_entropy(all_logits, targets)
```

- [ ] **Step 2: Quick sanity check (run locally, no GPU needed)**

```bash
python3 -c "
import torch
import torch.nn.functional as F

# Simulate what the loss does with random embeddings
B, D = 4, 16
moral = F.normalize(torch.randn(B, D), dim=-1)
pos   = F.normalize(torch.randn(B, D), dim=-1)
neg   = F.normalize(torch.randn(B, D), dim=-1)
τ = 0.05

fable_sim    = torch.mm(moral, pos.T) / τ
hard_neg_sim = (moral * neg).sum(dim=-1, keepdim=True) / τ
moral_sim    = torch.mm(moral, moral.T) / τ
moral_sim.fill_diagonal_(float('-inf'))

logits  = torch.cat([fable_sim, hard_neg_sim, moral_sim], dim=1)
targets = torch.arange(B)
loss    = F.cross_entropy(logits, targets)
print(f'Loss: {loss.item():.4f}  (random embeddings, should be ~log({2*B+1}) ≈ {__import__(\"math\").log(2*B+1):.2f})')
"
```

Expected output: `Loss: ~2.2` (log of 9 for B=4, since denominator has 2B+1=9 terms).

- [ ] **Step 3: Commit**

```bash
git add finetuning/lib/losses.py
git commit -m "feat(lib): add InfoNCE loss with hard negatives and moral-moral negatives"
```

---

## Task 4: Training script

**Files:**
- Create: `finetuning/ft_03_hard_neg/train.py`

The script:
1. Loads mined hard negatives from `data/hard_negatives.json`
2. Filters by `--distractor_type` (one of the 4 types, or "all")
3. Builds training triplets: for each (moral, fable, hard_neg_fable), uses the GroupKFold splits from ft_01
4. Trains with `InfoNCELoss` and the same LoRA + SentenceTransformerTrainer setup as ft_02
5. Evaluates using `finetuning/lib/eval.evaluate()` — same MRR pipeline as ft_02
6. Saves results JSON to `results/`

Key differences from train.py in ft_02:
- Dataset has 3 columns: anchor, positive, negative (triplets)
- Loss is `InfoNCELoss` (not `MultipleNegativesRankingLoss`)
- Temperature τ is a config/arg parameter

- [ ] **Step 1: Write train.py**

```python
"""
ft_03_hard_neg — 5-fold CV with hard-negative InfoNCE loss.

Usage:
    # All folds, injected_adjectives distractors:
    ./run.sh finetuning/ft_03_hard_neg/train.py --distractor_type injected_adjectives

    # Single fold for quick test:
    ./run.sh finetuning/ft_03_hard_neg/train.py --distractor_type injected_adjectives --fold 0

    # Temperature sweep:
    ./run.sh finetuning/ft_03_hard_neg/train.py --distractor_type injected_adjectives --tau 0.07

    # Remote GPU:
    ./run.sh finetuning/ft_03_hard_neg/train.py --distractor_type injected_adjectives --remote --gpu 2
"""
import argparse
import gc
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from finetuning.lib.data import load_pairs
from finetuning.lib.eval import evaluate
from finetuning.lib.losses import InfoNCELoss

CACHE_DIR = EXP_DIR / "cache"
RESULTS_DIR = EXP_DIR / "results"
CONFIG_PATH = EXP_DIR / "config.yaml"
SPLITS_PATH = EXP_DIR.parent / "ft_01_5fold_cv" / "cache" / "splits" / "folds.json"
HARD_NEG_PATH = EXP_DIR / "data" / "hard_negatives.json"

DISTRACTOR_TYPES = ["similar_characters", "based_on_adjectives", "injected_adjectives", "partial_story", "all"]
_EVAL_KEYS = ("MRR", "Recall@1", "Recall@5", "Recall@10")


def build_triplets(
    hard_negs: list[dict],
    distractor_type: str,
    moral_texts: list[str],
    doc_texts: list[str],
    ground_truth: dict[int, int],
    train_moral_idxs: list[int],
    instruction: str,
) -> tuple[list[str], list[str], list[str]]:
    """
    Build (anchor, positive, negative) lists for the training fold.

    Returns:
        anchors    instructed moral texts
        positives  fable texts (or fable+summary)
        negatives  hard negative fable texts
    """
    # Index hard_negs by moral_idx
    moral_idx_to_hard_negs: dict[int, list[dict]] = {}
    for rec in hard_negs:
        idx = rec["moral_idx"]
        moral_idx_to_hard_negs.setdefault(idx, []).append(rec)

    train_moral_set = set(train_moral_idxs)

    anchors, positives, negatives = [], [], []
    for moral_idx in train_moral_idxs:
        fable_idx = ground_truth[moral_idx]
        candidates = moral_idx_to_hard_negs.get(moral_idx, [])

        # Filter by distractor type
        if distractor_type != "all":
            candidates = [r for r in candidates if r["distractor_type"] == distractor_type]
        else:
            # One random distractor per anchor
            if candidates:
                candidates = [random.choice(candidates)]

        for rec in candidates:
            hard_neg_fable_idx = rec["hard_neg_fable_idx"]
            # Skip if hard neg is the correct fable or not in the doc corpus
            if hard_neg_fable_idx == fable_idx:
                continue

            anchors.append(f"{instruction}{moral_texts[moral_idx]}")
            positives.append(doc_texts[fable_idx])
            negatives.append(doc_texts[hard_neg_fable_idx])

    return anchors, positives, negatives


def run_fold(
    fold: dict,
    moral_texts: list[str],
    doc_texts: list[str],
    ground_truth: dict[int, int],
    hard_negs: list[dict],
    config: dict,
    distractor_type: str,
    force: bool,
    use_wandb: bool,
) -> dict:
    fold_idx = fold["fold"]
    train_idx, test_idx = fold["train"], fold["test"]

    instruction = config.get("query_instruction", "")
    τ = config.get("temperature", 0.05)

    anchors, positives, negatives = build_triplets(
        hard_negs, distractor_type, moral_texts, doc_texts, ground_truth, train_idx, instruction
    )
    print(
        f"\n  Fold {fold_idx + 1}/5  "
        f"train={len(train_idx)} morals → {len(anchors)} triplets  "
        f"test={len(test_idx)}  τ={τ}"
    )

    test_morals = [f"{instruction}{moral_texts[i]}" for i in test_idx]
    test_gt = {j: ground_truth[i] for j, i in enumerate(test_idx)}

    _model_root = Path(config["model_output_dir"]) if config.get("model_output_dir") else CACHE_DIR / "models"
    run_tag = f"{distractor_type}_tau{str(τ).replace('.', '')}"
    model_cache = _model_root / config["doc_mode"] / run_tag / f"fold_{fold_idx}"
    checkpoint_dir = CACHE_DIR / "checkpoints" / config["doc_mode"] / run_tag / f"fold_{fold_idx}"
    emb_cache = CACHE_DIR / "embeddings" / config["doc_mode"] / run_tag / f"fold_{fold_idx}"

    if model_cache.exists() and not force:
        print(f"    [cache hit] Loading model ← {model_cache}")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(model_cache))
    else:
        model = _train(
            anchors, positives, negatives, doc_texts, test_morals, test_gt,
            config, fold_idx, τ, model_cache, checkpoint_dir, use_wandb, distractor_type, run_tag, force
        )

    metrics = evaluate(model, test_morals, doc_texts, test_gt, cache_dir=emb_cache, force=force)
    print(
        f"  MRR={metrics['MRR']:.4f}  "
        f"R@1={metrics['Recall@1']:.4f}  "
        f"R@5={metrics['Recall@5']:.4f}  "
        f"R@10={metrics['Recall@10']:.4f}"
    )

    if use_wandb:
        wandb.log({"fold": fold_idx, **{f"eval/{k.lower().replace('@', '_at_')}": metrics[k] for k in _EVAL_KEYS}})
        wandb.finish()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def _train(
    anchors, positives, negatives, doc_texts, test_morals, test_gt,
    config, fold_idx, τ, model_cache, checkpoint_dir, use_wandb, distractor_type, run_tag, force
):
    import shutil
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from transformers import EarlyStoppingCallback
    from transformers.trainer_utils import get_last_checkpoint

    if force and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    checkpoint_to_resume = None
    if checkpoint_dir.exists():
        last_ckpt = get_last_checkpoint(str(checkpoint_dir))
        if last_ckpt:
            checkpoint_to_resume = last_ckpt
            print(f"    [resume] ← {checkpoint_to_resume}")

    model_kwargs = config.get("model_kwargs") or {}
    model = SentenceTransformer(config["model_name"], **({"model_kwargs": model_kwargs} if model_kwargs else {}))
    if config.get("max_seq_length"):
        model.max_seq_length = config["max_seq_length"]

    lora_cfg = config.get("lora")
    if lora_cfg:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model[0].auto_model = get_peft_model(model[0].auto_model, peft_config)
        model[0].auto_model.print_trainable_parameters()

    loss = InfoNCELoss(model, temperature=τ)
    train_dataset = Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives})

    evaluator = InformationRetrievalEvaluator(
        queries={str(j): text for j, text in enumerate(test_morals)},
        corpus={str(i): text for i, text in enumerate(doc_texts)},
        relevant_docs={str(j): {str(test_gt[j])} for j in range(len(test_morals))},
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        name=f"fold_{fold_idx}",
    )

    if use_wandb:
        wandb.init(
            project=config["wandb"]["project"],
            name=f"fold_{fold_idx}_{run_tag}",
            group=f"ft_03_hard_neg/{run_tag}",
            tags=["ft_03_hard_neg", distractor_type, f"tau_{τ}", f"fold_{fold_idx}"],
            config={k: v for k, v in config.items() if k != "wandb"},
        )

    best_metric = f"eval_fold_{fold_idx}_cosine_mrr@10"
    early_stop_patience = config.get("early_stopping_patience")
    steps_per_epoch = max(1, len(anchors) // config["batch_size"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        learning_rate=float(config["learning_rate"]),
        seed=config["seed"],
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        dataloader_pin_memory=False,
        logging_steps=max(1, steps_per_epoch // 2),
        report_to="wandb" if wandb.run is not None else "none",
    )

    callbacks = []
    if early_stop_patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stop_patience))

    import os
    if os.getenv("TG_BOT_TOKEN") and os.getenv("TG_CHAT_ID"):
        from finetuning.lib.notify import TelegramCallback
        label = f"ft_03/{run_tag}/fold_{fold_idx}"
        callbacks.append(TelegramCallback(label=label))

    SentenceTransformerTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        evaluator=evaluator,
        loss=loss,
        callbacks=callbacks or None,
    ).train(resume_from_checkpoint=checkpoint_to_resume)

    if lora_cfg:
        model[0].auto_model = model[0].auto_model.merge_and_unload()

    model_cache.mkdir(parents=True, exist_ok=True)
    model.save(str(model_cache))
    print(f"    [saved] → {model_cache}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distractor_type", choices=DISTRACTOR_TYPES, required=True,
                        help="Which distractor type to use for hard negatives")
    parser.add_argument("--tau", type=float, default=None, help="Override temperature τ in config")
    parser.add_argument("--fold", type=int, choices=range(5), metavar="0-4",
                        help="Run a single fold instead of all 5")
    parser.add_argument("--doc_mode", choices=["raw", "fable_plus_summary"], help="Override config doc_mode")
    parser.add_argument("--force", action="store_true", help="Re-train even if cached")
    parser.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    if args.doc_mode:
        config["doc_mode"] = args.doc_mode
    if args.tau is not None:
        config["temperature"] = args.tau

    if not HARD_NEG_PATH.exists():
        raise FileNotFoundError(
            f"Hard negatives not found: {HARD_NEG_PATH}\n"
            "Run first: ./run.sh finetuning/ft_03_hard_neg/mine_negatives.py"
        )
    with open(HARD_NEG_PATH) as f:
        hard_negs = json.load(f)

    with open(SPLITS_PATH) as f:
        all_folds = json.load(f)

    wandb_cfg = config.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", False) and not args.no_wandb

    τ = config.get("temperature", 0.05)
    print(
        f"\n[ft_03_hard_neg]  model={config['model_name']}  "
        f"distractor={args.distractor_type}  τ={τ}  "
        f"doc_mode={config['doc_mode']}  epochs={config['epochs']}"
    )

    notify.send(
        f"🚀 ft_03_hard_neg starting\n"
        f"distractor_type: {args.distractor_type}  τ: {τ}\n"
        f"doc_mode: {config['doc_mode']}  epochs: {config['epochs']}\n"
        f"folds: {[args.fold] if args.fold is not None else 'all 5'}"
    )

    moral_texts, doc_texts, ground_truth = load_pairs(config["doc_mode"])

    fold_indices = [args.fold] if args.fold is not None else list(range(len(all_folds)))
    fold_metrics = [
        run_fold(all_folds[i], moral_texts, doc_texts, ground_truth, hard_negs,
                 config, args.distractor_type, args.force, use_wandb)
        for i in fold_indices
    ]

    mrr_scores = [m["MRR"] for m in fold_metrics]
    mean_mrr = float(np.mean(mrr_scores))
    std_mrr = float(np.std(mrr_scores))
    print(f"\n  Final MRR: {mean_mrr:.4f} ± {std_mrr:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"fold{args.fold}" if args.fold is not None else "all_folds"
    out = RESULTS_DIR / f"{ts}_{args.distractor_type}_tau{str(τ).replace('.', '')}_{suffix}.json"
    with open(out, "w") as f:
        json.dump({
            "config": config,
            "distractor_type": args.distractor_type,
            "temperature": τ,
            "folds_run": fold_indices,
            "mean_mrr": mean_mrr,
            "std_mrr": std_mrr,
            "fold_mrrs": mrr_scores,
            "fold_metrics": fold_metrics,
        }, f, indent=2)
    print(f"  Results → {out}")

    notify.send(
        f"✅ ft_03_hard_neg done\n"
        f"distractor: {args.distractor_type}  τ: {τ}\n"
        f"Final MRR: {mean_mrr:.4f} ± {std_mrr:.4f}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test — single fold locally (no GPU)**

This will fail quickly on CPU (that's fine — we just want to verify the script parses args and loads data):

```bash
./run.sh finetuning/ft_03_hard_neg/train.py --distractor_type injected_adjectives --fold 0 --no-wandb
```

Expected: script starts, prints fold info, begins encoding (may OOM on CPU — that's ok, verify the print lines appear first).

- [ ] **Step 3: Commit**

```bash
git add finetuning/ft_03_hard_neg/train.py
git commit -m "feat(ft_03): training script with InfoNCE loss and distractor ablation"
```

---

## Task 5: Run Phase 1 — distractor type sweep

Run all 5 distractor type configurations (one per GPU call, all 5 folds each).

- [ ] **Step 1: Check GPU availability and disk space**

```bash
./run.sh status
ssh $GPU_USER@$GPU_HOST "df -h ~ && df -h /data/lior"
```

Require: ≥30GB free on physical disk before starting.

- [ ] **Step 2: Quick single-fold smoke test on GPU**

```bash
./run.sh finetuning/ft_03_hard_neg/train.py --distractor_type injected_adjectives --fold 0 --no-wandb --remote --gpu 2
```

Expected: completes without error, prints MRR for fold 0.

- [ ] **Step 3: Run all 5 distractor types (on available GPUs, can parallelize)**

```bash
# GPU 2
./run.sh finetuning/ft_03_hard_neg/train.py --distractor_type similar_characters --remote --gpu 2
# GPU 3 (when GPU 2 frees up, or if GPU 3 is available)
./run.sh finetuning/ft_03_hard_neg/train.py --distractor_type based_on_adjectives --remote --gpu 3
# ... etc for injected_adjectives, partial_story, all
```

- [ ] **Step 4: Pull results and compare**

```bash
./run.sh pull
```

Compare mean MRR across distractor types. Pick the best for Phase 2.

---

## Task 6: Run Phase 2 — temperature sweep

Use the best distractor type from Phase 1.

- [ ] **Step 1: Run τ ∈ {0.05, 0.07, 0.1} (τ=0.05 is already done from Phase 1)**

```bash
./run.sh finetuning/ft_03_hard_neg/train.py --distractor_type <best_type> --tau 0.07 --remote --gpu 2
./run.sh finetuning/ft_03_hard_neg/train.py --distractor_type <best_type> --tau 0.1  --remote --gpu 3
```

- [ ] **Step 2: Pull and record final results**

```bash
./run.sh pull
```

Record the winning configuration (distractor_type + τ) in the results summary.
