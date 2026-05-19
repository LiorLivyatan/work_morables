# Fine-Tuning Experiments

Contrastive fine-tuning of bi-encoder models on the MORABLES moral-to-fable
retrieval task. Each sub-directory is a self-contained experiment.

## Experiments

| Dir | Stage | Description | Status |
|-----|-------|-------------|--------|
| `ft_00_overfit/` | 0 | Train=test upper bound (sanity check) | Done |
| `ft_01_5fold_cv/` | 1 | 5-fold CV, BGE-base, raw fables | Done |
| `ft_02_linq_5fold_cv/` | 2 | 5-fold CV, Linq 7B + LoRA, GroupKFold fix | Done |
| `ft_03_hard_neg/` | 3 | Linq + custom InfoNCE (Type-2 moral negatives) | Done |
| `ft_04_storal_augment/` | 4 | Linq + STORAL data augmentation (fold 0) | Done |
| `ft_05_mixing_ratio/` | 5 | BGE-base STORAL mixing ratio sweep | Done |
| `ft_06_random_search/` | 6 | Hyperparameter random search | Done |
| `ft_07_storal_transfer/` | 7 | STORAL pre-training → MORABLES fine-tune | Done |
| `ft_08_clean_sanity/` | 8 | Linq+LoRA clean baseline (ft_03 settings, full data) | Done |
| `ft_09_fn_masking/` | 9 | Linq+LoRA + false-negative masking via sim matrix | Done |
| `ft_10_idiolink_transfer/` | 10 | IdioLink pre-training transfer to MORABLES (4 options) | In progress |

## Shared Library (`lib/`)

| Module | Responsibility |
|--------|---------------|
| `lib/data.py` | Load MORABLES pairs; `build_doc_text()` for document representation |
| `lib/trainer.py` | `train_model()` — SentenceTransformer fine-tuning with caching |
| `lib/eval.py` | `evaluate()` — encode + compute MRR/R@k with embedding caching |

## Adding a New Experiment

1. Copy an existing experiment dir as a template
2. Update `config.yaml` for your specific setup
3. Write `train.py` using the shared `lib/` — keep it a thin orchestrator
4. Add a `README.md` documenting purpose, how to run, and expected results
5. Add an entry to this table

---

## How It All Works — Technical Walkthrough

### 1. How 7B/8B Models Fit on a 24 GB Card

Linq-Embed-Mistral has 7B parameters. In float32 that's 28 GB — too large.
Three techniques stacked together bring peak VRAM to ~20–21 GB:

**bfloat16 precision** (`torch_dtype: auto` in every config)
7B × 2 bytes = 14 GB base weight footprint. Modern GPUs handle bf16 natively.

**LoRA — only train 0.76% of parameters** (every Linq experiment)
```yaml
lora:
  r: 64
  alpha: 128
  target_modules: [q_proj, k_proj, v_proj, o_proj]
```
All 7B weights are frozen. Instead, for each attention matrix W we add two small
rank-64 matrices A and B. The effective update is `W + A×B`. Only A and B receive
gradients. Frozen weights need no optimizer state, which is the main memory saving
(Adam keeps a momentum and variance tensor per trainable parameter — 2× the size
of the weights themselves).

`q/k/v/o_proj` are the self-attention projections — the most semantically important
part of the transformer. The FFN layers stay frozen.

**Gradient checkpointing + small batch**
```yaml
batch_size: 4
gradient_accumulation_steps: 4
gradient_checkpointing: true
```
Gradient checkpointing discards intermediate activations during the forward pass
and recomputes them during backprop. ~30% slower but halves activation memory.
`batch_size=4` with `gradient_accumulation_steps=4` simulates an effective batch
of 16 while keeping only 4 samples on GPU at once.

**Rough VRAM budget for Linq 7B:**

| Item | ~VRAM |
|------|-------|
| Model weights (bf16) | 14 GB |
| LoRA adapter + optimizer state (Adam) | 3–4 GB |
| Activations (batch=4, seq=512, checkpointed) | 3–4 GB |
| **Total** | **~20–21 GB** |

BGE-base (110M params) uses no LoRA — full fine-tune fits in ~2 GB, runs at `batch_size=32`.

---

### 2. The Loss Function

#### Basic version: MultipleNegativesRankingLoss (ft_01, ft_02)

The model trains on (moral, fable) pairs. With a batch of B pairs, every
other fable in the batch acts as a free negative — no mining needed.

For each moral_i the model computes cosine similarity against all B fables
in the batch, then runs cross-entropy: the correct fable should score highest.
This is InfoNCE loss:

```
L = -log( exp(sim(q_i, d_i) / τ) / Σ_j exp(sim(q_i, d_j) / τ) )
```

**Temperature τ=0.05** divides all similarities by 0.05 (= multiplies by 20),
making the distribution very sharp. Small similarity differences produce large
loss contributions, forcing the model to be precise rather than approximately correct.

#### Extended version: custom MaskedInfoNCELoss (ft_03+, ft_09)

Two additions on top of the basic loss:

**Type-2 negatives — moral vs moral** (`train.py` in ft_03+):
```python
moral_sim = torch.mm(moral_emb, moral_emb.T) / τ   # (B, B)
moral_sim.masked_fill_(torch.eye(B, ...), -inf)      # exclude self
all_logits = torch.cat([fable_sim, moral_sim], dim=1)
```
The model must also rank the query moral above other morals in the batch.
Without this, morals can collapse into a tight cluster that makes the
fable-matching task trivially easy during training but brittle at test time.

**False-negative masking** (ft_09 only):
```python
batch_sim = moral_sim_matrix[labels][:, labels]    # precomputed BGE-large sims
false_neg  = batch_sim > fn_mask_threshold         # sim > 0.85 = near-duplicate
false_neg.fill_diagonal_(False)                    # keep the actual positive
fable_sim  = fable_sim.masked_fill(false_neg, -inf)
```
Problem: if moral_i = "Union is strength" and moral_j = "Union gives strength"
appear in the same batch, without masking the model is penalized for being
attracted to fable_j when given query_i — but fable_j IS a valid answer.
Fix: a 709×709 cosine similarity matrix (computed once with BGE-large, cached
at `data/processed/moral_sim_matrix.npy`) tells us at training time which
in-batch pairs are semantically equivalent, and we exclude them from the denominator.

---

### 3. Cross-Validation: GroupKFold

27 morals in the dataset appear for multiple fables (same text, multiple stories).
Standard KFold could split these across train and test — the model would memorize
the string and report inflated MRR.

GroupKFold assigns every instance of the same moral text to the same fold:

```python
text_to_group: dict[str, int] = {}
for t in moral_texts:
    if t not in text_to_group:
        text_to_group[t] = len(text_to_group)
    groups.append(text_to_group[t])

GroupKFold(n_splits=5).split(indices, groups=groups)
```

All 709 fables stay in the corpus (retrieval targets) for every fold.
Only the *query* set is split — ~568 train queries, ~141 test queries per fold.

---

### 4. BestAdapterCallback — Saving Peak Performance

7B models overfit quickly on 568 training pairs. Peak MRR is typically at
epoch 2; by epoch 5 the model has overfit and MRR drops.

The standard HF `load_best_model_at_end=True` restores from a checkpoint file
on disk — for a 7B model that takes minutes and can cause OOM (two copies in
memory at once).

Our solution: snapshot only the LoRA adapter weights (~200 MB) to CPU RAM
at every MRR improvement:

```python
def on_evaluate(self, args, state, control, metrics=None, **kwargs):
    if mrr > self.best_mrr:
        self.best_state = {
            k: v.detach().cpu().clone()
            for k, v in self.peft_model.state_dict().items()
            if "lora" in k.lower()   # ~200 MB, not 14 GB
        }
```

At training end, `restore()` copies these CPU tensors back to GPU — fast,
no disk I/O, no double-load OOM.

EarlyStoppingCallback (patience=3) halts training when MRR stops improving,
so we don't waste GPU time on epochs that will be discarded anyway.

---

### 5. Subprocess Orchestration — Guaranteed GPU Memory Release

After fold 0 completes, calling `del model; gc.collect(); torch.cuda.empty_cache()`
is not enough. PyTorch/CUDA can leave allocations in its internal memory pool,
and when fold 1 tries to load a fresh 14 GB model it crashes with OOM.

The fix: each fold runs as a completely separate OS process.

```python
if args.fold is None and len(folds) > 1:
    for fold in folds:
        subprocess.run(
            [sys.executable, __file__, "--fold", str(fold["fold"])] + extra,
            check=True,
        )
```

When an OS process exits, all CUDA memory is unconditionally returned to the
driver. The parent process launches folds sequentially and collects results
from the per-fold JSON files written by each child.

---

### 6. Full Training Pipeline — Summary

```
709 morals × 709 fables (doc_mode: fable_plus_summary)
    ↓ GroupKFold (5 folds, grouped by moral text)
    ↓ ~568 train / ~141 test per fold
    ↓ subprocess per fold — guaranteed GPU memory release between folds

Linq-Embed-Mistral (7B, bfloat16) + LoRA r=64 α=128
    → 14 GB base weights (frozen) + ~53M trainable LoRA params
    → gradient_checkpointing: halves activation memory
    → batch_size=4 × grad_accum=4 = effective batch 16

MaskedInfoNCELoss (τ=0.05)
    → Type 1: moral_i vs all in-batch fables
    → Type 2: moral_i vs other in-batch morals (prevents collapse)
    → FN masking: exclude pairs with BGE-large sim > 0.85

BestAdapterCallback: CPU-snapshots LoRA weights at each MRR peak
EarlyStoppingCallback (patience=3): stops at ~epoch 4–5
merge_and_unload(): folds LoRA back into base model at save time
    → output is a standard SentenceTransformer, no PEFT needed at inference
```

---

## Result Reference

| Experiment | Model | MRR@10 | Notes |
|------------|-------|--------|-------|
| Zero-shot | Linq-Embed-Mistral | 0.210 | raw fables |
| Zero-shot | Linq + Gemini summaries | 0.360 | fable+summary |
| ft_02 | Linq+LoRA | 0.416 ± 0.033 | trained on raw, eval on fable+summary |
| ft_03 basic | Linq+LoRA | 0.438 ± 0.043 | + Type-2 moral negatives |
| ft_08 | Linq+LoRA | 0.440 ± 0.048 | clean baseline (same as ft_03) |
| ft_09 | Linq+LoRA | 0.435 ± 0.049 | + false-negative masking (no gain) |
| Oracle upper bound | — | 0.893 | fable concatenated with its own moral |
