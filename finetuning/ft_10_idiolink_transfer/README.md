# ft_10 — IdioLink Transfer Experiments

## Motivation

Both MORABLES and IdioLink are concept-level retrieval tasks: the query expresses an
idea in one surface form, and the relevant documents express the *same idea* in a
different surface form. In MORABLES a moral statement ("The strong should help the
weak") must match a fable that embodies it; in IdioLink a PIE usage ("cut and dried")
must match documents expressing the same literal or idiomatic meaning.

The hypothesis is that training on IdioLink teaches Linq-Embed-Mistral a general
inductive prior — *align meaning across form* — that transfers to MORABLES and
either improves fine-tuning efficiency or reduces the gap to the oracle ceiling.

**IdioLink training data available:**
- 440 triplets `(query, positive, [5 negatives])` across 22 PIEs
- Soft negatives: documents from other PIEs
- Hard negatives: same PIE, opposing literal/idiomatic interpretation
- Complementary structured data: `triplets_train_span.jsonl` (span-level queries)

**Current MORABLES best:** ft_09 (Linq+LoRA + FN masking) — MRR ≈ 0.448 (fold 4),
mean ~0.435 across all folds.

---

## Options

### Option A — Zero-Shot Transfer (`option_a_zero_shot/`)

**Setup:** Fine-tune Linq exclusively on IdioLink training data (440 triplets, no
MORABLES data). Evaluate the resulting model on the full MORABLES 5-fold CV.

**Purpose:** Pure diagnostic. Does concept-level idiom alignment generalize to
moral-fable retrieval with zero task-specific signal?

**Expected outcome:** Better than zero-shot baseline (0.210) but below ft_02
(0.416). Establishes a meaningful transfer lower bound.

**Cost:** One Linq training run (~1–2 hours on a single RTX 3090). No fold loop needed.

---

### Option B — Sequential Pre-training (`option_b_sequential/`)

**Setup:**
1. Stage 1: fine-tune Linq on IdioLink (440 triplets) — same LoRA config as ft_09
2. Stage 2: continue fine-tuning the Stage 1 model on MORABLES (5-fold CV, ft_09
   settings: MaskedInfoNCE, FN masking, Type-2 moral negatives)

**Purpose:** Tests whether an IdioLink-initialized model reaches higher MRR on MORABLES
than a Linq-initialized model — and whether the gap closes faster (fewer MORABLES
epochs needed).

**Key design question:** Should Stage 1 merge the LoRA adapter before Stage 2, or
keep LoRA parameters and continue training them on MORABLES? Merging first gives
Stage 2 a better base model; not merging risks adapter interference.

**Expected outcome:** MRR improvement over ft_09, or comparable MRR with faster
convergence (earlier early-stopping epoch).

**Cost:** 1 pre-training run + 5 MORABLES fold runs (~4–6 hours total).

---

### Option C — Data Mixing (`option_c_mixing/`)

**Setup:** Mix IdioLink triplets directly into the MORABLES training batches at a
fixed ratio. A single training run per fold. No staged pipeline.

**Why IdioLink negatives are richer than STORAL (ft_04/ft_05):**
- STORAL: (moral, story) pairs with only in-batch negatives — same structure as
  MORABLES training, just out-of-domain content.
- IdioLink: explicit hard negatives (same PIE, wrong interpretation) — already
  structured as triplets with meaningful contrastive signal.

**Ratio options to sweep (inspired by ft_05 findings):**
- 50 pairs (~11% of MORABLES)
- 110 pairs (25%)
- 220 pairs (50%)
- 440 pairs (100% IdioLink, 1:1 mix)

**Expected outcome:** Moderate gains at low ratios; degradation at high ratios
(domain shift). Based on ft_05 (STORAL), sweet spot is likely well below 50%.

**Cost:** 4 ratio × 1 fold = 4 short runs for sweep, then 5 folds at best ratio.

---

### Option D — IdioLink Curriculum: Explicit Hard Negatives (`option_d_hard_neg/`)

**Setup:** Don't use IdioLink text. Use its *curriculum design principle*: for each
MORABLES training query (moral), include a fable associated with a near-duplicate
moral as an **explicit hard negative** in the loss (rather than just masking it out
of the denominator as in ft_09).

**ft_09 vs Option D:**
- ft_09: near-duplicate moral pairs → masked to `-inf` in denominator (excluded from
  loss signal entirely)
- Option D: near-duplicate moral pairs → actively pushed apart as hard negatives
  (kept in denominator with their own loss contribution, similar to IdioLink's
  same-PIE hard negatives)

**Implementation:** Modify `MaskedInfoNCELoss` to use a weighted hard-negative term
instead of a mask. The similarity threshold (0.85) stays from ft_09; the change is
in how boundary pairs are treated.

**Expected outcome:** Harder training signal than ft_09; may improve or may destabilize
depending on threshold and weighting. A natural ablation of ft_09.

**Cost:** 5 fold runs — same cost as ft_09.

---

## Recommended Execution Order

1. **Option A first** — cheap, confirms whether IdioLink signal transfers at all
2. **Option B** — if A shows transfer, sequential pre-training is the main bet
3. **Option C** — parallel to B or as follow-up if B shows diminishing returns
4. **Option D** — if ft_09 vs ft_08 gap closes further, worth isolating

## Run Commands

```bash
# Option A (zero-shot transfer — no folds needed)
./run.sh finetuning/ft_10_idiolink_transfer/option_a_zero_shot/train.py --remote --gpu 2

# Option B (sequential — stage 1 then stage 2)
./run.sh finetuning/ft_10_idiolink_transfer/option_b_sequential/pretrain.py --remote --gpu 2
./run.sh finetuning/ft_10_idiolink_transfer/option_b_sequential/finetune.py --remote --gpu 2

# Option C (mixing ratio sweep, fold 0)
./run.sh finetuning/ft_10_idiolink_transfer/option_c_mixing/train.py --fold 0 --ratio 110 --remote --gpu 2

# Option D (hard negatives, full 5-fold)
./run.sh finetuning/ft_10_idiolink_transfer/option_d_hard_neg/train.py --remote --gpu 2
```
