# Exp 14: Contrastive Fine-Tuning — Detailed Plan
_April 2026_

---

## The Core Idea

Teach an embedding model what "moral similarity" means by training on (moral, fable) pairs, so it learns to pull morals and their fables close together in vector space. This is the most direct way to close the gap between MRR 0.21 (off-the-shelf) and the oracle 0.89.

---

## Why Fine-Tuning Requires Careful Data Handling

If we train on all 709 pairs and test on the same 709, we're cheating — the model memorizes the answers. This is **data leakage**. We use three strategies to avoid it.

---

## Exp 14a: MORABLES-Only with 5-Fold Cross-Validation

### What is 5-fold cross-validation?

Split the 709 pairs into 5 equal groups (~142 each):

| Round | Train on | Test on |
|-------|----------|---------|
| 1 | Groups 2+3+4+5 (567 pairs) | Group 1 |
| 2 | Groups 1+3+4+5 (567 pairs) | Group 2 |
| 3 | Groups 1+2+4+5 (567 pairs) | Group 3 |
| 4 | Groups 1+2+3+5 (567 pairs) | Group 4 |
| 5 | Groups 1+2+3+4 (567 pairs) | Group 5 |

Every pair gets tested exactly once, on a model that **never saw it during training**. Final MRR = average across all 5 test sets. Clean, no leakage.

### Training Signal

**Loss function:** `MultipleNegativesRankingLoss` (standard for bi-encoder retrieval fine-tuning)

For each moral in a training batch, the model sees:
- ✅ Its correct fable (positive)
- ❌ All other fables in the batch (in-batch negatives — free, no extra labeling needed)
- ❌ The MCQA distractors for that moral (hard negatives)

**Hard negatives are crucial.** MORABLES distractors are specifically designed to be similar-but-wrong — `similar_characters`, `based_on_adjectives`, etc. Easy random negatives don't teach the model anything useful. Hard negatives force it to make fine-grained moral distinctions.

**Limitation:** 567 training pairs is small for contrastive learning. Batch size is constrained, meaning fewer in-batch negatives per step.

---

## Exp 14b: External Datasets (Zero-Shot Transfer)

Train on moral/narrative datasets from other domains, evaluate zero-shot on MORABLES. No leakage risk at all.

| Dataset | Size | Format | Domain fit |
|---------|------|--------|-----------|
| STORAL (ACL 2022) | ~3K | story → moral lesson | High |
| Moral Stories (EMNLP 2021) | 12K | short narrative + moral annotation | High |
| ePiC | ~5K | proverb–narrative pairs | Medium |
| Project Gutenberg Aesop (additional) | ~300 | fable–moral pairs | Very high |

**Pros:** No leakage, tests generalization across moral domains.  
**Cons:** Domain shift — modern moral stories vs. ancient fables. The vocabulary and style differ significantly.

---

## Exp 14c: Synthetic Training Data (Novel Contribution)

### The Idea

For each of the 709 MORABLES morals, prompt Gemini/Claude:
> "Write 5 different short fables (~100 words each) that teach this moral: *{moral}*"

This yields **~3,545 (moral, synthetic fable) pairs** for training. Evaluate zero-shot on the real 709 MORABLES pairs — zero leakage.

### Why This Is Interesting

1. **Scale:** 5× more training data than MORABLES-only, at ~$2–3 total API cost
2. **Alignment:** Synthetic fables are conditioned on the exact MORABLES morals, so they're closer in style than STORAL or Moral Stories
3. **Hard negatives:** For each moral, the other 708 morals' synthetic fables act as semi-hard negatives
4. **Paper contribution:** First application of synthetic fable generation for moral retrieval fine-tuning — connects to the broader literature on synthetic data for abstract reasoning

### Expected Behavior

The model trained on synthetic fables should learn a richer mapping from moral → narrative structure, even though the synthetic fables are "fake." At test time, real fables share the same moral vocabulary, so the fine-tuned embedding space should transfer.

---

## Architecture

| Component | Choice |
|-----------|--------|
| Base model | `Linq-Embed-Mistral` (fine-tune from our best baseline) |
| Alternative base | `BAAI/bge-large-en-v1.5` (lighter, faster training) |
| Loss | `MultipleNegativesRankingLoss` |
| Framework | `sentence-transformers` `SentenceTransformerTrainer` |
| Batch size | 16–32 (limited by GPU/MPS memory) |
| Epochs | 3–5 (small dataset → few epochs to avoid overfitting) |
| Evaluation | MRR on held-out fold / MORABLES test set |

---

## Expected Results Hypothesis

| Sub-experiment | Expected MRR | Rationale |
|---------------|-------------|-----------|
| Baseline (no fine-tuning) | 0.210 | Current best |
| 14a: MORABLES CV | 0.28–0.35 | Task-specific but small data |
| 14b: External datasets | 0.22–0.28 | More data, domain shift |
| 14c: Synthetic data | 0.30–0.40 | Best data alignment + scale |

Fine-tuning should be our biggest result. The narrative: **"even modest task-specific supervision — synthetic or cross-domain — dramatically closes the gap that off-the-shelf embeddings leave open."**

---

## Implementation Notes

- Framework: `sentence-transformers >= 3.0` has a clean `SentenceTransformerTrainer` API
- Hard negatives: load from MORABLES MCQA config, format as `InputExample(texts=[moral, fable, distractor])`
- Cross-validation: use `sklearn.model_selection.KFold`, checkpoint best epoch per fold
- Synthetic generation: reuse Exp 07/09 Gemini API infrastructure, new prompt only
- Estimated implementation time: 3–4 days total for all three sub-experiments
