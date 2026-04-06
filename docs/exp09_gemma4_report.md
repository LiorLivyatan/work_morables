# Experiment 09: Gemma 4 Local Summarization
_April 2026 — MORABLES Moral-to-Fable Retrieval_

---

## Background

Our best off-the-shelf retrieval setup (Linq-Embed-Mistral on raw fables) achieves **MRR = 0.210**.
The theoretical oracle — concatenating the fable with its ground-truth moral — reaches **MRR = 0.893**.
That 4× gap is the core problem we are trying to close.

Previous work (Exp 07) showed that **Gemini 2.5 Pro API** summaries can push Config B to MRR = 0.360 — a significant jump. This experiment asks:
> *Can locally-run open-source models (Gemma 4) achieve comparable quality, without any API dependency?*

Gemma 4 was released in April 2026. We test three sizes locally on Apple Silicon (M2 Ultra) via [mlx-lm](https://github.com/ml-explore/mlx-lm).

---

## Models Tested

| Label | HuggingFace ID | Architecture | Active Params | Size on Disk |
|-------|----------------|-------------|---------------|-------------|
| **E2B** | `mlx-community/gemma-4-e2b-it-4bit` | Mixture-of-Experts (MoE) | ~1B | 3.6 GB |
| **E4B** | `mlx-community/gemma-4-e4b-it-4bit` | Mixture-of-Experts (MoE) | ~2B | 5.2 GB |
| **31B** | `mlx-community/gemma-4-31b-it-4bit` | Dense | 31B | 18.4 GB |

**MoE explained:** The E2B and E4B models have a larger total parameter count but route each token through only a subset of "expert" sub-networks. This means inference cost matches the *active* parameter count, not the total — making them fast even on modest hardware.

All models are 4-bit quantized for Apple Silicon via the `mlx-community` HuggingFace org.

---

## Retrieval Setup

**Every single result in the table uses the same retrieval model:**
> **Linq-Embed-Mistral** (`Linq-AI-Research/Linq-Embed-Mistral`) — a Mistral-7B model fine-tuned for dense retrieval via contrastive learning, currently our best embedding model.

The Gemma models are used **only to generate summaries**. The summaries are then embedded with Linq-Embed-Mistral and used for retrieval against the 709-fable corpus.

---

## Experiment Design

### The Three Configurations

| Config | Corpus document | Hypothesis |
|--------|----------------|-----------|
| **A — Summary only** | Just the generated summary | Can a one-sentence summary alone capture the moral for retrieval? |
| **B — Fable + Summary** | Full fable text + `\n\nMoral summary: {summary}` | Does the summary enrich the fable with moral signal? |
| **C — Fable + prefix** | `Fable: {full fable text}` | Does a text prefix alone help? (control condition) |

### The Three Prompts

For each fable, the model is called with a **system prompt** (below) and the user message:
```
Fable: {fable text}
```

---

#### Prompt 1: `direct_moral`

```
You are an expert in fables, parables, and moral philosophy.
When given a fable, extract its moral lesson as a single sentence.
Be as concise as possible.
```

**Goal:** Direct, short extraction. No reasoning chain. Outputs like:
> *"Compassion shown to others is repaid when you least expect it."*

---

#### Prompt 2: `cot_proverb`

```
You are an expert in fables and proverbs. When given a fable, reason step by step:
1. What is the central conflict or situation?
2. What does the outcome reveal about human nature?
3. What abstract principle does this illustrate?

After reasoning, state the moral as a proverb or maxim.
Output ONLY the proverb on the last line. Do NOT include your reasoning in the output.
Be as concise as possible.
```

**Goal:** Chain-of-thought reasoning that culminates in a proverb-style moral. Only the final line is kept. Outputs like:
> *"Kindness shown in times of need is repaid with loyalty when the opportunity arises."*

---

#### Prompt 3: `conceptual_abstract`

```
You are a moral philosopher. When given a fable, reason step by step:
1. What is the central conflict or situation?
2. What does the outcome reveal about human nature?
3. What abstract principle does this illustrate?

After reasoning, output ONLY the abstract moral principle as a single sentence on the last line.
Do NOT include your reasoning in the output. Do NOT add any label or prefix.
Be as concise as possible.
```

**Goal:** Similar CoT structure but framed from a philosophical angle, aiming for more abstract/universal principles. Outputs like:
> *"Compassion and reciprocal kindness can transform even the most antagonistic relationships."*

---

### Post-processing

- `<think>...</think>` blocks (from reasoning traces) are stripped
- `direct_moral`: first non-empty line is kept
- `cot_proverb`, `conceptual_abstract`: last non-empty line is kept (the conclusion)
- Average output lengths: direct_moral ~11 words, cot_proverb ~7 words, conceptual_abstract ~12 words

---

## Results

![Results table](../results/gemma4_results_table.png)

### Key Numbers

| Config | Variant | E2B | E4B | 31B | Gemini API |
|--------|---------|-----|-----|-----|-----------|
| Baseline | — | 0.210 | 0.210 | 0.210 | 0.210 |
| **A** | direct_moral | 0.138 | 0.166 | **0.263** | 0.337 |
| **A** | cot_proverb | 0.119 | 0.085 | 0.201 | 0.305 |
| **A** | conceptual_abs | 0.107 | 0.136 | 0.243 | **0.355** |
| **B** | direct_moral | 0.237 | 0.251 | **0.313** | 0.352 |
| **B** | cot_proverb | 0.220 | 0.235 | 0.288 | **0.360** |
| **B** | conceptual_abs | 0.229 | 0.247 | 0.304 | 0.349 |
| **C** | fable+prefix | 0.210 | 0.210 | 0.210 | — |

---

## Findings

### 1. Clear scaling with model size (Config B)

Config B (fable + summary) improves monotonically with model size:
- E2B: **+0.027** over baseline
- E4B: **+0.041** over baseline
- 31B: **+0.103** over baseline — nearly 50% relative improvement
- Gemini: **+0.150** over baseline

The gap between 31B and Gemini is only **0.047 MRR** — achieved without any API dependency or cloud cost.

### 2. Phase transition in Config A at 31B scale

Config A (summary only) is below baseline for E2B and E4B: the small models' summaries lose too much information for retrieval. At **31B, Config A crosses the baseline** for the first time (0.263 > 0.210). This suggests a qualitative shift in summarization quality — the 31B model captures enough of the moral structure that its summaries alone are useful retrieval documents.

### 3. E4B `cot_proverb` anomaly

E4B's `cot_proverb` Config A drops to **0.085** — dramatically worse than E2B (0.119) and 31B (0.201). Hypothesis: E4B is large enough to attempt multi-step reasoning but not large enough to conclude cleanly. The model emits reasoning text that leaks into the output, producing verbose summaries that embed poorly.

### 4. Config C adds nothing

The instruction prefix `Fable: {text}` produces identical MRR to raw fables across all models (0.210). The prefix adds no retrieval signal — confirming that the gain in Config B comes entirely from the generated summary, not the formatting.

### 5. `direct_moral` is most reliable

Despite being the simplest prompt, `direct_moral` is the most consistent performer across all model sizes in Config B. Chain-of-thought prompts (`cot_proverb`, `conceptual_abstract`) can produce better outputs at Gemini scale but are noisier at smaller sizes.

---

## Conclusions

Local Gemma 4 models are a viable alternative to API-based summarization for this task, especially at 31B scale. The results motivate two clear next steps:

1. **HyDE (Exp 12):** Instead of summarizing fables, use the model to generate a *hypothetical fable* from the moral (query side), directly fixing the vocabulary mismatch.
2. **Contrastive fine-tuning (Exp 14):** Explicitly train an embedding model on (moral, fable) pairs — potentially the largest single improvement.

---

## Reproducibility

```bash
# Generate summaries (one model at a time)
uv run python experiments/09_gemma4_summarization/generate_summaries.py --models gemma4-31b

# Run retrieval evaluation
uv run python experiments/09_gemma4_summarization/run.py \
  --summaries-path experiments/09_gemma4_summarization/results/generation_runs/<run_dir>/golden_summaries.json
```

All generation runs are checkpointed every 10 fables. Use `--resume` to continue an interrupted run.
Results stored in `experiments/09_gemma4_summarization/results/`.
