# Deep Walkthrough: `scripts/07_llm_experiments.py`

This script picks up where `05_model_comparison.py` left off. Where script 05 benchmarks many embedding models head-to-head, script 07 explores **three fundamentally different strategies** for improving over the best plain embedding model. It asks: *can we do better than a pure vector search?*

---

## 1. Big Picture: Three Approaches

```
╔══════════════════════════════════════════════════════════════════╗
║  Best embedding model from script 05 (Stella 1.5B, s2p_query)  ║
╚══════════════════════════════════════════════════════════════════╝
         │                    │                    │
         ▼                    ▼                    ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │   A     │          │   B     │          │   C     │
   │ Rerank  │          │ Summar. │          │ Enrich  │
   │ top-K   │          │ corpus  │          │ corpus  │
   │ w/Gemini│          │ w/local │          │ w/GT    │
   │  CoT    │          │  Qwen   │          │ morals  │
   └─────────┘          └─────────┘          └─────────┘
  API-based,            Local LLM,           No LLM —
  post-retrieval        pre-retrieval        upper bound
```

| Approach | What it changes | When LLM runs |
|---|---|---|
| A — CoT Reranking | **Query side**: asks Gemini to re-sort top-K embedding results | After retrieval (post-hoc) |
| B — Summarisation | **Corpus side**: replaces each fable text with an LLM-generated moral summary | Before retrieval (offline) |
| C — Enriched Corpus | **Corpus side**: appends the ground-truth moral directly to each fable | No LLM needed |

Approach C is the theoretical ceiling for Approach B — it answers "if the LLM always generated a perfect moral summary, how much would retrieval improve?"

---

## 2. Imports

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
```

- `matplotlib.use("Agg")` — switches matplotlib to a non-interactive backend that renders to file without a display. Critical when running on a headless server (no GUI). Must be called **before** `import matplotlib.pyplot` — this is an initialization setting.
- `cosine_similarity` is imported at the module level but used inside `retrieval_utils.compute_metrics()`. The import here is a leftover or for potential direct use.
- `SentenceTransformer` is imported at the top level because it's used in the shared helpers that all three approaches call.
- The heavy model-specific imports (`torch`, `transformers`, `google.generativeai`) are **deferred inside functions** — they're only imported when that approach actually runs. This avoids crashing the script if a library isn't installed when you only want to run one approach.

---

## 3. Paths and Directory Setup

```python
DATA_DIR     = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR  = Path(__file__).parent.parent / "results"
RUNS_DIR     = RESULTS_DIR / "runs"           # output from script 05
LLM_RUNS_DIR = RESULTS_DIR / "llm_runs"       # output from this script
SUMMARIES_CACHE = RESULTS_DIR / "summaries"   # cached LLM-generated summaries

for _d in (LLM_RUNS_DIR, SUMMARIES_CACHE):
    _d.mkdir(parents=True, exist_ok=True)
```

Two separate output trees:
- `results/runs/` — owned by script 05 (embedding benchmarks). Approach A reads from here.
- `results/llm_runs/` — owned by this script (LLM experiments).
- `results/summaries/` — persistent cache for Approach B's generated summaries. Generating summaries with a local LLM is slow, so this cache means you only pay that cost once per (model, prompt) combination.

---

## 4. `.env` Loader

```python
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    with open(_env) as f:
        for _line in f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _v = _v.split("#")[0].strip()   # strip inline comments
                os.environ.setdefault(_k.strip(), _v)
```

A minimal `.env` file parser written from scratch (without `python-dotenv`). Key details:
- `split("=", 1)` — the `1` means only split on the **first** `=`. This handles values that contain `=` themselves (e.g., base64-encoded API keys).
- `_v.split("#")[0].strip()` — strips inline comments (`KEY=value  # this is a comment`).
- `os.environ.setdefault(...)` — only sets the variable if it's **not already set**. This means real environment variables (e.g., set in your shell or CI) take precedence over the `.env` file, which is correct behavior.
- Used by Approach A to load `GEMINI_API_KEY`.

---

## 5. Global Configuration Constants

```python
DEFAULT_EMBED_MODEL  = "dunzhang/stella_en_1.5B_v5"
DEFAULT_QUERY_PROMPT = "s2p_query"
DEFAULT_TOP_K        = 20
```

These encode a key result from script 05: the **best embedding model found was Stella 1.5B** with the `s2p_query` prompt. This is hardcoded as the default here so the LLM experiments automatically build on top of the best baseline.

`DEFAULT_TOP_K = 20` — Approach A reranks the top 20 candidates. This is a deliberate choice:
- Too small (e.g., 5): you might exclude the correct answer before Gemini even sees it.
- Too large (e.g., 100): prompt gets very long, Gemini's reasoning quality degrades, and API costs increase.
- 20 is a common sweet spot in the LLM-reranking literature.

---

## 6. Approach B: Qwen Models

```python
QWEN_MODELS = {
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B-Instruct",
    "qwen3.5-2b":   "Qwen/Qwen3.5-2B-Instruct",
    "qwen3.5-4b":   "Qwen/Qwen3.5-4B-Instruct",
    "qwen3.5-9b":   "Qwen/Qwen3.5-9B-Instruct",
}
```

Four sizes of the same model family. The experiment systematically tests whether larger generation models produce better moral summaries for retrieval. Qwen3.5 Instruct models are **chat-tuned** — they follow instructions, which is needed for structured outputs like "write one sentence".

---

## 7. Approach B: Summary Prompt Templates

```python
SUMMARY_PROMPTS = {
    "direct":   "What is the moral of this fable? Answer in one sentence.\n\n...",
    "detailed": "Read this fable and identify the deeper lesson...",
    "cot":      "Think step by step... MORAL: [one sentence]",
    "few_shot": "Here are two examples... Now write the moral...",
}
```

Four prompting strategies, each testing a different theory about how to elicit high-quality moral summaries:

**`direct`** — minimal instruction. Tests the model's default behavior and vocabulary for moral summarization. Outputs a plain sentence.

**`detailed`** — explicitly asks the model to identify the virtue, vice, or life wisdom. Tests whether richer framing improves precision of the extracted moral. The phrase "Be specific about what virtue, vice, or life wisdom" pushes the model toward moral-philosophy vocabulary, which may better match the query morals.

**`cot`** (Chain-of-Thought) — makes the model reason in three explicit steps before stating the moral:
1. What happens in the story?
2. What conflict is illustrated?
3. What abstract principle does it demonstrate?

The hypothesis: forcing explicit intermediate reasoning steps causes the model to "commit" to a clearer interpretation before writing the final sentence, reducing vague outputs. The structured `MORAL: [sentence]` ending also makes parsing easier (see `_generate_summary`).

**`few_shot`** — provides two worked examples (Tortoise/Hare and Crow/Fox) before asking for the new fable's moral. The examples calibrate both the **format** (one sentence) and the **style** (abstract moral principle, not plot summary). Few-shot prompting is particularly effective at format control.

---

## 8. Approach C: Corpus Templates

```python
CORPUS_TEMPLATES = {
    "plain":       "{fable}",
    "fable_moral": "Fable/Parable: {fable}; Moral: {moral}",
    "fable_tag":   "Fable: {fable}",
}
```

Three progressive levels of information added to each corpus document:

- `plain` — raw fable text. Equivalent to the baseline from script 05.
- `fable_tag` — just adds a `"Fable: "` prefix. Tests whether domain-labeling the document type changes the embedding.
- `fable_moral` — the full enriched document: fable text + semicolon + the **ground-truth moral**. This is the upper bound: it answers "if an LLM always summarized fables perfectly, how much would retrieval improve?" The `{moral}` slot is filled with the actual correct moral from the dataset.

---

## 9. Module-Level Data Loading

```python
with open(DATA_DIR / "fables_corpus.json") as f:  fables_corpus = json.load(f)
with open(DATA_DIR / "morals_corpus.json") as f:  morals_corpus = json.load(f)
with open(DATA_DIR / "qrels_moral_to_fable.json") as f:  qrels_m2f = json.load(f)

gt_m2f: dict[int, int] = {}
for qrel in qrels_m2f:
    moral_idx = int(qrel["query_id"].split("_")[1])
    fable_idx = int(qrel["doc_id"].split("_")[1])
    gt_m2f[moral_idx] = fable_idx
```

Notice: this script reads `qrels_moral_to_fable.json` (moral is the query), while script 05 reads `qrels_fable_to_moral.json` and then **inverts it**. This is the same ground truth, just stored differently — this script uses the already-inverted version.

```python
fable_to_moral_text: dict[int, str] = {
    fable_idx: moral_texts[moral_idx]
    for moral_idx, fable_idx in gt_m2f.items()
}
```

This reverse mapping (fable index → moral text) is needed only for Approach C to fill the `{moral}` slot in the `fable_moral` template. It's constructed once at module level so it's available to `run_approach_c` without being recomputed.

---

## 10. Shared Helpers

### `load_embed_model()`

```python
def load_embed_model(model_id: str) -> SentenceTransformer:
    return SentenceTransformer(model_id, trust_remote_code=True)
```

`trust_remote_code=True` is needed because Stella executes custom Python code for its `s2p_query` prompt template.

### `encode_morals()` and `encode_corpus()`

```python
def encode_morals(model, query_prompt):
    kwargs = dict(show_progress_bar=True, normalize_embeddings=True,
                  convert_to_numpy=True, batch_size=32)
    if query_prompt:
        kwargs["prompt_name"] = query_prompt
    return model.encode(moral_texts, **kwargs)

def encode_corpus(model, texts):
    return model.encode(texts, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
```

Two separate functions that make the **asymmetry explicit**: morals are encoded as queries (with optional prompt), corpus items are encoded as documents (plain). This matches Stella's intended use: `s2p_query` stands for "Sentence-to-Passage query" — the query gets a special prompt, the passages don't.

`normalize_embeddings=True` ensures L2-normalization, which makes cosine similarity = dot product — the same contract as script 05.

---

## 11. Approach A: CoT Reranking with Gemini

### Concept

```
Moral (query) ─► embedding search ─► top-20 fable candidates
                                              │
                                              ▼
                                        Gemini (CoT prompt)
                                              │
                                              ▼
                                     Reranked top-20 fables
                                              │
                                              ▼
                                         Metrics
```

### The Reranking Prompt

```python
COT_RERANK_PROMPT = """\
You are an expert in fables, parables, and moral reasoning.
...
MORAL LESSON:
{moral}

CANDIDATE FABLES (numbered):
{fables_list}

Think step by step:
1. What does this moral teach?
2. What kind of story scenario would illustrate this?
3. Which candidate fable most directly embodies this principle?

After your reasoning, output ONLY a line starting with "RANKING:" followed by
fable numbers in order from best to worst match, separated by commas.
Example: RANKING: 3, 1, 5, 2, 4
"""
```

The prompt is carefully structured:
- **Role framing**: "You are an expert in fables" — activates relevant knowledge.
- **Three-step CoT**: forces the model to reason before ranking. Without this, LLMs tend to anchor on superficial text similarity.
- **Structured output**: "ONLY a line starting with RANKING:" — makes parsing robust.
- **Example**: shows the expected format to prevent format variation.

Each fable is truncated to 400 characters to keep the prompt short and reduce cost. The full prompt for 20 candidates is roughly 2000–3000 tokens.

### `_find_latest_predictions()`

```python
def _find_latest_predictions(model_run_dir):
    if model_run_dir and model_run_dir.exists():
        return model_run_dir
    run_dirs = sorted(RUNS_DIR.iterdir()) if RUNS_DIR.exists() else []
    return run_dirs[-1]
```

If no `--run-dir` is passed on the CLI, this auto-discovers the most recent run from script 05 by sorting the timestamped directory names. ISO timestamps sort lexicographically = chronologically, so `sorted()` returns them oldest-first and `[-1]` picks the newest.

### `_best_predictions_file()`

```python
best = max(results, key=lambda r: r.get("MRR", 0))
run_key = best["run_key"]
preds_file = run_dir / "predictions" / f"{run_key}.json"
```

Scans `results.json` from script 05 to find which run had the highest MRR, then loads that run's prediction file. The prediction file contains `top_k_indices` per query — the pre-ranked candidate list that Gemini will rerank.

### `_rerank_with_gemini()`

```python
for line in text.strip().split("\n"):
    if line.strip().upper().startswith("RANKING:"):
        ranking_str = line.split(":", 1)[1].strip()
        ranked_nums = []
        for part in ranking_str.split(","):
            part = part.strip().rstrip(".")
            try:
                num = int(part)
                if 1 <= num <= len(candidate_fables):
                    ranked_nums.append(num - 1)
            except ValueError:
                continue
        ranked_global = [candidate_indices[i] for i in ranked_nums if i < len(candidate_indices)]
        for idx in candidate_indices:
            if idx not in ranked_global:
                ranked_global.append(idx)
        return ranked_global
```

Parsing is defensive by design:
- `line.strip().upper()` — handles both `RANKING:` and `ranking:`.
- `rstrip(".")` — handles `RANKING: 3, 1, 5.` (trailing punctuation).
- `1 <= num <= len(candidate_fables)` — validates bounds (rejects hallucinated numbers).
- The fallback loop at the end: any candidates Gemini didn't mention are appended in their original embedding order. This ensures the output always covers all candidates even if Gemini's response was partial.
- On API failure (exception): `return list(candidate_indices)` falls back to the original embedding ranking — no crash, just no improvement for that query.

### Baseline Computation

```python
embed_ranks = []
for m_idx in moral_indices:
    top_k_indices = queries[m_idx]["top_k_indices"][:top_k]
    correct = gt_m2f.get(m_idx)
    rank = top_k_indices.index(correct) if correct in top_k_indices else len(fable_texts)
```

Before reranking, the baseline is computed: what is the embedding model's rank within the **top-K candidates**? If the correct fable isn't in the top-K, `rank = len(fable_texts)` (a large penalty). This baseline is tighter than full-corpus MRR because it measures what's achievable by reranking — you can never recover a correct answer that wasn't in the top-K.

```python
time.sleep(0.5)  # rate limiting
```

0.5 seconds between Gemini calls limits the rate to ~120 queries/minute, staying safely below typical API quota limits.

---

## 12. Approach B: LLM Moral Summarisation

### Concept

```
                    Approach B Architecture
┌──────────────────────────────────────────────────────────────┐
│  OFFLINE (once per model×prompt combination)                  │
│                                                               │
│  Fable text ──► Qwen 3.5 (0.8B/2B/4B/9B) ──► moral summary  │
│  "The fox and the..."    "direct"/"cot"...  "Greed leads..."  │
│                                                               │
│  Cached in: results/summaries/{model_label}/{prompt}.json     │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  RETRIEVAL (per combination)                                  │
│                                                               │
│  Moral query ──► Stella embed ──► query vector               │
│  Summary corpus ──► Stella embed ──► corpus vectors           │
│  cosine similarity ──► metrics                               │
└──────────────────────────────────────────────────────────────┘
```

### Loading the Generation Model

```python
def _load_generation_model(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.train(False)
    return tokenizer, model
```

Key decisions:
- `torch.float16` on GPU, `torch.float32` on CPU — float16 halves memory usage and is supported by MPS/CUDA. On CPU, float16 can be slower than float32 due to lack of hardware support.
- `device_map="auto"` — HuggingFace automatically places layers across available devices. For large models (9B) that don't fit in a single GPU's VRAM, it can split across multiple GPUs or spill to CPU. For single-GPU scenarios, it just puts everything on GPU.
- `model.train(False)` — sets the model to inference mode (equivalent to `.eval()`). Disables dropout and batch normalization updates. Required for deterministic, correct inference.

### Generating a Summary

```python
def _generate_summary(tokenizer, model, fable_text, prompt_template):
    prompt = prompt_template.format(fable=fable_text)
    messages = [{"role": "user", "content": prompt}]

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,        # Qwen3.5-specific: disable CoT tokens
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
```

`apply_chat_template` converts the `{"role": "user", "content": ...}` dict into the model's expected raw text format (which includes special tokens like `<|im_start|>user`, `<|im_end|>`, etc. for Qwen). Different models have different chat templates, and this API handles all of them correctly.

`enable_thinking=False` — Qwen3.5 has a special "thinking mode" where it generates a long internal reasoning trace inside `<think>...</think>` tags before the final answer. For short-output tasks like one-sentence moral extraction, this is wasteful. The `try/except TypeError` handles older model versions that don't support this parameter.

```python
out = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
```

- `max_new_tokens=80` — limits output to ~80 tokens. A one-sentence moral is typically 10–20 tokens. The cap prevents runaway generation.
- `do_sample=False` — greedy decoding (always pick the highest-probability token). Makes outputs **deterministic** — the same fable+prompt always produces the same summary. Important for reproducibility and for the cache to be valid across runs.
- `pad_token_id=tokenizer.eos_token_id` — tells the model to treat the end-of-sequence token as padding. Prevents warnings when generating with batch_size=1 and no explicit pad token.

```python
response = tokenizer.decode(
    out[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
).strip()
```

`out[0]` is the full token sequence (prompt + generated tokens). `inputs["input_ids"].shape[1]` is the prompt length. Slicing `[prompt_length:]` extracts only the **newly generated tokens** — the model's response, not the input prompt echoed back.

### Post-Processing the Response

```python
# Strip any <think>...</think> blocks
response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

# Extract from CoT-style "MORAL: ..." output
for line in response.split("\n"):
    if line.strip().upper().startswith("MORAL:"):
        response = line.split(":", 1)[1].strip()
        break

# Take first non-empty sentence
lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
return lines[0] if lines else response
```

Three-stage cleaning:
1. `re.DOTALL` makes `.` match newlines, so `<think>long multi-line reasoning</think>` is stripped even if it spans many lines.
2. The `MORAL:` extraction handles the `cot` prompt style which formats output as `MORAL: [sentence]`.
3. Taking the first non-empty line handles all remaining cases: the model may output a one-liner directly, or it may follow with extra explanation (which we discard).

### Cache System

```python
cache_file = cache_dir / f"{prompt_label}.json"

if cache_file.exists():
    with open(cache_file) as f:
        cached = json.load(f)
    if len(cached) == len(fable_texts):
        return [cached[str(i)] for i in range(len(fable_texts))]
    print(f"    [cache] Stale ({len(cached)}/{len(fable_texts)}) — regenerating")
```

Cache structure: `results/summaries/{model_label}/{prompt_label}.json` → a dict with string keys `"0"`, `"1"`, ... (JSON doesn't support integer keys, so indices are stored as strings). The length check ensures cache validity: if the fables corpus was later expanded, a partial cache is detected and regenerated instead of silently returning incomplete results.

```python
del gen_model, tokenizer
if device == "mps":  torch.mps.empty_cache()
elif device == "cuda": torch.cuda.empty_cache()
```

After generating summaries, the generation model is explicitly deleted and GPU memory is freed before loading the embedding model for the next step. Without this, both a 9B generation model and a 1.5B embedding model would need to coexist in memory simultaneously.

### The 4×4 Experiment Grid

```python
for model_label, model_id in QWEN_MODELS.items():          # 4 model sizes
    for prompt_label, prompt_template in SUMMARY_PROMPTS.items():  # 4 prompts
        summaries = _load_or_generate_summaries(...)
        summary_embs = encode_corpus(embed_model, summaries)
        m = compute_metrics(moral_embs, summary_embs, gt_m2f)
```

16 combinations total (4 models × 4 prompts). The embed model is **shared** across all 16 — moral embeddings are computed once before the loop. Only the corpus embeddings change (different summaries for each combination). This is efficient: `encode_morals()` is called only once.

---

## 13. Approach C: Enriched Corpus Embedding

### Concept

```python
corpus_texts = [
    template.format(
        fable=fable_texts[i],
        moral=fable_to_moral_text.get(i, ""),
    )
    for i in range(len(fable_texts))
]
```

For the `fable_moral` template, each document becomes:
```
"Fable/Parable: [full fable text]; Moral: [ground-truth moral]"
```

When the embedding model encodes this, it sees both the narrative content of the fable AND the abstract moral principle explicitly stated. This dramatically improves retrieval because the moral query now has vocabulary overlap with the corpus document.

The `fable_to_moral_text.get(i, "")` fallback handles fables that might not have a ground-truth moral in the dataset (a defensive measure for data integrity).

### Why "Fable/Parable" not just "Fable"?

The string `"Fable/Parable: "` in the template signals to the embedding model that this is a narrative with a lesson. The slash construction covers both categories. Stella was trained on diverse text and may have learned that documents tagged this way relate to moral content.

### Interpreting the Results

- `plain` ≈ Stella's baseline from script 05
- `fable_tag` — measures the effect of domain labeling alone (no new content)
- `fable_moral` — **upper bound**: if Approach B achieves 80% of the gain from `plain` to `fable_moral`, the LLM summarization is working well. If only 20%, the generated summaries are too noisy.

---

## 14. Visualization

### Approach A Plot

```python
def _plot_approach_a(base_mrr, base_r1, rerr_mrr, rerr_r1, run_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (b_val, r_val, metric) in zip(
        axes, [(base_mrr, rerr_mrr, "MRR"), (base_r1, rerr_r1, "Recall@1")]
    ):
        bars = ax.bar(["Embedding\n(baseline)", "Gemini CoT\nReranking"],
                      [b_val, r_val], color=["#4C72B0", "#DD8452"])
```

Side-by-side bar charts for MRR and Recall@1. Two bars each: embedding baseline vs. Gemini reranked. The y-limit is `max * 1.3 + 0.01` to leave room for the value labels above the bars.

### Approach B Plot

```python
def _plot_approach_b(baseline, results, run_dir):
    x = np.arange(len(prompt_labels))
    width = 0.18
    for mi, (mlabel, color) in enumerate(zip(model_labels, colors)):
        offset = (mi - len(model_labels) / 2 + 0.5) * width
        ax.bar(x + offset, mat[mi], width, ...)
    ax.axhline(baseline[metric_key], color="grey", linestyle="--", ...)
```

**Grouped bar chart**: x-axis has 4 prompt groups (`direct`, `detailed`, `cot`, `few_shot`). Within each group, 4 bars for the 4 model sizes, offset by `width = 0.18` each. The offset formula `(mi - len/2 + 0.5) * width` centers the group of bars around each tick mark.

A dashed horizontal line marks the plain-embedding baseline, so you can see at a glance which (model, prompt) combinations beat or miss the baseline.

The `mat` 2D array `(n_models × n_prompts)` is built by indexing into the results list, allowing clean bar positioning.

### Approach C Plot

3 bars per metric (one per template), with value labels. The `fable_moral` bar visually shows the upper bound.

All three plots use `plt.close(fig)` after saving. This is important when generating multiple figures in a script — without it, matplotlib accumulates figures in memory, which eventually causes memory leaks or display issues.

---

## 15. Main Block

### CLI

```python
parser.add_argument("--approach", nargs="+", choices=["A", "B", "C"],
                    default=["A", "B", "C"])
parser.add_argument("--run-dir",     type=Path,  default=None)
parser.add_argument("--embed-model", type=str,   default=DEFAULT_EMBED_MODEL)
parser.add_argument("--embed-prompt",type=str,   default=DEFAULT_QUERY_PROMPT)
parser.add_argument("--top-k",       type=int,   default=DEFAULT_TOP_K)
parser.add_argument("--sample",      type=int,   default=None)
```

- `--approach A B C` — can run any subset. `nargs="+"` requires at least one. `choices=` enforces valid values.
- `--run-dir` — path to a previous script 05 output. Only used by Approach A.
- `--embed-model` / `--embed-prompt` — allow overriding the best model from script 05.
- `--sample N` — run on only the first N queries. `--sample 20` lets you test the full pipeline in minutes instead of hours.

### Metadata

```python
metadata = {
    "timestamp": ..., "approaches": args.approach,
    "embed_model": args.embed_model, "embed_prompt": args.embed_prompt,
    "top_k": args.top_k, "sample": args.sample,
    "run_dir_for_approach_a": str(args.run_dir) if args.run_dir else "auto",
}
```

`run_dir_for_approach_a: "auto"` records that the latest script 05 run was used automatically, so you know exactly which embedding results backed this LLM experiment.

### Final Save

```python
with open(run_dir / "all_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
```

`default=str` is a safety net: if any value in the dict isn't JSON-serializable (e.g., a Path object, a numpy float), `default=str` converts it to its string representation instead of raising a TypeError.

---

## 16. How the Three Approaches Relate to Each Other

```
     BASELINE (script 05)
          │
          ▼
     Stella 1.5B, s2p_query
     MRR = X.XXXX
          │
    ┌─────┼─────┐
    │     │     │
    ▼     ▼     ▼
    A     B     C
    │     │     │
 reranks  summ. upper
 post-hoc offline bound
          │
    Compare all three
    to baseline
```

| Research Question | Answered by |
|---|---|
| Can an LLM reason better than cosine similarity? | Approach A vs. its baseline |
| Do generated moral summaries improve retrieval? | Approach B vs. baseline |
| Does larger LLM → better summaries? | 0.8B vs. 9B rows in Approach B |
| Does prompt engineering matter? | `direct` vs. `cot` vs. `few_shot` columns in Approach B |
| What is the theoretical ceiling if summaries were perfect? | Approach C (`fable_moral`) |
| How close did Approach B get to the ceiling? | Best Approach B result vs. C `fable_moral` |
