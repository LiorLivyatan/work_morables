"""
07_llm_experiments.py — LLM-enhanced moral→fable retrieval

Three approaches to improve over plain embedding retrieval:

  ┌─ Approach A ─────────────────────────────────────────────────────────────┐
  │  CoT Reranking                                                           │
  │  Take the top-K fable candidates from the best embedding model, then    │
  │  ask Gemini (with chain-of-thought) to rerank them by relevance to the  │
  │  query moral. Requires GEMINI_API_KEY in .env.                          │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Approach B ─────────────────────────────────────────────────────────────┐
  │  LLM Moral Summarisation → Embed Summary                                │
  │  Use a small local Qwen 3.5 model to generate a one-sentence moral      │
  │  for each fable, then embed the generated summary (instead of the raw   │
  │  fable text) with the best embedding model.                             │
  │  Experiments: 4 model sizes (0.8B/2B/4B/9B) × 4 prompt styles → 16    │
  │  combinations shown in a bar plot.                                      │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Approach C ─────────────────────────────────────────────────────────────┐
  │  Enriched Corpus Embedding                                               │
  │  Embed "Fable/Parable: {fable}; Moral: {moral}" directly, combining     │
  │  fable + ground-truth moral into one enriched representation. Tests the │
  │  upper-bound of Approach B (if LLM summaries were perfect).            │
  │  Also tests simpler prefix-only variants.                               │
  └──────────────────────────────────────────────────────────────────────────┘

Requires:
  • GEMINI_API_KEY in .env or environment  (Approach A only)
  • Sufficient RAM/VRAM for Qwen 3.5 generation  (Approach B only)

Usage:
  python scripts/07_llm_experiments.py                      # all approaches
  python scripts/07_llm_experiments.py --approach A         # single approach
  python scripts/07_llm_experiments.py --approach B C       # two approaches
  python scripts/07_llm_experiments.py --approach B --sample 20  # quick test
  python scripts/07_llm_experiments.py --approach A --run-dir results/runs/2026-03-15_09-34-52_combined
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from retrieval_utils import compute_metrics

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RUNS_DIR    = RESULTS_DIR / "runs"
LLM_RUNS_DIR = RESULTS_DIR / "llm_runs"
SUMMARIES_CACHE = RESULTS_DIR / "summaries"

for _d in (LLM_RUNS_DIR, SUMMARIES_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# ── load .env ──────────────────────────────────────────────────────────────────
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    with open(_env) as f:
        for _line in f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _v = _v.split("#")[0].strip()
                os.environ.setdefault(_k.strip(), _v)

# ── embedding model (best from 05_model_comparison.py) ────────────────────────
# linq-embed-mistral__general is the top model (MRR=0.2105).
# It uses E5-instruct style: prepend "Instruct: {task}\nQuery: {text}" to queries.
DEFAULT_EMBED_MODEL       = "Linq-AI-Research/Linq-Embed-Mistral"
DEFAULT_QUERY_PROMPT      = None        # not used for this model (no prompt_name)
DEFAULT_QUERY_INSTRUCTION = "Given a text, retrieve the most relevant passage that answers the query"
DEFAULT_TOP_K             = 20          # candidates to rerank in Approach A

# ── Approach B: generation models (Qwen2.5 Instruct, fully supported) ─────────
# Qwen3.5 requires transformers>=5.x which breaks the stella embedding model.
# Qwen2.5-Instruct covers the same size range and is state-of-the-art.
QWEN_MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b":   "Qwen/Qwen2.5-7B-Instruct",
}

# ── Approach B: prompt templates ──────────────────────────────────────────────
# All prompts explicitly forbid "The moral is..." / "The lesson is..." prefixes.
# Post-processing in _generate_summary strips them anyway as a safety net.
SUMMARY_PROMPTS = {
    # Direct one-shot instruction
    "direct": (
        "What is the moral of this fable? Write it as a single short sentence. "
        "Do NOT start with 'The moral is' or 'The lesson is' — state it directly.\n\n"
        "Fable: {fable}\n\n"
        "Moral:"
    ),
    # Richer context: ask for specific virtue/vice
    "detailed": (
        "Read this fable and identify the deeper lesson or moral principle it teaches. "
        "Be specific about what virtue, vice, or life wisdom the story illustrates. "
        "Write one concise sentence. Do NOT use phrases like 'The moral is' or "
        "'This fable teaches' — state the principle directly as a standalone truth.\n\n"
        "Fable: {fable}\n\n"
        "Moral:"
    ),
    # Chain-of-thought: three guided questions lead to the moral
    "cot": (
        "Given a fable, think step by step, then return ONLY the moral on the last line, without any prefix or suffix.\n\n"
        "Think through:\n"
        "1. What happens in the story?\n"
        "2. What does the main character do, and what results from it?\n"
        "3. What does the story seem to be warning against or encouraging?\n\n"
        "Fable: {fable}\n\n"
        "Moral:"
    ),
    # Few-shot: examples show the exact clean format expected
    "few_shot": (
        "Here are two examples of fables and their morals:\n\n"
        "Example 1:\n"
        "Fable: A hare laughed at a tortoise for being slow. The tortoise challenged the hare "
        "to a race. The hare, confident in his speed, stopped to rest midway. The tortoise "
        "plodded on steadily and won.\n"
        "Moral: Consistent effort overcomes natural talent when the talented grow complacent.\n\n"
        "Example 2:\n"
        "Fable: A crow held a piece of cheese in her beak. A fox told her she must have a "
        "beautiful singing voice. When she opened her beak to sing, the cheese fell to the fox.\n"
        "Moral: Vanity makes us easy prey for flatterers.\n\n"
        "Now write the moral for this fable. Follow the exact format above — "
        "a short standalone phrase, no 'The moral is...' prefix.\n\n"
        "Fable: {fable}\n\n"
        "Moral:"
    ),
}

# ── Approach C: corpus templates ──────────────────────────────────────────────
# {moral} = ground-truth moral for that fable  (upper bound for Approach B)
CORPUS_TEMPLATES = {
    "plain":       "{fable}",
    "fable_tag":   "Fable: {fable}",
    "fable_moral": "Fable/Parable: {fable}; Moral: {moral}",
}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

with open(DATA_DIR / "fables_corpus.json") as f:
    fables_corpus = json.load(f)
with open(DATA_DIR / "morals_corpus.json") as f:
    morals_corpus = json.load(f)
with open(DATA_DIR / "qrels_moral_to_fable.json") as f:
    qrels_m2f = json.load(f)

fable_texts = [f["text"] for f in fables_corpus]
moral_texts = [m["text"] for m in morals_corpus]

# Ground truth: moral_idx → fable_idx
gt_m2f: dict[int, int] = {}
for qrel in qrels_m2f:
    moral_idx = int(qrel["query_id"].split("_")[1])
    fable_idx = int(qrel["doc_id"].split("_")[1])
    gt_m2f[moral_idx] = fable_idx

# Reverse mapping: fable_idx → moral_text  (used in Approach C)
fable_to_moral_text: dict[int, str] = {
    fable_idx: moral_texts[moral_idx]
    for moral_idx, fable_idx in gt_m2f.items()
}

print(f"Loaded {len(fable_texts)} fables, {len(moral_texts)} morals, "
      f"{len(gt_m2f)} qrels")


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def detect_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_embed_model(model_id: str) -> SentenceTransformer:
    print(f"Loading embedding model: {model_id}")
    return SentenceTransformer(model_id, trust_remote_code=True)


def encode_morals(model: SentenceTransformer, query_prompt: str | None,
                  query_instruction: str | None = None) -> np.ndarray:
    """
    Encode all morals as queries (normalised float32).

    Supports two instruction styles:
      - prompt_name (stella):  pass query_prompt="s2p_query"
      - E5-instruct prefix (linq-embed-mistral, e5-mistral, SFR-mistral):
        pass query_instruction="Given a text, retrieve..."
        → prepends "Instruct: {task}\\nQuery: {text}" to each moral
    """
    texts = moral_texts
    if query_instruction:
        texts = [f"Instruct: {query_instruction}\nQuery: {t}" for t in texts]

    kwargs: dict = dict(show_progress_bar=True, normalize_embeddings=True,
                        convert_to_numpy=True, batch_size=8)
    if query_prompt:
        kwargs["prompt_name"] = query_prompt
    return model.encode(texts, **kwargs)


def encode_corpus(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Encode corpus texts (no query prompt)."""
    return model.encode(
        texts, show_progress_bar=True, normalize_embeddings=True,
        convert_to_numpy=True, batch_size=32,
    )


def metrics_summary(m: dict) -> str:
    return (f"MRR={m['MRR']:.4f}  R@1={m['Recall@1']:.4f}  "
            f"R@5={m['Recall@5']:.4f}  R@10={m['Recall@10']:.4f}")


def make_run_dir() -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = LLM_RUNS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ══════════════════════════════════════════════════════════════════════════════
#   █████╗     ██████╗ ██████╗ ██████╗  █████╗  ██████╗██╗  ██╗    █████╗
#  ██╔══██╗   ██╔══██╗██╔══██╗██╔═══██╗██╔══██╗██╔════╝██║  ██║   ██╔══██╗
#  ███████║   ██████╔╝██████╔╝██║   ██║███████║██║     ███████║   ███████║
#  ██╔══██║   ██╔═══╝ ██╔═══╝ ██║   ██║██╔══██║██║     ██╔══██║   ██╔══██║
#  ██║  ██║   ██║     ██║     ╚██████╔╝██║  ██║╚██████╗██║  ██║   ██║  ██║
#  ╚═╝  ╚═╝   ╚═╝     ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝  ╚═╝
#
#  CoT Reranking — use Gemini to rerank top-K embedding candidates
# ══════════════════════════════════════════════════════════════════════════════

COT_RERANK_PROMPT = """\
You are an expert in fables, parables, and moral reasoning.

Given a moral lesson and a list of candidate fables, think step by step about \
which fable best illustrates this moral.

MORAL LESSON:
{moral}

CANDIDATE FABLES (numbered):
{fables_list}

Think step by step:
1. What does this moral teach? What behaviour, virtue, or principle does it describe?
2. What kind of story scenario would naturally illustrate this lesson?
3. Which candidate fable most directly embodies this principle?

After your reasoning, output ONLY a line starting with "RANKING:" followed by the \
fable numbers in order from best to worst match, separated by commas.
Example: RANKING: 3, 1, 5, 2, 4\
"""


def _find_latest_predictions(model_run_dir: Path | None) -> Path:
    """Return the run directory that contains predictions."""
    if model_run_dir and model_run_dir.exists():
        return model_run_dir
    run_dirs = sorted(RUNS_DIR.iterdir()) if RUNS_DIR.exists() else []
    if not run_dirs:
        raise RuntimeError(
            "No run directories found. Run 05_model_comparison.py first, "
            "or pass --run-dir explicitly."
        )
    return run_dirs[-1]


def _best_predictions_file(run_dir: Path) -> tuple[Path, str]:
    """Return (predictions_file, run_key) of the run with highest MRR."""
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results.json in {run_dir}")
    with open(results_path) as f:
        results = [r for r in json.load(f) if "error" not in r]
    best = max(results, key=lambda r: r.get("MRR", 0))
    run_key = best["run_key"]
    preds_file = run_dir / "predictions" / f"{run_key}.json"
    if not preds_file.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {preds_file}\n"
            "Re-run 05_model_comparison.py to regenerate."
        )
    return preds_file, run_key


def _rerank_with_gemini(moral_text: str, candidate_fables: list[str],
                        candidate_indices: list[int], gemini_model) -> list[int]:
    """Ask Gemini to rerank candidate fables for a moral query."""
    fables_list = "\n".join(
        f"{i + 1}. {text[:400]}{'...' if len(text) > 400 else ''}"
        for i, text in enumerate(candidate_fables)
    )
    prompt = COT_RERANK_PROMPT.format(moral=moral_text, fables_list=fables_list)
    try:
        response = gemini_model.generate_content(prompt)
        text = response.text
        for line in text.strip().split("\n"):
            if line.strip().upper().startswith("RANKING:"):
                ranking_str = line.split(":", 1)[1].strip()
                ranked_nums: list[int] = []
                for part in ranking_str.split(","):
                    part = part.strip().rstrip(".")
                    try:
                        num = int(part)
                        if 1 <= num <= len(candidate_fables):
                            ranked_nums.append(num - 1)
                    except ValueError:
                        continue
                ranked_global = [candidate_indices[i] for i in ranked_nums
                                 if i < len(candidate_indices)]
                for idx in candidate_indices:
                    if idx not in ranked_global:
                        ranked_global.append(idx)
                return ranked_global
    except Exception as e:
        print(f"    Gemini error: {e}")
    return list(candidate_indices)


def run_approach_a(run_dir: Path, embed_run_dir: Path | None, top_k: int,
                   sample_size: int | None) -> dict:
    """
    Approach A — CoT Reranking with Gemini.

    Loads pre-computed top-K predictions from the best embedding model run,
    then uses Gemini with chain-of-thought reasoning to rerank candidates.
    """
    print("\n" + "═" * 70)
    print("APPROACH A — CoT Reranking with Gemini")
    print("═" * 70)

    _embed_run_dir = _find_latest_predictions(embed_run_dir)
    preds_file, best_key = _best_predictions_file(_embed_run_dir)
    print(f"  Using predictions from: {best_key}  ({preds_file})")

    with open(preds_file) as f:
        preds_data = json.load(f)
    queries = {q["query_idx"]: q for q in preds_data["queries"]}

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env or environment.")
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gemini_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
    gemini_model = genai.GenerativeModel(gemini_model_name)
    print(f"  Gemini model: {gemini_model_name}")
    print(f"  Reranking top-{top_k} candidates per query")

    moral_indices = sorted(gt_m2f.keys())
    if sample_size:
        moral_indices = moral_indices[:sample_size]
        print(f"  Sample mode: {sample_size} queries")

    # Baseline: embedding P@1 within the top-K candidates
    embed_ranks = []
    for m_idx in moral_indices:
        if m_idx not in queries:
            continue
        top_k_indices = queries[m_idx]["top_k_indices"][:top_k]
        correct = gt_m2f.get(m_idx)
        rank = top_k_indices.index(correct) if correct in top_k_indices else len(fable_texts)
        embed_ranks.append(rank)
    baseline_recall_1 = float(np.mean([1.0 if r == 0 else 0.0 for r in embed_ranks]))
    baseline_mrr = float(np.mean([1.0 / (r + 1) for r in embed_ranks]))
    print(f"\n  Baseline (embedding top-{top_k}):"
          f"  MRR={baseline_mrr:.4f}  R@1={baseline_recall_1:.4f}")

    # Gemini reranking
    reranked_results: dict[int, list[int]] = {}
    for m_idx in tqdm(moral_indices, desc="  CoT reranking"):
        if m_idx not in queries:
            continue
        candidates = queries[m_idx]["top_k_indices"][:top_k]
        candidate_texts = [fable_texts[i] for i in candidates]
        reranked_results[m_idx] = _rerank_with_gemini(
            moral_texts[m_idx], candidate_texts, candidates, gemini_model
        )
        time.sleep(0.5)  # rate limiting

    reranked_ranks = []
    for m_idx, ranking in reranked_results.items():
        correct = gt_m2f.get(m_idx)
        rank = ranking.index(correct) if correct in ranking else len(fable_texts)
        reranked_ranks.append(rank)

    reranked_recall_1 = float(np.mean([1.0 if r == 0 else 0.0 for r in reranked_ranks]))
    reranked_mrr = float(np.mean([1.0 / (r + 1) for r in reranked_ranks]))
    improvement = reranked_recall_1 - baseline_recall_1
    print(f"\n  After CoT reranking:  MRR={reranked_mrr:.4f}  R@1={reranked_recall_1:.4f}")
    print(f"  R@1 change: {improvement:+.4f}  ({'↑' if improvement >= 0 else '↓'} {abs(improvement):.1%})")

    results = {
        "approach": "A",
        "embed_model": best_key,
        "gemini_model": gemini_model_name,
        "top_k": top_k,
        "n_queries": len(reranked_ranks),
        "baseline_MRR": baseline_mrr,
        "baseline_Recall@1": baseline_recall_1,
        "reranked_MRR": reranked_mrr,
        "reranked_Recall@1": reranked_recall_1,
    }
    _plot_approach_a(baseline_mrr, baseline_recall_1, reranked_mrr,
                     reranked_recall_1, run_dir)
    out_path = run_dir / "approach_a_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results → {out_path}")
    return results


def _plot_approach_a(base_mrr, base_r1, rerr_mrr, rerr_r1, run_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Approach A — CoT Reranking vs Embedding Baseline", fontsize=12)
    for ax, (b_val, r_val, metric) in zip(
        axes,
        [(base_mrr, rerr_mrr, "MRR"), (base_r1, rerr_r1, "Recall@1")]
    ):
        bars = ax.bar(["Embedding\n(baseline)", "Gemini CoT\nReranking"],
                      [b_val, r_val],
                      color=["#4C72B0", "#DD8452"], edgecolor="white")
        ax.set_title(metric)
        ax.set_ylim(0, max(b_val, r_val) * 1.3 + 0.01)
        for bar, v in zip(bars, [b_val, r_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = run_dir / "approach_a_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#   █████╗    ██████╗     ██████╗  ██████╗  ██████╗ █████╗  ██████╗██╗  ██╗
#  ██╔══██╗   ██╔══██╗    ██╔══██╗██╔══██╗██╔═══██╗██╔══██╗██╔════╝██║  ██║
#  ███████║   ██████╔╝    ██████╔╝██████╔╝██║   ██║███████║██║     ███████║
#  ██╔══██║   ██╔═══╝     ██╔═══╝ ██╔══██╗██║   ██║██╔══██║██║     ██╔══██║
#  ██║  ██║   ██║         ██║     ██║  ██║╚██████╔╝██║  ██║╚██████╗██║  ██║
#  ╚═╝  ╚═╝   ╚═╝         ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
#
#  LLM Summarisation — generate moral summary per fable, then embed & retrieve
# ══════════════════════════════════════════════════════════════════════════════

def _load_generation_model(model_id: str, device: str):
    """Load a Qwen 3.5 Instruct model for text generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading generation model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.train(False)  # equivalent to model.eval() — sets inference mode
    return tokenizer, model


def _generate_summary(tokenizer, model, fable_text: str,
                      prompt_template: str) -> str:
    """Generate a one-sentence moral summary for a single fable."""
    import torch

    prompt = prompt_template.format(fable=fable_text)
    messages = [{"role": "user", "content": prompt}]

    # Apply chat template; disable Qwen3.5 thinking mode for short outputs
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # Strip any <think>…</think> blocks
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Extract from CoT-style "MORAL: ..." output
    for line in response.split("\n"):
        if line.strip().upper().startswith("MORAL:"):
            response = line.split(":", 1)[1].strip()
            break

    # Take first non-empty line
    lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
    response = lines[0] if lines else response

    # NOTE: prefix stripping intentionally disabled — let the model generate naturally.
    # To re-enable: response = _strip_summary_prefix(response)
    return response


# Prefixes that models add despite being told not to — stripped in post-processing
_SUMMARY_PREFIXES = [
    "the moral of this fable is that ",
    "the moral of the fable is that ",
    "the moral of this story is that ",
    "the moral of the story is that ",
    "the moral is that ",
    "the moral is: ",
    "the lesson of this fable is that ",
    "the lesson taught by this fable is that ",
    "the lesson taught by the fable is that ",
    "the lesson is that ",
    "the deeper lesson of this fable is that ",
    "the deeper lesson or moral principle illustrated in this fable is that ",
    "the deeper lesson or moral principle illustrated in this fable is ",
    "the key lesson is that ",
    "the key lesson is: ",
    "this fable teaches that ",
    "this fable teaches us that ",
    "this story teaches that ",
    "in conclusion, ",
    "overall, ",
    "therefore, ",
    "thus, ",
]


def _strip_summary_prefix(text: str) -> str:
    """Remove common formulaic prefixes and capitalise the result."""
    t = text.strip()
    lower = t.lower()
    for prefix in _SUMMARY_PREFIXES:
        if lower.startswith(prefix):
            t = t[len(prefix):]
            break
    # Capitalise first letter
    return t[0].upper() + t[1:] if t else t


def _load_summaries_from_cache(model_label: str, prompt_label: str) -> list[str]:
    """
    Load pre-generated summaries from cache (written by 07_generate_summaries.py).
    Raises FileNotFoundError if the cache is missing or incomplete.
    """
    cache_file = SUMMARIES_CACHE / model_label / f"{prompt_label}.json"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Summary cache not found: {cache_file}\n"
            f"Run first:  python scripts/07_generate_summaries.py --models {model_label} --prompts {prompt_label}"
        )
    with open(cache_file) as f:
        cached = json.load(f)
    if len(cached) != len(fable_texts):
        raise ValueError(
            f"Cache is incomplete: {len(cached)}/{len(fable_texts)} entries in {cache_file}\n"
            f"Re-run:  python scripts/07_generate_summaries.py --models {model_label} --prompts {prompt_label}"
        )
    return [cached[str(i)] for i in range(len(fable_texts))]


def run_approach_b(run_dir: Path, embed_model_id: str, query_prompt: str | None,
                   query_instruction: str | None) -> list[dict]:
    """
    Approach B — LLM Moral Summarisation → Embed & Retrieve.

    Loads pre-generated summaries from cache (run 07_generate_summaries.py first),
    embeds them with the best embedding model, and measures retrieval quality.
    """
    print("\n" + "═" * 70)
    print("APPROACH B — LLM Summarisation × Embedding")
    print("═" * 70)
    print("  (Summaries loaded from cache — run 07_generate_summaries.py to regenerate)")

    embed_model = load_embed_model(embed_model_id)
    print("  Encoding morals (queries)...")
    moral_embs = encode_morals(embed_model, query_prompt, query_instruction)

    print("  Baseline: encoding raw fable texts...")
    fable_embs = encode_corpus(embed_model, fable_texts)
    baseline = compute_metrics(moral_embs, fable_embs, gt_m2f)
    print(f"  Baseline (plain fable):  {metrics_summary(baseline)}")

    all_results: list[dict] = []

    for model_label in QWEN_MODELS:
        print(f"\n  ── Model: {model_label}")
        for prompt_label in SUMMARY_PROMPTS:
            combo_key = f"{model_label}__{prompt_label}"
            try:
                summaries = _load_summaries_from_cache(model_label, prompt_label)
            except (FileNotFoundError, ValueError) as e:
                print(f"    [{combo_key}] SKIPPED — {e}")
                continue
            n_non_empty = sum(1 for s in summaries if s)
            print(f"    [{combo_key}] {n_non_empty}/{len(fable_texts)} non-empty summaries")

            summary_embs = encode_corpus(embed_model, summaries)
            m = compute_metrics(moral_embs, summary_embs, gt_m2f)
            m["combo"] = combo_key
            m["model"] = model_label
            m["prompt"] = prompt_label
            all_results.append(m)
            print(f"    {metrics_summary(m)}")

    _plot_approach_b(baseline, all_results, run_dir)
    out_path = run_dir / "approach_b_results.json"
    with open(out_path, "w") as f:
        json.dump({"baseline": baseline, "results": all_results}, f, indent=2)
    print(f"\n  Saved results → {out_path}")
    return all_results


def _plot_approach_b(baseline: dict, results: list[dict], run_dir: Path):
    """
    Grouped bar chart: x-axis = prompt style, bars = Qwen model sizes.
    Dashed horizontal line marks the plain-embedding baseline.
    """
    prompt_labels = list(SUMMARY_PROMPTS.keys())
    model_labels  = list(QWEN_MODELS.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x = np.arange(len(prompt_labels))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Approach B — LLM Moral Summarisation: MRR & Recall@1 per Model × Prompt",
        fontsize=12,
    )

    for ax, metric_key in zip(axes, ["MRR", "Recall@1"]):
        mat = np.zeros((len(model_labels), len(prompt_labels)))
        for r in results:
            mi = model_labels.index(r["model"])
            pi = prompt_labels.index(r["prompt"])
            mat[mi, pi] = r[metric_key]

        for mi, (mlabel, color) in enumerate(zip(model_labels, colors)):
            offset = (mi - len(model_labels) / 2 + 0.5) * width
            ax.bar(x + offset, mat[mi], width, label=mlabel,
                   color=color, edgecolor="white", linewidth=0.5)

        ax.axhline(baseline[metric_key], color="grey", linestyle="--",
                   linewidth=1.2, alpha=0.8, label="Baseline (plain fable)")
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_labels, fontsize=9)
        ax.set_ylabel(metric_key)
        ax.set_title(metric_key)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = run_dir / "approach_b_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#   █████╗      ██████╗    ██████╗  ██████╗  ██████╗  █████╗  ██████╗██╗  ██╗
#  ██╔══██╗    ██╔════╝   ██╔══██╗██╔══██╗██╔═══██╗██╔══██╗██╔════╝██║  ██║
#  ███████║    ██║        ██████╔╝██████╔╝██║   ██║███████║██║     ███████║
#  ██╔══██║    ██║        ██╔═══╝ ██╔══██╗██║   ██║██╔══██║██║     ██╔══██║
#  ██║  ██║    ╚██████╗   ██║     ██║  ██║╚██████╔╝██║  ██║╚██████╗██║  ██║
#  ╚═╝  ╚═╝    ╚═════╝   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
#
#  Enriched Corpus — embed "Fable: {text}; Moral: {moral}" directly
# ══════════════════════════════════════════════════════════════════════════════

def run_approach_c(run_dir: Path, embed_model_id: str, query_prompt: str | None,
                   query_instruction: str | None) -> list[dict]:
    """
    Approach C — Enriched Corpus Embedding.

    Embeds fable corpus using three templates that progressively add more
    semantic information, with 'fable_moral' using the ground-truth moral as
    an upper bound (equivalent to a perfect Approach B summary).
    """
    print("\n" + "═" * 70)
    print("APPROACH C — Enriched Corpus Embedding")
    print("═" * 70)
    print("  Templates:")
    for k, v in CORPUS_TEMPLATES.items():
        preview = v.replace("{fable}", "<fable>").replace("{moral}", "<moral>")
        print(f"    {k:<14} → {preview[:70]}")

    embed_model = load_embed_model(embed_model_id)
    print("\n  Encoding morals (queries)...")
    moral_embs = encode_morals(embed_model, query_prompt, query_instruction)

    results = []
    for template_key, template in CORPUS_TEMPLATES.items():
        corpus_texts = [
            template.format(
                fable=fable_texts[i],
                moral=fable_to_moral_text.get(i, ""),
            )
            for i in range(len(fable_texts))
        ]
        print(f"\n  Encoding corpus [{template_key}]...")
        corpus_embs = encode_corpus(embed_model, corpus_texts)
        m = compute_metrics(moral_embs, corpus_embs, gt_m2f)
        m["template"] = template_key
        results.append(m)
        print(f"    {metrics_summary(m)}")

    _plot_approach_c(results, run_dir)
    out_path = run_dir / "approach_c_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results → {out_path}")
    return results


def _plot_approach_c(results: list[dict], run_dir: Path):
    labels = [r["template"] for r in results]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        "Approach C — Enriched Corpus Templates  "
        "(fable_moral = ground-truth upper bound)",
        fontsize=11,
    )
    for ax, metric in zip(axes, ["MRR", "Recall@1"]):
        vals = [r[metric] for r in results]
        bars = ax.bar(labels, vals, color=colors[:len(labels)], edgecolor="white")
        ax.set_title(metric)
        ax.set_ylim(0, max(vals) * 1.35 + 0.01)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = run_dir / "approach_c_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-enhanced moral→fable retrieval experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--approach", nargs="+", choices=["A", "B", "C"],
        default=["A", "B", "C"],
        help="Which approaches to run (default: all three).",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Path to 05_model_comparison run dir for Approach A predictions. "
             "Defaults to the most recent run in results/runs/.",
    )
    parser.add_argument(
        "--embed-model", type=str, default=DEFAULT_EMBED_MODEL,
        help=f"HuggingFace model ID for embedding (default: {DEFAULT_EMBED_MODEL}).",
    )
    parser.add_argument(
        "--embed-prompt", type=str, default=DEFAULT_QUERY_PROMPT,
        help="SentenceTransformer prompt_name for query encoding (e.g. 's2p_query' for stella). "
             "Mutually exclusive with --query-instruction.",
    )
    parser.add_argument(
        "--query-instruction", type=str, default=DEFAULT_QUERY_INSTRUCTION,
        help="E5-instruct style task string prepended as 'Instruct: {task}\\nQuery: {text}'. "
             f"Default (for linq-embed-mistral): '{DEFAULT_QUERY_INSTRUCTION}'.",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of candidates to rerank in Approach A (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Run on only the first N queries (quick testing).",
    )
    args = parser.parse_args()

    print(f"\nApproaches to run: {args.approach}")
    run_dir = make_run_dir()
    print(f"Output directory:  {run_dir}\n")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "approaches": args.approach,
        "embed_model": args.embed_model,
        "embed_prompt": args.embed_prompt,
        "query_instruction": args.query_instruction,
        "top_k": args.top_k,
        "sample": args.sample,
        "run_dir_for_approach_a": str(args.run_dir) if args.run_dir else "auto",
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    all_results: dict[str, object] = {}

    if "A" in args.approach:
        all_results["A"] = run_approach_a(
            run_dir, args.run_dir, args.top_k, args.sample
        )

    if "B" in args.approach:
        all_results["B"] = run_approach_b(
            run_dir, args.embed_model, args.embed_prompt,
            args.query_instruction,
        )

    if "C" in args.approach:
        all_results["C"] = run_approach_c(
            run_dir, args.embed_model, args.embed_prompt, args.query_instruction
        )

    with open(run_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("SUMMARY")
    print("═" * 70)

    if "A" in all_results and isinstance(all_results["A"], dict):
        a = all_results["A"]
        print(f"Approach A — CoT Reranking (top-{a.get('top_k', '?')}):")
        print(f"  Baseline   MRR={a['baseline_MRR']:.4f}  R@1={a['baseline_Recall@1']:.4f}")
        print(f"  Reranked   MRR={a['reranked_MRR']:.4f}  R@1={a['reranked_Recall@1']:.4f}")

    if "B" in all_results and all_results["B"]:
        b_results = all_results["B"]
        best_b = max(b_results, key=lambda r: r.get("MRR", 0))
        print(f"\nApproach B — Best combo:  {best_b['combo']}")
        print(f"  {metrics_summary(best_b)}")

    if "C" in all_results and all_results["C"]:
        print(f"\nApproach C — Template comparison:")
        for r in all_results["C"]:
            print(f"  {r['template']:<16} {metrics_summary(r)}")

    print(f"\nAll outputs → {run_dir}/")
