"""
08_qwen3_embedding_retrieval.py — Qwen3-Embedding with instruction-steered reasoning layer.

Hypothesis: Qwen3-Embedding (a decoder LLM fine-tuned for embedding) can outperform
Linq-Embed-Mistral because its instruction field acts as a "reasoning layer" that
steers the model's hidden representation toward moral meaning.

Reference: Zhang et al. (2025), "Qwen3 Embedding" (arXiv:2506.05176v1)

Task: Moral → Fable retrieval (709 pairs, MORABLES dataset)

Models: Qwen3-Embedding-0.6B, 4B, 8B (all via transformers + last-token pooling)
Instructions: baseline, moral_focused, analytical, abstract

Usage:
  python scripts/08_qwen3_embedding_retrieval.py                    # all models & instructions
  python scripts/08_qwen3_embedding_retrieval.py --models 0.6B      # single model
  python scripts/08_qwen3_embedding_retrieval.py --models 0.6B 4B   # two models
  python scripts/08_qwen3_embedding_retrieval.py --instructions baseline moral_focused
  python scripts/08_qwen3_embedding_retrieval.py --sample 20        # quick test on 20 queries
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from retrieval_utils import compute_metrics, compute_rankings

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RUNS_DIR    = RESULTS_DIR / "qwen3_embedding_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "4B":   "Qwen/Qwen3-Embedding-4B",
    "8B":   "Qwen/Qwen3-Embedding-8B",
}

# ── Instructions (moral/query side only — documents are encoded plain) ────────
# Qwen3-Embedding format: "Instruct: {instruction}\nQuery:{text}"
# NOTE: no space between "Query:" and query text — Qwen3 specific.
INSTRUCTIONS = {
    "baseline": (
        "Given a text, retrieve the most relevant passage that matches this text"
    ),
    "moral_focused": (
        "Given a moral principle or life lesson, retrieve the fable or parable "
        "that teaches this exact lesson through its characters, conflict, and "
        "resolution. Focus on the underlying meaning, not surface-level details "
        "like character names or settings."
    ),
    "analytical": (
        "Given a moral statement about human nature, retrieve the fable that "
        "illustrates this principle. Consider: the fable will dramatize this "
        "lesson through a narrative arc where characters' choices lead to "
        "consequences that reveal the truth of this moral. The fable may use "
        "animals, people, or objects as allegories. Look for the structural "
        "match between the abstract lesson and the narrative pattern."
    ),
    "abstract": (
        "Given a concise moral truth — a statement about virtue, vice, or the "
        "human condition — retrieve the fable that serves as its narrative "
        "embodiment. The fable will not state this moral explicitly; instead, "
        "the moral emerges from the interplay of characters' actions and their "
        "consequences. Look past literal content to find the fable whose "
        "deeper meaning aligns with this principle."
    ),
}

# ── Baselines (from previous experiments, for comparison chart) ───────────────
BASELINES = {
    "Linq-Embed-Mistral": {"MRR": 0.2105, "Recall@1": 0.141},
    "Approach B best\n(Qwen3.5-9b+CoT)": {"MRR": 0.215, "Recall@1": 0.141},
}

# ── Data ──────────────────────────────────────────────────────────────────────
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

print(f"Loaded {len(fable_texts)} fables, {len(moral_texts)} morals, "
      f"{len(gt_m2f)} qrels")


# ── Device ────────────────────────────────────────────────────────────────────

def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Qwen3-Embedding encoder ──────────────────────────────────────────────────
# Follows the QwenAdapter pattern from 05_model_comparison.py:
# - Causal LLM with last-token pooling (NOT mean pooling)
# - Query format: "Instruct: {task}\nQuery:{text}" (no space before text)
# - Document format: plain text

class Qwen3Encoder:
    def __init__(self, model_id: str, device: str):
        from transformers import AutoTokenizer, AutoModel
        print(f"  Loading {model_id} on {device}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, padding_side="left"
        )
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.train(False)
        print(f"  Loaded. Embedding dim = {self.model.config.hidden_size}")

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        seq_lengths = attention_mask.sum(dim=1) - 1
        return last_hidden_states[
            torch.arange(len(last_hidden_states), device=last_hidden_states.device),
            seq_lengths,
        ]

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="    encoding"):
            batch = texts[i: i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=8192, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**encoded)
            embs = self._last_token_pool(out.last_hidden_state, encoded["attention_mask"])
            embs = F.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)

    def encode_queries(self, texts: list[str], instruction: str,
                       batch_size: int = 16) -> np.ndarray:
        formatted = [f"Instruct: {instruction}\nQuery:{t}" for t in texts]
        return self._encode(formatted, batch_size)

    def encode_corpus(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return self._encode(texts, batch_size)

    def unload(self):
        del self.model
        del self.tokenizer
        torch.mps.empty_cache() if self.device == "mps" else None
        import gc; gc.collect()


# ── Helpers ───────────────────────────────────────────────────────────────────

def metrics_summary(m: dict) -> str:
    return (f"MRR={m['MRR']:.4f}  R@1={m['Recall@1']:.4f}  "
            f"R@5={m['Recall@5']:.4f}  R@10={m['Recall@10']:.4f}")


def make_run_dir() -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)
    return run_dir


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(
    model_labels: list[str],
    instruction_labels: list[str],
    sample_size: int | None,
    run_dir: Path,
) -> list[dict]:
    device = detect_device()
    print(f"\nDevice: {device}")
    print(f"Models: {model_labels}")
    print(f"Instructions: {instruction_labels}")
    if sample_size:
        print(f"Sample mode: {sample_size} queries")
    print(f"Run dir: {run_dir}\n")

    # Determine which queries to evaluate
    moral_indices = sorted(gt_m2f.keys())
    if sample_size:
        moral_indices = moral_indices[:sample_size]

    # Build ground truth for the subset
    gt_subset = {idx: gt_m2f[idx] for idx in moral_indices}
    query_texts = [moral_texts[i] for i in moral_indices]

    all_results: list[dict] = []

    for model_label in model_labels:
        model_id = MODELS[model_label]
        print(f"\n{'═' * 70}")
        print(f"  Model: {model_label}  ({model_id})")
        print(f"{'═' * 70}")

        encoder = Qwen3Encoder(model_id, device)

        # Encode corpus once per model (fables are always plain text)
        print("\n  Encoding fable corpus (plain text, no instruction)...")
        t0 = time.time()
        fable_embs = encoder.encode_corpus(fable_texts)
        corpus_time = time.time() - t0
        print(f"  Corpus encoded in {corpus_time:.1f}s  "
              f"({fable_embs.shape[0]} x {fable_embs.shape[1]})")

        for instr_label in instruction_labels:
            instruction = INSTRUCTIONS[instr_label]
            run_key = f"{model_label}__{instr_label}"
            print(f"\n  ── Instruction: {instr_label}")
            print(f"     \"{instruction[:80]}...\"")

            # Encode morals (queries) with this instruction
            t0 = time.time()
            moral_embs = encoder.encode_queries(query_texts, instruction)
            query_time = time.time() - t0
            print(f"     Queries encoded in {query_time:.1f}s  "
                  f"({moral_embs.shape[0]} x {moral_embs.shape[1]})")

            # Compute metrics
            m = compute_metrics(moral_embs, fable_embs, gt_subset)
            m["run_key"] = run_key
            m["model"] = model_label
            m["model_id"] = model_id
            m["instruction"] = instr_label
            m["instruction_text"] = instruction
            m["embedding_dim"] = int(fable_embs.shape[1])
            m["corpus_encode_time_s"] = round(corpus_time, 1)
            m["query_encode_time_s"] = round(query_time, 1)
            all_results.append(m)

            print(f"     {metrics_summary(m)}")

            # Save per-query predictions
            rankings = compute_rankings(moral_embs, fable_embs, top_k=50)
            preds = {}
            for qi, m_idx in enumerate(moral_indices):
                correct_fable = gt_m2f[m_idx]
                ranked = rankings[qi]["indices"]
                correct_rank = ranked.index(correct_fable) if correct_fable in ranked else -1
                preds[str(m_idx)] = {
                    "moral_text": moral_texts[m_idx][:100],
                    "correct_fable_idx": correct_fable,
                    "correct_rank": correct_rank,
                    "top_10": ranked[:10],
                    "top_10_scores": rankings[qi]["scores"][:10],
                }
            pred_path = run_dir / "predictions" / f"{run_key}.json"
            with open(pred_path, "w") as f:
                json.dump(preds, f, indent=2, ensure_ascii=False)

            # Save results incrementally (so partial runs are recoverable)
            with open(run_dir / "results.json", "w") as f:
                json.dump(all_results, f, indent=2)

        # Unload model before loading the next one
        encoder.unload()
        print(f"\n  Unloaded {model_label}")

    return all_results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: list[dict], run_dir: Path):
    """Grouped bar chart: instruction variants x model sizes, with baseline lines."""
    if not results:
        return

    model_labels = sorted(set(r["model"] for r in results),
                          key=lambda x: list(MODELS.keys()).index(x))
    instr_labels = sorted(set(r["instruction"] for r in results),
                          key=lambda x: list(INSTRUCTIONS.keys()).index(x))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    x = np.arange(len(instr_labels))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Qwen3-Embedding: Instruction-Steered Moral → Fable Retrieval",
        fontsize=12,
    )

    for ax, metric_key in zip(axes, ["MRR", "Recall@1"]):
        # Build matrix
        mat = np.zeros((len(model_labels), len(instr_labels)))
        for r in results:
            mi = model_labels.index(r["model"])
            ii = instr_labels.index(r["instruction"])
            mat[mi, ii] = r[metric_key]

        # Plot bars
        for mi, (mlabel, color) in enumerate(zip(model_labels, colors)):
            offset = (mi - len(model_labels) / 2 + 0.5) * width
            bars = ax.bar(x + offset, mat[mi], width, label=f"Qwen3-Emb-{mlabel}",
                          color=color, edgecolor="white", linewidth=0.5)
            # Value labels on bars
            for bar, v in zip(bars, mat[mi]):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        # Baseline lines
        line_styles = ["--", ":"]
        line_colors = ["grey", "darkred"]
        for (bname, bvals), ls, lc in zip(BASELINES.items(), line_styles, line_colors):
            ax.axhline(bvals[metric_key], color=lc, linestyle=ls,
                       linewidth=1.2, alpha=0.7, label=bname)

        ax.set_xticks(x)
        ax.set_xticklabels(instr_labels, fontsize=9)
        ax.set_ylabel(metric_key)
        ax.set_title(metric_key)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = run_dir / "comparison_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved chart → {out}")


def print_summary_table(results: list[dict]):
    """Print a formatted comparison table."""
    print(f"\n{'═' * 80}")
    print("  RESULTS SUMMARY — Moral → Fable Retrieval")
    print(f"{'═' * 80}")
    print(f"  {'Model':<10} {'Instruction':<16} {'MRR':>8} {'R@1':>8} "
          f"{'R@5':>8} {'R@10':>8} {'NDCG@10':>8}")
    print(f"  {'─' * 10} {'─' * 16} {'─' * 8} {'─' * 8} "
          f"{'─' * 8} {'─' * 8} {'─' * 8}")

    # Baselines
    for bname, bvals in BASELINES.items():
        label = bname.replace("\n", " ")
        print(f"  {label:<27} {bvals['MRR']:>8.4f} {bvals['Recall@1']:>8.4f} "
              f"{'—':>8} {'—':>8} {'—':>8}")
    print(f"  {'─' * 10} {'─' * 16} {'─' * 8} {'─' * 8} "
          f"{'─' * 8} {'─' * 8} {'─' * 8}")

    # Sort: by model size, then instruction
    model_order = list(MODELS.keys())
    instr_order = list(INSTRUCTIONS.keys())
    sorted_results = sorted(results, key=lambda r: (
        model_order.index(r["model"]),
        instr_order.index(r["instruction"]),
    ))

    best_mrr = max(r["MRR"] for r in results) if results else 0
    for r in sorted_results:
        marker = " *" if r["MRR"] == best_mrr else "  "
        print(f"  {r['model']:<10} {r['instruction']:<16} {r['MRR']:>8.4f} "
              f"{r['Recall@1']:>8.4f} {r['Recall@5']:>8.4f} "
              f"{r['Recall@10']:>8.4f} {r['NDCG@10']:>8.4f}{marker}")

    print(f"\n  * = best MRR")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-Embedding retrieval with instruction-steered reasoning layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Which Qwen3-Embedding sizes to run (default: all three).",
    )
    parser.add_argument(
        "--instructions", nargs="+", choices=list(INSTRUCTIONS.keys()),
        default=list(INSTRUCTIONS.keys()),
        help="Which instruction variants to run (default: all four).",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Evaluate only N queries for quick testing (does not affect corpus encoding).",
    )
    args = parser.parse_args()

    total = len(args.models) * len(args.instructions)
    print(f"\nQwen3-Embedding Experiment")
    print(f"  {len(args.models)} models x {len(args.instructions)} instructions = {total} runs")
    if args.sample:
        print(f"  Sample mode: {args.sample} queries")
    print()

    run_dir = make_run_dir()

    # Save metadata
    metadata = {
        "experiment": "qwen3_embedding_retrieval",
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "device": detect_device(),
        "task": "moral_to_fable",
        "n_fables": len(fable_texts),
        "n_morals": len(moral_texts),
        "n_qrels": len(gt_m2f),
        "sample_size": args.sample,
        "models": {k: MODELS[k] for k in args.models},
        "instructions": {k: INSTRUCTIONS[k] for k in args.instructions},
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Run
    results = run_experiment(args.models, args.instructions, args.sample, run_dir)

    # Plot & summarize
    plot_results(results, run_dir)
    print_summary_table(results)

    print(f"\n  All results saved to: {run_dir}")
