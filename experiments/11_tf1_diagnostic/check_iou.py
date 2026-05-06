"""
TF1-EN-3M vs MORABLES — moral↔fable lexical overlap (IoU) diagnostic.

Question: Does training on TF1 actually teach semantic moral→fable matching,
or only lexical matching? In TF1 the moral is a SEED phrase the LLM weaves
into the fable, so we expect higher IoU than MORABLES (literary fables, IoU≈0.011).

Steps:
1. Load MORABLES (709 fable–moral pairs from data/raw/fables.json) and
   compute IoU on it using the exact tokenization from scripts/02_eda.py
   so the baseline number we compare against is recomputed in this run.
2. Stream N rows from klusai/ds-tf1-en-3m, extract the moral from the
   prompt's "- Teaching: ..." bullet, compute IoU against the fable body.
3. Save summary stats + per-row samples + a small set of human-readable
   examples to a gitignored timestamped run dir.

Run:
    ./run.sh experiments/11_tf1_diagnostic/check_iou.py            # default N=5000
    ./run.sh experiments/11_tf1_diagnostic/check_iou.py --n 1000   # smaller
"""
import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import load_dataset

EXP_DIR = Path(__file__).parent
RUNS_DIR = EXP_DIR / "results" / "runs"
MORABLES_FABLES = EXP_DIR.parent.parent / "data" / "raw" / "fables.json"

TF1_HF_ID = "klusai/ds-tf1-en-3m"

# Same stopword list and tokenization as scripts/02_eda.py for apples-to-apples
WORD_RE = re.compile(r"\b\w+\b")
STOP_WORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "and",
    "but", "or", "nor", "not", "so", "if", "than", "that",
    "this", "it", "its", "his", "her", "their", "who", "which",
    "what", "when", "where", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such",
    "no", "only", "own", "same", "he", "she", "they", "them",
    "him", "we", "you", "i", "me", "my", "your", "our",
}

# TF1 prompts use bullet "- Teaching: <moral>" — confirmed from dataset card example
TEACHING_RE = re.compile(r"-\s*Teaching:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def word_set(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def iou(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def compute_iou_pair(moral: str, fable: str) -> tuple[float, float]:
    m, f = word_set(moral), word_set(fable)
    return iou(m, f), iou(m - STOP_WORDS, f - STOP_WORDS)


def stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def morables_baseline() -> dict:
    fables = json.loads(MORABLES_FABLES.read_text())
    all_iou, no_stop_iou = [], []
    for entry in fables:
        a, b = compute_iou_pair(entry["moral"], entry["story"])
        all_iou.append(a)
        no_stop_iou.append(b)
    return {
        "source": "MORABLES (data/raw/fables.json)",
        "n_pairs": len(fables),
        "iou_all_words": stats(all_iou),
        "iou_no_stopwords": stats(no_stop_iou),
    }


def tf1_diagnostic(
    n: int, chunks: int, total_rows: int
) -> tuple[dict, list[dict]]:
    """
    Stream `n` rows total, split into `chunks` evenly-spaced windows across
    the first `total_rows` rows of TF1's train split. With chunks=1 this
    reads the first n rows contiguously (legacy behaviour). With chunks>1
    this stratifies sampling so a sorted-by-seed shard cannot bias the
    unique-morals estimate.

    Also tracks the cumulative count of unique morals as more rows are
    seen, so we can detect whether the pool size has plateaued.
    """
    chunk_size = n // chunks
    stride = total_rows // chunks
    print(
        f"Streaming {n} rows from {TF1_HF_ID} as {chunks} chunk(s) of "
        f"{chunk_size} rows, stride={stride:,}"
    )

    ds = load_dataset(TF1_HF_ID, split="train", streaming=True)

    rows: list[dict] = []
    extraction_failures = 0
    seen_morals: Counter = Counter()
    unique_growth: list[tuple[int, int, int]] = []  # (rows_seen, unique_count, chunk_idx)
    seen_total = 0

    for chunk_idx in range(chunks):
        offset = chunk_idx * stride
        print(f"  chunk {chunk_idx + 1}/{chunks}: offset={offset:,}, taking {chunk_size} rows ...")
        chunk_iter = ds.skip(offset).take(chunk_size) if offset > 0 else ds.take(chunk_size)

        for i, ex in enumerate(chunk_iter):
            seen_total += 1
            prompt = ex.get("prompt", "")
            fable = ex.get("fable", "")
            m = TEACHING_RE.search(prompt)
            if not m:
                extraction_failures += 1
                continue
            moral = m.group(1).strip()
            seen_morals[moral.lower()] += 1

            # Log unique-growth checkpoints (log-spaced early, linear late)
            if seen_total <= 1000 and seen_total % 100 == 0:
                unique_growth.append((seen_total, len(seen_morals), chunk_idx))
            elif seen_total % 1000 == 0:
                unique_growth.append((seen_total, len(seen_morals), chunk_idx))

            iou_all, iou_ns = compute_iou_pair(moral, fable)
            rows.append({
                "idx": offset + i,
                "chunk": chunk_idx,
                "prompt_hash": ex.get("prompt_hash"),
                "moral": moral,
                "fable": fable,
                "moral_word_count": len(moral.split()),
                "fable_word_count": len(fable.split()),
                "iou_all": iou_all,
                "iou_no_stop": iou_ns,
            })
        print(
            f"    chunk done: cumulative rows={seen_total}, "
            f"unique morals so far={len(seen_morals)}"
        )

    # Per-chunk unique counts (helps detect cross-chunk seed bias)
    per_chunk = []
    for c in range(chunks):
        chunk_rows = [r for r in rows if r["chunk"] == c]
        chunk_morals = Counter(r["moral"].lower() for r in chunk_rows)
        per_chunk.append({
            "chunk": c,
            "offset": c * stride,
            "n": len(chunk_rows),
            "unique": len(chunk_morals),
            "top_5": chunk_morals.most_common(5),
        })

    summary = {
        "source": TF1_HF_ID,
        "rows_streamed": seen_total,
        "chunks": chunks,
        "chunk_size": chunk_size,
        "stride": stride,
        "extraction_failures": extraction_failures,
        "extraction_success_rate": (seen_total - extraction_failures) / seen_total,
        "n_pairs_used": len(rows),
        "moral_unique_count": len(seen_morals),
        "moral_top_20_repeats": seen_morals.most_common(20),
        "unique_growth_curve": unique_growth,
        "per_chunk": per_chunk,
        "all_unique_morals_with_counts": sorted(seen_morals.items(), key=lambda kv: -kv[1]),
        "iou_all_words": stats([r["iou_all"] for r in rows]),
        "iou_no_stopwords": stats([r["iou_no_stop"] for r in rows]),
    }
    return summary, rows


def write_examples_md(out_path: Path, rows: list[dict], k: int = 20) -> None:
    rng = np.random.default_rng(42)
    sampled_idx = rng.choice(len(rows), size=min(k, len(rows)), replace=False)
    sampled = [rows[i] for i in sampled_idx]
    sampled.sort(key=lambda r: r["iou_no_stop"])

    lines = ["# TF1 random samples (sorted by content-word IoU, ascending)\n"]
    for r in sampled:
        lines.append(f"## idx={r['idx']}  iou_no_stop={r['iou_no_stop']:.3f}  iou_all={r['iou_all']:.3f}\n")
        lines.append(f"**Moral:** {r['moral']}\n")
        fable_excerpt = r["fable"][:600] + ("..." if len(r["fable"]) > 600 else "")
        lines.append(f"**Fable (first 600 chars):** {fable_excerpt}\n")
        lines.append("---\n")
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="total TF1 rows to stream")
    parser.add_argument("--chunks", type=int, default=1,
                        help="split --n into K windows spaced evenly across the dataset")
    parser.add_argument("--total-rows", type=int, default=2_800_000,
                        help="size of the train split (for stride calc)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {out_dir}")

    print("\n[1/2] MORABLES baseline IoU ...")
    morables = morables_baseline()
    print(f"  MORABLES IoU (all words):     mean={morables['iou_all_words']['mean']:.4f}")
    print(f"  MORABLES IoU (no stopwords):  mean={morables['iou_no_stopwords']['mean']:.4f}")

    print(f"\n[2/2] TF1-EN-3M (n={args.n}, chunks={args.chunks}) ...")
    tf1, rows = tf1_diagnostic(args.n, args.chunks, args.total_rows)
    print(f"  extraction success: {tf1['extraction_success_rate']:.3%} ({tf1['n_pairs_used']}/{args.n})")
    print(f"  TF1 IoU (all words):          mean={tf1['iou_all_words']['mean']:.4f}")
    print(f"  TF1 IoU (no stopwords):       mean={tf1['iou_no_stopwords']['mean']:.4f}")
    print(f"  unique morals: {tf1['moral_unique_count']} / {tf1['n_pairs_used']}")
    print(f"  per-chunk unique counts: {[c['unique'] for c in tf1['per_chunk']]}")

    summary = {
        "timestamp": timestamp,
        "args": vars(args),
        "morables": morables,
        "tf1": tf1,
        "ratio_no_stopwords_tf1_over_morables":
            tf1["iou_no_stopwords"]["mean"] / morables["iou_no_stopwords"]["mean"]
            if morables["iou_no_stopwords"]["mean"] > 0 else None,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "samples.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    write_examples_md(out_dir / "examples.md", rows)

    print(f"\nDone. summary.json, samples.jsonl, examples.md in {out_dir}")


if __name__ == "__main__":
    main()
