"""
Build a MORABLES-shaped derivative of TF1-EN-3M under
data/external/tf1_synthetic/processed/ from the diagnostic's samples.jsonl
cache. See docs/superpowers/specs/2026-05-06-tf1-synthetic-corpus-design.md.
"""
import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from finetuning.lib import notify  # noqa: E402

DEFAULT_RUNS_DIR = ROOT / "experiments" / "11_tf1_diagnostic" / "results" / "runs"
DEFAULT_OUT = ROOT / "data" / "external" / "tf1_synthetic"

LEAKAGE_PATTERNS = [
    re.compile(r"the\s+(moral|lesson|teaching|takeaway)\s+(of|is|here\s+is)", re.IGNORECASE),
    re.compile(r"this\s+(story|fable|tale)\s+teaches\s+(us|that)", re.IGNORECASE),
    re.compile(r"^\s*moral\s*:", re.IGNORECASE | re.MULTILINE),
]

WORD_RE = re.compile(r"\b\w+\b")
STOP_WORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "and", "but", "or", "nor", "not", "so", "if", "than",
    "that", "this", "it", "its", "his", "her", "their", "who", "which", "what", "when",
    "where", "how", "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "only", "own", "same", "he", "she", "they", "them", "him", "we", "you",
    "i", "me", "my", "your", "our",
}


def has_explicit_moral(fable_text: str, moral_text: str) -> bool:
    """True if the fable contains an explicit restatement of the moral.

    Two layers:
    1. Regex patterns catching common LLM closing phrases like
       "The moral of the story is...".
    2. Per-sentence content-word overlap of at least 70% with the moral
       (catches near-verbatim moral restatements without the regex tells).

    Morals with fewer than 2 content words are exempt from layer 2 to
    avoid spurious matches on tiny morals.
    """
    if not fable_text or not moral_text:
        return False
    for pattern in LEAKAGE_PATTERNS:
        if pattern.search(fable_text):
            return True
    moral_content = set(WORD_RE.findall(moral_text.lower())) - STOP_WORDS
    if len(moral_content) < 2:
        return False
    for sentence in re.split(r"[.!?]", fable_text):
        sent_content = set(WORD_RE.findall(sentence.lower())) - STOP_WORDS
        if moral_content and len(moral_content & sent_content) / len(moral_content) >= 0.70:
            return True
    return False


def group_by_moral(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        key = r["moral"].lower().strip()
        out.setdefault(key, []).append(r)
    return out


def first_seen_order(rows: list[dict]) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for r in rows:
        key = r["moral"].lower().strip()
        if key not in seen:
            seen.add(key)
            order.append(key)
    return order


def assign_moral_ids(unique_morals: list[str]) -> dict[str, str]:
    return {m: f"moral_tf1_{i:03d}" for i, m in enumerate(unique_morals)}


def sample_n_per_moral(grouped: dict[str, list[dict]], n: int, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {}
    for moral, rows in grouped.items():
        if len(rows) < n:
            raise ValueError(
                f"Moral has only {len(rows)} cached rows, need {n}: {moral!r}"
            )
        sorted_rows = sorted(rows, key=lambda r: r["idx"])
        out[moral] = rng.sample(sorted_rows, n)
    return out


def select_low_iou_clean(grouped: dict[str, list[dict]], n: int) -> dict[str, list[dict]]:
    """Pick the n fables with lowest iou_no_stop per moral, after filtering
    out fables that explicitly restate the moral.

    Each row must contain an `iou_no_stop` float field (produced by
    experiments/11_tf1_diagnostic/check_iou.py during sample dumping).
    """
    out: dict[str, list[dict]] = {}
    for moral, rows in grouped.items():
        clean = [r for r in rows if not has_explicit_moral(r["fable"], r["moral"])]
        if len(clean) < n:
            n_word = f"{len(clean)} fable" + ("" if len(clean) == 1 else "s")
            raise ValueError(
                f"After leakage filter, only {n_word} remain for "
                f"{moral!r}; need {n}. Consider re-streaming more TF1 rows."
            )
        clean.sort(key=lambda r: r["iou_no_stop"])
        out[moral] = clean[:n]
    return out


def build_morals_corpus(unique_morals: list[str], moral_ids: dict[str, str]) -> list[dict]:
    return [{"doc_id": moral_ids[m], "text": m} for m in unique_morals]


def build_fables_corpus(
    sampled: dict[str, list[dict]],
    unique_morals: list[str],
    moral_ids: dict[str, str],
    n: int,
) -> list[dict]:
    out: list[dict] = []
    for moral_idx, moral_text in enumerate(unique_morals):
        rows = sampled[moral_text]
        assert len(rows) == n, (
            f"Expected {n} rows for moral {moral_text!r}, got {len(rows)}"
        )
        for i, row in enumerate(rows):
            fable_id = f"fable_tf1_{moral_idx * n + i:05d}"
            out.append({
                "doc_id": fable_id,
                "text": row["fable"],
                "moral_id": moral_ids[moral_text],
                "source_idx": row["idx"],
                "source_chunk": row["chunk"],
                "prompt_hash": row["prompt_hash"],
            })
    return out


def build_qrels_moral_to_fable(fables_corpus: list[dict]) -> list[dict]:
    return [
        {"query_id": f["moral_id"], "doc_id": f["doc_id"], "relevance": 1}
        for f in fables_corpus
    ]


def build_qrels_fable_to_moral(fables_corpus: list[dict]) -> list[dict]:
    return [
        {"query_id": f["doc_id"], "doc_id": f["moral_id"], "relevance": 1}
        for f in fables_corpus
    ]


def _latest_samples_path() -> Path:
    runs = sorted(DEFAULT_RUNS_DIR.glob("*/samples.jsonl"))
    if not runs:
        raise FileNotFoundError(
            f"No samples.jsonl found under {DEFAULT_RUNS_DIR}. "
            "Run check_iou.py first."
        )
    return runs[-1]


def _read_samples(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _write_readme(out_dir: Path, n: int, seed: int, source: Path,
                  n_morals: int, selection: str = "random") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_source = source.relative_to(ROOT) if source.is_relative_to(ROOT) else source

    # Only the random strategy depends on the seed; for low_iou_clean the
    # selection is deterministic (sorted by iou) and seed is irrelevant.
    if selection == "random":
        seed_line = f"- Seed: {seed}\n"
        build_cmd = (
            f"./run.sh experiments/11_tf1_diagnostic/build_tf1_corpus.py "
            f"--selection {selection} --n {n} --seed {seed}"
        )
    else:
        seed_line = ""
        build_cmd = (
            f"./run.sh experiments/11_tf1_diagnostic/build_tf1_corpus.py "
            f"--selection {selection} --n {n}"
        )

    readme = f"""# TF1-EN-3M synthetic — MORABLES-shaped derivative

Source: https://huggingface.co/datasets/klusai/ds-tf1-en-3m (MIT)
Paper: https://arxiv.org/abs/2504.20605

Built from {rel_source}
via experiments/11_tf1_diagnostic/check_iou.py (--n 50000 --chunks 10).

## Snapshot (this build)

- Selection strategy: {selection}
- N per moral: {n}
- Total morals: {n_morals}
- Total fables: {n * n_morals}
{seed_line}- Built: {datetime.now().isoformat(timespec='seconds')}

## Build commands

{build_cmd}
./run.sh experiments/11_tf1_diagnostic/cluster_tf1_morals.py --mode exact --in {out_dir}

See experiments/11_tf1_diagnostic/REPORT.md for the analysis that motivated this derivative.
"""
    (out_dir / "README.md").write_text(readme)


def dedup_by_prompt_hash(rows: list[dict]) -> list[dict]:
    """Keep the first-seen row per prompt_hash.

    The multi-chunk sampling in check_iou.py re-reads the same TF1 source
    rows at different chunk offsets, producing identical (prompt_hash, fable)
    pairs stored under different idx values.  Deduplicating here restores the
    invariant that every fable in the corpus is unique.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        h = r["prompt_hash"]
        if h not in seen:
            seen.add(h)
            out.append(r)
    return out


def run_build(
    samples_path: Path,
    n: int,
    seed: int,
    out_dir: Path,
    expected_unique_morals: int = 100,
    selection: str = "random",
) -> dict:
    rows = dedup_by_prompt_hash(_read_samples(samples_path))
    unique_morals = first_seen_order(rows)
    if len(unique_morals) != expected_unique_morals:
        raise ValueError(
            f"expected {expected_unique_morals} unique morals, got {len(unique_morals)}"
        )

    grouped = group_by_moral(rows)
    moral_ids = assign_moral_ids(unique_morals)
    if selection == "random":
        sampled = sample_n_per_moral(grouped, n=n, seed=seed)
    elif selection == "low_iou_clean":
        sampled = select_low_iou_clean(grouped, n=n)
    else:
        raise ValueError(
            f"unknown selection {selection!r}; expected 'random' or 'low_iou_clean'"
        )

    morals_corpus = build_morals_corpus(unique_morals, moral_ids)
    fables_corpus = build_fables_corpus(sampled, unique_morals, moral_ids, n=n)
    qrels_mtf = build_qrels_moral_to_fable(fables_corpus)
    qrels_ftm = build_qrels_fable_to_moral(fables_corpus)

    hashes = [f["prompt_hash"] for f in fables_corpus]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate prompt_hash in sampled fables")

    processed = out_dir / "processed"
    _write_json(processed / "morals_corpus.json", morals_corpus)
    _write_json(processed / "fables_corpus.json", fables_corpus)
    _write_json(processed / "qrels_moral_to_fable.json", qrels_mtf)
    _write_json(processed / "qrels_fable_to_moral.json", qrels_ftm)
    _write_readme(out_dir, n=n, seed=seed, source=samples_path,
                  n_morals=len(unique_morals), selection=selection)

    return {
        "n_morals": len(unique_morals),
        "n_fables": len(fables_corpus),
        "selection": selection,
        "out_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument(
        "--selection", choices=["random", "low_iou_clean"], default="random",
        help="How to pick n fables per moral. random = uniform sample. "
             "low_iou_clean = filter explicit-moral restatements, then take "
             "the n lowest by iou_no_stop.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output corpus dir. Defaults to data/external/tf1_synthetic/ "
             "for --selection random, or data/external/tf1_synthetic_low_iou/ "
             "for --selection low_iou_clean.",
    )
    parser.add_argument(
        "--expected-unique-morals", type=int, default=100,
        help="invariant from the diagnostic; lower for testing on small samples",
    )
    args = parser.parse_args()

    samples_path = args.source or _latest_samples_path()
    if args.out is not None:
        out_dir = args.out
    elif args.selection == "low_iou_clean":
        out_dir = ROOT / "data" / "external" / "tf1_synthetic_low_iou"
    else:
        out_dir = DEFAULT_OUT

    print(f"Reading samples from {samples_path}")
    print(f"Selection: {args.selection}  Output dir: {out_dir}")
    notify.send(
        f"🛠 build_tf1_corpus starting\n"
        f"source: {samples_path.name}  n={args.n}  seed={args.seed}  selection={args.selection}"
    )
    result = run_build(
        samples_path=samples_path,
        n=args.n,
        seed=args.seed,
        out_dir=out_dir,
        expected_unique_morals=args.expected_unique_morals,
        selection=args.selection,
    )
    print(f"Wrote {result['n_fables']} fables across {result['n_morals']} morals to {result['out_dir']}")
    notify.send(
        f"✅ build_tf1_corpus done\n"
        f"{result['n_fables']} fables across {result['n_morals']} morals  selection={args.selection}"
    )


if __name__ == "__main__":
    main()
