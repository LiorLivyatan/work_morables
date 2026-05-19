"""
Build a MORABLES-shaped derivative of TF1-EN-3M under
data/external/tf1_synthetic/processed/ from the diagnostic's samples.jsonl
cache. See docs/superpowers/specs/2026-05-06-tf1-synthetic-corpus-design.md.
"""
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DEFAULT_RUNS_DIR = ROOT / "experiments" / "11_tf1_diagnostic" / "results" / "runs"
DEFAULT_OUT = ROOT / "data" / "external" / "tf1_synthetic"


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


def _write_readme(out_dir: Path, n: int, seed: int, source: Path, n_morals: int) -> None:
    try:
        rel_source = source.relative_to(ROOT)
    except ValueError:
        rel_source = source
    readme = f"""# TF1-EN-3M synthetic — MORABLES-shaped derivative

Source: https://huggingface.co/datasets/klusai/ds-tf1-en-3m (MIT)
Paper: https://arxiv.org/abs/2504.20605

Built from {rel_source}
via experiments/11_tf1_diagnostic/check_iou.py (--n 50000 --chunks 10).

## Build commands

./run.sh experiments/11_tf1_diagnostic/build_tf1_corpus.py --n <N> --seed <S>
./run.sh experiments/11_tf1_diagnostic/cluster_tf1_morals.py --threshold <T>

## Snapshot (this build)

- N per moral: {n}
- Total morals: {n_morals}
- Total fables: {n * n_morals}
- Seed: {seed}
- Built: {datetime.now().isoformat(timespec='seconds')}

See experiments/11_tf1_diagnostic/REPORT.md for the analysis that motivated this derivative.
"""
    (out_dir / "README.md").write_text(readme)


def run_build(
    samples_path: Path,
    n: int,
    seed: int,
    out_dir: Path,
    expected_unique_morals: int = 100,
) -> dict:
    rows = _read_samples(samples_path)
    unique_morals = first_seen_order(rows)
    assert len(unique_morals) == expected_unique_morals, (
        f"expected {expected_unique_morals} unique morals, got {len(unique_morals)}"
    )

    grouped = group_by_moral(rows)
    moral_ids = assign_moral_ids(unique_morals)
    sampled = sample_n_per_moral(grouped, n=n, seed=seed)

    morals_corpus = build_morals_corpus(unique_morals, moral_ids)
    fables_corpus = build_fables_corpus(sampled, unique_morals, moral_ids, n=n)
    qrels_mtf = build_qrels_moral_to_fable(fables_corpus)
    qrels_ftm = build_qrels_fable_to_moral(fables_corpus)

    hashes = [f["prompt_hash"] for f in fables_corpus]
    assert len(set(hashes)) == len(hashes), "duplicate prompt_hash in sampled fables"

    processed = out_dir / "processed"
    _write_json(processed / "morals_corpus.json", morals_corpus)
    _write_json(processed / "fables_corpus.json", fables_corpus)
    _write_json(processed / "qrels_moral_to_fable.json", qrels_mtf)
    _write_json(processed / "qrels_fable_to_moral.json", qrels_ftm)
    _write_readme(out_dir, n=n, seed=seed, source=samples_path, n_morals=len(unique_morals))

    return {
        "n_morals": len(unique_morals),
        "n_fables": len(fables_corpus),
        "out_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--expected-unique-morals", type=int, default=100,
        help="invariant from the diagnostic; set lower for testing on small samples",
    )
    args = parser.parse_args()

    samples_path = args.source or _latest_samples_path()
    print(f"Reading samples from {samples_path}")
    result = run_build(
        samples_path=samples_path,
        n=args.n,
        seed=args.seed,
        out_dir=args.out,
        expected_unique_morals=args.expected_unique_morals,
    )
    print(f"Wrote {result['n_fables']} fables across {result['n_morals']} morals to {result['out_dir']}")


if __name__ == "__main__":
    main()
