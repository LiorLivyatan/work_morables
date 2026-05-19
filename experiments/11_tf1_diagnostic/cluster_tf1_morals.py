"""
Cluster the 100 unique morals from data/external/tf1_synthetic/processed/ into
semantic groups and emit data/external/tf1_synthetic/clustered/, mirroring
data/clustered/ for MORABLES. See the design spec.
"""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).parent.parent.parent
DEFAULT_IN = ROOT / "data" / "external" / "tf1_synthetic"
DEFAULT_INSPECT_ROOT = ROOT / "experiments" / "11_tf1_diagnostic" / "cluster_inspection"
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
EXACT_THRESHOLD = 0.999


def agglomerative_clusters(sim_matrix: np.ndarray, threshold: float) -> list[list[int]]:
    n = sim_matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    dist = np.clip(1.0 - sim_matrix, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="single")
    labels = fcluster(z, t=1.0 - threshold, criterion="distance")
    groups: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(idx)
    return list(groups.values())


def classify_cluster_type(members: list[int], sim_matrix: np.ndarray) -> str:
    if len(members) == 1:
        return "singleton"
    all_exact = all(
        sim_matrix[a, b] >= EXACT_THRESHOLD
        for i, a in enumerate(members)
        for b in members[i + 1:]
    )
    return "exact" if all_exact else "near"


def pick_canonical_text(members: list[int], moral_texts: list[str], counts: dict[str, int]) -> str:
    def key(i: int) -> tuple[int, int]:
        return (-counts.get(moral_texts[i], 0), i)
    best = sorted(members, key=key)[0]
    return moral_texts[best]


def build_clustered_outputs(
    morals: list[dict],
    qmf: list[dict],
    sim_matrix: np.ndarray,
    counts: dict[str, int],
    threshold: float,
) -> dict[str, list[dict]]:
    moral_texts = [m["text"] for m in morals]
    moral_ids = [m["doc_id"] for m in morals]

    # fable_id grouped by source moral_id (from input qrels)
    fables_for_moral: dict[str, list[str]] = {}
    for row in qmf:
        fables_for_moral.setdefault(row["query_id"], []).append(row["doc_id"])

    clusters = agglomerative_clusters(sim_matrix, threshold=threshold)

    morals_unique: list[dict] = []
    cluster_mapping: list[dict] = []
    moral_to_cluster: list[dict] = []
    qrels_mtf_clustered: list[dict] = []
    qrels_ftm_clustered: list[dict] = []

    for cluster_idx, members in enumerate(sorted(clusters, key=lambda c: (-len(c), c[0]))):
        cluster_id = f"cluster_{cluster_idx:03d}"
        cluster_type = classify_cluster_type(members, sim_matrix)
        canonical_text = pick_canonical_text(members, moral_texts, counts)
        member_moral_ids = [moral_ids[i] for i in members]
        member_texts = [moral_texts[i] for i in members]
        relevant_fable_ids = [
            fid for mid in member_moral_ids for fid in fables_for_moral.get(mid, [])
        ]

        unique_doc_id = f"moral_tf1_unique_{cluster_idx:04d}"
        morals_unique.append({
            "doc_id": unique_doc_id,
            "text": canonical_text,
            "cluster_id": cluster_id,
            "cluster_type": cluster_type,
            "relevant_fable_ids": relevant_fable_ids,
            "cluster_moral_set": member_texts,
        })
        cluster_mapping.append({
            "cluster_id": cluster_id,
            "type": cluster_type,
            "moral_set": member_texts,
            "fables": relevant_fable_ids,
            "n_morals": len(member_texts),
            "n_fables": len(relevant_fable_ids),
        })
        for mid, text in zip(member_moral_ids, member_texts):
            moral_to_cluster.append({
                "query_id": mid,
                "text": text,
                "cluster_id": cluster_id,
            })
        for fid in relevant_fable_ids:
            qrels_mtf_clustered.append({
                "query_id": unique_doc_id, "doc_id": fid, "relevance": 1,
            })
            qrels_ftm_clustered.append({
                "query_id": fid, "doc_id": unique_doc_id, "relevance": 1,
            })

    return {
        "morals_unique_corpus.json": morals_unique,
        "cluster_mapping.json": cluster_mapping,
        "moral_to_cluster.json": moral_to_cluster,
        "qrels_moral_to_fable_clustered.json": qrels_mtf_clustered,
        "qrels_fable_to_moral_clustered.json": qrels_ftm_clustered,
    }


def _load_counts_from_samples(default: dict[str, int] | None = None) -> dict[str, int]:
    """
    Compute per-moral occurrence count from the latest samples.jsonl.
    When `default` is given (testing path), returns it directly.
    """
    if default is not None:
        return default
    runs_root = ROOT / "experiments" / "11_tf1_diagnostic" / "results" / "runs"
    samples = sorted(runs_root.glob("*/samples.jsonl"))
    if not samples:
        raise FileNotFoundError(f"No samples.jsonl under {runs_root}")
    counts: Counter = Counter()
    with samples[-1].open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            counts[r["moral"].lower().strip()] += 1
    return counts


def _embed_morals(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return emb @ emb.T


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def run_cluster(
    in_dir: Path,
    threshold: float,
    inspect_thresholds: list[float],
    inspection_root: Path,
    sim_matrix: np.ndarray | None = None,
    counts_override: dict[str, int] | None = None,
    model_name: str = DEFAULT_MODEL,
) -> Path:
    morals = json.loads((in_dir / "processed" / "morals_corpus.json").read_text())
    qmf = json.loads((in_dir / "processed" / "qrels_moral_to_fable.json").read_text())
    moral_texts = [m["text"] for m in morals]

    if sim_matrix is None:
        print(f"Embedding {len(moral_texts)} morals with {model_name} ...")
        sim_matrix = _embed_morals(moral_texts, model_name)

    counts = _load_counts_from_samples(counts_override)

    # Side-by-side inspection dumps
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inspect_dir = inspection_root / timestamp
    inspect_dir.mkdir(parents=True, exist_ok=True)
    for t in inspect_thresholds:
        clusters_at_t = agglomerative_clusters(sim_matrix, threshold=t)
        payload = [
            {
                "size": len(c),
                "members": [moral_texts[i] for i in c],
            }
            for c in sorted(clusters_at_t, key=lambda c: -len(c))
        ]
        _write_json(inspect_dir / f"clusters_at_{t:.2f}.json", payload)

    # Canonical clustered outputs
    out = build_clustered_outputs(
        morals=morals, qmf=qmf, sim_matrix=sim_matrix,
        counts=counts, threshold=threshold,
    )
    clustered_dir = in_dir / "clustered"
    for filename, data in out.items():
        _write_json(clustered_dir / filename, data)

    # Append clustering summary to the existing README
    n_clusters = len(out["cluster_mapping.json"])
    types = Counter(c["type"] for c in out["cluster_mapping.json"])
    readme_path = in_dir / "README.md"
    readme = readme_path.read_text() if readme_path.exists() else ""
    readme += (
        f"\n\n## Clustering (this run)\n\n"
        f"- Threshold: {threshold}\n"
        f"- Model: {model_name}\n"
        f"- Clusters: {n_clusters}  "
        f"(singleton={types.get('singleton', 0)}, "
        f"near={types.get('near', 0)}, "
        f"exact={types.get('exact', 0)})\n"
        f"- Inspection dumps: experiments/11_tf1_diagnostic/cluster_inspection/{timestamp}/\n"
    )
    readme_path.write_text(readme)

    return clustered_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument(
        "--inspect-thresholds", type=str, default="0.80,0.85,0.90",
        help="comma-separated cosine thresholds to dump alongside the canonical run",
    )
    parser.add_argument("--in", dest="in_dir", type=Path, default=DEFAULT_IN)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    inspect = [float(t) for t in args.inspect_thresholds.split(",")]
    out = run_cluster(
        in_dir=args.in_dir,
        threshold=args.threshold,
        inspect_thresholds=inspect,
        inspection_root=DEFAULT_INSPECT_ROOT,
        model_name=args.model,
    )
    print(f"Wrote clustered outputs to {out}")


if __name__ == "__main__":
    main()
