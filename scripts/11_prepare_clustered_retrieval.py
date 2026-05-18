"""
Prepare clustered MORABLES retrieval data.

Creates a benchmark with:
- 709 fable documents
- 669 unique moral-text queries
- multi-label qrels from each moral text to every fable in its semantic cluster

The existing data/processed files are not modified.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
CLUSTERS_PATH = ROOT / "analysis" / "clusters_full.json"
OUT_DIR = ROOT / "data" / "clustered"


def _norm(text: str) -> str:
    return text.strip()


def _idx_from_id(doc_id: str) -> int:
    return int(doc_id.split("_")[1])


def load_inputs() -> tuple[list[dict], list[dict], list[dict]]:
    with open(PROCESSED_DIR / "fables_corpus.json") as f:
        fables = json.load(f)
    with open(PROCESSED_DIR / "morals_corpus.json") as f:
        morals = json.load(f)
    with open(CLUSTERS_PATH) as f:
        clusters = json.load(f)
    return fables, morals, clusters


def validate_clusters(clusters: list[dict], fables: list[dict], morals: list[dict]) -> None:
    cluster_ids = [c["cluster_id"] for c in clusters]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("Cluster IDs are not unique")

    expected_fables = {f["doc_id"] for f in fables}
    cluster_fables = [fid for c in clusters for fid in c["fables"]]
    if len(cluster_fables) != len(set(cluster_fables)):
        raise ValueError("A fable appears in more than one cluster")
    if set(cluster_fables) != expected_fables:
        missing = sorted(expected_fables - set(cluster_fables))[:10]
        extra = sorted(set(cluster_fables) - expected_fables)[:10]
        raise ValueError(f"Cluster fable coverage mismatch; missing={missing}, extra={extra}")

    processed_morals = {_norm(m["text"]) for m in morals}
    cluster_morals = {_norm(m) for c in clusters for m in c["moral_set"]}
    canonical_only = cluster_morals - processed_morals
    if canonical_only:
        print(
            "Note: "
            f"{len(canonical_only)} cluster moral text(s) are canonical-only "
            "and do not appear verbatim in data/processed/morals_corpus.json."
        )


def build_clustered_data(
    fables: list[dict],
    morals: list[dict],
    clusters: list[dict],
) -> dict[str, list[dict]]:
    moral_to_cluster: dict[str, dict] = {}
    fable_to_cluster: dict[str, dict] = {}

    for cluster in clusters:
        for moral_text in cluster["moral_set"]:
            moral_to_cluster[_norm(moral_text)] = cluster
        for fable_id in cluster["fables"]:
            fable_to_cluster[fable_id] = cluster

    first_seen_moral_order = {}
    for idx, moral in enumerate(morals):
        first_seen_moral_order.setdefault(_norm(moral["text"]), idx)

    cluster_moral_texts = [_norm(m) for c in clusters for m in c["moral_set"]]
    ordered_cluster_morals = sorted(
        set(cluster_moral_texts),
        key=lambda text: first_seen_moral_order.get(text, len(first_seen_moral_order)),
    )

    unique_morals = []
    for text in ordered_cluster_morals:
        cluster = moral_to_cluster[text]
        query_id = f"moral_unique_{len(unique_morals):04d}"
        unique_morals.append(
            {
                "doc_id": query_id,
                "text": text,
                "cluster_id": cluster["cluster_id"],
                "cluster_type": cluster["type"],
                "relevant_fable_ids": cluster["fables"],
                "cluster_moral_set": cluster["moral_set"],
            }
        )

    qrels_moral_to_fable = []
    moral_to_cluster_rows = []
    for moral in unique_morals:
        moral_to_cluster_rows.append(
            {
                "query_id": moral["doc_id"],
                "text": moral["text"],
                "cluster_id": moral["cluster_id"],
            }
        )
        for fable_id in moral["relevant_fable_ids"]:
            qrels_moral_to_fable.append(
                {
                    "query_id": moral["doc_id"],
                    "doc_id": fable_id,
                    "relevance": 1,
                }
            )

    qrels_fable_to_moral = []
    for fable in fables:
        fable_id = fable["doc_id"]
        cluster = fable_to_cluster[fable_id]
        for moral_text in cluster["moral_set"]:
            query_id = next(
                m["doc_id"] for m in unique_morals if m["text"] == _norm(moral_text)
            )
            qrels_fable_to_moral.append(
                {
                    "query_id": fable_id,
                    "doc_id": query_id,
                    "relevance": 1,
                }
            )

    cluster_mapping = []
    for cluster in clusters:
        cluster_mapping.append(
            {
                "cluster_id": cluster["cluster_id"],
                "type": cluster["type"],
                "moral_set": cluster["moral_set"],
                "fables": cluster["fables"],
                "n_morals": len(cluster["moral_set"]),
                "n_fables": len(cluster["fables"]),
            }
        )

    return {
        "fables_corpus.json": fables,
        "morals_unique_corpus.json": unique_morals,
        "qrels_moral_to_fable_clustered.json": qrels_moral_to_fable,
        "qrels_fable_to_moral_clustered.json": qrels_fable_to_moral,
        "moral_to_cluster.json": moral_to_cluster_rows,
        "cluster_mapping.json": cluster_mapping,
    }


def write_outputs(outputs: dict[str, list[dict]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, data in outputs.items():
        path = OUT_DIR / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {path.relative_to(ROOT)}: {len(data)} entries")


def main() -> None:
    fables, morals, clusters = load_inputs()
    validate_clusters(clusters, fables, morals)
    outputs = build_clustered_data(fables, morals, clusters)
    write_outputs(outputs)

    qrels = outputs["qrels_moral_to_fable_clustered.json"]
    cluster_mapping = outputs["cluster_mapping.json"]
    print("\nClustered retrieval summary:")
    print(f"  Fable documents: {len(outputs['fables_corpus.json'])}")
    print(f"  Unique moral queries: {len(outputs['morals_unique_corpus.json'])}")
    print(f"  Moral clusters: {len(cluster_mapping)}")
    print(f"  Moral->fable qrel rows: {len(qrels)}")
    print(f"  Fables covered: {sum(c['n_fables'] for c in cluster_mapping)}")


if __name__ == "__main__":
    main()
