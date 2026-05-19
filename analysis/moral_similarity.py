"""
Compute pairwise cosine similarities between all 709 morals.
Outputs a sorted table so you can spot near-duplicates at a glance.

Run via:  ./run.sh analysis/moral_similarity.py
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA = Path(__file__).parent.parent / "data/processed/morals_corpus.json"
OUT_CSV = Path(__file__).parent / "moral_similarity_pairs.csv"
OUT_MD = Path(__file__).parent / "moral_similarity_report.md"

SIM_THRESHOLD_HIGH = 0.90
SIM_THRESHOLD_MED = 0.80


def load_morals():
    with open(DATA) as f:
        rows = json.load(f)
    return [(r["doc_id"], r["fable_id"], r["text"]) for r in rows]


def embed(morals, texts):
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    print(f"Embedding {len(texts)} morals...")
    # BGE uses this prefix for symmetric (passage↔passage) similarity
    prefixed = ["Represent this sentence: " + t for t in texts]
    embeddings = model.encode(prefixed, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


def compute_pairs(ids, fable_ids, texts, embeddings):
    n = len(texts)
    # Full cosine similarity matrix (embeddings already L2-normalised)
    sim_matrix = np.dot(embeddings, embeddings.T)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            same_fable = fable_ids[i] == fable_ids[j]
            pairs.append((sim, ids[i], ids[j], fable_ids[i], fable_ids[j], texts[i], texts[j], same_fable))
    pairs.sort(key=lambda x: -x[0])
    return pairs


def write_csv(pairs):
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["similarity", "id_a", "id_b", "fable_a", "fable_b", "same_fable", "moral_a", "moral_b"])
        for sim, id_a, id_b, fa, fb, ta, tb, same in pairs:
            writer.writerow([f"{sim:.4f}", id_a, id_b, fa, fb, same, ta, tb])
    print(f"Wrote {len(pairs)} pairs → {OUT_CSV}")


def write_report(pairs):
    exact = [(s, a, b, fa, fb, ta, tb, sf) for s, a, b, fa, fb, ta, tb, sf in pairs if s >= 0.9999]
    near = [(s, a, b, fa, fb, ta, tb, sf) for s, a, b, fa, fb, ta, tb, sf in pairs if SIM_THRESHOLD_HIGH <= s < 0.9999]
    med = [(s, a, b, fa, fb, ta, tb, sf) for s, a, b, fa, fb, ta, tb, sf in pairs if SIM_THRESHOLD_MED <= s < SIM_THRESHOLD_HIGH]

    # Only cross-fable pairs are problematic
    near_diff = [(s, a, b, fa, fb, ta, tb, sf) for s, a, b, fa, fb, ta, tb, sf in near if not sf]
    med_diff = [(s, a, b, fa, fb, ta, tb, sf) for s, a, b, fa, fb, ta, tb, sf in med if not sf]

    lines = [
        "# Moral Similarity Report",
        "",
        f"**Total morals:** 709  |  **Total pairs checked:** {len(pairs):,}",
        "",
        "## Quick summary",
        "",
        "| Tier | Count | Cross-fable (problematic) |",
        "|------|-------|--------------------------|",
        f"| Exact duplicates (sim = 1.000) | {len(exact)} | {sum(1 for p in exact if not p[7])} |",
        f"| Near-duplicates (0.90 ≤ sim < 1.00) | {len(near)} | **{len(near_diff)}** |",
        f"| Highly similar (0.80 ≤ sim < 0.90) | {len(med)} | {len(med_diff)} |",
        "",
        "---",
        "",
        "## Exact duplicates (sim = 1.000) — same text, different fable",
        "",
        "These are the hardest annotation problems: identical moral string → two different correct fables.",
        "The model is penalised for picking either one.",
        "",
        "| Moral text | Fable A | Fable B |",
        "|------------|---------|---------|",
    ]
    seen_exact = set()
    for sim, id_a, id_b, fa, fb, ta, tb, same in exact:
        if same:
            continue
        key = tuple(sorted([ta, tb]))
        if key in seen_exact:
            continue
        seen_exact.add(key)
        # deduplicate fable ids
        lines.append(f"| {ta} | {fa} | {fb} |")

    lines += [
        "",
        "---",
        "",
        "## Near-duplicate pairs (0.90 ≤ sim < 1.00) — different fables",
        "",
        "Sorted by similarity descending. These morals say the same thing in slightly different words,",
        "yet map to different fables. Retrieval errors on these are almost certainly annotation ambiguity.",
        "",
        "| Sim | Moral A | Moral B |",
        "|-----|---------|---------|",
    ]
    for sim, id_a, id_b, fa, fb, ta, tb, same in near_diff:
        lines.append(f"| {sim:.3f} | {ta} | {tb} |")

    lines += [
        "",
        "---",
        "",
        "## Highly similar pairs (0.80 ≤ sim < 0.90) — different fables",
        "",
        "These overlap semantically. Retrieval errors here may be model errors OR annotation ambiguity — needs human review.",
        "",
        "| Sim | Moral A | Moral B |",
        "|-----|---------|---------|",
    ]
    for sim, id_a, id_b, fa, fb, ta, tb, same in med_diff:
        lines.append(f"| {sim:.3f} | {ta} | {tb} |")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote report → {OUT_MD}")


def main():
    morals = load_morals()
    ids = [m[0] for m in morals]
    fable_ids = [m[1] for m in morals]
    texts = [m[2] for m in morals]

    embeddings = embed(morals, texts)
    pairs = compute_pairs(ids, fable_ids, texts, embeddings)

    write_csv(pairs)
    write_report(pairs)

    high = [p for p in pairs if p[0] >= SIM_THRESHOLD_HIGH]
    print(f"\nTop 30 most similar pairs:")
    print(f"{'Sim':>6}  {'SameFable':>9}  Moral A  ↔  Moral B")
    print("-" * 90)
    for sim, id_a, id_b, fa, fb, ta, tb, same in pairs[:30]:
        flag = "same" if same else "DIFF"
        short_a = ta[:45].ljust(45)
        short_b = tb[:45]
        print(f"{sim:6.3f}  {flag:>9}  {short_a}  ↔  {short_b}")


if __name__ == "__main__":
    main()
