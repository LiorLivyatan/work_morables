"""
06_layer_probing/analyze.py
==============================
Where in embedding space do morals and fables sit? UMAP visualisation +
intra/inter pair distance analysis. Optional: MRR as a function of layer index.

NOTE: The UMAP plot and distance analysis only need the final .npy embeddings
(same as all other analyses). The layer-by-layer MRR curve requires re-running
the model with output_hidden_states=True — this is GPU-intensive and should only
be done if the other analyses suggest a geometric issue.

Outputs:
  umap_morals_fables.png       — 2D UMAP: morals + fables, coloured by gt pair match
  intra_inter_distance.png     — intra-pair (moral, gt-fable) vs inter-pair distances
  distance_stats.csv           — mean/std intra vs inter distances
  [optional] layer_mrr_curve.png — MRR@10 at each transformer layer (requires --run_layers)

FIXED:
  - UMAP projection to 2D
  - Intra-pair = (moral_i, gt_fable_i) cosine distance
  - Inter-pair = random sample of (moral_i, fable_j≠gt) pairs

CONFIGURABLE:
  --moral_embs     path to moral_embs.npy
  --doc_embs       path to doc_embs.npy
  --label          experiment name
  --n_neighbors    UMAP n_neighbors (default: 15)
  --colour_by      gt_pair | fable_length (default: gt_pair)
  --run_layers     if set, also run per-layer MRR (requires model + GPU)
  --model_id       HF model ID (only needed with --run_layers)
  --output_dir     where to save outputs (default: results/)

TODO (layer probing):
  Implement per-layer embedding extraction when --run_layers is passed.
  Load the model with output_hidden_states=True, extract embeddings at each layer,
  compute MRR@10 per layer, and plot. This tells us at which layer the moral/fable
  separation emerges.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.lib.loader import ExperimentConfig, load_dataset, load_embeddings, compute_rankings
from analysis.lib.plotting import setup_style, save_fig, COLOURS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--moral_embs",  required=True)
    p.add_argument("--doc_embs",    required=True)
    p.add_argument("--label",       required=True)
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--colour_by",   default="gt_pair",
                   choices=["gt_pair", "fable_length"])
    p.add_argument("--run_layers",  action="store_true",
                   help="Also run per-layer MRR curve (GPU required)")
    p.add_argument("--model_id",    default=None,
                   help="HF model ID for --run_layers")
    p.add_argument("--output_dir",  default=str(Path(__file__).parent / "results"))
    return p.parse_args()


def main():
    args = parse_args()
    setup_style()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[06_layer_probing] {args.label}")

    try:
        from umap import UMAP
    except ImportError:
        print("  umap-learn not installed. Run: uv add umap-learn")
        return

    fables, morals, qrels = load_dataset()
    cfg = ExperimentConfig(
        moral_embs_path=args.moral_embs,
        doc_embs_path=args.doc_embs,
        label=args.label,
    )
    moral_embs, doc_embs = load_embeddings(cfg)
    rankings = compute_rankings(moral_embs, doc_embs, qrels)

    fable_texts = [f["text"] for f in fables]
    n_morals = moral_embs.shape[0]
    n_fables = doc_embs.shape[0]

    # ── UMAP ─────────────────────────────────────────────────────────────────
    print("  Running UMAP …")
    all_embs = np.vstack([moral_embs, doc_embs])   # (n_morals + n_fables, D)
    reducer  = UMAP(n_neighbors=args.n_neighbors, min_dist=0.1, random_state=42)
    coords   = reducer.fit_transform(all_embs)

    moral_coords = coords[:n_morals]
    fable_coords = coords[n_morals:]

    fig, ax = plt.subplots(figsize=(10, 7))
    # Draw connecting lines between each (moral, gt fable) pair
    for q_idx, gt_fable in qrels.items():
        ax.plot(
            [moral_coords[q_idx, 0], fable_coords[gt_fable, 0]],
            [moral_coords[q_idx, 1], fable_coords[gt_fable, 1]],
            alpha=0.1, color=COLOURS["neutral"], linewidth=0.5,
        )

    if args.colour_by == "fable_length":
        lengths = [len(t.split()) for t in fable_texts]
        sc = ax.scatter(fable_coords[:, 0], fable_coords[:, 1],
                        c=lengths, cmap="viridis", s=15, alpha=0.8, label="Fables")
        plt.colorbar(sc, ax=ax, label="Word count")
        ax.scatter(moral_coords[:, 0], moral_coords[:, 1],
                   c=COLOURS["secondary"], s=15, alpha=0.8, marker="^", label="Morals")
    else:
        ax.scatter(fable_coords[:, 0], fable_coords[:, 1],
                   c=COLOURS["primary"], s=15, alpha=0.6, label="Fables")
        ax.scatter(moral_coords[:, 0], moral_coords[:, 1],
                   c=COLOURS["secondary"], s=15, alpha=0.8, marker="^", label="Morals")

    ax.set_title(f"UMAP — {args.label}")
    ax.legend()
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    save_fig(str(out / "umap_morals_fables.png"), fig)

    # ── Intra vs inter distance ───────────────────────────────────────────────
    rng = np.random.default_rng(42)
    intra_sims = [
        float(moral_embs[q] @ doc_embs[gt])
        for q, gt in qrels.items()
    ]
    # Random inter-pair sample
    q_idxs = rng.integers(0, n_morals, 500)
    d_idxs = rng.integers(0, n_fables, 500)
    inter_sims = [
        float(moral_embs[q] @ doc_embs[d])
        for q, d in zip(q_idxs, d_idxs)
        if qrels.get(int(q)) != int(d)
    ]

    with open(out / "distance_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_type", "mean_sim", "std_sim", "n"])
        w.writerow(["intra (moral, gt-fable)",  f"{np.mean(intra_sims):.4f}",
                    f"{np.std(intra_sims):.4f}", len(intra_sims)])
        w.writerow(["inter (moral, random-fable)", f"{np.mean(inter_sims):.4f}",
                    f"{np.std(inter_sims):.4f}", len(inter_sims)])
    print(f"  [saved] {out / 'distance_stats.csv'}")

    fig, ax = plt.subplots()
    bins = np.linspace(-0.2, 1.0, 40)
    ax.hist(inter_sims, bins=bins, alpha=0.6, color=COLOURS["neutral"],
            label=f"Inter-pair (random) mean={np.mean(inter_sims):.3f}", density=True)
    ax.hist(intra_sims, bins=bins, alpha=0.7, color=COLOURS["gt"],
            label=f"Intra-pair (gt) mean={np.mean(intra_sims):.3f}", density=True)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"Intra vs inter-pair similarity — {args.label}")
    ax.legend()
    save_fig(str(out / "intra_inter_distance.png"), fig)

    if args.run_layers:
        print("  --run_layers not yet implemented. See TODO in script header.")

    print("  Done.\n")


if __name__ == "__main__":
    main()
