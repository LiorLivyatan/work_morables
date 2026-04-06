"""lib/pipeline/matrix_aggregator.py — Aggregate stage 2 retrieval results into final analysis."""

import json
from pathlib import Path
import numpy as np


def aggregate_matrix(run_dir: Path):
    run_dir = Path(run_dir)
    results_dir = run_dir / "retrieval_results"

    if not results_dir.exists():
        print("  [matrix_aggregator] No retrieval_results directory found.")
        return

    # Results keyed by: ablation -> metric -> gen_model -> embed_model -> value
    ablations = ["raw_raw", "summary_only", "paraphrase_only", "full", "rrf"]
    metrics_to_track = ["Recall@1", "Recall@5", "MRR"]

    gen_models = set()
    embed_models = set()

    # We will load all results into memory
    results_data = []

    for file_path in results_dir.glob("*.json"):
        with open(file_path) as f:
            data = json.load(f)
            results_data.append(data)
            gen_models.add(data["gen_model"])
            embed_models.add(data["embed_model"])

    gen_models = sorted(list(gen_models))
    embed_models = sorted(list(embed_models))

    # Helper to construct grids
    def _build_grid(ablation: str, metric: str):
        grid = []
        for g in gen_models:
            row = []
            for e in embed_models:
                # Find matching record
                val = next((r["metrics"].get(metric, 0) for r in results_data
                            if r["gen_model"] == g and r["embed_model"] == e and r["ablation"] == ablation), 0.0)
                row.append(val)
            grid.append(row)
        return {
            "rows": gen_models,
            "cols": embed_models,
            "values": grid
        }

    all_ablations = {}
    for ab in ablations:
        all_ablations[ab] = {}
        for m in metrics_to_track:
            all_ablations[ab][m] = _build_grid(ab, m)

    # Main matrix summary file
    matrix_summary = {
        "ablation": "full",
        "metric": "Recall@1",
        "matrix": all_ablations.get("full", {}).get("Recall@1", {}),
        "all_ablations": all_ablations
    }

    with open(run_dir / "matrix_summary.json", "w") as f:
        json.dump(matrix_summary, f, indent=2)

    # Compute Rankings
    # For rankings, we compute averages across all embed_models for a given gen_model in the "full" ablation, and vice versa.
    # best pair is naturally the max in "full" ablation.
    full_results = [r for r in results_data if r["ablation"] == "full"]

    best_pair = None
    best_pair_r1 = -1

    gen_scores = {g: {"r1": [], "mrr": []} for g in gen_models}
    embed_scores = {e: {"r1": [], "mrr": []} for e in embed_models}

    for r in full_results:
        g = r["gen_model"]
        e = r["embed_model"]
        r1 = r["metrics"].get("Recall@1", 0)
        mrr = r["metrics"].get("MRR", 0)

        gen_scores[g]["r1"].append(r1)
        gen_scores[g]["mrr"].append(mrr)

        embed_scores[e]["r1"].append(r1)
        embed_scores[e]["mrr"].append(mrr)

        if r1 > best_pair_r1:
            best_pair_r1 = r1
            best_pair = {
                "gen_model": g,
                "embed_model": e,
                "ablation": "full",
                "Recall@1": r1,
                "MRR": mrr
            }

    # Average scores
    gen_avgs = []
    for g, vals in gen_scores.items():
        if vals["r1"]:
            gen_avgs.append({
                "name": g,
                "avg_recall_at_1": round(float(np.mean(vals["r1"])), 4),
                "avg_mrr": round(float(np.mean(vals["mrr"])), 4)
            })

    embed_avgs = []
    for e, vals in embed_scores.items():
        if vals["r1"]:
            embed_avgs.append({
                "name": e,
                "avg_recall_at_1": round(float(np.mean(vals["r1"])), 4),
                "avg_mrr": round(float(np.mean(vals["mrr"])), 4)
            })

    gen_avgs.sort(key=lambda x: x["avg_recall_at_1"], reverse=True)
    embed_avgs.sort(key=lambda x: x["avg_recall_at_1"], reverse=True)

    best_gen_model = gen_avgs[0] if gen_avgs else None
    best_embed_model = embed_avgs[0] if embed_avgs else None

    # Impact analysis: variance across the marginal avgs
    gen_variance = round(float(np.var([x["avg_recall_at_1"] for x in gen_avgs])), 5) if gen_avgs else 0.0
    embed_variance = round(float(np.var([x["avg_recall_at_1"] for x in embed_avgs])), 5) if embed_avgs else 0.0
    dominant_factor = "generation" if gen_variance > embed_variance else "embedding"

    rankings = {
        "best_gen_model": best_gen_model,
        "best_embed_model": best_embed_model,
        "best_pair": best_pair,
        "impact_analysis": {
            "gen_model_variance": gen_variance,
            "embed_model_variance": embed_variance,
            "dominant_factor": dominant_factor
        }
    }

    with open(run_dir / "rankings.json", "w") as f:
        json.dump(rankings, f, indent=2)

    print("\n  ── Aggregation Complete ──")
    if best_pair:
        print(f"  Best Pair (R@1={best_pair['Recall@1']}): {best_pair['gen_model']} x {best_pair['embed_model']}")
    print(f"  Dominant Factor: {dominant_factor}")
