"""
Steps 3-4-5 orchestrator. For each chosen concept (3 targets + 1 placebo):
  - extract hidden states at the 5 layers (one forward pass over fables);
  - build CAA matched-pair v_C and mean_diff v_C; log quality metrics;
  - run the (layer × alpha) intervention sweep;
  - run null controls in candidate_only mode at cells where targets look like
    they pass;
  - summarise + plot.

Run via: ./run.sh analysis/08_concept_steering/run_intervention.py [--remote --gpu N]
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from lib.config import load_config
from lib.data import load_corpus, build_tag_index
from lib.io import load_json, save_json, save_npy, load_npy, text_hash
from lib.model import load_model, encode, extract_hidden_states, encode_with_intervention
from lib.retrieval import compute_rankings, mrr_at_k
from lib.vectors import (build_matched_pairs, build_caa_vector,
                          build_mean_diff_vector, cosine,
                          matched_pair_quality_metrics)
from lib.intervene import sweep_concept
from lib.nulls import shuffled_tag_indices, random_unit_direction_at_norm
from lib.eval import reciprocal_rank_per_query, summarize_run, stage2_go_no_go
from lib.plotting import plot_specificity_summary


def _candidate_cells(summary: dict, target_concepts: list[str]) -> list[tuple[str, int, float]]:
    out: list[tuple[str, int, float]] = []
    for concept in target_concepts:
        if concept not in summary["cells"]:
            continue
        for layer_key, lc in summary["cells"][concept].items():
            for ai, alpha in enumerate(summary["alphas"]):
                hi = lc["S_ci_hi"][ai]
                if hi is not None and hi < 0.0:
                    layer_int = int(layer_key)
                    out.append((concept, layer_int, alpha))
    return out


def _run_random_direction_check(
    *, cfg, handle, corpus, moral_embs, gt_indices,
    candidate_cells, results_dir, fable_idx_by_id, rr_baseline,
    null_envelopes,
) -> bool:
    """For each candidate (concept, layer, alpha), draw `n_seeds` random unit
    directions at matched L2 norm to the true v_C, intervene, compute S.
    Pass iff the MEDIAN random-direction S lies inside the shuffled-tag null
    envelope at every candidate cell (spec §6.3 criterion 3)."""
    rng = np.random.default_rng(1)
    n_seeds = cfg["null_controls"]["random_direction"]["n_seeds"]

    all_ok = True
    for cname, layer, alpha in candidate_cells:
        try:
            envelope_lo, envelope_hi = null_envelopes[cname][layer][alpha]
        except (KeyError, TypeError):
            all_ok = False
            continue

        v_caa_path = results_dir / "concept_vectors" / f"{cname}_layer{layer}_caa_matched.npy"
        if not v_caa_path.exists():
            all_ok = False
            continue
        v_caa = np.load(v_caa_path)
        target_norm = float(np.linalg.norm(v_caa))

        field, value = cname.split("__", 1)
        tag_index = build_tag_index(
            ROOT / cfg["data"]["metadata_path"],
            fields=[field],
        )
        positives = tag_index[field][value]
        pos_indices = np.array([fable_idx_by_id[p] for p in positives if p in fable_idx_by_id])
        target_mask = np.array([gt in set(pos_indices) for gt in gt_indices])

        S_random: list[float] = []
        for _ in range(n_seeds):
            v_rand = random_unit_direction_at_norm(
                hidden_dim=handle.hidden_dim, target_norm=target_norm, rng=rng,
            )
            embs, _ = encode_with_intervention(
                handle, corpus.fable_texts,
                layer_idx=layer, direction=v_rand, alpha=alpha,
                batch_size=cfg["model"]["batch_size"],
            )
            rr = reciprocal_rank_per_query(compute_rankings(moral_embs, embs), gt_indices)
            dt = (rr[target_mask] - rr_baseline[target_mask]).mean()
            dc = (rr[~target_mask] - rr_baseline[~target_mask]).mean()
            S_random.append(float(dt - dc))

        median_S = float(np.median(S_random))
        if not (envelope_lo <= median_S <= envelope_hi):
            all_ok = False
    return all_ok


def _run_shuffled_tag_nulls(
    *, cfg, handle, corpus, hs_by_layer, moral_embs, gt_indices,
    candidate_cells, results_dir, fable_idx_by_id, rr_baseline,
) -> dict:
    """For each candidate (concept, layer, α), build a null distribution of S
    by replacing the positive set with a random subset of the same size."""
    rng = np.random.default_rng(0)
    n_perms = cfg["null_controls"]["shuffled_tag_caa"]["n_permutations"]

    by_cl = defaultdict(list)
    for c, l, a in candidate_cells:
        by_cl[(c, l)].append(a)

    out: dict = defaultdict(lambda: defaultdict(dict))
    for (cname, layer), alphas_for_cell in by_cl.items():
        field, value = cname.split("__", 1)
        v_caa_path = results_dir / "concept_vectors" / f"{cname}_layer{layer}_caa_matched.npy"
        if not v_caa_path.exists():
            continue
        v_caa = np.load(v_caa_path)
        target_norm = float(np.linalg.norm(v_caa))

        # Recover positives doc-id set via concept name
        from lib.data import build_tag_index
        tag_index = build_tag_index(
            ROOT / cfg["data"]["metadata_path"],
            fields=cfg["discovery"]["metadata_fields"],
        )
        positives = tag_index[field][value]
        n_pos = len([p for p in positives if p in fable_idx_by_id])

        pos_indices = np.array([fable_idx_by_id[p] for p in positives if p in fable_idx_by_id])
        target_mask = np.array([gt in set(pos_indices) for gt in gt_indices])

        perm_sets = shuffled_tag_indices(
            n_total=len(corpus.fable_texts), n_positives=n_pos, n_perms=n_perms, rng=rng,
        )
        S_perms: dict[float, list[float]] = {a: [] for a in alphas_for_cell}
        for perm_pos in perm_sets:
            perm_neg = np.setdiff1d(np.arange(len(corpus.fable_texts)), perm_pos)
            v_perm = (hs_by_layer[layer][perm_pos].mean(axis=0)
                       - hs_by_layer[layer][perm_neg].mean(axis=0))
            v_perm /= max(np.linalg.norm(v_perm), 1e-12)
            v_perm *= target_norm
            for alpha in alphas_for_cell:
                embs, _ = encode_with_intervention(
                    handle, corpus.fable_texts,
                    layer_idx=layer, direction=v_perm, alpha=alpha,
                    batch_size=cfg["model"]["batch_size"],
                )
                rr = reciprocal_rank_per_query(compute_rankings(moral_embs, embs), gt_indices)
                dt = (rr[target_mask] - rr_baseline[target_mask]).mean()
                dc = (rr[~target_mask] - rr_baseline[~target_mask]).mean()
                S_perms[alpha].append(float(dt - dc))

        for alpha in alphas_for_cell:
            arr = np.array(S_perms[alpha])
            out[cname][layer][alpha] = (float(np.quantile(arr, 0.025)),
                                         float(np.quantile(arr, 0.975)))
    return {k: dict(v) for k, v in out.items()}


def main(config_path: Path, force: bool = False) -> int:
    cfg = load_config(config_path)
    cache_dir   = ROOT / cfg["output"]["cache_dir"]
    results_dir = ROOT / cfg["output"]["results_dir"]
    raw_layers = cfg["vectors"]["layers"]
    alphas = cfg["intervention"]["alphas"]

    corpus = load_corpus(
        morals_path=ROOT / cfg["data"]["morals_path"],
        fables_path=ROOT / cfg["data"]["fables_path"],
        qrels_path =ROOT / "data/processed/qrels_moral_to_fable.json",
    )
    baseline = load_json(results_dir / "ranks_baseline.json")
    tag_index = build_tag_index(ROOT / cfg["data"]["metadata_path"],
                                  fields=cfg["discovery"]["metadata_fields"])
    metadata = load_json(ROOT / cfg["data"]["metadata_path"])

    handle = load_model(cfg)
    # Canonicalise to block indices in [0, n_layers - 1]. -1 -> n_layers - 1.
    # This single resolution is reused for: hidden-state extraction, vector
    # building, sweep, summary keying, and null cell lookup.
    layers = [(handle.n_layers - 1) if l == -1 else l for l in raw_layers]
    notify.send(
        f"🚀 08_concept_steering: run_intervention starting\n"
        f"targets: {cfg['concepts']['targets']}\n"
        f"placebo: {cfg['concepts']['placebo']}\n"
        f"layers={layers} alphas={alphas}"
    )

    # 1. Hidden states for all fables at the requested block indices (single pass)
    hs_cache = cache_dir / f"fable_hidden_states_{text_hash(corpus.fable_texts)}.npz"
    if not force and hs_cache.exists():
        npz = np.load(hs_cache)
        hs_by_layer = {int(k): npz[k] for k in npz.files}
    else:
        hs_by_layer = extract_hidden_states(
            handle, corpus.fable_texts, layers=layers,
            batch_size=cfg["model"]["batch_size"],
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(hs_cache, **{str(l): h for l, h in hs_by_layer.items()})

    # 2. Moral embeddings
    moral_cache = cache_dir / f"moral_embs_{text_hash(corpus.moral_texts)}.npy"
    if moral_cache.exists():
        moral_embs = load_npy(moral_cache)
    else:
        moral_embs = encode(handle, corpus.moral_texts, cfg["model"]["batch_size"])
        save_npy(moral_cache, moral_embs)

    # 3. Per-concept vector build + sweep
    concepts_to_run = list(cfg["concepts"]["targets"]) + list(cfg["concepts"]["placebo"])
    fable_token_lengths = [len(t.split()) for t in corpus.fable_texts]
    fable_idx_by_id = {fid: i for i, fid in enumerate(corpus.fable_doc_ids)}
    gt_indices = np.array([q["gt_fable_idx"] for q in baseline["queries"]])
    # Cap at k=10 so this matches reciprocal_rank_per_query (MRR@10 truthfulness).
    rr_baseline = np.array(
        [(1.0 / q["gt_rank"]) if q["gt_rank"] <= 10 else 0.0 for q in baseline["queries"]],
        dtype=np.float64,
    )

    cells_per_concept: dict[str, list[dict]] = {}
    target_mask_per_concept: dict[str, np.ndarray] = {}

    for spec_entry in concepts_to_run:
        field, value = spec_entry["field"], spec_entry["value"]
        cname = f"{field}__{value}"
        positives = tag_index[field][value]

        cf = ("moral_category" if field in {"characters", "character_roles"}
              else "character_roles")
        cf = cf if cfg["vectors"]["matching"]["cross_field_matching"] else None
        pairs = build_matched_pairs(
            positives=positives, fable_doc_ids=corpus.fable_doc_ids,
            metadata=metadata, fable_token_lengths=fable_token_lengths,
            match_fields=cfg["vectors"]["matching"]["fields"],
            cross_field=cf,
            length_tolerance=cfg["vectors"]["matching"]["length_tolerance"],
        )
        if len(pairs) < cfg["vectors"]["matching"]["min_matched_pairs"]:
            notify.send(f"⚠️ {cname}: only {len(pairs)} matched pairs — skipping")
            continue

        pos_idx = np.array([fable_idx_by_id[p] for p, _ in pairs])
        neg_idx = np.array([fable_idx_by_id[n] for _, n in pairs])
        pos_h = {l: hs_by_layer[l][pos_idx] for l in hs_by_layer}
        neg_h = {l: hs_by_layer[l][neg_idx] for l in hs_by_layer}

        v_caa  = build_caa_vector(pos_h, neg_h)
        all_pos_idx = np.array([fable_idx_by_id[p] for p in positives if p in fable_idx_by_id])
        all_neg_idx = np.setdiff1d(np.arange(len(corpus.fable_texts)), all_pos_idx)
        v_mean = build_mean_diff_vector(hs_by_layer, all_pos_idx, all_neg_idx)
        cos_per_layer = {l: cosine(v_caa[l], v_mean[l]) for l in v_caa}

        for l, vec in v_caa.items():
            save_npy(results_dir / "concept_vectors" / f"{cname}_layer{l}_caa_matched.npy", vec)
        for l, vec in v_mean.items():
            save_npy(results_dir / "concept_vectors" / f"{cname}_layer{l}_mean_diff.npy", vec)

        # Quality metrics
        target_mask = np.array([
            gt in set(all_pos_idx.tolist()) for gt in gt_indices
        ])
        pos_mrr = float(rr_baseline[target_mask].mean()) if target_mask.any() else float("nan")
        neg_mrr = float(rr_baseline[~target_mask].mean()) if (~target_mask).any() else float("nan")
        quality = matched_pair_quality_metrics(
            pairs=pairs, metadata=metadata,
            fable_token_lengths=fable_token_lengths,
            fable_doc_ids=corpus.fable_doc_ids,
            cross_field=cf,
            pos_baseline_mrr=pos_mrr, neg_baseline_mrr=neg_mrr,
            cos_caa_meandiff_per_layer=cos_per_layer,
        )
        save_json(results_dir / "concept_vectors" / f"{cname}.meta.json", quality)
        target_mask_per_concept[cname] = target_mask

        notify.send(f"▶ {cname}: sweeping {len(layers)} layers × {len(alphas)} alphas")
        summary = sweep_concept(
            handle=handle, fable_texts=corpus.fable_texts,
            moral_embs=moral_embs, gt_indices=gt_indices,
            concept_name=cname, direction_per_layer=v_caa,
            layers=layers, alphas=alphas,
            output_dir=results_dir / "ranks_intervened",
            batch_size=cfg["model"]["batch_size"],
        )
        cells_per_concept[cname] = summary["cells"]

    if not cells_per_concept:
        notify.send("❌ run_intervention: no concepts produced enough matched pairs")
        return 1

    # 4. First-pass summary (no nulls yet) to find candidate cells
    summary_no_nulls = summarize_run(
        cells_per_concept=cells_per_concept,
        target_query_mask_per_concept=target_mask_per_concept,
        rr_baseline=rr_baseline, layers=layers, alphas=alphas,
        null_envelopes=None,
        n_bootstrap=cfg["eval"]["primary_statistic"]["n_bootstrap"],
    )

    # 5. Null controls at candidate cells (skip in 'skip' run_mode)
    target_names = [f"{c['field']}__{c['value']}" for c in cfg["concepts"]["targets"]]
    placebo_names = [f"{c['field']}__{c['value']}" for c in cfg["concepts"]["placebo"]]
    candidate_cells = _candidate_cells(summary_no_nulls, target_names)
    null_envelopes = None
    random_dir_within_null = False
    if cfg["null_controls"]["run_mode"] != "skip" and candidate_cells:
        notify.send(f"▶ null controls: {len(candidate_cells)} candidate cells")
        null_envelopes = _run_shuffled_tag_nulls(
            cfg=cfg, handle=handle, corpus=corpus,
            hs_by_layer=hs_by_layer, moral_embs=moral_embs,
            gt_indices=gt_indices, candidate_cells=candidate_cells,
            results_dir=results_dir, fable_idx_by_id=fable_idx_by_id,
            rr_baseline=rr_baseline,
        )
        random_dir_within_null = _run_random_direction_check(
            cfg=cfg, handle=handle, corpus=corpus,
            moral_embs=moral_embs, gt_indices=gt_indices,
            candidate_cells=candidate_cells,
            results_dir=results_dir, fable_idx_by_id=fable_idx_by_id,
            rr_baseline=rr_baseline, null_envelopes=null_envelopes,
        )

    # 6. Final summary with null envelopes + Stage-2 inputs + plot + decision
    final_summary = summarize_run(
        cells_per_concept=cells_per_concept,
        target_query_mask_per_concept=target_mask_per_concept,
        rr_baseline=rr_baseline, layers=layers, alphas=alphas,
        null_envelopes=null_envelopes,
        n_bootstrap=cfg["eval"]["primary_statistic"]["n_bootstrap"],
    )
    # Stage-2 cond3/cond4 inputs (must be populated explicitly).
    final_summary["random_dir_within_null"] = bool(random_dir_within_null)
    # Max pooled cosine across passing cells (cond4 should be < 0.99 for GO).
    passing_pooled_cos: list[float] = []
    for concept, cell_list in cells_per_concept.items():
        if concept not in target_names:
            continue
        layer_cells = final_summary["cells"][concept]
        for cell_json in cell_list:
            layer_key = str(cell_json["layer"])
            alpha = cell_json["alpha"]
            try:
                ai = final_summary["alphas"].index(alpha)
                if layer_cells[layer_key]["S_ci_hi"][ai] is not None \
                        and layer_cells[layer_key]["S_ci_hi"][ai] < 0.0:
                    passing_pooled_cos.append(cell_json["pooled_cosine_mean"])
            except (KeyError, ValueError):
                continue
    final_summary["passing_pooled_cosine_max"] = (
        float(max(passing_pooled_cos)) if passing_pooled_cos else 1.0
    )

    save_json(results_dir / "specificity_summary.json", final_summary)
    plot_specificity_summary(summary=final_summary,
                              save_path=results_dir / "specificity_summary.png")

    decision = stage2_go_no_go(
        final_summary,
        target_concepts=target_names,
        placebo_concepts=placebo_names,
    )
    save_json(results_dir / "stage2_decision.json", decision)

    notify.send(
        f"✅ 08_concept_steering: run_intervention done\n"
        f"Stage 2 GO: {decision['go']}\n"
        f"targets passing: {decision['targets_passing']}\n"
        f"reasons: {decision['reasons']}"
    )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    sys.exit(main(Path(args.config), force=args.force))
