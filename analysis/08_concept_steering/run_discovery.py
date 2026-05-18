"""
Step 2: rank candidate failure-associated concepts via failure overrepresentation
(Fisher's exact + BH-FDR), then select 3 targets (one per metadata field) and
1 difficulty-matched placebo. Updates concepts.* in config.yaml.

Run via: ./run.sh analysis/08_concept_steering/run_discovery.py
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import yaml
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from lib.config import load_config
from lib.data import load_corpus, build_tag_index
from lib.io import load_json, save_json
from lib.discovery import rank_problematic_concepts, per_tag_baseline_mrr


TARGET_FIELDS = ("characters", "character_roles", "moral_category")


def main(config_path: Path) -> int:
    cfg = load_config(config_path)
    results_dir = ROOT / cfg["output"]["results_dir"]

    baseline = load_json(results_dir / "ranks_baseline.json")
    fable_doc_ids = baseline["fable_doc_ids"]
    failed_doc_ids = {q["gt_fable_doc_id"] for q in baseline["queries"] if q["gt_rank"] > 1}

    corpus = load_corpus(
        morals_path=ROOT / cfg["data"]["morals_path"],
        fables_path=ROOT / cfg["data"]["fables_path"],
        qrels_path =ROOT / "data/processed/qrels_moral_to_fable.json",
    )
    tag_index = build_tag_index(
        ROOT / cfg["data"]["metadata_path"],
        fields=cfg["discovery"]["metadata_fields"],
    )

    # 1. Discovery
    df = rank_problematic_concepts(
        tag_index=tag_index, fable_doc_ids=fable_doc_ids,
        failed_doc_ids=failed_doc_ids,
        min_tagged_fables=cfg["discovery"]["min_tagged_fables"],
        fdr_alpha=cfg["discovery"]["fdr_alpha"],
    )
    print("\n=== Top 20 problematic concepts ===")
    print(df.head(20).to_string(index=False))

    # 2. Per-tag baseline MRR (truncated rankings — sufficient for MRR@10)
    rankings = np.array([q["top_50_indices"] for q in baseline["queries"]])
    gt = np.array([q["gt_fable_idx"] for q in baseline["queries"]])
    mrr_df = per_tag_baseline_mrr(
        tag_index=tag_index, fable_doc_ids=fable_doc_ids,
        moral_gt_idx=gt.tolist(), rankings=rankings,
        min_tagged_fables=cfg["discovery"]["min_tagged_fables"],
    )

    # 3. Pick targets — one per metadata field, FDR-significant, lowest p
    targets: list[dict] = []
    used_fields: set[str] = set()
    for _, row in df[df["fdr_significant"]].iterrows():
        if row["field"] in used_fields or row["field"] not in TARGET_FIELDS:
            continue
        targets.append({"field": row["field"], "value": row["value"]})
        used_fields.add(row["field"])
        if len(targets) == 3:
            break

    if not targets:
        notify.send("❌ run_discovery: no FDR-significant target concepts found")
        save_json(results_dir / "discovery_report.json", {
            "all_concepts": df.to_dict(orient="records"),
            "per_tag_mrr": mrr_df.to_dict(orient="records"),
            "selected": {"targets": [], "placebo": []},
        })
        return 1

    # 4. Difficulty-matched placebo selection
    target_mean_mrr = float(np.mean([
        mrr_df.loc[(mrr_df["field"] == t["field"]) & (mrr_df["value"] == t["value"]),
                    "baseline_mrr"].iloc[0]
        for t in targets
    ]))
    target_n = float(np.mean([
        df.loc[(df["field"] == t["field"]) & (df["value"] == t["value"]),
                "n_tagged"].iloc[0]
        for t in targets
    ]))
    candidates = mrr_df.merge(df[["field", "value", "n_tagged", "fdr_significant"]],
                                 on=["field", "value"])
    candidates = candidates[~candidates["fdr_significant"]]
    candidates = candidates[~candidates["field"].isin(used_fields)]
    if len(candidates) == 0:
        notify.send(
            "❌ run_discovery: no non-significant placebo candidate exists "
            "outside the target fields. Cannot proceed."
        )
        save_json(results_dir / "discovery_report.json", {
            "all_concepts": df.to_dict(orient="records"),
            "per_tag_mrr": mrr_df.to_dict(orient="records"),
            "selected": {"targets": targets, "placebo": [],
                          "target_mean_baseline_mrr": target_mean_mrr,
                          "target_mean_n_tagged": target_n,
                          "failure_reason": "no_placebo_candidate"},
        })
        return 2

    strict = candidates[
        (abs(candidates["baseline_mrr"] - target_mean_mrr) <= 0.05) &
        (candidates["n_tagged"] >= 0.5 * target_n) &
        (candidates["n_tagged"] <= 1.5 * target_n)
    ]
    if len(strict) == 0:
        notify.send("⚠️ run_discovery: strict placebo match failed, loosening")
        candidates = candidates.assign(
            mrr_dist=(candidates["baseline_mrr"] - target_mean_mrr).abs()
        ).sort_values("mrr_dist")
        placebo_row = candidates.iloc[0]
    else:
        placebo_row = strict.iloc[0]

    placebo = [{"field": str(placebo_row["field"]), "value": str(placebo_row["value"])}]

    # 5. Save report
    save_json(results_dir / "discovery_report.json", {
        "all_concepts": df.to_dict(orient="records"),
        "per_tag_mrr": mrr_df.to_dict(orient="records"),
        "selected": {
            "targets": targets,
            "placebo": placebo,
            "target_mean_baseline_mrr": target_mean_mrr,
            "target_mean_n_tagged": target_n,
        },
    })

    # 6. Update config.yaml in place
    with open(config_path) as f:
        live = yaml.safe_load(f)
    live["concepts"]["targets"] = targets
    live["concepts"]["placebo"] = placebo
    with open(config_path, "w") as f:
        yaml.safe_dump(live, f, sort_keys=False)

    notify.send(
        f"✅ 08_concept_steering: run_discovery done\n"
        f"targets: {targets}\n"
        f"placebo: {placebo}"
    )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    args = p.parse_args()
    sys.exit(main(Path(args.config)))
