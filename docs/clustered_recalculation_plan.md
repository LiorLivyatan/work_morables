# Clustered Benchmark Recalculation Plan

Date: 2026-05-17

Current clustered benchmark status:

- Fable documents: 709
- Unique moral queries: 668
- Moral clusters: 554
- Clustered moral-to-fable qrel rows: 1122
- Source cluster file: `analysis/clusters_full.json`
- Derived benchmark folder: `data/clustered/`

This file answers one question: after the clustered data is finalized, which old results can be recalculated from saved artifacts, and which experiments need another scoring/evaluation run.

## Decision Rules

1. If full rankings were saved, we can recalculate clustered metrics without rerunning the model.
   Example: `results/runs/2026-03-15_combined_v2/predictions/*.json`.

2. If only top-10 predictions were saved, we can recalculate top-10 metrics only.
   This supports `MRR@10`, `MAP@10`, `Hit@10`, `Recall@10`, and `NDCG@10`, but not `Recall@100`, `Recall@200`, or `Recall@300`.

3. If only aggregate metrics were saved, we must rerun scoring.
   This does not always mean expensive regeneration. For embedding experiments, it usually means encoding the final 668 moral queries and ranking the same 709 fables.

4. Fine-tuned models do not need training again just to evaluate the corrected clustered benchmark.
   If model weights exist, re-evaluate them. If model weights are missing, first try to pull them from remote/long-term storage. Retrain only if the model cannot be recovered or if we want a new cluster-aware training objective.

5. Any result produced from old rankings is exact only if the query/document texts are unchanged.
   If the final reviewer changes canonical moral text, rerun scoring for official final numbers.

## Highest Priority After Final Review

1. Recalculate `exp_02` from saved full rankings.
   This is the fastest high-value update and covers 45 old embedding runs with full-depth metrics.

2. Re-evaluate or rerun `exp_12` zero-shot comprehensive.
   It has 120 aggregate rows but no local rankings/cache, so it needs a scoring rerun. Make the rerun save full rankings.

3. Re-evaluate the best fine-tuned model family.
   Prioritize `ft_07` LINQ `s500` / `sfull`. Existing metrics cannot be recalculated because rankings were not saved. Recover model weights first; retrain only if recovery fails.

4. Recalculate top-10 LLM retrieval metrics from saved rows.
   GPT-5.4-Nano and Gemini 3.1 Flash Lite full-709 batches have top-10 ranked IDs. This is enough for top-10 clustered metrics, not deep recall.

5. Rerun official LLM batches only after the clustered data is final.
   Use the final 668 unique moral prompts and minimal prompt. Ask for more than top 10 only if we care about deep recall.

## Inventory

The machine-readable inventory is in:

`docs/clustered_recalculation_inventory.csv`

Summary by action:

| Action | Experiments |
|---|---|
| Full metric-only recalculation now | `exp_02` original embedding model comparison |
| Partial top-10 recalculation now | `exp_05`, GPT-5.4-Nano batch, Gemini 3.1 Flash Lite batch, LLM smoke rows |
| Rerun scoring/eval, no training | `exp_01`, `exp_06`, `exp_07`, `exp_09`, `exp_11`, `exp_12`, `exp_13`, most saved fine-tuned models if weights exist |
| Pull models or retrain before evaluation | `ft_02`, `ft_03`, `ft_04`, `ft_07`, `ft_09` if model weights are not recoverable |
| New training only if making a new claim | Cluster-aware 5-fold CV / new `ft_11` style experiment |
| Low-priority or optional | `exp_03`, `exp_04`, `exp_08` pilots, `ft_08`, `ft_10`, future planned experiments |

## Fine-Tuning Guidance

For old fine-tuning claims:

- Do not retrain automatically.
- Re-evaluate old trained models on the clustered benchmark.
- If weights are missing, pull from remote or rerun only the selected important models.

For new cluster-aware fine-tuning claims:

- Create a new experiment rather than overwriting old 5FCV.
- Split by cluster, not by fable row.
- Treat all fables in the same cluster as positives.
- Exclude same-cluster fables from negatives.

Recommended future experiment name:

`finetuning/ft_11_clustered_5fold_cv`

## Implementation To-Do

1. Extend `scripts/12_recalculate_clustered_from_predictions.py` to batch-process a whole predictions directory and write a CSV summary.
2. Add a top-10 LLM rows recalculator for `llm_retrieval/results/**/rows.csv`.
3. Add a clustered mode to embedding eval scripts so reruns save:
   - full ranked IDs
   - score matrix or top-709 rankings
   - clustered metrics with `Hit@k`, `Recall@k`, `MAP@10`, and `NDCG@k`
4. Add a fine-tuned model clustered evaluator that can load saved fold models and evaluate on `data/clustered`.
5. After final reviewer approval, run in this order:
   - `exp_02` recalculation
   - `exp_12` clustered rerun
   - `ft_07` clustered re-eval
   - GPT/Gemini top-10 recalculation
   - selected official LLM reruns
