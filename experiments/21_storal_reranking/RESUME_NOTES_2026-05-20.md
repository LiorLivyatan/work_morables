# Reranker Resume Notes - 2026-05-20

## Stop State

GPU 0 was stopped on 2026-05-20 after freeing GPUs for the team.

Stopped job:

```bash
./run.sh experiments/21_storal_reranking/rerank.py --rerankers bge_reranker_v2_m3 --candidate-k 100 --remote --gpu 0
```

The job was interrupted after completing several `bge_reranker_v2_m3` rows. The final combined CSV/JSON for this interrupted run was not written, but each completed row saved its ranking artifact under:

```text
/media/eimtest/data/lior/ParabeLink/experiments/21_storal_reranking/results/rankings/
```

GPU 0 was released successfully. At stop check:

```text
GPU 0 | Mem: 991/24155 MB | Util: 0%
```

The remaining 991 MiB was a stale `[Not Found]` process shown by `nvidia-smi`, not our active reranker process.

## Completed `bge_reranker_v2_m3` Rows

These ranking artifacts exist on the server:

```text
bge_reranker_v2_m3__linq__s1000__conceptual_abstract_fable__gemini__k100.json
bge_reranker_v2_m3__linq__s1000__fable_proverb__gemini__k100.json
bge_reranker_v2_m3__linq__s500__fable_cot_proverb__gemini__k100.json
bge_reranker_v2_m3__linq__s500__fable_proverb__gemini__k100.json
bge_reranker_v2_m3__linq__s200__fable_proverb__gemini__k100.json
bge_reranker_v2_m3__linq__s500__conceptual_abstract_fable__gemini__k100.json
bge_reranker_v2_m3__linq__s200__fable_direct_moral__gemini__k100.json
bge_reranker_v2_m3__linq__s200__fable_cot_proverb__gemini__k100.json
```

Latest logged metrics before stopping:

```text
bge_reranker_v2_m3__linq__s500__fable_cot_proverb__gemini__k100
MRR@10=0.3451 MAP@10=0.2988 NDCG@10=0.3371 Hit@10=0.5419 Recall@100=0.9051

bge_reranker_v2_m3__linq__s500__fable_proverb__gemini__k100
MRR@10=0.3359 MAP@10=0.2924 NDCG@10=0.3303 Hit@10=0.5404 Recall@100=0.9015

bge_reranker_v2_m3__linq__s200__fable_proverb__gemini__k100
MRR@10=0.3389 MAP@10=0.2955 NDCG@10=0.3348 Hit@10=0.5509 Recall@100=0.9010

bge_reranker_v2_m3__linq__s500__conceptual_abstract_fable__gemini__k100
MRR@10=0.3905 MAP@10=0.3370 NDCG@10=0.3833 Hit@10=0.6003 Recall@100=0.8927

bge_reranker_v2_m3__linq__s200__fable_direct_moral__gemini__k100
MRR@10=0.3756 MAP@10=0.3244 NDCG@10=0.3643 Hit@10=0.5868 Recall@100=0.8920

bge_reranker_v2_m3__linq__s200__fable_cot_proverb__gemini__k100
MRR@10=0.3538 MAP@10=0.3077 NDCG@10=0.3431 Hit@10=0.5404 Recall@100=0.8920
```

Earlier rows should be recalculated from the saved ranking artifacts when aggregating final metrics.

## Interrupted Row

The job had just started the next row when stopped:

```text
bge_reranker_v2_m3 linq s1000 fable_cot_proverb/gemini
```

It reached roughly:

```text
query 100/668
```

Treat this row as incomplete and rerun it later.

## Resume Plan

The current reranker script does not resume inside a partially completed row. Resume means:

1. Keep the completed ranking artifacts listed above.
2. Recalculate metrics from those artifacts if needed.
3. Run only the remaining `bge_reranker_v2_m3` plan rows, starting with:

```text
linq s1000 fable_cot_proverb/gemini
```

Recommended next code improvement before resuming:

- Add a `--skip-existing-rankings` option to `experiments/21_storal_reranking/rerank.py`.
- It should skip rows whose ranking file already exists in `results/rankings/`.
- Then rerun the same command safely:

```bash
./run.sh experiments/21_storal_reranking/rerank.py --rerankers bge_reranker_v2_m3 --candidate-k 100 --skip-existing-rankings --remote --gpu 0
```

Without that option, rerunning the same command will recompute completed rows.

