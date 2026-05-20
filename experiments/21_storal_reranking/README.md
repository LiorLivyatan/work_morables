# Experiment 21: STORAL Reranking

Second-stage reranking for the strongest STORAL fine-tuned retrieval runs.

The first-stage retrievers have already produced full rankings, so this experiment
does not rerun dense retrieval. It takes the best existing FT12 STORAL configs,
keeps the top candidate fables per moral, reranks those candidate pairs with a
cross-encoder/reranker, then recomputes the same clustered MORABLES metrics.

## Research Question

Dense retrieval is optimized for candidate recall. A reranker should improve the
top of the list by moving relevant fables from the candidate pool into the top
10. We therefore expect the clearest movement in:

- `MAP@10`
- `MRR@10`
- `NDCG@10`
- `Hit@1/5/10`
- `Recall@10`

`Recall@100` and `Recall@200` are treated mainly as candidate-pool diagnostics.
For a fixed `candidate_k=100`, reranking cannot add new documents to the top-100
candidate set; it can only reorder it.

## Files

- `config.yaml` - rerankers, candidate pool sizes, and source paths.
- `prepare_plan.py` - selects the top FT12 retrieval configs by `Recall@100`.
- `rerank.py` - reranks saved first-stage rankings and writes metrics.
- `run_plans/top_recall100_storal.csv` - generated plan file.
- `results/` - reranking result CSV/JSON files.

## Local Dry Run

Use the built-in mock reranker to validate the pipeline without downloading any
model:

```bash
./run.sh experiments/21_storal_reranking/prepare_plan.py --top-n 15 --local
./run.sh experiments/21_storal_reranking/rerank.py \
  --plan experiments/21_storal_reranking/run_plans/top_recall100_storal.csv \
  --rerankers mock_lexical \
  --candidate-k 20 \
  --limit-runs 1 \
  --limit-queries 5 \
  --local
```

## Intended First Wave

- Select top 10-15 FT12 configs by `Recall@100`.
- Rerank top-100 candidates.
- Compare top-heavy metrics before/after reranking.

Candidate rerankers in `config.yaml`:

- `BAAI/bge-reranker-large`
- `BAAI/bge-reranker-v2-m3`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `jinaai/jina-reranker-v2-base-multilingual`
- `mixedbread-ai/mxbai-rerank-large-v1`
- `Qwen/Qwen3-Reranker-0.6B`
- `Qwen/Qwen3-Reranker-4B`
- `Qwen/Qwen3-Reranker-8B`

The Qwen rerankers use Qwen's official CausalLM yes/no scoring path rather than
the generic `CrossEncoder` path. This avoids older `sentence-transformers`
loading the model as `Qwen3ForSequenceClassification` with a newly initialized
`score.weight` head. The score is computed as the probability of the next token
being `yes` versus `no` after the Qwen chat-template reranking prompt.

Before launching overnight, run one real reranker on one config with a tiny
query limit to verify model loading on the server.
