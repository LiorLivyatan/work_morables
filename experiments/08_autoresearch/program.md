# Retrieval Autoresearch — Agent Instructions

## Your Task
Improve MRR on the MORABLES moral-to-fable retrieval benchmark.
Starting baseline: ~0.210 MRR (Linq-Embed-Mistral, raw embedding).
Oracle ceiling: 0.893 MRR (fable+moral concatenation).

## The Loop

Each iteration:
1. Read `results.tsv` to see what's been tried and the current best MRR
2. Read `retrieval_pipeline.py` to understand the current state
3. Propose ONE focused change to `retrieval_pipeline.py` (edit it with your Edit tool)
4. Run: `python experiments/08_autoresearch/runner.py "description of change"`
5. Read the output — runner.py handles git keep/discard and logs the result
6. Repeat from step 1

You are the agent. runner.py is your tool. Never touch runner.py.

## What You May Modify

Only `experiments/08_autoresearch/retrieval_pipeline.py`.

Inside it, you may:
- Change `MODEL_ID` to any locally available sentence-transformers model
- Change `QUERY_INSTRUCTION` and/or `CORPUS_INSTRUCTION`
- Change `CHUNKING`, `CHUNK_AGG` for chunking strategies
- Set `SPARSE_WEIGHT > 0` to blend BM25 (requires: `pip install rank_bm25` — OK to add)
- Set `RERANKER_ID` to a cross-encoder model id for reranking
- Modify `rewrite_query()` for query reformulation (no API calls)
- Add any logic inside `run_pipeline()` (score normalization, ensembles, etc.)

## What You Must Not Change

- `runner.py` — never touch it
- `lib/` — never touch retrieval_utils.py, embedding_cache.py, data.py
- The `run_pipeline() -> dict` return contract (must include key "MRR")
- The stdout format: `mrr: 0.XXXX`, `r@1: 0.XXXX`, etc. (parsed by runner)
- Do NOT make API calls inside `retrieval_pipeline.py` (local models only)
- Do NOT install packages other than `rank_bm25` (already in environment or simple add)

## Exploration Priority

Work through these axes roughly in order — earlier axes have higher impact/cost ratio:

1. **Query instructions** — Try 5-10 different phrasings. Short, abstract instructions
   often outperform verbose ones (see Exp 05 finding: specificity hurts). Examples:
   - `"Retrieve a fable that teaches this lesson"`
   - `"Find the story behind this moral"`
   - `"moral lesson"` (ultra-short)
   - `""` (empty instruction — try it)
   - `None` (no instruction at all)

2. **Corpus instructions** — Try applying instructions to fables too, not just queries.
   Asymmetric instructions (query has instruction, corpus doesn't, or vice versa) sometimes
   help with instruction-tuned models.

3. **Embedding model swap** — Try these (all available via sentence-transformers):
   - `"intfloat/e5-large-v2"` with instruction prefix `"query: "` / `"passage: "`
   - `"BAAI/bge-large-en-v1.5"` with `"Represent this sentence: "`
   - `"thenlper/gte-large"`
   - `"nomic-ai/nomic-embed-text-v1"` (requires `trust_remote_code=True`)
   - `"intfloat/multilingual-e5-large-instruct"`
   Models are cached by sentence-transformers in `~/.cache/torch/sentence_transformers/`.

4. **Query rewriting** — Reformulate morals to sound more like narrative content.
   No API calls: use templates. Examples:
   - `f"A story about: {moral}"`
   - `f"The lesson '{moral}' can be learned from a fable about"`
   - Add filler words that match fable vocabulary ("once upon a time", "a tale of")

5. **Sparse fusion (BM25)** — Add keyword matching alongside dense retrieval.
   Install: `pip install rank_bm25`
   Then add BM25 scoring to `run_pipeline()` and blend with cosine similarity.
   Try `SPARSE_WEIGHT` values: 0.1, 0.2, 0.3, 0.5.

6. **Reranking** — Use a cross-encoder on the top-50 candidates from dense retrieval.
   Set `RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"` as a starting point.
   This runs slower (~2-3 min extra) but can significantly improve precision.

## Research Heuristics

- **One change at a time**: Isolate the effect. If you change model AND instruction together
  and it improves, you don't know which helped.
- **Combine near-misses**: Two discarded experiments that each got close may combine well.
- **Don't repeat failures**: If an exact config was tried and discarded, don't retry it.
- **Simplest first**: A 1-line instruction change before a full model swap.
- **If stuck (5 consecutive discards)**: Try something radical — different model family,
  corpus-side instruction, or a query rewriting approach you haven't tried.
- **Lower is not always better for instructions**: Verbose instructions can hurt.
  Counter-intuitively, `None` (no instruction) sometimes beats a well-crafted one.

## Reference: What's Been Tried

| Exp | Best config | MRR |
|-----|-------------|-----|
| 02  | Linq-Embed-Mistral + generic instruction | 0.210 |
| 05  | Qwen3-Embedding + task-specific instructions | 0.183 (worse!) |
| 06  | Sentence chunking | 0.151 (worse!) |

Key learnings:
- Task-specific instructions HURT on this dataset (Exp 05)
- Sentence-level chunking HURTS — morals need full fable context (Exp 06)
- Linq-Embed-Mistral outperformed 20+ other models in Exp 02

## Never Stop
Run the loop indefinitely until manually interrupted (Ctrl-C or end of session).
When one session ends, the next session can resume by reading results.tsv.
