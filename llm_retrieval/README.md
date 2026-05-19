# LLM Retrieval Experiment

Tests whether frontier LLMs can retrieve the correct fable from a 709-fable corpus given a moral statement — an "oracle baseline" against fine-tuned bi-encoder retrievers.

## How It Works

Each query: the model receives the full corpus (709 fables formatted as `[fable_id] text`) and a moral statement, and must return the 10 most relevant fable IDs as a JSON array.

**Corpus size:** ~120k tokens. Models with 131K context (GPT-OSS-20B/120B) are near capacity — monitor for overflow errors.

## Usage

```bash
# Test mode — smoke test one model, one variant, 100 morals
./run.sh llm_retrieval/run.py --test 100 --models GPT-4o-mini --variants minimal

# Full run — all models, all variants
./run.sh llm_retrieval/run.py

# Skip already-completed runs (safe to re-run after interruption)
./run.sh llm_retrieval/run.py --skip-existing
```

## Required `.env` keys

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
TOGETHER_API_KEY=...
```

## Results

- `results/runs/YYYY-MM-DD_<model>_<variant>.csv` — per-query rows
- `results/unified.csv` — aggregate metrics per (model × variant) run

## Results Table

*(populated after runs)*

| Model | Variant | MRR@10 | R@1 | R@10 | NDCG@10 |
|-------|---------|--------|-----|------|---------|
| — | — | — | — | — | — |
