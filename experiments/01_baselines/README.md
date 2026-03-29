# Experiment 01: Baseline Embedding Retrieval

## What it tests
Naive cosine similarity retrieval with small sentence-transformer models and instruction-aware embedding models.

## Method
Encode morals and fables independently with off-the-shelf embedding models. Retrieve by cosine similarity.

Models tested:
- `all-MiniLM-L6-v2`, `all-mpnet-base-v2` (small sentence-BERT)
- `BAAI/bge-large-en-v1.5` (with instruction prefix variants)
- `intfloat/e5-large-v2`, `intfloat/multilingual-e5-large`

## How to run
```bash
python experiments/01_baselines/run.py
```

## Key results

| Model | Variant | MRR | R@1 |
|-------|---------|-----|-----|
| e5-large-v2 | no_prefix | 0.080 | 4.4% |
| all-mpnet-base-v2 | plain | 0.065 | 2.7% |
| all-MiniLM-L6-v2 | plain | 0.063 | 3.4% |

## Key findings
- Small embedding models perform poorly on this task (MRR < 0.10)
- Instruction prefixes for BGE hurt rather than help
- Established that stronger models are needed
