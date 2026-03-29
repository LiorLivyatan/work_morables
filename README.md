# MORABLES: Moral Retrieval Benchmark

Retrieval benchmark testing whether embedding models can capture abstract moral meaning — going beyond surface-level lexical matching to evaluate analogical moral reasoning.

## Dataset

**MORABLES**: 709 fable-moral pairs from the Western literary tradition (Aesop, Gibbs, Perry, Abstemius).
- Fables: narrative stories (mean 117 words)
- Morals: abstract life lessons (mean 10 words)
- Lexical overlap (IoU): **0.011** — near zero, confirming lexical matching fails
- 678 unique morals (27 shared across multiple fables)

## Task

**Moral-to-fable retrieval**: given a moral (query), find the fable that teaches it (document).

## Results Summary

| # | Method | Best Config | MRR | R@1 |
|---|--------|-------------|-----|-----|
| 1 | Oracle (fable+moral concat) | — | **0.893** | **82.7%** |
| 2 | Off-the-shelf embedding | Linq-Embed-Mistral | 0.210 | 14.1% |
| 3 | LLM Summarisation | Qwen3.5-9b + CoT | 0.215 | 14.1% |
| 4 | Qwen3-Embedding (instruction-steered) | 8B + baseline | 0.183 | 11.0% |
| 5 | Sentence-level chunking | sliding window | 0.180 | 10.7% |
| 6 | Prompt repetition | baseline x3 | 0.183 | 10.7% |

**Gap to close**: 0.210 (best real) vs 0.893 (oracle) — 4.3x improvement theoretically possible.

## Repo Structure

```
work_morables/
├── README.md
├── requirements.txt
├── lib/                          # Shared code
│   ├── retrieval_utils.py        #   Metrics (MRR, Recall@k, NDCG, etc.)
│   └── data.py                   #   Common data loading
├── data/
│   ├── raw/                      #   Original MORABLES dataset
│   └── processed/                #   Retrieval-formatted corpus + qrels
├── experiments/                   # One directory per experiment
│   ├── 01_baselines/             #   Small model baselines
│   ├── 02_model_comparison/      #   20+ model comparison
│   ├── 03_llm_summarisation/     #   LLM moral summaries (Approach B + C oracle)
│   ├── 04_llm_reranking/         #   Gemini CoT reranking (Approach A)
│   ├── 05_qwen3_embedding/       #   Instruction-steered embeddings + prompt repetition
│   └── 06_sentence_chunking/     #   Sentence-level RAG-style retrieval
├── docs/                          # Documentation and analysis
├── papers/                        # Reference PDFs
├── meeting_materials/             # Professor meeting prep
└── archive/                       # Old planning docs
```

Each experiment directory contains:
- `run.py` — main experiment script
- `results/` — output data (summary files tracked, large run outputs gitignored)
- `README.md` — method description, how to run, key results and findings

## Quick Start

```bash
# Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run an experiment
python experiments/02_model_comparison/run.py --models linq-embed-mistral
```
