# Experiment 10: Local Model Matrix Design

Evaluate combinations of local generation models × embedding models on the MORABLES retrieval task. Identifies the best generation model, best embedding model, and best pair under multiple ablation modes.

## Models Evaluated

**Generation Models:**
- Phi-3.5-mini (`microsoft/Phi-3.5-mini-instruct`)
- Qwen3.5-4B (`Qwen/Qwen3.5-4B`)

**Embedding Models:**
- Linq-Embed-Mistral (`Linq-AI-Research/Linq-Embed-Mistral`) — exp08 model
- Qwen3-Embed-8B (`Qwen/Qwen3-Embedding-8B`)
- BGE-M3 (`BAAI/bge-m3`)
- BGE-Large-EN (`BAAI/bge-large-en-v1.5`)
- Nomic-Embed-Text (`nomic-ai/nomic-embed-text-v1.5`)

## Run Instruction
```bash
python experiments/10_model_matrix/run_pipeline.py
```

To add a single embedding model to an existing run (skips generation):
```bash
python experiments/10_model_matrix/add_embed_model.py \
  --run-dir <path/to/run_dir> \
  --embed-alias Linq-Embed-Mistral
```

## Results (sample50, `full` ablation — summary corpus + query paraphrases)

| Gen \ Embed       | BGE-Large-EN | BGE-M3 | Linq-Embed-Mistral | Nomic-Embed-Text | Qwen3-Embed-8B |
|-------------------|:------------:|:------:|:------------------:|:----------------:|:--------------:|
| **Phi-3.5-mini**  | 0.18         | 0.14   | 0.16               | 0.16             | 0.20           |
| **Qwen3.5-4B**    | 0.16         | 0.26   | **0.28**           | 0.26             | **0.32**       |

Metric: Recall@1. Best overall: **Qwen3.5-4B × Qwen3-Embed-8B = 0.32**.

### All Ablations (Recall@1)

| Gen \ Embed       | Ablation        | BGE-Large-EN | BGE-M3 | Linq-Embed-Mistral | Nomic-Embed-Text | Qwen3-Embed-8B |
|-------------------|-----------------|:------------:|:------:|:------------------:|:----------------:|:--------------:|
| **Phi-3.5-mini**  | raw_raw         | 0.14         | 0.06   | **0.46**           | 0.10             | 0.18           |
|                   | summary_only    | 0.12         | 0.16   | 0.24               | 0.18             | 0.16           |
|                   | paraphrase_only | 0.08         | 0.04   | 0.42               | 0.12             | 0.20           |
|                   | full            | 0.18         | 0.14   | 0.16               | 0.16             | 0.20           |
|                   | rrf             | 0.08         | 0.14   | 0.26               | 0.12             | 0.16           |
| **Qwen3.5-4B**    | raw_raw         | 0.14         | 0.06   | **0.46**           | 0.10             | 0.18           |
|                   | summary_only    | 0.20         | 0.22   | 0.26               | 0.24             | 0.36           |
|                   | paraphrase_only | 0.12         | 0.12   | 0.40               | 0.16             | 0.16           |
|                   | full            | 0.16         | 0.26   | 0.28               | 0.26             | 0.32           |
|                   | rrf             | 0.22         | 0.28   | 0.36               | 0.20             | 0.24           |

### Key Findings

- **Linq-Embed-Mistral `raw_raw`** hits R@1=**0.46** for both gen models — the highest single cell in the entire matrix. This suggests Linq-Embed-Mistral excels at direct moral-to-fable matching without summarization.
- On the stable `full` ablation (summary + paraphrases), **Qwen3.5-4B × Qwen3-Embed-8B** is best at **0.32**.
- **Qwen3.5-4B** outperforms Phi-3.5-mini on summarization-dependent ablations; on `raw_raw` both models produce the same raw fable texts so scores are identical.
- Dominant factor: **embedding model** for `raw_raw`; **generation model** for `full`/`summary_only`.
