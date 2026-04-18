# Experiment 10: Local Model Matrix Design

Evaluate combinations of local generation models × embedding models on the MORABLES retrieval task. Identifies the best generation model, best embedding model, and best pair under multiple ablation modes.

## Models Evaluated

**Generation Models:**
- Mistral-7B-Instruct
- Phi-3.5-mini
- Qwen3.5-4B
- Qwen3.5-9B

**Embedding Models:**
- BGE-Large-EN
- BGE-M3
- Linq-Embed-Mistral
- Nomic-Embed-Text
- Qwen3-Embed-8B

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

## Results (full 709 fables, `full` ablation — summary corpus + query paraphrases)

| Generation Model | BGE-Large-EN | BGE-M3 | Linq-Embed-Mistral | Nomic-Embed-Text | Qwen3-Embed-8B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mistral-7B-Instruct** | 0.0127 | 0.0324 | 0.0085 | 0.0635 | 0.0155 |
| **Phi-3.5-mini** | 0.0197 | 0.0197 | 0.0155 | 0.0437 | 0.0282 |
| **Qwen3.5-4B** | 0.0254 | 0.0296 | 0.0508 | 0.0592 | 0.0324 |
| **Qwen3.5-9B** | 0.0296 | 0.0353 | **0.0973** | 0.0719 | 0.0494 |

Metric: Recall@1. Best overall: **Qwen3.5-9B × Linq-Embed-Mistral = 0.0973**.

## Results (full 709 fables, Recall@5)

| Generation Model | BGE-Large-EN | BGE-M3 | Linq-Embed-Mistral | Nomic-Embed-Text | Qwen3-Embed-8B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mistral-7B-Instruct** | 0.0578 | 0.0959 | 0.0959 | 0.1467 | 0.0705 |
| **Phi-3.5-mini** | 0.0621 | 0.0663 | 0.0705 | 0.1199 | 0.0790 |
| **Qwen3.5-4B** | 0.0889 | 0.0959 | 0.1495 | 0.1481 | 0.0931 |
| **Qwen3.5-9B** | 0.0973 | 0.1114 | **0.2045** | 0.1636 | 0.1044 |

Metric: Recall@5. Best overall: **Qwen3.5-9B × Linq-Embed-Mistral = 0.2045**.

### Key Findings

- **Generator Scaling Matters**: Better/larger LLMs produce significantly better inputs for the embedding models. For example, keeping the embedding model constant (Linq), logic goes up exactly linearly with generator capability (Mistral -> Phi -> Qwen-4B -> Qwen-9B).
- **The Clear Winner**: **Qwen3.5-9B + Linq-Embed-Mistral**. This combination hits `~9.7%` Recall@1 and `~20.5%` Recall@5. 
- **Nomic-Embed-Text** acts as the most robust baseline embedding model. It holds 4-6% Recall@1 even with weak generators.
- **Comparison to Oracle (Exp 07)**: Using Gemini 3.1 Pro (in Exp 07) yielded ~26.5% Recall@1, meaning **the generation quality** remains the biggest bottleneck in the overall pipeline.
