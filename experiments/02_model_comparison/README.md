# Experiment 02: Large-Scale Model Comparison

## What it tests
Comprehensive comparison of 20+ embedding models on moral-to-fable retrieval with various instruction configurations.

## Method
Test each model with multiple instruction variants (no instruction, general, task-specific). Models range from small BERT-based to 7B+ parameter instruction-following models.

## How to run
```bash
python experiments/02_model_comparison/run.py                        # all models
python experiments/02_model_comparison/run.py --models linq-embed    # specific model
```

## Key results (top 5)

| Model | Variant | MRR | R@1 | R@10 |
|-------|---------|-----|-----|------|
| Linq-Embed-Mistral | general | **0.210** | **14.1%** | **36.4%** |
| Linq-Embed-Mistral | no_instr | 0.184 | 11.7% | 30.2% |
| SFR-Embedding-Mistral | general | 0.178 | 11.3% | 31.2% |
| facebook/drama-1b | prompted | 0.171 | 10.3% | 29.3% |
| stella_en_1.5B_v5 | prompted | 0.169 | 10.7% | 28.6% |

## Key findings
- **Linq-Embed-Mistral is the best model** (MRR=0.210) with a generic instruction
- Larger models generally outperform smaller ones
- Task-specific instructions ("moral", "fable") often hurt vs. generic instructions
- Mistral-based models dominate the top ranks
