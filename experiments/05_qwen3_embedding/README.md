# Experiment 05: Qwen3-Embedding with Instruction Steering

## What it tests
Whether instruction-steered representations from Qwen3-Embedding (a decoder-based embedding model) can improve moral-to-fable retrieval. Also tests prompt repetition (Leviathan et al., 2025).

## Method
Qwen3-Embedding uses the format `Instruct: {instruction}\nQuery:{text}` for queries. We test various instructions to steer the embedding toward moral meaning. Documents (fables) are always encoded as plain text.

**Instruction variants tested:**
- `baseline`: Generic retrieval instruction (71 chars)
- `moral_focused`, `analytical`, `abstract`: Detailed moral-specific instructions (243-396 chars)
- `summarize_match`, `cot_style`, `summary_retrieval`: Short task-specific instructions (72-81 chars)
- `baseline_x2`, `baseline_x3`: Prompt repetition (repeat full query 2x/3x)

## How to run
```bash
python experiments/05_qwen3_embedding/run.py --models 8B --instructions baseline cot_style
```

## Key results (8B model)

| Instruction | Length | MRR | R@1 |
|-------------|--------|-----|-----|
| baseline | 71 chars | 0.183 | 11.0% |
| baseline_x3 (repeat) | 71 x3 | 0.183 | 10.7% |
| baseline_x2 (repeat) | 71 x2 | 0.180 | 10.6% |
| cot_style | 81 chars | 0.162 | 8.9% |
| abstract | 380 chars | 0.105 | 5.5% |

## Key findings
- **All task-specific instructions HURT performance** vs. the generic baseline
- Perfect inverse correlation between instruction length and retrieval quality
- Detailed instructions cause "instruction dominance" — they overwhelm short moral queries in the last-token representation
- **Prompt repetition (x2, x3) has no effect** on embedding quality
- Qwen3-Embedding-8B (MRR=0.183) underperforms Linq-Embed-Mistral (MRR=0.210)
