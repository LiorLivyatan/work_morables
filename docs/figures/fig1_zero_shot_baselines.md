# Fig 1 — Zero-Shot Embedding Baselines

![Zero-Shot Baselines](fig1_zero_shot_baselines.png)

## What this shows

Two phases of zero-shot evaluation — no fine-tuning at any point.

**Task:** Given a moral statement, retrieve the Aesop fable that best conveys it.  
**Dataset:** 709 fables, 709 moral queries.  
**Metric:** MRR@10 (Mean Reciprocal Rank at cutoff 10).

---

### Panel A — Early Models (Exp 01 / 02, raw fables corpus)

Tested ~15 embedding models on the raw fable text as the retrieval corpus.  
No instruction prefix was used for most models; the winning setup for each was recorded.

| Model | MRR@10 |
|---|---|
| Qwen3-Embedding-0.6B | **0.104** |
| Multilingual-E5-large | 0.095 |
| Instructor-base | 0.088 |
| E5-base-v2 | 0.086 |
| TART | 0.084 |
| Contriever | 0.083 |
| all-MiniLM-L6-v2 | 0.079 |
| all-mpnet-base-v2 | 0.077 |
| BGE-base-en-v1.5 | 0.069 |

**Key insight:** Sentence-level models plateau around 0.10 MRR on raw fable text.  
Fables are long, narrative documents — short query vs. long document asymmetry hurts retrieval.

---

### Panel B — Large Models (Exp 11, fable+summary corpus)

Tested 7 large embedding models (475M–8B) using the **fable+summary** corpus, where each fable is augmented with a Gemini-generated conceptual summary. Linq-Embed-Mistral is included as a reference since it was the anchor for all fine-tuning.

| Model | MRR@10 | Notes |
|---|---|---|
| Linq-Embed-Mistral | 0.360 | Reference |
| BGE-en-ICL | 0.319 | Surprisingly strong; uses in-context learning |
| Qwen3-Embedding-8B | 0.264 | Good instruction following |
| Nomic-Embed-v2-MoE | 0.221 | MoE 475M active params |
| Llama-Embed-Nemotron-8B (no instr) | 0.086 | Broken — bidirectional attn incompatible with ST |
| NV-Embed-v2 | ERROR | transformers v5.5 breaking change in custom code |
| GTE-Qwen2-7B | ERROR | Same transformers v5 compatibility issue |

**Key insight:** No model beats Linq zero-shot. BGE-en-ICL is the strongest challenger at 0.319.  
Linq's 7B scale + asymmetric instruction tuning gives it an edge on this domain.
