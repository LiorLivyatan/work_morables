# Direction 1: Improved Retrieval — Stronger Models, Instructions, CoT

## Goal
Show that stronger embedding models with task-specific instructions, and LLM-based CoT reranking, yield significantly better moral retrieval than the ~3% R@1 baseline.

## Tasks

### Task 1: Add instruction-aware embedding models
Create `scripts/06_improved_retrieval.py` that tests:
- `BAAI/bge-large-en-v1.5` (supports `"Represent this sentence: "` prefix)
- `intfloat/e5-large-v2` (supports `"query: "` / `"passage: "` prefixes)
- `Alibaba-NLP/gte-large-en-v1.5`
- `intfloat/multilingual-e5-large` (for future cross-lingual work)

Each model needs its correct instruction/prefix format. Reuse the same evaluation pipeline from `04_baseline_retrieval.py`.

### Task 2: Test different instruction prefixes
For instruction-aware models, compare:
- No instruction (raw text)
- Generic instruction
- Task-specific instruction (e.g., "Given this fable, retrieve the abstract moral lesson it teaches")
- Moral-focused instruction variations

### Task 3: Add LLM-based CoT reranking
Create `scripts/07_llm_reranking.py`:
- Take top-50 candidates from best embedding model
- Use Claude API with CoT prompt to rerank
- Prompt: "Read this fable. Think step by step about the deeper lesson. Then rank which moral best matches."
- Also try: Generate a moral summary via CoT, then embed the summary

### Task 4: Consolidate results
- Update `results/` with all new metrics
- Create comparison table showing improvement over baseline
- Update TASKS.md with findings
