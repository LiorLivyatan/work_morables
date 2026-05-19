# MORABLES Experiment Catalog

Date: 2026-05-17

Purpose: one human-readable inventory of experiments we have run or planned, with the models, configurations, result files, and current status. This is meant as an audit checklist before the final clustered-data reruns.

Current benchmark variants:

- Original benchmark: 709 moral rows as queries, 709 fables as documents, single-label qrels.
- Clustered benchmark draft: 668 unique moral queries, 709 fables, 554 clusters, 1122 multi-label qrel rows.
- Clustered files: `data/clustered/*`.
- Cluster source: `analysis/clusters_full.json`.

## Data And Analysis Setup

### Raw / Processed Data

- Source data: `data/raw/fables.json`, `data/raw/mcqa.json`.
- Original processed retrieval files: `data/processed/fables_corpus.json`, `morals_corpus.json`, `qrels_moral_to_fable.json`, `qrels_fable_to_moral.json`, augmented MCQA moral corpus.
- Clustered processed files: `data/clustered/fables_corpus.json`, `morals_unique_corpus.json`, `qrels_moral_to_fable_clustered.json`, `cluster_mapping.json`.

### Cluster Review / Ambiguity Analysis

- Files: `analysis/clusters_full.json`, `analysis/moral_similarity*.csv`, `analysis/review_decisions.json`, `analysis/07_soft_mrr/*`.
- Main purpose: identify exact/near-equivalent morals and build multi-label clustered qrels.
- Current important stats: 668 unique moral queries, 554 clusters, 709 fables covered once.
- Related outputs: `results/clustered_recalculation/linq-embed-mistral__general__clustered_metrics.json`.

## Retrieval / Embedding Experiments

### Exp 01 — Baseline Embedding Retrieval

- Folder: `experiments/01_baselines/`.
- Purpose: initial small sentence-transformer baselines.
- Tasks/configs:
  - Fable -> moral retrieval.
  - Moral -> fable retrieval.
  - Fable -> augmented moral retrieval with MCQA distractors.
- Models actually in `run.py`:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `sentence-transformers/all-mpnet-base-v2`
- README also mentions broader early baselines:
  - `BAAI/bge-large-en-v1.5`
  - `intfloat/e5-large-v2`
  - `intfloat/multilingual-e5-large`
- Result files:
  - `experiments/01_baselines/results/baseline_results.json`
  - `experiments/01_baselines/results/improved_retrieval_results.json`
- Clustered status: aggregate-only; rerun scoring if we want final clustered numbers.

### Exp 02 — Large-Scale Embedding Model Comparison

- Folder: `experiments/02_model_comparison/`.
- Main result run: `results/runs/2026-03-15_combined_v2/`.
- Purpose: broad off-the-shelf embedding sweep on original moral-to-fable retrieval.
- Result artifacts:
  - `results/runs/2026-03-15_combined_v2/results.json`
  - `results/runs/2026-03-15_combined_v2/predictions/*.json`
- Completed runs in `combined_v2`: 45.
- Full rankings saved: yes, top_k=709. This is the easiest full clustered recalculation target.
- Completed model/run keys include:
  - `Linq-AI-Research/Linq-Embed-Mistral`: `no_instr`, `general`, `specific`
  - `Salesforce/SFR-Embedding-Mistral`: `no_instr`, `general`, `specific`
  - `Qwen/Qwen3-Embedding-4B`: `no_instr`, `general`, `specific`
  - `Qwen/Qwen3-Embedding-0.6B`: `no_instr`, `general`, `specific`
  - `intfloat/e5-mistral-7b-instruct`: `no_instr`, `general`, `specific`
  - `Alibaba-NLP/gte-Qwen2-1.5B-instruct`: `no_instr`, `general`, `specific`
  - `Alibaba-NLP/gte-Qwen2-7B-instruct`: `no_instr`, `general`
  - `NovaSearch/stella_en_1.5B_v5`: `no_instr`, `prompted`
  - `facebook/drama-1b`: `no_instr`, `prompted`
  - `hkunlp/instructor-base`: `no_instr`, `general`, `specific`
  - `hkunlp/instructor-xl`: `no_instr`, `general`, `specific`
  - `intfloat/multilingual-e5-large`: `plain`
  - `intfloat/multilingual-e5-large-instruct`: `no_instr`, `general`, `specific`
  - `BAAI/bge-base-en-v1.5`: `plain`
  - `BAAI/bge-m3`: `plain`
  - `intfloat/e5-base-v2`: `plain`
  - `facebook/contriever`: `plain`
  - `orionweller/tart-dual-contriever-msmarco`: `no_instr`, `specific`
  - `nomic-ai/nomic-embed-text-v2-moe`: `no_instr`, `prefixed`
  - `vec-ai/lychee-embed`: `no_instr`, `general`, `specific`
- Registry also included attempted/planned adapters not all present in final completed JSON:
  - `jinaai/jina-embeddings-v3`
  - `GritLM/GritLM-7B`
  - `nvidia/llama-embed-nemotron-8b`
  - `BAAI/bge-multilingual-gemma2`
  - `nvidia/NV-Embed-v2`
- Instruction variants:
  - `no_instr`
  - `general`: "Given a text, retrieve the most relevant passage that answers the query"
  - `specific`: "Given a moral principle or lesson, retrieve the fable that illustrates it"
  - model-specific prompt/prefix variants where needed.
- Clustered status: can fully recalculate from saved rankings.

### Exp 03 — LLM Summarisation, Approach B

- Folder: `experiments/03_llm_summarisation/`.
- Purpose: generate fable moral summaries with local Qwen models, then retrieve using Linq embeddings.
- Generator models:
  - Qwen3.5 0.8B
  - Qwen3.5 2B
  - Qwen3.5 4B
  - Qwen3.5 9B
- Prompt styles:
  - `direct`
  - `detailed`
  - `cot`
  - `few_shot`
- Retriever:
  - `Linq-AI-Research/Linq-Embed-Mistral`
- Special oracle:
  - fable + ground-truth moral concatenation, upper-bound style result.
- Result files:
  - `experiments/03_llm_summarisation/results/approach_b_results.csv`
  - `experiments/03_llm_summarisation/results/approach_b_summary.md`
- Clustered status: aggregate-only; generated summaries are not clearly stored in the results folder, so rerun/reconstruct only if this historical experiment remains important.

### Exp 04 — LLM CoT Reranking

- Folder: `experiments/04_llm_reranking/`.
- Purpose: rerank embedding top-K candidates using Gemini reasoning/scoring.
- Stage-1 retrieval:
  - Linq-Embed-Mistral and/or BGE variants.
- Stage-2 reranker:
  - Gemini flash-lite / Gemini 2.5 flash style API models, depending on run.
- Configs:
  - top-20 reranking.
  - sample sizes: 5 and 100 queries in recorded README.
- Result file:
  - `experiments/04_llm_reranking/results/llm_reranking_results.json`
- Clustered status: small-sample aggregate-only; rerun only if reranking is back in scope.

### Exp 05 — Qwen3 Embedding Instruction Steering

- Folder: `experiments/05_qwen3_embedding/`.
- Purpose: test whether Qwen3 embedding instructions improve moral-to-fable retrieval.
- Main model:
  - `Qwen/Qwen3-Embedding-8B`
- Instruction variants:
  - `baseline`
  - `moral_focused`
  - `analytical`
  - `abstract`
  - `summarize_match`
  - `cot_style`
  - `summary_retrieval`
  - `baseline_x2`
  - `baseline_x3`
- Documents:
  - raw fable text.
- Result files:
  - `experiments/05_qwen3_embedding/results/*/results.json`
  - `experiments/05_qwen3_embedding/results/*/predictions/*.json`
  - mirrored under `results/qwen3_embedding_runs/*/predictions/*.json`
- Prediction depth:
  - top-10 only.
- Clustered status:
  - can partially recalculate top-10 clustered metrics.
  - must rerun scoring for full rankings / Recall@100+.

### Exp 06 — Sentence-Level Chunking Retrieval

- Folder: `experiments/06_sentence_chunking/`.
- Purpose: test fable sentence/chunk indexing instead of whole-fable embeddings.
- Main model:
  - Linq-Embed-Mistral baseline style.
- Chunking strategies:
  - full fable baseline.
  - sentence chunks.
  - last-N sentence variants.
  - sliding windows.
- Aggregation methods:
  - max.
  - top-k mean.
  - weighted, favoring later sentences.
- Result file:
  - `experiments/06_sentence_chunking/results/2026-03-29_11-23-18/results.json`
- Clustered status: aggregate-only; rerun if sentence/chunking is included in final comparisons.

### Exp 07 — SOTA Summarization Oracle

- Folder: `experiments/07_sota_summarization_oracle/`.
- Purpose: use Gemini 3.1 Pro to create high-quality moral summaries for each fable, then embed/retrieve.
- Generator:
  - Gemini 3.1 Pro.
- Summary variants:
  - `direct_moral`
  - `narrative_distillation`
  - `conceptual_abstract`
  - later full result also contains exp-07-compatible variants such as `cot_proverb`.
- Retrieval model:
  - `Linq-AI-Research/Linq-Embed-Mistral`
- Corpus/eval configs:
  - summary only.
  - fable + summary.
  - fable + instruction / baseline raw fable.
- Result artifacts:
  - `experiments/07_sota_summarization_oracle/results/generation_runs/full_709/golden_summaries.json`
  - `experiments/07_sota_summarization_oracle/results/generation_runs/full_709/retrieval_results_all_variants.json`
- Clustered status:
  - summaries are saved; rerun embedding scoring only.
  - no need to regenerate summaries unless the fable text changes.

### Exp 08 — Symmetric Moral Matching

- Folder: `experiments/08_symmetric_moral_matching/`.
- Purpose: make both sides moral-like: summarize fables into aphorisms and expand moral queries.
- Sample sizes:
  - sample1, sample10, sample50.
- Corpus variants:
  - `ground_truth_style`
  - `declarative_universal`
- Query expansion variants:
  - `moral_rephrase`
  - `moral_elaborate`
  - `moral_abstract`
- Retrieval configs:
  - `A`
  - `B`
  - `A_expand`
  - `B_expand`
  - `RRF_all`, reciprocal rank fusion with k=60.
- Baseline:
  - exp07 `conceptual_abstract`.
- Result files:
  - `experiments/08_symmetric_moral_matching/results/generation_runs/*/retrieval_results.json`
  - plus `query_expansions.json`, `corpus_summaries.json`, token usage files.
- Clustered status: sample-only; rerun/ignore depending on whether this pilot remains part of the final story.

### Exp 08 Autoresearch — Local Retrieval Search Loop

- Folder: `experiments/08_autoresearch/`.
- Purpose: automated local search over retrieval pipeline tweaks.
- Editable axes:
  - model ID.
  - query instruction.
  - corpus instruction.
  - chunking strategy.
  - BM25 sparse fusion.
  - cross-encoder reranker.
  - query rewrite template.
- Starting baseline:
  - Linq-Embed-Mistral generic instruction.
- Result file:
  - `experiments/08_autoresearch/results.tsv`
- Status: barely started / exploratory.
- Clustered status: not a final experiment unless resumed after data freeze.

### Exp 09 — Gemma 4 Local Summarization Oracle

- Folder: `experiments/09_gemma4_summarization/`.
- Purpose: use local Gemma 4 open-weight models to generate moral summaries and compare with Gemini oracle.
- Generator models from README:
  - `mlx-community/gemma-4-e2b-it-4bit`
  - `mlx-community/gemma-4-e4b-it-4bit`
  - `mlx-community/gemma-4-31b-it-4bit`
- Prompt variants:
  - `direct_moral`
  - `narrative_distillation`
  - `conceptual_abstract`
- Retriever:
  - Linq-Embed-Mistral.
- Result artifacts:
  - `experiments/09_gemma4_summarization/results/generation_runs/*/golden_summaries.json`
  - `experiments/09_gemma4_summarization/results/runs/*/results.json`
  - `experiments/09_gemma4_summarization/results/embedding_cache/*.npy`
- Clustered status: generated summaries/cache exist; rerun scoring if historical Gemma 4 numbers are needed.

### Exp 09 Thinking Smoke

- Folder: `experiments/09_gemma4_thinking_smoke/`.
- Purpose: smoke tests for Gemma 4 thinking mode and generation behavior.
- Generator models:
  - Gemma 4 E2B/E4B/31B variants as smoke-tested.
- Result artifacts:
  - `experiments/09_gemma4_thinking_smoke/results/generation_runs/*/golden_summaries.json`
  - `experiments/09_gemma4_thinking_smoke/results/runs/*/results.json`
- Status: smoke / historical, not central.

### Exp 10 — Local Model Matrix

- Folder: `experiments/10_model_matrix/`.
- Purpose: matrix of local generation models x embedding models, with ablations.
- Generation models:
  - `Qwen/Qwen3.5-9B`
  - `Qwen/Qwen3.5-4B`
  - `microsoft/Phi-3.5-mini-instruct`
  - `mistralai/Mistral-7B-Instruct-v0.3`
- Embedding models:
  - `Linq-AI-Research/Linq-Embed-Mistral`
  - `Qwen/Qwen3-Embedding-8B`
  - `BAAI/bge-m3`
  - `BAAI/bge-large-en-v1.5`
  - `nomic-ai/nomic-embed-text-v1.5`
- Ablation modes:
  - raw/raw.
  - summary only.
  - paraphrase only.
  - full: summary corpus + query paraphrases.
  - RRF fusion.
- Result status:
  - documented in `experiments/10_model_matrix/README.md`.
  - not included in the master embedding ledger as a primary final result.
- Clustered status: rerun only if this matrix is needed.

### Exp 11 — Additional Embedding Baselines

- Folder: `experiments/11_embedding_baselines/`.
- Purpose: later baseline sweep on `fable_plus_summary`.
- Doc mode:
  - `fable_plus_summary`.
- Configured models:
  - `Linq-AI-Research/Linq-Embed-Mistral`
  - `nvidia/NV-Embed-v2`
  - `Alibaba-NLP/gte-Qwen2-7B-instruct`
  - `BAAI/bge-en-icl`
  - `google/embeddinggemma-300m`
  - `nomic-ai/nomic-embed-text-v2-moe`
  - `Qwen/Qwen3-Embedding-8B`
  - `nvidia/llama-embed-nemotron-8b`
  - `nvidia/llama-embed-nemotron-8b` no-instruction variant
  - `tencent/KaLM-Embedding-Gemma3-12B-2511`
  - `GritLM/GritLM-7B`
- Completed/result-bearing models found:
  - `Nomic-Embed-v2-MoE`
  - `Llama-Embed-Nemotron-8B`
  - `Llama-Embed-Nemotron-8B-NoInstr`
  - `BGE-en-ICL`
  - `Qwen3-Embedding-8B`
  - some rows for `EmbeddingGemma-300M`, `NV-Embed-v2`, `GTE-Qwen2-7B` have no metric in inspected JSON.
- Result files:
  - `experiments/11_embedding_baselines/results/*.json`
  - partial embedding cache under `experiments/11_embedding_baselines/cache/embeddings/`.
- Clustered status:
  - if cache exists, encode final unique moral queries and recompute.
  - otherwise rerun scoring.

### Exp 12 — Zero-Shot Comprehensive Matrix

- Folder: `experiments/12_zero_shot_comprehensive/`.
- Purpose: comprehensive zero-shot matrix of embedding models x corpus configs x instruction variants.
- Models, 10:
  - `Linq-AI-Research/Linq-Embed-Mistral`
  - `BAAI/bge-en-icl`
  - `Qwen/Qwen3-Embedding-8B`
  - `nomic-ai/nomic-embed-text-v2-moe`
  - `Qwen/Qwen3-Embedding-0.6B`
  - `intfloat/multilingual-e5-large`
  - `intfloat/multilingual-e5-large-instruct`
  - `BAAI/bge-m3`
  - `hkunlp/instructor-base`
  - `intfloat/e5-base-v2`
- Corpus configs, 6:
  - `raw`: `{fable}`
  - `fable_cot`: `{fable}` + `cot_proverb`
  - `fable_direct`: `{fable}` + `direct_moral`
  - `fable_abstract`: `{fable}` + `conceptual_abstract`
  - `summary_only_cot`: `cot_proverb` only
  - `summary_fable_cot`: `cot_proverb` first, then fable
- Instruction variants, 2:
  - `default`: each model's own configured instruction.
  - `generic`: `Instruct: Given a text, retrieve the most relevant passage that answers the query\nQuery: `
- Total completed rows:
  - 120.
- Result file:
  - `experiments/12_zero_shot_comprehensive/results/zero_shot_comprehensive.csv`
- Clustered status:
  - aggregate-only locally; rerun scoring.
  - no LLM generation needed.
  - rerun should save full rankings.

### Exp 13 — Gemma 4 GPU Summarization Evaluation

- Folder: `experiments/13_gemma4_gpu_summarization/`.
- Purpose: full GPU Gemma/Gemini summary generation and retrieval evaluation.
- Generator models:
  - `google/gemma-4-E2B-it`
  - `google/gemma-4-E4B-it`
  - `google/gemma-4-26B-A4B-it`
  - `google/gemma-4-31B-it`
  - result CSV also includes `gemini` as comparator.
- Prompt variants:
  - `cot_proverb`
  - `direct_moral`
  - `conceptual_abstract`
  - `proverb`
  - `thinking_cot_proverb`
  - `thinking_direct_moral`
- Retrieval models:
  - `Qwen/Qwen3-Embedding-8B`
  - `Linq-AI-Research/Linq-Embed-Mistral`
- Result files:
  - `experiments/13_gemma4_gpu_summarization/results/2026-05-06_11-54-49_eval_all.csv`
  - per-retriever CSVs in same folder.
- Completed rows:
  - 112.
- Clustered status:
  - aggregate-only in current repo; rerun eval/scoring.
  - check whether generated summaries exist remotely before regenerating.

## Full-Context LLM Retrieval

### LLM Retrieval Framework

- Folder: `llm_retrieval/`.
- Purpose: send the whole fable corpus to an LLM and ask it to return top-k fable IDs for each moral.
- Current default prompt variant:
  - `minimal`: system says "You are a retrieval system. Return only the requested JSON."
- Historical prompt variants:
  - `minimal`
  - `detailed`
  - `cot`
- Top-k:
  - 10.
- Configured models:
  - OpenAI: `gpt-4o-mini`, `gpt-4o`, `gpt-5.4-nano`, `gpt-5.5`
  - OpenAI OSS via OpenRouter: `openai/gpt-oss-20b:free`, `openai/gpt-oss-120b:free`
  - Anthropic: `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`, `claude-opus-4-7`
  - Google: `gemini-2.5-pro`, `gemini-3-flash`, `gemini-3.1-flash-lite-preview`, `gemini-3.1-pro`
  - Qwen via OpenRouter: `qwen/qwen3.5-flash`, `qwen/qwen3.6-plus`, `qwen/qwen3.6-27b`
  - Meta via OpenRouter: `meta-llama/Llama-4-Scout-17B-Instruct`

### LLM Runs Actually Present

- Single/smoke CSVs:
  - `llm_retrieval/results/runs/2026-05-14_GPT-4o-mini_minimal.csv`
  - `llm_retrieval/results/runs/2026-05-14_Gemini-2.5-Pro_minimal.csv`
  - `llm_retrieval/results/runs/2026-05-15_GPT-5.4-Nano_minimal.csv`
  - `llm_retrieval/results/runs/2026-05-15_Gemini-3.1-Flash-Lite-Preview_minimal.csv`
- OpenAI batch:
  - `llm_retrieval/results/batches/merged/2026-05-15_GPT-5.4-Nano_minimal_batch_n709_rows.csv`
  - summary JSON in same merged folder.
- Gemini batch:
  - `llm_retrieval/results/gemini_batches/2026-05-16_Gemini-3.1-Flash-Lite-Preview_minimal_batch_n709/rows.csv`
  - summary JSON in same folder.
- Comparison files against ft07:
  - `llm_retrieval/results/comparisons/*Gemini-3.1-Flash-Lite-Preview*`.
- Clustered status:
  - top-10 clustered metrics can be recalculated from rows CSV.
  - official final LLM benchmark should rerun after cluster data is final, using 668 final query texts.
  - cannot compute Recall@100/200/300 unless we request more than top 10.

## Fine-Tuning Experiments

### FT 00 — Overfit Sanity

- Folder: `finetuning/ft_00_overfit/`.
- Purpose: sanity check that training can memorize / improve on a tiny or direct setup.
- Model:
  - `BAAI/bge-base-en-v1.5`
- Config:
  - `doc_mode: raw`
  - epochs 30
  - batch size 64
  - LR 2e-5
- Result:
  - `finetuning/ft_00_overfit/results/2026-04-06_12-43-03_raw.json`
- Local model artifact:
  - yes, `finetuning/ft_00_overfit/cache/models/raw/model.safetensors`.

### FT 01 — BGE 5-Fold CV

- Folder: `finetuning/ft_01_5fold_cv/`.
- Purpose: 5-fold supervised fine-tuning on MORABLES.
- Model:
  - `BAAI/bge-base-en-v1.5`
- Config:
  - `doc_mode: raw`, with possible `fable_plus_summary` override.
  - epochs 30
  - batch size 32
  - LR 2e-5
  - 5 folds.
- Result:
  - `finetuning/ft_01_5fold_cv/results/2026-04-06_14-25-34_raw_all_folds.json`
- Local model artifacts:
  - yes, folds 0-4 under `finetuning/ft_01_5fold_cv/cache/models/raw/`.
- Clustered status:
  - re-evaluate existing fold models; no immediate retraining required.

### FT 02 — LINQ 5-Fold CV

- Folder: `finetuning/ft_02_linq_5fold_cv/`.
- Purpose: 5-fold MORABLES fine-tuning of LINQ with LoRA.
- Model:
  - `Linq-AI-Research/Linq-Embed-Mistral`
- Config:
  - doc modes: raw and fable_plus_summary evaluated.
  - epochs 10
  - batch size 4
  - gradient accumulation 4
  - LR 1e-4
  - LoRA r=64, alpha=128, dropout=0.05, target modules q/k/v/o projections.
  - query instruction: moral-to-fable task-specific instruction.
- Result files:
  - multiple under `finetuning/ft_02_linq_5fold_cv/results/*.json`.
- Local model artifacts:
  - current workspace has incomplete-looking fold dirs without model weights.
- Clustered status:
  - pull/recover models if possible; otherwise rerun selected eval/training.

### FT 03 — Hard Negative Fine-Tuning

- Folder: `finetuning/ft_03_hard_neg/`.
- Purpose: LINQ LoRA with hard negative variants.
- Model:
  - `Linq-AI-Research/Linq-Embed-Mistral`
- Doc mode:
  - `fable_plus_summary`.
- Hard negative variants/results:
  - `basic`
  - `hard_neg_based_on_adjectives`
  - `hard_neg_injected_adjectives`
  - `hard_neg_partial_story`
- Config:
  - epochs 10
  - batch size 4
  - gradient accumulation 4
  - LR 1e-4
  - temperature 0.05
  - LoRA r=64, alpha=128, dropout=0.05.
- Result files:
  - `finetuning/ft_03_hard_neg/results/*.json`.
- Clustered status:
  - no saved rankings; recover model weights or rerun selected variants after final data.

### FT 04 — STORAL Augment

- Folder: `finetuning/ft_04_storal_augment/`.
- Purpose: augment MORABLES training with STORAL external moral-story pairs.
- Model:
  - `Linq-AI-Research/Linq-Embed-Mistral`
- Doc mode:
  - `fable_plus_summary`.
- Config:
  - epochs 10
  - batch size 4
  - LR 1e-4
  - temperature 0.05
  - LoRA r=64.
- Result:
  - one fold0 result in `finetuning/ft_04_storal_augment/results/*.json`.
- Clustered status:
  - optional; recover/re-evaluate only if needed.

### FT 05 — STORAL Mixing Ratio

- Folder: `finetuning/ft_05_mixing_ratio/`.
- Purpose: BGE-base with varying amounts of STORAL added to MORABLES fold0 training.
- Model:
  - `BAAI/bge-base-en-v1.5`
- Doc mode:
  - `fable_plus_summary`.
- Ratios:
  - 0
  - 200
  - 500
  - 1000
  - 1675
- Config:
  - epochs 15
  - batch size 32
  - LR 2e-5
  - temperature 0.05.
- Result:
  - `finetuning/ft_05_mixing_ratio/results/2026-04-24_12-24-34_mixing_ratio_fold0.json`
- Local model artifacts:
  - yes, ratio models under `finetuning/ft_05_mixing_ratio/cache/models/`.
- Clustered status:
  - re-evaluate existing local models if this ablation remains useful.

### FT 06 — LINQ Random Search

- Folder: `finetuning/ft_06_random_search/`.
- Purpose: fold0 hyperparameter search for LINQ LoRA.
- Model:
  - `Linq-AI-Research/Linq-Embed-Mistral`
- Doc mode:
  - `fable_plus_summary`.
- Search space:
  - LoRA rank: 32, 64, 128.
  - temperature: 0.02, 0.05, 0.07, 0.1.
  - learning rate: 5e-5, 1e-4, 2e-4.
- Fixed config:
  - epochs 10
  - batch size 4
  - gradient accumulation 4
  - LoRA dropout 0.05.
- Local artifacts:
  - some checkpoint/model artifacts exist for r32/r64 variants.
- Result status:
  - master ledger marks no completed consolidated result file.

### FT 07 — STORAL Transfer

- Folder: `finetuning/ft_07_storal_transfer/`.
- Purpose: train on STORAL only, evaluate zero-shot on all 709 MORABLES fables.
- Models:
  - `BAAI/bge-base-en-v1.5`
  - `Linq-AI-Research/Linq-Embed-Mistral`
  - `Qwen/Qwen3-Embedding-8B`
- Dataset sizes:
  - `s500`: 500 STORAL pairs, MORABLES p10-p90 length range.
  - `s1000`: 1000 STORAL pairs, wider length range.
  - `sfull`: all 1675 clean STORAL pairs.
- Config:
  - epochs 10
  - temperature 0.05
  - BGE full fine-tune, batch 32, LR 2e-5.
  - LINQ/Qwen3 LoRA r=64, alpha=128, dropout=0.05, batch 4, gradient accumulation 4, LR 1e-4.
- Result files:
  - nine JSONs under `finetuning/ft_07_storal_transfer/results/*.json`.
- Notable previous best:
  - LINQ `s500`, especially with `fable_plus_summary`.
- Clustered status:
  - no saved rankings in result files.
  - recover/pull model weights first; retrain only if recovery fails.

### FT 08 — Clean Sanity

- Folder: `finetuning/ft_08_clean_sanity/`.
- Purpose: compare noisy all-709 training versus clean 615-query subset with ambiguous cases removed.
- Model:
  - `Linq-AI-Research/Linq-Embed-Mistral`
- Config:
  - same as ft07 LINQ best style.
  - `doc_mode: fable_plus_summary`
  - 5 folds
  - epochs 10
  - LoRA r=64.
- Status:
  - planned/configured; no completed result found in current master ledger.

### FT 09 — False-Negative Masking

- Folder: `finetuning/ft_09_fn_masking/`.
- Purpose: LINQ LoRA with moral-similarity false-negative masking during contrastive training.
- Model:
  - `Linq-AI-Research/Linq-Embed-Mistral`
- Config:
  - `doc_mode: fable_plus_summary`
  - 5 folds
  - epochs 10
  - temperature 0.05
  - false-negative mask threshold: 0.85
  - LoRA r=64, alpha=128, dropout=0.05.
- Result files:
  - `finetuning/ft_09_fn_masking/results/*.json`
  - per-fold JSONs exist.
- Clustered status:
  - conceptually relevant to new cluster labels.
  - no rankings; recover model weights or rerun selected version after final data.

### FT 10 — IdioLink Transfer

- Folder: `finetuning/ft_10_idiolink_transfer/`.
- Purpose: transfer from IdioLink idiom/literal retrieval data to MORABLES.

Option A — Zero-shot transfer:

- Config: `option_a_zero_shot/config.yaml`.
- Training data:
  - 440 IdioLink training triplets.
- Models:
  - `Linq-AI-Research/Linq-Embed-Mistral`
  - `BAAI/bge-en-icl`
  - `Qwen/Qwen3-Embedding-8B`
  - `Qwen/Qwen3-Embedding-0.6B`
  - `intfloat/multilingual-e5-large`
  - `intfloat/multilingual-e5-large-instruct`
  - `BAAI/bge-m3`
  - `intfloat/e5-base-v2`
  - `nomic-ai/nomic-embed-text-v2-moe`
  - `hkunlp/instructor-base`
- LoRA for large models:
  - LINQ, BGE-en-ICL, Qwen3-8B.
- Result files:
  - `finetuning/ft_10_idiolink_transfer/option_a_zero_shot/results/*.csv`.

Option B — Sequential:

- Stage 1:
  - pretrain LINQ LoRA on IdioLink.
  - config: `option_b_sequential/config_pretrain.yaml`.
- Stage 2:
  - fine-tune on MORABLES from the IdioLink-pretrained checkpoint.
  - config: `option_b_sequential/config_finetune.yaml`.
- Status:
  - planned/configured, not completed in current ledger.

Option C / D:

- Option C mixing and Option D hard negatives are documented in README files.
- Status:
  - planned/configured, not completed.

## Planned / Future Experiments

### Exp 14 — Contrastive Fine-Tuning Plan

- File: `docs/exp14_finetuning_plan.md`.
- Planned sub-experiments:
  - 14a: MORABLES-only 5-fold CV.
  - 14b: external dataset transfer.
  - 14c: synthetic fable generation.
- Candidate base models:
  - `Linq-AI-Research/Linq-Embed-Mistral`
  - `BAAI/bge-large-en-v1.5`
- Loss:
  - `MultipleNegativesRankingLoss`.
- Data sources discussed:
  - MORABLES.
  - STORAL.
  - Moral Stories.
  - ePiC.
  - Project Gutenberg Aesop.
  - synthetic LLM-generated fables.
- Current status:
  - plan only. With clustered data, this should become a new cluster-aware CV experiment rather than replacing older 5FCV.

### Other Future Directions

- Cross-encoder full-scale reranking:
  - candidate: `cross-encoder/ms-marco-MiniLM-L-6-v2` or L-12.
- ColBERT / late interaction:
  - candidate: `colbert-ir/colbertv2.0`.
- HyDE / hypothetical fable query generation:
  - generator: Gemini / Claude / Gemma.
  - embed generated hypothetical fable against real fable corpus.
- Projection layer:
  - MLP over frozen LINQ embeddings.
- MCQA distractor confusion analysis:
  - use MORABLES MCQA distractor labels.
- Adversarial robustness:
  - use `data/raw/adversarial_*.json`.

## Master Result Ledgers

- Main embedding/fine-tuning ledger:
  - `docs/embedding_experiments_master.csv`
  - 458 rows.
- Clustered recalculation inventory:
  - `docs/clustered_recalculation_inventory.csv`
  - 28 rows.
- General old summary:
  - `docs/results_all_experiments.csv`
- FT07 summary:
  - `docs/results_ft07_comprehensive.csv`
- Fine-tuning config summary:
  - `docs/finetuning_configs.csv`
