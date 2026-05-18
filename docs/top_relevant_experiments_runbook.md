# Top Relevant Retrieval Experiments Runbook

Generated from local configs/results on 2026-05-17.

This file is the working checklist for the clustered MORABLES benchmark. It focuses on the experiments we are most likely to keep: zero-shot embedding retrieval, summary-augmented zero-shot retrieval, and fine-tuning / transfer learning.

Current clustered benchmark target:
- Queries: 668 unique moral texts after clustering/union.
- Documents: 709 fables.
- Relevance rows: 1122 moral-query to fable links.
- Official metrics should include MRR, MAP@10, Hit@10, Recall@10, Recall@100, Recall@200, Recall@300, and NDCG@10 where the artifact depth allows it.

## Rerun Flag Semantics

| Flag | Meaning |
| --- | --- |
| No | We have enough local artifact to rescore against the clustered qrels without calling the embedding model again. Usually this means full rankings, or full cached query/document embeddings. |
| Partial | We can rescore only shallow metrics, usually top-10, because the saved artifact is only top-10. Not enough for Recall@100/200/300. |
| Eval only | Training/generation does not need to be redone if the model/summaries are available, but we must run evaluation again to create clustered rankings. |
| Yes | We need to rerun model scoring/evaluation, or retrain for official cluster-aware CV. Aggregate metrics alone are not enough. |

Important: MORABLES-trained cross-validation runs are not just a metric-recalculation problem. With clustered labels, the folds should be cluster-aware so equivalent morals do not leak across train/test. Old MORABLES-CV results can be diagnostic, but official final CV should be rerun with cluster-aware splits once the clustered data is final.

## Immediate Shortlist

| Priority | Experiment/config | Why keep it | Clustered action |
| --- | --- | --- | --- |
| 1 | Exp02 Linq-Embed-Mistral, raw fable, general instruction | Best old raw zero-shot baseline and already has full 709 rankings. | No rerun; already recalc-compatible. |
| 1 | Exp12 Qwen3-Embedding-8B, summary_fable_cot, generic instruction | Best zero-shot summary-augmented aggregate result in exp12. | Yes; aggregate CSV only, rerun full scoring on clustered data. |
| 1 | Exp12 Linq-Embed-Mistral, fable_cot, generic instruction | Best/near-best Linq summary-augmented zero-shot setting. | Yes; aggregate CSV only, rerun full scoring on clustered data. |
| 1 | FT07 Linq STORAL s500, fable_plus_summary | Best completed fine-tuned/transfer result and has local full embeddings for this one config. | No model rerun for diagnostic recalc; rerun official only if we change evaluation corpus or want fresh artifacts. |
| 2 | FT07 Qwen3-Embedding-8B STORAL sfull/s500 | Top non-Linq fine-tuning candidate. | Eval only if weights can be pulled; otherwise retrain selected config. |
| 2 | FT09 Linq false-negative masking | Best MORABLES-CV style method so far, conceptually aligned with clusters. | Yes for official cluster-aware CV retraining. |
| 3 | Synthetic fable fine-tuning | Not done; likely a thesis contribution candidate. | Yes; design and run after clustered data is final. |

## Zero-Shot Embedding Experiments

These runs use moral texts as queries and fables or fable-derived documents as the indexed corpus. For zero-shot runs, there is no MORABLES train/test leakage, so saved full rankings or embeddings are enough to rescore the new clustered qrels.

### Exp01 Early Baselines

Early sanity baselines in `experiments/01_baselines` and root-level result files. These are mostly useful as historical context; they do not have the full ranked artifacts needed for final clustered metrics.
| Config family | Models/configs | Document/query setup | Existing artifact | Rerun required? | Use going forward |
| --- | --- | --- | --- | --- | --- |
| Sentence-transformer baselines | all-MiniLM-L6-v2; all-mpnet-base-v2 | fable->moral and moral->fable sanity retrieval on raw text | aggregate JSON/CSV only | Yes | Historical only. Do not prioritize unless we need small-model controls. |
| Early stronger baselines | BGE-large-en-v1.5; E5-large-v2; multilingual-E5-large | raw fable documents; model-native prefixes where relevant | aggregate results only | Yes | Covered better by exp02/exp12. |
| Augmented moral / MCQA baseline | same small baseline family | fable to augmented moral with MCQA distractor context | aggregate results only | Yes | Historical only; not part of final primary benchmark. |

### Exp02 Broad Raw-Fable Embedding Sweep

Raw fable corpus, 709 moral queries, full ranked fable lists saved under `results/runs/2026-03-15_combined_v2/predictions`. These are the cleanest historical zero-shot artifacts for direct clustered rescoring.
| Run key | Model ID | Corpus | Instruction/config | Old MRR@10 | Artifact depth | Rerun required? |
| --- | --- | --- | --- | --- | --- | --- |
| bge-base__plain | BAAI/bge-base-en-v1.5 | raw fable | plain | 0.069066 | full 709 ranking | No |
| bge-m3__plain | BAAI/bge-m3 | raw fable | plain | 0.088894 | full 709 ranking | No |
| contriever__plain | facebook/contriever | raw fable | plain | 0.083705 | full 709 ranking | No |
| drama__no_instr | facebook/drama-1b | raw fable | no_instr | 0.084990 | full 709 ranking | No |
| drama__prompted | facebook/drama-1b | raw fable | prompted | 0.091843 | full 709 ranking | No |
| e5-base__plain | intfloat/e5-base-v2 | raw fable | plain | 0.086181 | full 709 ranking | No |
| e5-mistral-7b__general | intfloat/e5-mistral-7b-instruct | raw fable | general | 0.145910 | full 709 ranking | No |
| e5-mistral-7b__no_instr | intfloat/e5-mistral-7b-instruct | raw fable | no_instr | 0.100199 | full 709 ranking | No |
| e5-mistral-7b__specific | intfloat/e5-mistral-7b-instruct | raw fable | specific | 0.051967 | full 709 ranking | No |
| gte-qwen2-1.5b__general | Alibaba-NLP/gte-Qwen2-1.5B-instruct | raw fable | general | 0.080615 | full 709 ranking | No |
| gte-qwen2-1.5b__no_instr | Alibaba-NLP/gte-Qwen2-1.5B-instruct | raw fable | no_instr | 0.056056 | full 709 ranking | No |
| gte-qwen2-1.5b__specific | Alibaba-NLP/gte-Qwen2-1.5B-instruct | raw fable | specific | 0.064312 | full 709 ranking | No |
| gte-qwen2-7b__general | Alibaba-NLP/gte-Qwen2-7B-instruct | raw fable | general | 0.044619 | full 709 ranking | No |
| gte-qwen2-7b__no_instr | Alibaba-NLP/gte-Qwen2-7B-instruct | raw fable | no_instr | 0.014980 | full 709 ranking | No |
| instructor-base__general | hkunlp/instructor-base | raw fable | general | 0.084896 | full 709 ranking | No |
| instructor-base__no_instr | hkunlp/instructor-base | raw fable | no_instr | 0.087654 | full 709 ranking | No |
| instructor-base__specific | hkunlp/instructor-base | raw fable | specific | 0.073340 | full 709 ranking | No |
| instructor-xl__general | hkunlp/instructor-xl | raw fable | general | 0.121256 | full 709 ranking | No |
| instructor-xl__no_instr | hkunlp/instructor-xl | raw fable | no_instr | 0.133228 | full 709 ranking | No |
| instructor-xl__specific | hkunlp/instructor-xl | raw fable | specific | 0.115489 | full 709 ranking | No |
| linq-embed-mistral__general | Linq-AI-Research/Linq-Embed-Mistral | raw fable | general | 0.210495 | full 709 ranking | No |
| linq-embed-mistral__no_instr | Linq-AI-Research/Linq-Embed-Mistral | raw fable | no_instr | 0.184208 | full 709 ranking | No |
| linq-embed-mistral__specific | Linq-AI-Research/Linq-Embed-Mistral | raw fable | specific | 0.080086 | full 709 ranking | No |
| lychee__general | vec-ai/lychee-embed | raw fable | general | 0.100587 | full 709 ranking | No |
| lychee__no_instr | vec-ai/lychee-embed | raw fable | no_instr | 0.099674 | full 709 ranking | No |
| lychee__specific | vec-ai/lychee-embed | raw fable | specific | 0.068538 | full 709 ranking | No |
| multilingual-e5-instruct__general | intfloat/multilingual-e5-large-instruct | raw fable | general | 0.088898 | full 709 ranking | No |
| multilingual-e5-instruct__no_instr | intfloat/multilingual-e5-large-instruct | raw fable | no_instr | 0.079227 | full 709 ranking | No |
| multilingual-e5-instruct__specific | intfloat/multilingual-e5-large-instruct | raw fable | specific | 0.093780 | full 709 ranking | No |
| multilingual-e5__plain | intfloat/multilingual-e5-large | raw fable | plain | 0.094876 | full 709 ranking | No |
| nomic-v2-moe__no_instr | nomic-ai/nomic-embed-text-v2-moe | raw fable | no_instr | 0.102181 | full 709 ranking | No |
| nomic-v2-moe__prefixed | nomic-ai/nomic-embed-text-v2-moe | raw fable | prefixed | 0.082499 | full 709 ranking | No |
| qwen3-0.6b__general | Qwen/Qwen3-Embedding-0.6B | raw fable | general | 0.103476 | full 709 ranking | No |
| qwen3-0.6b__no_instr | Qwen/Qwen3-Embedding-0.6B | raw fable | no_instr | 0.096653 | full 709 ranking | No |
| qwen3-0.6b__specific | Qwen/Qwen3-Embedding-0.6B | raw fable | specific | 0.088521 | full 709 ranking | No |
| qwen3-4b__general | Qwen/Qwen3-Embedding-4B | raw fable | general | 0.161961 | full 709 ranking | No |
| qwen3-4b__no_instr | Qwen/Qwen3-Embedding-4B | raw fable | no_instr | 0.141876 | full 709 ranking | No |
| qwen3-4b__specific | Qwen/Qwen3-Embedding-4B | raw fable | specific | 0.123907 | full 709 ranking | No |
| sfr-mistral__general | Salesforce/SFR-Embedding-Mistral | raw fable | general | 0.178017 | full 709 ranking | No |
| sfr-mistral__no_instr | Salesforce/SFR-Embedding-Mistral | raw fable | no_instr | 0.156188 | full 709 ranking | No |
| sfr-mistral__specific | Salesforce/SFR-Embedding-Mistral | raw fable | specific | 0.100143 | full 709 ranking | No |
| stella-1.5b__no_instr | NovaSearch/stella_en_1.5B_v5 | raw fable | no_instr | 0.144757 | full 709 ranking | No |
| stella-1.5b__prompted | NovaSearch/stella_en_1.5B_v5 | raw fable | prompted | 0.168635 | full 709 ranking | No |
| tart__no_instr | orionweller/tart-dual-contriever-msmarco | raw fable | no_instr | 0.084350 | full 709 ranking | No |
| tart__specific | orionweller/tart-dual-contriever-msmarco | raw fable | specific | 0.067345 | full 709 ranking | No |

### Exp02 Registry Entries Without Final Combined Results

These appeared in the exp02 model registry but do not have rows in the final combined result inventory. Keep them visible so we do not confuse "configured" with "completed".
| Run key | Model ID | Status | Rerun required? | Note |
| --- | --- | --- | --- | --- |
| jina-v3__no_instr | jinaai/jina-embeddings-v3 | configured/no final combined result | Yes | Run only if we explicitly want this model in the final sweep. |
| jina-v3__retrieval | jinaai/jina-embeddings-v3 | configured/no final combined result | Yes | Run only if we explicitly want this model in the final sweep. |
| gritlm-7b__no_instr | GritLM/GritLM-7B | configured/no final combined result | Yes | Run only if we explicitly want this model in the final sweep. |
| gritlm-7b__specific | GritLM/GritLM-7B | configured/no final combined result | Yes | Run only if we explicitly want this model in the final sweep. |
| gte-qwen2-7b__specific | Alibaba-NLP/gte-Qwen2-7B-instruct | configured/no final combined result | Yes | Run only if we explicitly want this model in the final sweep. |
| nemotron-8b__no_instr | nvidia/llama-embed-nemotron-8b | configured/no final combined result | Yes | Run only if we explicitly want this model in the final sweep. |
| nemotron-8b__general | nvidia/llama-embed-nemotron-8b | configured/no final combined result | Yes | Run only if we explicitly want this model in the final sweep. |
| nemotron-8b__specific | nvidia/llama-embed-nemotron-8b | configured/no final combined result | Yes | Run only if we explicitly want this model in the final sweep. |
| bge-gemma2__no_instr | BAAI/bge-multilingual-gemma2 | configured/no final combined result | Yes | Run only if we explicitly want this model in the final sweep. |
| bge-gemma2__general | BAAI/bge-multilingual-gemma2 | configured/no final combined result | Yes | Run only if we explicitly want this model in the final sweep. |
| bge-gemma2__specific | BAAI/bge-multilingual-gemma2 | configured/no final combined result | Yes | Run only if we explicitly want this model in the final sweep. |
| nv-embed-v2__no_instr | nvidia/NV-Embed-v2 | configured/no final combined result | Yes | Run only if we explicitly want this model in the final sweep. |
| nv-embed-v2__general | nvidia/NV-Embed-v2 | configured/no final combined result | Yes | Run only if we explicitly want this model in the final sweep. |
| nv-embed-v2__specific | nvidia/NV-Embed-v2 | configured/no final combined result | Yes | Run only if we explicitly want this model in the final sweep. |

### Exp05 Qwen3-Embedding-8B Instruction Steering

Qwen3-specific query-side instruction steering. Documents were always raw fable text. Saved predictions contain only `top_10`, so they are not enough for the final deep-recall clustered metrics.
| Run key | Model ID | Corpus | Instruction variant | Instruction text | Old n | Old MRR@10 | Artifact depth | Rerun required? | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8B__abstract | Qwen/Qwen3-Embedding-8B | raw fable | abstract | Given a concise moral truth, retrieve the fable that serves as its narrative embodiment. | 5 | 0.249272 | top-10 only | Yes | sample only |
| 8B__analytical | Qwen/Qwen3-Embedding-8B | raw fable | analytical | Given a moral statement about human nature, retrieve the fable that illustrates this principle. | 5 | 0.262785 | top-10 only | Yes | sample only |
| 8B__baseline | Qwen/Qwen3-Embedding-8B | raw fable | baseline | Given a text, retrieve the most relevant passage that matches this text | 5 | 0.304529 | top-10 only | Yes | sample only |
| 8B__moral_focused | Qwen/Qwen3-Embedding-8B | raw fable | moral_focused | Given a moral principle or life lesson, retrieve the fable/parable that teaches this exact lesson. | 5 | 0.266370 | top-10 only | Yes | sample only |
| 2026-03-28_20-00-39 | Qwen/Qwen3-Embedding-8B | raw fable | - | - | - | - | metadata only | Yes | no results.json |
| 8B__abstract | Qwen/Qwen3-Embedding-8B | raw fable | abstract | Given a concise moral truth, retrieve the fable that serves as its narrative embodiment. | 709 | 0.104555 | top-10 only | Partial | official rerun needed for Recall@100/200/300 |
| 8B__analytical | Qwen/Qwen3-Embedding-8B | raw fable | analytical | Given a moral statement about human nature, retrieve the fable that illustrates this principle. | 709 | 0.110030 | top-10 only | Partial | official rerun needed for Recall@100/200/300 |
| 8B__baseline | Qwen/Qwen3-Embedding-8B | raw fable | baseline | Given a text, retrieve the most relevant passage that matches this text | 709 | 0.183359 | top-10 only | Partial | official rerun needed for Recall@100/200/300 |
| 8B__moral_focused | Qwen/Qwen3-Embedding-8B | raw fable | moral_focused | Given a moral principle or life lesson, retrieve the fable/parable that teaches this exact lesson. | 709 | 0.126956 | top-10 only | Partial | official rerun needed for Recall@100/200/300 |
| 8B__cot_style | Qwen/Qwen3-Embedding-8B | raw fable | cot_style | Given a moral, reason about what story would teach this lesson, then retrieve it. | 709 | 0.161661 | top-10 only | Partial | official rerun needed for Recall@100/200/300 |
| 8B__summarize_match | Qwen/Qwen3-Embedding-8B | raw fable | summarize_match | Given a moral lesson, retrieve the fable whose core message matches this moral. | 709 | 0.110949 | top-10 only | Partial | official rerun needed for Recall@100/200/300 |
| 8B__summary_retrieval | Qwen/Qwen3-Embedding-8B | raw fable | summary_retrieval | Given a moral, retrieve the fable whose summary best matches this moral. | 709 | 0.130029 | top-10 only | Partial | official rerun needed for Recall@100/200/300 |
| 8B__baseline_x2 | Qwen/Qwen3-Embedding-8B | raw fable | baseline_x2 | baseline instruction, formatted query repeated 2x. | 709 | 0.180434 | top-10 only | Partial | official rerun needed for Recall@100/200/300 |
| 8B__baseline_x3 | Qwen/Qwen3-Embedding-8B | raw fable | baseline_x3 | baseline instruction, formatted query repeated 3x. | 709 | 0.183497 | top-10 only | Partial | official rerun needed for Recall@100/200/300 |

### Exp11 Additional Embedding Baselines On Fable+Summary

Fable+summary document mode, model-specific query instruction/prefixes. Only a subset has successful metrics, and only three models have full local embeddings that can be rescored without model calls.
| Model alias | Model ID | Corpus/doc mode | Query instruction/prefix | Old MRR@10 | Status | Artifact | Rerun required? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | Instruct: Given a moral statement, retrieve the fable that best conveys this moral. Query: | - | no local result row | none | Yes | Retry only if this model remains relevant. |
| NV-Embed-v2 | nvidia/NV-Embed-v2 | fable_plus_summary | Instruct: Given a moral statement, retrieve the fable that best conveys this moral. Query: | - | error | 2026-04-27_15-06-17_NV-Embed-v2.json | Yes | Retry only if this model remains relevant. |
| GTE-Qwen2-7B | Alibaba-NLP/gte-Qwen2-7B-instruct | fable_plus_summary | Instruct: Given a moral statement, retrieve the fable that best conveys this moral. Query: | - | error | 2026-04-27_15-14-24_GTE-Qwen2-7B.json | Yes | Retry only if this model remains relevant. |
| BGE-en-ICL | BAAI/bge-en-icl | fable_plus_summary | Given a moral statement, retrieve the fable that best conveys this moral. | 0.319178 | completed | full embeddings cached | No | Recalculate rankings from cached embeddings. |
| EmbeddingGemma-300M | google/embeddinggemma-300m | fable_plus_summary | - | - | error | 2026-04-27_14-49-59_EmbeddingGemma-300M_Llama-Embed-Nemotron-8B.json | Yes | Retry only if this model remains relevant. |
| Nomic-Embed-v2-MoE | nomic-ai/nomic-embed-text-v2-moe | fable_plus_summary | search_query: | 0.221127 | completed | full embeddings cached | No | Recalculate rankings from cached embeddings. |
| Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | fable_plus_summary | Instruct: Given a moral statement, retrieve the fable that best conveys this moral. Query: | 0.263596 | completed | full embeddings cached | No | Recalculate rankings from cached embeddings. |
| Llama-Embed-Nemotron-8B | nvidia/llama-embed-nemotron-8b | fable_plus_summary | Instruct: Given a moral statement, retrieve the fable that best conveys this moral. Query: | 0.055116 | completed | 2026-04-27_14-49-59_EmbeddingGemma-300M_Llama-Embed-Nemotron-8B.json | Yes | Rerun scoring because only aggregate metrics are local. |
| Llama-Embed-Nemotron-8B-NoInstr | nvidia/llama-embed-nemotron-8b | fable_plus_summary | - | 0.086049 | completed | 2026-04-27_15-00-41_Llama-Embed-Nemotron-8B-NoInstr.json | Yes | Rerun scoring because only aggregate metrics are local. |
| KaLM-Gemma3-12B | tencent/KaLM-Embedding-Gemma3-12B-2511 | fable_plus_summary | Instruct: Given a moral statement, retrieve the fable that best conveys this moral. Query: | - | no local result row | none | Yes | Retry only if this model remains relevant. |
| GritLM-7B | GritLM/GritLM-7B | fable_plus_summary | <\|user\|> Given a moral statement, retrieve the fable that best conveys this moral. <\|em... | - | no local result row | none | Yes | Retry only if this model remains relevant. |

### Exp12 Comprehensive Zero-Shot Matrix

This is the main zero-shot matrix: 10 embedding models x 6 corpus configurations x 2 instruction variants = 120 completed aggregate rows. No full local ranking artifacts were found, so official clustered metrics require rerunning scoring.
| Model | Model ID | Corpus config | Document representation | Instruction variant | Query instruction | Old MRR@10 | Artifact | Rerun required? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | raw | raw fable text only | default | model default instruction/prefix | 0.095004 | aggregate CSV only | Yes |
| Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | fable_cot | fable + cot_proverb summary | default | model default instruction/prefix | 0.211060 | aggregate CSV only | Yes |
| Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | fable_direct | fable + direct_moral summary | default | model default instruction/prefix | 0.212461 | aggregate CSV only | Yes |
| Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | fable_abstract | fable + conceptual_abstract summary | default | model default instruction/prefix | 0.207749 | aggregate CSV only | Yes |
| Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | summary_only_cot | cot_proverb summary only (no fable) | default | model default instruction/prefix | 0.230353 | aggregate CSV only | Yes |
| Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | summary_fable_cot | cot_proverb first, then fable (reversed order) | default | model default instruction/prefix | 0.229335 | aggregate CSV only | Yes |
| BGE-en-ICL | BAAI/bge-en-icl | raw | raw fable text only | default | model default instruction/prefix | 0.165869 | aggregate CSV only | Yes |
| BGE-en-ICL | BAAI/bge-en-icl | fable_cot | fable + cot_proverb summary | default | model default instruction/prefix | 0.318431 | aggregate CSV only | Yes |
| BGE-en-ICL | BAAI/bge-en-icl | fable_direct | fable + direct_moral summary | default | model default instruction/prefix | 0.319882 | aggregate CSV only | Yes |
| BGE-en-ICL | BAAI/bge-en-icl | fable_abstract | fable + conceptual_abstract summary | default | model default instruction/prefix | 0.323494 | aggregate CSV only | Yes |
| BGE-en-ICL | BAAI/bge-en-icl | summary_only_cot | cot_proverb summary only (no fable) | default | model default instruction/prefix | 0.289746 | aggregate CSV only | Yes |
| BGE-en-ICL | BAAI/bge-en-icl | summary_fable_cot | cot_proverb first, then fable (reversed order) | default | model default instruction/prefix | 0.300152 | aggregate CSV only | Yes |
| Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | raw | raw fable text only | default | model default instruction/prefix | 0.106328 | aggregate CSV only | Yes |
| Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | fable_cot | fable + cot_proverb summary | default | model default instruction/prefix | 0.263275 | aggregate CSV only | Yes |
| Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | fable_direct | fable + direct_moral summary | default | model default instruction/prefix | 0.269123 | aggregate CSV only | Yes |
| Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | fable_abstract | fable + conceptual_abstract summary | default | model default instruction/prefix | 0.271328 | aggregate CSV only | Yes |
| Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | summary_only_cot | cot_proverb summary only (no fable) | default | model default instruction/prefix | 0.258634 | aggregate CSV only | Yes |
| Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | summary_fable_cot | cot_proverb first, then fable (reversed order) | default | model default instruction/prefix | 0.293282 | aggregate CSV only | Yes |
| Nomic-Embed-v2-MoE | nomic-ai/nomic-embed-text-v2-moe | raw | raw fable text only | default | model default instruction/prefix | 0.082476 | aggregate CSV only | Yes |
| Nomic-Embed-v2-MoE | nomic-ai/nomic-embed-text-v2-moe | fable_cot | fable + cot_proverb summary | default | model default instruction/prefix | 0.220586 | aggregate CSV only | Yes |
| Nomic-Embed-v2-MoE | nomic-ai/nomic-embed-text-v2-moe | fable_direct | fable + direct_moral summary | default | model default instruction/prefix | 0.219189 | aggregate CSV only | Yes |
| Nomic-Embed-v2-MoE | nomic-ai/nomic-embed-text-v2-moe | fable_abstract | fable + conceptual_abstract summary | default | model default instruction/prefix | 0.225653 | aggregate CSV only | Yes |
| Nomic-Embed-v2-MoE | nomic-ai/nomic-embed-text-v2-moe | summary_only_cot | cot_proverb summary only (no fable) | default | model default instruction/prefix | 0.236206 | aggregate CSV only | Yes |
| Nomic-Embed-v2-MoE | nomic-ai/nomic-embed-text-v2-moe | summary_fable_cot | cot_proverb first, then fable (reversed order) | default | model default instruction/prefix | 0.252393 | aggregate CSV only | Yes |
| Qwen3-Embedding-0.6B | Qwen/Qwen3-Embedding-0.6B | raw | raw fable text only | default | model default instruction/prefix | 0.087977 | aggregate CSV only | Yes |
| Qwen3-Embedding-0.6B | Qwen/Qwen3-Embedding-0.6B | fable_cot | fable + cot_proverb summary | default | model default instruction/prefix | 0.235567 | aggregate CSV only | Yes |
| Qwen3-Embedding-0.6B | Qwen/Qwen3-Embedding-0.6B | fable_direct | fable + direct_moral summary | default | model default instruction/prefix | 0.255880 | aggregate CSV only | Yes |
| Qwen3-Embedding-0.6B | Qwen/Qwen3-Embedding-0.6B | fable_abstract | fable + conceptual_abstract summary | default | model default instruction/prefix | 0.256111 | aggregate CSV only | Yes |
| Qwen3-Embedding-0.6B | Qwen/Qwen3-Embedding-0.6B | summary_only_cot | cot_proverb summary only (no fable) | default | model default instruction/prefix | 0.202584 | aggregate CSV only | Yes |
| Qwen3-Embedding-0.6B | Qwen/Qwen3-Embedding-0.6B | summary_fable_cot | cot_proverb first, then fable (reversed order) | default | model default instruction/prefix | 0.255760 | aggregate CSV only | Yes |
| Multilingual-E5-large | intfloat/multilingual-e5-large | raw | raw fable text only | default | model default instruction/prefix | 0.093338 | aggregate CSV only | Yes |
| Multilingual-E5-large | intfloat/multilingual-e5-large | fable_cot | fable + cot_proverb summary | default | model default instruction/prefix | 0.214650 | aggregate CSV only | Yes |
| Multilingual-E5-large | intfloat/multilingual-e5-large | fable_direct | fable + direct_moral summary | default | model default instruction/prefix | 0.222608 | aggregate CSV only | Yes |
| Multilingual-E5-large | intfloat/multilingual-e5-large | fable_abstract | fable + conceptual_abstract summary | default | model default instruction/prefix | 0.229506 | aggregate CSV only | Yes |
| Multilingual-E5-large | intfloat/multilingual-e5-large | summary_only_cot | cot_proverb summary only (no fable) | default | model default instruction/prefix | 0.230043 | aggregate CSV only | Yes |
| Multilingual-E5-large | intfloat/multilingual-e5-large | summary_fable_cot | cot_proverb first, then fable (reversed order) | default | model default instruction/prefix | 0.279467 | aggregate CSV only | Yes |
| Multilingual-E5-large-instruct | intfloat/multilingual-e5-large-instruct | raw | raw fable text only | default | model default instruction/prefix | 0.088987 | aggregate CSV only | Yes |
| Multilingual-E5-large-instruct | intfloat/multilingual-e5-large-instruct | fable_cot | fable + cot_proverb summary | default | model default instruction/prefix | 0.239634 | aggregate CSV only | Yes |
| Multilingual-E5-large-instruct | intfloat/multilingual-e5-large-instruct | fable_direct | fable + direct_moral summary | default | model default instruction/prefix | 0.238617 | aggregate CSV only | Yes |
| Multilingual-E5-large-instruct | intfloat/multilingual-e5-large-instruct | fable_abstract | fable + conceptual_abstract summary | default | model default instruction/prefix | 0.231134 | aggregate CSV only | Yes |
| Multilingual-E5-large-instruct | intfloat/multilingual-e5-large-instruct | summary_only_cot | cot_proverb summary only (no fable) | default | model default instruction/prefix | 0.222179 | aggregate CSV only | Yes |
| Multilingual-E5-large-instruct | intfloat/multilingual-e5-large-instruct | summary_fable_cot | cot_proverb first, then fable (reversed order) | default | model default instruction/prefix | 0.267789 | aggregate CSV only | Yes |
| BGE-M3 | BAAI/bge-m3 | raw | raw fable text only | default | model default instruction/prefix | 0.089027 | aggregate CSV only | Yes |
| BGE-M3 | BAAI/bge-m3 | fable_cot | fable + cot_proverb summary | default | model default instruction/prefix | 0.257381 | aggregate CSV only | Yes |
| BGE-M3 | BAAI/bge-m3 | fable_direct | fable + direct_moral summary | default | model default instruction/prefix | 0.249424 | aggregate CSV only | Yes |
| BGE-M3 | BAAI/bge-m3 | fable_abstract | fable + conceptual_abstract summary | default | model default instruction/prefix | 0.263634 | aggregate CSV only | Yes |
| BGE-M3 | BAAI/bge-m3 | summary_only_cot | cot_proverb summary only (no fable) | default | model default instruction/prefix | 0.235532 | aggregate CSV only | Yes |
| BGE-M3 | BAAI/bge-m3 | summary_fable_cot | cot_proverb first, then fable (reversed order) | default | model default instruction/prefix | 0.276119 | aggregate CSV only | Yes |
| Instructor-base | hkunlp/instructor-base | raw | raw fable text only | default | model default instruction/prefix | 0.083859 | aggregate CSV only | Yes |
| Instructor-base | hkunlp/instructor-base | fable_cot | fable + cot_proverb summary | default | model default instruction/prefix | 0.220395 | aggregate CSV only | Yes |
| Instructor-base | hkunlp/instructor-base | fable_direct | fable + direct_moral summary | default | model default instruction/prefix | 0.218020 | aggregate CSV only | Yes |
| Instructor-base | hkunlp/instructor-base | fable_abstract | fable + conceptual_abstract summary | default | model default instruction/prefix | 0.232844 | aggregate CSV only | Yes |
| Instructor-base | hkunlp/instructor-base | summary_only_cot | cot_proverb summary only (no fable) | default | model default instruction/prefix | 0.242335 | aggregate CSV only | Yes |
| Instructor-base | hkunlp/instructor-base | summary_fable_cot | cot_proverb first, then fable (reversed order) | default | model default instruction/prefix | 0.236561 | aggregate CSV only | Yes |
| E5-base-v2 | intfloat/e5-base-v2 | raw | raw fable text only | default | model default instruction/prefix | 0.086715 | aggregate CSV only | Yes |
| E5-base-v2 | intfloat/e5-base-v2 | fable_cot | fable + cot_proverb summary | default | model default instruction/prefix | 0.166489 | aggregate CSV only | Yes |
| E5-base-v2 | intfloat/e5-base-v2 | fable_direct | fable + direct_moral summary | default | model default instruction/prefix | 0.168909 | aggregate CSV only | Yes |
| E5-base-v2 | intfloat/e5-base-v2 | fable_abstract | fable + conceptual_abstract summary | default | model default instruction/prefix | 0.170442 | aggregate CSV only | Yes |
| E5-base-v2 | intfloat/e5-base-v2 | summary_only_cot | cot_proverb summary only (no fable) | default | model default instruction/prefix | 0.224454 | aggregate CSV only | Yes |
| E5-base-v2 | intfloat/e5-base-v2 | summary_fable_cot | cot_proverb first, then fable (reversed order) | default | model default instruction/prefix | 0.252476 | aggregate CSV only | Yes |
| Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | raw | raw fable text only | generic | generic shared instruction | 0.210713 | aggregate CSV only | Yes |
| Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | fable_cot | fable + cot_proverb summary | generic | generic shared instruction | 0.359747 | aggregate CSV only | Yes |
| Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | fable_direct | fable + direct_moral summary | generic | generic shared instruction | 0.350240 | aggregate CSV only | Yes |
| Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | fable_abstract | fable + conceptual_abstract summary | generic | generic shared instruction | 0.348126 | aggregate CSV only | Yes |
| Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | summary_only_cot | cot_proverb summary only (no fable) | generic | generic shared instruction | 0.303064 | aggregate CSV only | Yes |
| Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | summary_fable_cot | cot_proverb first, then fable (reversed order) | generic | generic shared instruction | 0.349903 | aggregate CSV only | Yes |
| BGE-en-ICL | BAAI/bge-en-icl | raw | raw fable text only | generic | generic shared instruction | 0.145494 | aggregate CSV only | Yes |
| BGE-en-ICL | BAAI/bge-en-icl | fable_cot | fable + cot_proverb summary | generic | generic shared instruction | 0.309028 | aggregate CSV only | Yes |
| BGE-en-ICL | BAAI/bge-en-icl | fable_direct | fable + direct_moral summary | generic | generic shared instruction | 0.321160 | aggregate CSV only | Yes |
| BGE-en-ICL | BAAI/bge-en-icl | fable_abstract | fable + conceptual_abstract summary | generic | generic shared instruction | 0.318651 | aggregate CSV only | Yes |
| BGE-en-ICL | BAAI/bge-en-icl | summary_only_cot | cot_proverb summary only (no fable) | generic | generic shared instruction | 0.296334 | aggregate CSV only | Yes |
| BGE-en-ICL | BAAI/bge-en-icl | summary_fable_cot | cot_proverb first, then fable (reversed order) | generic | generic shared instruction | 0.291770 | aggregate CSV only | Yes |
| Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | raw | raw fable text only | generic | generic shared instruction | 0.179986 | aggregate CSV only | Yes |
| Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | fable_cot | fable + cot_proverb summary | generic | generic shared instruction | 0.328267 | aggregate CSV only | Yes |
| Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | fable_direct | fable + direct_moral summary | generic | generic shared instruction | 0.332793 | aggregate CSV only | Yes |
| Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | fable_abstract | fable + conceptual_abstract summary | generic | generic shared instruction | 0.346068 | aggregate CSV only | Yes |
| Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | summary_only_cot | cot_proverb summary only (no fable) | generic | generic shared instruction | 0.295644 | aggregate CSV only | Yes |
| Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | summary_fable_cot | cot_proverb first, then fable (reversed order) | generic | generic shared instruction | 0.362549 | aggregate CSV only | Yes |
| Nomic-Embed-v2-MoE | nomic-ai/nomic-embed-text-v2-moe | raw | raw fable text only | generic | generic shared instruction | 0.038727 | aggregate CSV only | Yes |
| Nomic-Embed-v2-MoE | nomic-ai/nomic-embed-text-v2-moe | fable_cot | fable + cot_proverb summary | generic | generic shared instruction | 0.101078 | aggregate CSV only | Yes |
| Nomic-Embed-v2-MoE | nomic-ai/nomic-embed-text-v2-moe | fable_direct | fable + direct_moral summary | generic | generic shared instruction | 0.104235 | aggregate CSV only | Yes |
| Nomic-Embed-v2-MoE | nomic-ai/nomic-embed-text-v2-moe | fable_abstract | fable + conceptual_abstract summary | generic | generic shared instruction | 0.105723 | aggregate CSV only | Yes |
| Nomic-Embed-v2-MoE | nomic-ai/nomic-embed-text-v2-moe | summary_only_cot | cot_proverb summary only (no fable) | generic | generic shared instruction | 0.167394 | aggregate CSV only | Yes |
| Nomic-Embed-v2-MoE | nomic-ai/nomic-embed-text-v2-moe | summary_fable_cot | cot_proverb first, then fable (reversed order) | generic | generic shared instruction | 0.125214 | aggregate CSV only | Yes |
| Qwen3-Embedding-0.6B | Qwen/Qwen3-Embedding-0.6B | raw | raw fable text only | generic | generic shared instruction | 0.105336 | aggregate CSV only | Yes |
| Qwen3-Embedding-0.6B | Qwen/Qwen3-Embedding-0.6B | fable_cot | fable + cot_proverb summary | generic | generic shared instruction | 0.263401 | aggregate CSV only | Yes |
| Qwen3-Embedding-0.6B | Qwen/Qwen3-Embedding-0.6B | fable_direct | fable + direct_moral summary | generic | generic shared instruction | 0.282671 | aggregate CSV only | Yes |
| Qwen3-Embedding-0.6B | Qwen/Qwen3-Embedding-0.6B | fable_abstract | fable + conceptual_abstract summary | generic | generic shared instruction | 0.293502 | aggregate CSV only | Yes |
| Qwen3-Embedding-0.6B | Qwen/Qwen3-Embedding-0.6B | summary_only_cot | cot_proverb summary only (no fable) | generic | generic shared instruction | 0.248871 | aggregate CSV only | Yes |
| Qwen3-Embedding-0.6B | Qwen/Qwen3-Embedding-0.6B | summary_fable_cot | cot_proverb first, then fable (reversed order) | generic | generic shared instruction | 0.287398 | aggregate CSV only | Yes |
| Multilingual-E5-large | intfloat/multilingual-e5-large | raw | raw fable text only | generic | generic shared instruction | 0.090448 | aggregate CSV only | Yes |
| Multilingual-E5-large | intfloat/multilingual-e5-large | fable_cot | fable + cot_proverb summary | generic | generic shared instruction | 0.201914 | aggregate CSV only | Yes |
| Multilingual-E5-large | intfloat/multilingual-e5-large | fable_direct | fable + direct_moral summary | generic | generic shared instruction | 0.208922 | aggregate CSV only | Yes |
| Multilingual-E5-large | intfloat/multilingual-e5-large | fable_abstract | fable + conceptual_abstract summary | generic | generic shared instruction | 0.218152 | aggregate CSV only | Yes |
| Multilingual-E5-large | intfloat/multilingual-e5-large | summary_only_cot | cot_proverb summary only (no fable) | generic | generic shared instruction | 0.241444 | aggregate CSV only | Yes |
| Multilingual-E5-large | intfloat/multilingual-e5-large | summary_fable_cot | cot_proverb first, then fable (reversed order) | generic | generic shared instruction | 0.262288 | aggregate CSV only | Yes |
| Multilingual-E5-large-instruct | intfloat/multilingual-e5-large-instruct | raw | raw fable text only | generic | generic shared instruction | 0.090142 | aggregate CSV only | Yes |
| Multilingual-E5-large-instruct | intfloat/multilingual-e5-large-instruct | fable_cot | fable + cot_proverb summary | generic | generic shared instruction | 0.251579 | aggregate CSV only | Yes |
| Multilingual-E5-large-instruct | intfloat/multilingual-e5-large-instruct | fable_direct | fable + direct_moral summary | generic | generic shared instruction | 0.254043 | aggregate CSV only | Yes |
| Multilingual-E5-large-instruct | intfloat/multilingual-e5-large-instruct | fable_abstract | fable + conceptual_abstract summary | generic | generic shared instruction | 0.252979 | aggregate CSV only | Yes |
| Multilingual-E5-large-instruct | intfloat/multilingual-e5-large-instruct | summary_only_cot | cot_proverb summary only (no fable) | generic | generic shared instruction | 0.270427 | aggregate CSV only | Yes |
| Multilingual-E5-large-instruct | intfloat/multilingual-e5-large-instruct | summary_fable_cot | cot_proverb first, then fable (reversed order) | generic | generic shared instruction | 0.278849 | aggregate CSV only | Yes |
| BGE-M3 | BAAI/bge-m3 | raw | raw fable text only | generic | generic shared instruction | 0.063235 | aggregate CSV only | Yes |
| BGE-M3 | BAAI/bge-m3 | fable_cot | fable + cot_proverb summary | generic | generic shared instruction | 0.194008 | aggregate CSV only | Yes |
| BGE-M3 | BAAI/bge-m3 | fable_direct | fable + direct_moral summary | generic | generic shared instruction | 0.205809 | aggregate CSV only | Yes |
| BGE-M3 | BAAI/bge-m3 | fable_abstract | fable + conceptual_abstract summary | generic | generic shared instruction | 0.202140 | aggregate CSV only | Yes |
| BGE-M3 | BAAI/bge-m3 | summary_only_cot | cot_proverb summary only (no fable) | generic | generic shared instruction | 0.223197 | aggregate CSV only | Yes |
| BGE-M3 | BAAI/bge-m3 | summary_fable_cot | cot_proverb first, then fable (reversed order) | generic | generic shared instruction | 0.216154 | aggregate CSV only | Yes |
| Instructor-base | hkunlp/instructor-base | raw | raw fable text only | generic | generic shared instruction | 0.085339 | aggregate CSV only | Yes |
| Instructor-base | hkunlp/instructor-base | fable_cot | fable + cot_proverb summary | generic | generic shared instruction | 0.215036 | aggregate CSV only | Yes |
| Instructor-base | hkunlp/instructor-base | fable_direct | fable + direct_moral summary | generic | generic shared instruction | 0.214098 | aggregate CSV only | Yes |
| Instructor-base | hkunlp/instructor-base | fable_abstract | fable + conceptual_abstract summary | generic | generic shared instruction | 0.229554 | aggregate CSV only | Yes |
| Instructor-base | hkunlp/instructor-base | summary_only_cot | cot_proverb summary only (no fable) | generic | generic shared instruction | 0.235009 | aggregate CSV only | Yes |
| Instructor-base | hkunlp/instructor-base | summary_fable_cot | cot_proverb first, then fable (reversed order) | generic | generic shared instruction | 0.230136 | aggregate CSV only | Yes |
| E5-base-v2 | intfloat/e5-base-v2 | raw | raw fable text only | generic | generic shared instruction | 0.084113 | aggregate CSV only | Yes |
| E5-base-v2 | intfloat/e5-base-v2 | fable_cot | fable + cot_proverb summary | generic | generic shared instruction | 0.174436 | aggregate CSV only | Yes |
| E5-base-v2 | intfloat/e5-base-v2 | fable_direct | fable + direct_moral summary | generic | generic shared instruction | 0.178844 | aggregate CSV only | Yes |
| E5-base-v2 | intfloat/e5-base-v2 | fable_abstract | fable + conceptual_abstract summary | generic | generic shared instruction | 0.186593 | aggregate CSV only | Yes |
| E5-base-v2 | intfloat/e5-base-v2 | summary_only_cot | cot_proverb summary only (no fable) | generic | generic shared instruction | 0.226524 | aggregate CSV only | Yes |
| E5-base-v2 | intfloat/e5-base-v2 | summary_fable_cot | cot_proverb first, then fable (reversed order) | generic | generic shared instruction | 0.254940 | aggregate CSV only | Yes |

## Summary-Augmented Zero-Shot Retrieval

These experiments generate or use summaries as document-side text, then retrieve with embeddings. They are zero-shot with respect to MORABLES training, but clustered metrics need fresh rankings unless full rankings were saved.

### Exp07 Gemini Oracle Summaries

Gemini-generated summaries are available locally under `experiments/07_sota_summarization_oracle/results/generation_runs/full_709/`. We should not regenerate them unless we change the prompt; just rerun clustered evaluation.
| Run/config | Generator/source | Prompt variant | Retriever | Eval doc mode | Corpus config | Old MRR@10 | Artifact | Rerun required? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_raw_fable | Gemini oracle / ground truth / baseline | - | Linq-Embed-Mistral | raw | raw | 0.209771 | summaries exist; rankings not saved | Eval only |
| conceptual_abstract__fable_plus_summary | Gemini API | conceptual_abstract | Linq-Embed-Mistral | fable_plus_summary_undefined | conceptual_abstract__fable_plus_summary_undefined | 0.349055 | summaries exist; rankings not saved | Eval only |
| conceptual_abstract__summary_only | Gemini API | conceptual_abstract | Linq-Embed-Mistral | summary_only_undefined | conceptual_abstract__summary_only_undefined | 0.355457 | summaries exist; rankings not saved | Eval only |
| cot_proverb__fable_plus_summary | Gemini API | cot_proverb | Linq-Embed-Mistral | fable_plus_summary_undefined | cot_proverb__fable_plus_summary_undefined | 0.360134 | summaries exist; rankings not saved | Eval only |
| cot_proverb__summary_only | Gemini API | cot_proverb | Linq-Embed-Mistral | summary_only_undefined | cot_proverb__summary_only_undefined | 0.304774 | summaries exist; rankings not saved | Eval only |
| direct_moral__fable_plus_summary | Gemini API | direct_moral | Linq-Embed-Mistral | fable_plus_summary_undefined | direct_moral__fable_plus_summary_undefined | 0.351789 | summaries exist; rankings not saved | Eval only |
| direct_moral__summary_only | Gemini API | direct_moral | Linq-Embed-Mistral | summary_only_undefined | direct_moral__summary_only_undefined | 0.336776 | summaries exist; rankings not saved | Eval only |
| proverb__fable_plus_summary | Gemini API | proverb | Linq-Embed-Mistral | fable_plus_summary_undefined | proverb__fable_plus_summary_undefined | 0.350828 | summaries exist; rankings not saved | Eval only |
| proverb__summary_only | Gemini API | proverb | Linq-Embed-Mistral | summary_only_undefined | proverb__summary_only_undefined | 0.278281 | summaries exist; rankings not saved | Eval only |
| oracle__exp_07 | Gemini oracle / ground truth / baseline | - | Linq-Embed-Mistral | fable+ground_truth_moral | fable+ground_truth_moral | 0.893000 | summaries exist; rankings not saved | Eval only |

### Exp13 Gemini/Gemma Summary Matrix

GPU summary generation/evaluation matrix. Current local artifact is aggregate CSV only; generated summary JSONs are not present locally in this folder, so either pull summaries from remote or regenerate before clustered evaluation.
| Generator | Prompt variant | Corpus type | Retriever | Retriever ID | Old MRR@10 | Artifact | Rerun required? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemini | cot_proverb | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.328267 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemini | cot_proverb | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.295644 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemini | direct_moral | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.332793 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemini | direct_moral | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.337420 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemini | conceptual_abstract | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.346068 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemini | conceptual_abstract | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.350463 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemini | proverb | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.316358 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemini | proverb | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.270838 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | cot_proverb | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.213127 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | cot_proverb | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.134819 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | direct_moral | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.225415 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | direct_moral | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.159305 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | conceptual_abstract | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.221103 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | conceptual_abstract | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.116666 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | proverb | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.212658 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | proverb | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.131735 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | thinking_cot_proverb | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.246817 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | thinking_cot_proverb | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.176150 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | thinking_direct_moral | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.243525 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | thinking_direct_moral | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.171380 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | cot_proverb | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.253580 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | cot_proverb | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.209671 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | direct_moral | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.250965 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | direct_moral | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.181042 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | conceptual_abstract | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.261433 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | conceptual_abstract | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.210468 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | proverb | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.225675 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | proverb | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.132270 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | thinking_cot_proverb | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.248547 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | thinking_cot_proverb | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.163084 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | thinking_direct_moral | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.296306 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | thinking_direct_moral | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.232045 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | cot_proverb | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.217800 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | cot_proverb | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.010028 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | direct_moral | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.217800 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | direct_moral | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.010028 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | conceptual_abstract | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.217800 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | conceptual_abstract | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.010028 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | proverb | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.217800 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | proverb | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.010028 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | thinking_cot_proverb | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.217800 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | thinking_cot_proverb | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.010028 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | thinking_direct_moral | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.217800 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | thinking_direct_moral | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.010028 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | cot_proverb | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.276005 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | cot_proverb | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.209693 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | direct_moral | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.297223 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | direct_moral | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.275715 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | conceptual_abstract | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.303532 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | conceptual_abstract | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.267312 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | proverb | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.271697 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | proverb | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.198862 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | thinking_cot_proverb | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.217800 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | thinking_cot_proverb | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.010028 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | thinking_direct_moral | fable_summary | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.217800 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | thinking_direct_moral | summary_only | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 0.010028 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemini | cot_proverb | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.359747 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemini | cot_proverb | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.303064 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemini | direct_moral | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.350240 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemini | direct_moral | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.332607 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemini | conceptual_abstract | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.348126 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemini | conceptual_abstract | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.354532 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemini | proverb | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.350732 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemini | proverb | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.281152 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | cot_proverb | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.192102 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | cot_proverb | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.123459 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | direct_moral | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.221206 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | direct_moral | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.141121 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | conceptual_abstract | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.207233 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | conceptual_abstract | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.118412 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | proverb | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.218781 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | proverb | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.110887 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | thinking_cot_proverb | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.233360 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | thinking_cot_proverb | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.185218 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | thinking_direct_moral | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.213416 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E2B | thinking_direct_moral | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.170437 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | cot_proverb | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.222409 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | cot_proverb | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.182697 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | direct_moral | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.233525 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | direct_moral | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.166817 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | conceptual_abstract | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.232619 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | conceptual_abstract | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.190219 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | proverb | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.215549 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | proverb | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.122589 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | thinking_cot_proverb | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.215801 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | thinking_cot_proverb | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.177153 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | thinking_direct_moral | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.250208 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-E4B | thinking_direct_moral | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.230557 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | cot_proverb | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.251252 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | cot_proverb | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.009844 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | direct_moral | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.251252 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | direct_moral | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.009844 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | conceptual_abstract | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.251252 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | conceptual_abstract | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.009844 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | proverb | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.251252 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | proverb | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.009844 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | thinking_cot_proverb | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.251252 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | thinking_cot_proverb | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.009844 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | thinking_direct_moral | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.251252 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-26B-A4B | thinking_direct_moral | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.009844 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | cot_proverb | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.269708 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | cot_proverb | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.185207 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | direct_moral | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.290593 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | direct_moral | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.267324 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | conceptual_abstract | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.301458 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | conceptual_abstract | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.261908 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | proverb | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.278885 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | proverb | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.172867 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | thinking_cot_proverb | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.251252 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | thinking_cot_proverb | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.009844 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | thinking_direct_moral | fable_summary | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.251252 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |
| gemma4-31B | thinking_direct_moral | summary_only | Linq-Embed-Mistral | Linq-AI-Research/Linq-Embed-Mistral | 0.009844 | aggregate CSV only; generated-summary JSON not local | Eval only or pull+eval |

### Exp13 Optional / Not Yet Run

| Planned run | Retriever(s) | Corpus config | Status | Rerun required? | Note |
| --- | --- | --- | --- | --- | --- |
| gemma4_fable_prefix_config_c | Linq-Embed-Mistral / Qwen3-Embedding-8B | fable_prefix | planned_optional | Yes | Current exp_13 evaluates summary_only and fable_summary. Older exp09 report discussed Config C fable+prefix; no current CSV artifact for GPU matrix Config C. |
| gemma4_more_retrievers | BGE-en-ICL / Nomic / other embeddings | Gemma summaries | planned_optional | Yes | The configured exp_13 grid is complete for Qwen3-Embedding-8B and Linq; optional reruns could evaluate Gemma summaries with the rest of exp_12 embeddings. |

## Fine-Tuning And Transfer Experiments

For final reporting, split these into two categories:

- External-transfer fine-tunes, such as STORAL-only or IdioLink-only training: no MORABLES label leakage, so old models can be reevaluated if weights/embeddings exist.
- MORABLES-supervised CV fine-tunes: official clustered benchmark should be retrained with cluster-aware folds after the data is final. Old aggregate results remain useful as diagnostics and for choosing hyperparameters.

Primary model candidates to carry forward: Linq-Embed-Mistral and Qwen3-Embedding-8B. BGE-base/BGE-M3 are useful as light controls but should not dominate the final rerun budget.

### FT00 Overfit Sanity
| Run | Model | Train doc mode | Eval doc mode | Old MRR@10 | Artifact | Rerun required? | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_before_overfit | BAAI/bge-base-en-v1.5 | raw | raw | 0.063400 | model + embeddings local | No | Not a valid benchmark; sanity only. |
| trained_overfit_same_709 | BAAI/bge-base-en-v1.5 | raw | raw | 0.973554 | model + embeddings local | No | Not a valid benchmark; sanity only. |

### FT01 BGE-Base 5-Fold CV
| Run | Model | Train data | Config | Old MRR@10 | Artifact | Retrain for official clustered CV? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-06_14-25-34_raw_all_folds | BAAI/bge-base-en-v1.5 | MORABLES | raw doc_mode; 5 folds; epochs=30; batch=32; lr=2e-5 | 0.122500 | fold models + fold embeddings local | Yes | Use as BGE control only. |

### FT02 Linq 5-Fold CV
| Run | Model | Train doc mode | Eval doc mode | Train config | Old MRR@10 | Artifact | Retrain for official clustered CV? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-15_11-46-11_raw_all_folds | Linq-AI-Research/Linq-Embed-Mistral | raw | raw | LoRA r=64 alpha=128 dropout=0.05; epochs=10; batch=4; grad_accum=4; lr=1e-4; query moral-specific instruction | 0.312146 | result JSON only; local cache is empty | Yes | For final clustered CV, retrain with cluster-aware folds or pull old weights only for diagnostics. |
| 2026-04-15_12-34-13_raw_eval_fable_plus_summary_all_folds_evaluate | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | fable_plus_summary | LoRA r=64 alpha=128 dropout=0.05; epochs=10; batch=4; grad_accum=4; lr=1e-4; query moral-specific instruction | - | result JSON only; local cache is empty | Yes | For final clustered CV, retrain with cluster-aware folds or pull old weights only for diagnostics. |
| 2026-04-15_12-59-59_raw_eval_fable_plus_summary_all_folds_evaluate | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | fable_plus_summary | LoRA r=64 alpha=128 dropout=0.05; epochs=10; batch=4; grad_accum=4; lr=1e-4; query moral-specific instruction | - | result JSON only; local cache is empty | Yes | For final clustered CV, retrain with cluster-aware folds or pull old weights only for diagnostics. |
| 2026-04-15_13-14-09_raw_eval_fable_plus_summary_all_folds_evaluate | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | fable_plus_summary | LoRA r=64 alpha=128 dropout=0.05; epochs=10; batch=4; grad_accum=4; lr=1e-4; query moral-specific instruction | - | result JSON only; local cache is empty | Yes | For final clustered CV, retrain with cluster-aware folds or pull old weights only for diagnostics. |
| 2026-04-15_13-23-21_raw_all_folds_evaluate | Linq-AI-Research/Linq-Embed-Mistral | raw | raw | LoRA r=64 alpha=128 dropout=0.05; epochs=10; batch=4; grad_accum=4; lr=1e-4; query moral-specific instruction | - | result JSON only; local cache is empty | Yes | For final clustered CV, retrain with cluster-aware folds or pull old weights only for diagnostics. |
| 2026-04-21_08-14-30_raw_all_folds | Linq-AI-Research/Linq-Embed-Mistral | raw | raw | LoRA r=64 alpha=128 dropout=0.05; epochs=10; batch=4; grad_accum=4; lr=1e-4; query moral-specific instruction | 0.317722 | result JSON only; local cache is empty | Yes | For final clustered CV, retrain with cluster-aware folds or pull old weights only for diagnostics. |
| 2026-04-21_08-36-22_raw_all_folds_evaluate | Linq-AI-Research/Linq-Embed-Mistral | raw | raw | LoRA r=64 alpha=128 dropout=0.05; epochs=10; batch=4; grad_accum=4; lr=1e-4; query moral-specific instruction | - | result JSON only; local cache is empty | Yes | For final clustered CV, retrain with cluster-aware folds or pull old weights only for diagnostics. |
| 2026-04-21_08-48-48_raw_eval_fable_plus_summary_all_folds_evaluate | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | fable_plus_summary | LoRA r=64 alpha=128 dropout=0.05; epochs=10; batch=4; grad_accum=4; lr=1e-4; query moral-specific instruction | - | result JSON only; local cache is empty | Yes | For final clustered CV, retrain with cluster-aware folds or pull old weights only for diagnostics. |

### FT03 Linq Hard-Negative Variants
| Run | Model | Eval doc mode | Config | Old MRR@10 | Artifact | Retrain for official clustered CV? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-22_18-47-48_basic_tau005_fable_plus_summary_fold0 | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | mode=basic; tau=0.05 | 0.373676 | result JSON only | Yes | Diagnostic; cluster-aware masking may replace these old distractor choices. |
| 2026-04-23_10-22-49_basic_tau005_fable_plus_summary_fold0 | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | mode=basic; tau=0.05 | 0.390771 | result JSON only | Yes | Diagnostic; cluster-aware masking may replace these old distractor choices. |
| 2026-04-23_11-34-49_basic_tau005_fable_plus_summary_fold0 | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | mode=basic; tau=0.05 | 0.437873 | result JSON only | Yes | Diagnostic; cluster-aware masking may replace these old distractor choices. |
| 2026-04-23_13-01-04_basic_tau005_fable_plus_summary_all_folds | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | mode=basic; tau=0.05 | 0.438112 | result JSON only | Yes | Diagnostic; cluster-aware masking may replace these old distractor choices. |
| 2026-04-23_13-43-10_basic_tau005_fable_plus_summary_fold0 | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | mode=basic; tau=0.05 | 0.428353 | result JSON only | Yes | Diagnostic; cluster-aware masking may replace these old distractor choices. |
| 2026-04-23_18-47-07_hard_neg_based_on_adjectives_tau005_fable_plus_summary_all_folds | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | mode=hard_neg; distractor_type=based_on_adjectives; tau=0.05 | 0.356028 | result JSON only | Yes | Diagnostic; cluster-aware masking may replace these old distractor choices. |
| 2026-04-23_20-30-35_hard_neg_injected_adjectives_tau005_fable_plus_summary_all_folds | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | mode=hard_neg; distractor_type=injected_adjectives; tau=0.05 | 0.431742 | result JSON only | Yes | Diagnostic; cluster-aware masking may replace these old distractor choices. |
| 2026-04-23_22-11-43_hard_neg_partial_story_tau005_fable_plus_summary_all_folds | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | mode=hard_neg; distractor_type=partial_story; tau=0.05 | 0.438032 | result JSON only | Yes | Diagnostic; cluster-aware masking may replace these old distractor choices. |

### FT04 STORAL Augmentation
| Run | Model | Training data | Eval doc mode | Config | Old MRR@10 | Artifact | Rerun required? | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-24_12-52-04_storal_augment_tau005_fable_plus_summary_fold0 | Linq-AI-Research/Linq-Embed-Mistral | MORABLES+STORAL | fable_plus_summary | Linq LoRA; MORABLES+STORAL; fold0 only; tau=0.05 | 0.420123 | fold0 result + embeddings only | Yes | Superseded by FT07 external transfer; keep as historical. |

### FT05 BGE STORAL Mixing Ratio
| Run/config | Model | Training data | Eval doc mode | Config | Old fold0 MRR@10 | Artifact | Rerun required? | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ratio_0 | BAAI/bge-base-en-v1.5 | MORABLES + 0 STORAL pairs | fable_plus_summary | fold0 only; epochs=15; batch=32; lr=2e-5; tau=0.05 | 0.277371 | fold0 model + embeddings local | Yes | If kept, rerun cluster-aware and across all folds. |
| ratio_200 | BAAI/bge-base-en-v1.5 | MORABLES + 200 STORAL pairs | fable_plus_summary | fold0 only; epochs=15; batch=32; lr=2e-5; tau=0.05 | 0.287210 | fold0 model + embeddings local | Yes | If kept, rerun cluster-aware and across all folds. |
| ratio_500 | BAAI/bge-base-en-v1.5 | MORABLES + 500 STORAL pairs | fable_plus_summary | fold0 only; epochs=15; batch=32; lr=2e-5; tau=0.05 | 0.277694 | fold0 model + embeddings local | Yes | If kept, rerun cluster-aware and across all folds. |
| ratio_1000 | BAAI/bge-base-en-v1.5 | MORABLES + 1000 STORAL pairs | fable_plus_summary | fold0 only; epochs=15; batch=32; lr=2e-5; tau=0.05 | 0.273979 | fold0 model + embeddings local | Yes | If kept, rerun cluster-aware and across all folds. |
| ratio_1675 | BAAI/bge-base-en-v1.5 | MORABLES + 1675 STORAL pairs | fable_plus_summary | fold0 only; epochs=15; batch=32; lr=2e-5; tau=0.05 | 0.273961 | fold0 model + embeddings local | Yes | If kept, rerun cluster-aware and across all folds. |

### FT06 Linq Random Search
| Local config | Model | Training data | Eval doc mode | Config | Old MRR@10 | Artifact | Rerun required? | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r32_t002_lr1e-4 | Linq-Embed-Mistral | MORABLES fold0 | fable_plus_summary | LoRA r=32; tau=0.02; lr=1e-4; epochs=10; batch=4; grad_accum=4 | not consolidated in docs | model + embeddings local for fold0 | Yes | Use to choose config only; official final needs cluster-aware CV. |
| r32_t007_lr1e-4 | Linq-Embed-Mistral | MORABLES fold0 | fable_plus_summary | LoRA r=32; tau=0.07; lr=1e-4; epochs=10; batch=4; grad_accum=4 | not consolidated in docs | model + embeddings local for fold0 | Yes | Use to choose config only; official final needs cluster-aware CV. |
| r64_t005_lr2e-4 | Linq-Embed-Mistral | MORABLES fold0 | fable_plus_summary | LoRA r=64; tau=0.05; lr=2e-4; epochs=10; batch=4; grad_accum=4 | not consolidated in docs | model + embeddings local for fold0 | Yes | Use to choose config only; official final needs cluster-aware CV. |

### FT07 STORAL Transfer
| Run | Model | Training data | n train | Train doc mode | Eval doc mode | Config | Old MRR@10 | Artifact | Clustered eval rerun? | Retrain required? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-29_09-46-38_bge_s500__fable_plus_summary | BGE-base-en-v1.5 | STORAL_s500 | 500 | STORAL_stories | fable_plus_summary | size=s500; tau=0.05; length_range=56-195 | 0.255038 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_09-46-38_bge_s500__raw | BGE-base-en-v1.5 | STORAL_s500 | 500 | STORAL_stories | raw | size=s500; tau=0.05; length_range=56-195 | 0.127279 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_09-49-00_bge_s1000__fable_plus_summary | BGE-base-en-v1.5 | STORAL_s1000 | 1000 | STORAL_stories | fable_plus_summary | size=s1000; tau=0.05; length_range=30-300 | 0.255840 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_09-49-00_bge_s1000__raw | BGE-base-en-v1.5 | STORAL_s1000 | 1000 | STORAL_stories | raw | size=s1000; tau=0.05; length_range=30-300 | 0.145982 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_09-52-50_bge_sfull__fable_plus_summary | BGE-base-en-v1.5 | STORAL_sfull | 1675 | STORAL_stories | fable_plus_summary | size=sfull; tau=0.05; length_range= | 0.260304 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_09-52-50_bge_sfull__raw | BGE-base-en-v1.5 | STORAL_sfull | 1675 | STORAL_stories | raw | size=sfull; tau=0.05; length_range= | 0.149021 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_10-13-37_linq_s500__fable_plus_summary | Linq-Embed-Mistral | STORAL_s500 | 500 | STORAL_stories | fable_plus_summary | size=s500; tau=0.05; length_range=56-195 | 0.450427 | full embeddings local | No | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_10-13-37_linq_s500__raw | Linq-Embed-Mistral | STORAL_s500 | 500 | STORAL_stories | raw | size=s500; tau=0.05; length_range=56-195 | 0.356577 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_10-48-14_linq_s1000__fable_plus_summary | Linq-Embed-Mistral | STORAL_s1000 | 1000 | STORAL_stories | fable_plus_summary | size=s1000; tau=0.05; length_range=30-300 | 0.421946 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_10-48-14_linq_s1000__raw | Linq-Embed-Mistral | STORAL_s1000 | 1000 | STORAL_stories | raw | size=s1000; tau=0.05; length_range=30-300 | 0.306790 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_11-50-27_linq_sfull__fable_plus_summary | Linq-Embed-Mistral | STORAL_sfull | 1675 | STORAL_stories | fable_plus_summary | size=sfull; tau=0.05; length_range= | 0.432894 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_11-50-27_linq_sfull__raw | Linq-Embed-Mistral | STORAL_sfull | 1675 | STORAL_stories | raw | size=sfull; tau=0.05; length_range= | 0.328084 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_12-28-46_qwen3_s500__fable_plus_summary | Qwen3-Embedding-8B | STORAL_s500 | 500 | STORAL_stories | fable_plus_summary | size=s500; tau=0.05; length_range=56-195 | 0.425404 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_12-28-46_qwen3_s500__raw | Qwen3-Embedding-8B | STORAL_s500 | 500 | STORAL_stories | raw | size=s500; tau=0.05; length_range=56-195 | 0.314263 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_13-04-37_qwen3_s1000__fable_plus_summary | Qwen3-Embedding-8B | STORAL_s1000 | 1000 | STORAL_stories | fable_plus_summary | size=s1000; tau=0.05; length_range=30-300 | 0.430220 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_13-04-37_qwen3_s1000__raw | Qwen3-Embedding-8B | STORAL_s1000 | 1000 | STORAL_stories | raw | size=s1000; tau=0.05; length_range=30-300 | 0.329233 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_14-05-26_qwen3_sfull__fable_plus_summary | Qwen3-Embedding-8B | STORAL_sfull | 1675 | STORAL_stories | fable_plus_summary | size=sfull; tau=0.05; length_range= | 0.435099 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |
| 2026-04-29_14-05-26_qwen3_sfull__raw | Qwen3-Embedding-8B | STORAL_sfull | 1675 | STORAL_stories | raw | size=sfull; tau=0.05; length_range= | 0.318032 | result JSON only locally | Eval only | No if weights can be pulled; otherwise selected retrain |

### FT08 Clean/Noisy Sanity
| Run | Model | Training data | Config | Old MRR@10 | Artifact | Rerun required? | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ft_08_clean_sanity | Linq-Embed-Mistral | MORABLES clean/noisy | clean 615 vs noisy 709; Linq LoRA; fable_plus_summary; 5 folds | - | config/checkpoints only; no completed result | Yes | Recast as cluster-aware split if still useful. |

### FT09 Linq False-Negative Masking
| Run | Model | Eval doc mode | Config | Old MRR@10 | Artifact | Retrain for official clustered CV? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-09_16-41-50_fn_masking | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | threshold=0.85 | 0.373496 | result JSON/fold files only; no local model weights | Yes | Most relevant MORABLES-CV idea; rerun official with cluster-aware folds and clustered qrels. |
| 2026-05-09_16-57-13_fn_masking | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | threshold=0.85 | 0.463657 | result JSON/fold files only; no local model weights | Yes | Most relevant MORABLES-CV idea; rerun official with cluster-aware folds and clustered qrels. |
| 2026-05-09_17-11-46_fn_masking | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | threshold=0.85 | 0.503975 | result JSON/fold files only; no local model weights | Yes | Most relevant MORABLES-CV idea; rerun official with cluster-aware folds and clustered qrels. |
| 2026-05-09_17-26-25_fn_masking | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | threshold=0.85 | 0.386409 | result JSON/fold files only; no local model weights | Yes | Most relevant MORABLES-CV idea; rerun official with cluster-aware folds and clustered qrels. |
| 2026-05-09_20-14-04_fn_masking | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | threshold=0.85 | 0.447747 | result JSON/fold files only; no local model weights | Yes | Most relevant MORABLES-CV idea; rerun official with cluster-aware folds and clustered qrels. |
| aggregate_from_fold_files | Linq-AI-Research/Linq-Embed-Mistral | fable_plus_summary | Derived from fold_0..fold_4 result files; threshold=0.85 | 0.435057 | result JSON/fold files only; no local model weights | Yes | Most relevant MORABLES-CV idea; rerun official with cluster-aware folds and clustered qrels. |

### FT10 IdioLink Transfer
Option A is zero-shot transfer after IdioLink fine-tuning. It is external training, so no MORABLES split leakage, but most configs do not have complete local model/ranking artifacts.
| Run/config | Model | Training data | Eval doc mode | Corpus config | Prompt variant | Old MRR@10 | Artifact | Rerun required? | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| option_a_original_results__BGE-M3__gemini/conceptual_abstract/fable+summary | BGE-M3 | IdioLink | fable_summary | gemini/conceptual_abstract/fable+summary | conceptual_abstract | 0.233100 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2636; delta=-0.0305; source column=original_ft_MRR@10 |
| option_a_original_results__BGE-M3__gemini/cot_proverb/fable+summary | BGE-M3 | IdioLink | fable_summary | gemini/cot_proverb/fable+summary | cot_proverb | 0.230900 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2574; delta=-0.0265; source column=original_ft_MRR@10 |
| option_a_original_results__BGE-M3__gemini/cot_proverb/summary_only | BGE-M3 | IdioLink | summary_only | gemini/cot_proverb/summary_only | cot_proverb | 0.231300 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2355; delta=-0.0042; source column=original_ft_MRR@10 |
| option_a_original_results__BGE-M3__gemini/cot_proverb/summary+fable | BGE-M3 | IdioLink | summary+fable | gemini/cot_proverb/summary+fable | cot_proverb | 0.254600 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2761; delta=-0.0215; source column=original_ft_MRR@10 |
| option_a_original_results__BGE-M3__gemini/direct_moral/fable+summary | BGE-M3 | IdioLink | fable_summary | gemini/direct_moral/fable+summary | direct_moral | 0.229000 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2494; delta=-0.0204; source column=original_ft_MRR@10 |
| option_a_original_results__BGE-M3__gemma4-31B/conceptual_abstract/fable+summary | BGE-M3 | IdioLink | fable_summary | gemma4-31B/conceptual_abstract/fable+summary | conceptual_abstract | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=; delta=; source column=original_ft_MRR@10 |
| option_a_original_results__BGE-M3__gemma4-31B/conceptual_abstract/summary_only | BGE-M3 | IdioLink | summary_only | gemma4-31B/conceptual_abstract/summary_only | conceptual_abstract | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=; delta=; source column=original_ft_MRR@10 |
| option_a_original_results__BGE-M3__gemma4-E2B/direct_moral/fable+summary | BGE-M3 | IdioLink | fable_summary | gemma4-E2B/direct_moral/fable+summary | direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=; delta=; source column=original_ft_MRR@10 |
| option_a_original_results__BGE-M3__gemma4-E2B/direct_moral/summary_only | BGE-M3 | IdioLink | summary_only | gemma4-E2B/direct_moral/summary_only | direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=; delta=; source column=original_ft_MRR@10 |
| option_a_original_results__BGE-M3__gemma4-E4B/thinking_direct_moral/fable+summary | BGE-M3 | IdioLink | fable_summary | gemma4-E4B/thinking_direct_moral/fable+summary | thinking_direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=; delta=; source column=original_ft_MRR@10 |
| option_a_original_results__BGE-M3__gemma4-E4B/thinking_direct_moral/summary_only | BGE-M3 | IdioLink | summary_only | gemma4-E4B/thinking_direct_moral/summary_only | thinking_direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=; delta=; source column=original_ft_MRR@10 |
| option_a_original_results__BGE-M3__raw | BGE-M3 | IdioLink | raw | raw | - | 0.080200 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.0890; delta=-0.0088; source column=original_ft_MRR@10 |
| option_a_original_results__Linq-Embed-Mistral__gemini/conceptual_abstract/fable+summary | Linq-Embed-Mistral | IdioLink | fable_summary | gemini/conceptual_abstract/fable+summary | conceptual_abstract | 0.358000 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.3481; delta=+0.0099; source column=original_ft_MRR@10 |
| option_a_original_results__Linq-Embed-Mistral__gemini/cot_proverb/fable+summary | Linq-Embed-Mistral | IdioLink | fable_summary | gemini/cot_proverb/fable+summary | cot_proverb | 0.367200 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.3597; delta=+0.0075; source column=original_ft_MRR@10 |
| option_a_original_results__Linq-Embed-Mistral__gemini/cot_proverb/summary_only | Linq-Embed-Mistral | IdioLink | summary_only | gemini/cot_proverb/summary_only | cot_proverb | 0.305600 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.3031; delta=+0.0025; source column=original_ft_MRR@10 |
| option_a_original_results__Linq-Embed-Mistral__gemini/cot_proverb/summary+fable | Linq-Embed-Mistral | IdioLink | summary+fable | gemini/cot_proverb/summary+fable | cot_proverb | 0.351700 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.3499; delta=+0.0018; source column=original_ft_MRR@10 |
| option_a_original_results__Linq-Embed-Mistral__gemini/direct_moral/fable+summary | Linq-Embed-Mistral | IdioLink | fable_summary | gemini/direct_moral/fable+summary | direct_moral | 0.354700 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.3502; delta=+0.0045; source column=original_ft_MRR@10 |
| option_a_original_results__Linq-Embed-Mistral__gemma4-31B/conceptual_abstract/fable+summary | Linq-Embed-Mistral | IdioLink | fable_summary | gemma4-31B/conceptual_abstract/fable+summary | conceptual_abstract | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.3015; delta=; source column=original_ft_MRR@10 |
| option_a_original_results__Linq-Embed-Mistral__gemma4-31B/conceptual_abstract/summary_only | Linq-Embed-Mistral | IdioLink | summary_only | gemma4-31B/conceptual_abstract/summary_only | conceptual_abstract | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2619; delta=; source column=original_ft_MRR@10 |
| option_a_original_results__Linq-Embed-Mistral__gemma4-E2B/direct_moral/fable+summary | Linq-Embed-Mistral | IdioLink | fable_summary | gemma4-E2B/direct_moral/fable+summary | direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2212; delta=; source column=original_ft_MRR@10 |
| option_a_original_results__Linq-Embed-Mistral__gemma4-E2B/direct_moral/summary_only | Linq-Embed-Mistral | IdioLink | summary_only | gemma4-E2B/direct_moral/summary_only | direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.1411; delta=; source column=original_ft_MRR@10 |
| option_a_original_results__Linq-Embed-Mistral__gemma4-E4B/thinking_direct_moral/fable+summary | Linq-Embed-Mistral | IdioLink | fable_summary | gemma4-E4B/thinking_direct_moral/fable+summary | thinking_direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2502; delta=; source column=original_ft_MRR@10 |
| option_a_original_results__Linq-Embed-Mistral__gemma4-E4B/thinking_direct_moral/summary_only | Linq-Embed-Mistral | IdioLink | summary_only | gemma4-E4B/thinking_direct_moral/summary_only | thinking_direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2306; delta=; source column=original_ft_MRR@10 |
| option_a_original_results__Linq-Embed-Mistral__raw | Linq-Embed-Mistral | IdioLink | raw | raw | - | 0.209300 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2107; delta=-0.0014; source column=original_ft_MRR@10 |
| option_a_results__BGE-M3__gemini/conceptual_abstract/fable+summary | BGE-M3 | IdioLink | fable_summary | gemini/conceptual_abstract/fable+summary | conceptual_abstract | 0.256000 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2636; delta=-0.0076; source column=idiolink_ft_MRR@10 |
| option_a_results__BGE-M3__gemini/cot_proverb/fable+summary | BGE-M3 | IdioLink | fable_summary | gemini/cot_proverb/fable+summary | cot_proverb | 0.252100 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2574; delta=-0.0053; source column=idiolink_ft_MRR@10 |
| option_a_results__BGE-M3__gemini/cot_proverb/summary_only | BGE-M3 | IdioLink | summary_only | gemini/cot_proverb/summary_only | cot_proverb | 0.232900 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2355; delta=-0.0026; source column=idiolink_ft_MRR@10 |
| option_a_results__BGE-M3__gemini/cot_proverb/summary+fable | BGE-M3 | IdioLink | summary+fable | gemini/cot_proverb/summary+fable | cot_proverb | 0.264000 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2761; delta=-0.0121; source column=idiolink_ft_MRR@10 |
| option_a_results__BGE-M3__gemini/direct_moral/fable+summary | BGE-M3 | IdioLink | fable_summary | gemini/direct_moral/fable+summary | direct_moral | 0.251200 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2494; delta=+0.0018; source column=idiolink_ft_MRR@10 |
| option_a_results__BGE-M3__gemma4-31B/conceptual_abstract/fable+summary | BGE-M3 | IdioLink | fable_summary | gemma4-31B/conceptual_abstract/fable+summary | conceptual_abstract | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=; delta=; source column=idiolink_ft_MRR@10 |
| option_a_results__BGE-M3__gemma4-31B/conceptual_abstract/summary_only | BGE-M3 | IdioLink | summary_only | gemma4-31B/conceptual_abstract/summary_only | conceptual_abstract | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=; delta=; source column=idiolink_ft_MRR@10 |
| option_a_results__BGE-M3__gemma4-E2B/direct_moral/fable+summary | BGE-M3 | IdioLink | fable_summary | gemma4-E2B/direct_moral/fable+summary | direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=; delta=; source column=idiolink_ft_MRR@10 |
| option_a_results__BGE-M3__gemma4-E2B/direct_moral/summary_only | BGE-M3 | IdioLink | summary_only | gemma4-E2B/direct_moral/summary_only | direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=; delta=; source column=idiolink_ft_MRR@10 |
| option_a_results__BGE-M3__gemma4-E4B/thinking_direct_moral/fable+summary | BGE-M3 | IdioLink | fable_summary | gemma4-E4B/thinking_direct_moral/fable+summary | thinking_direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=; delta=; source column=idiolink_ft_MRR@10 |
| option_a_results__BGE-M3__gemma4-E4B/thinking_direct_moral/summary_only | BGE-M3 | IdioLink | summary_only | gemma4-E4B/thinking_direct_moral/summary_only | thinking_direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=; delta=; source column=idiolink_ft_MRR@10 |
| option_a_results__BGE-M3__raw | BGE-M3 | IdioLink | raw | raw | - | 0.089400 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.0890; delta=+0.0004; source column=idiolink_ft_MRR@10 |
| option_a_results__Linq-Embed-Mistral__gemini/conceptual_abstract/fable+summary | Linq-Embed-Mistral | IdioLink | fable_summary | gemini/conceptual_abstract/fable+summary | conceptual_abstract | 0.362900 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.3481; delta=+0.0148; source column=idiolink_ft_MRR@10 |
| option_a_results__Linq-Embed-Mistral__gemini/cot_proverb/fable+summary | Linq-Embed-Mistral | IdioLink | fable_summary | gemini/cot_proverb/fable+summary | cot_proverb | 0.371400 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.3597; delta=+0.0117; source column=idiolink_ft_MRR@10 |
| option_a_results__Linq-Embed-Mistral__gemini/cot_proverb/summary_only | Linq-Embed-Mistral | IdioLink | summary_only | gemini/cot_proverb/summary_only | cot_proverb | 0.294200 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.3031; delta=-0.0089; source column=idiolink_ft_MRR@10 |
| option_a_results__Linq-Embed-Mistral__gemini/cot_proverb/summary+fable | Linq-Embed-Mistral | IdioLink | summary+fable | gemini/cot_proverb/summary+fable | cot_proverb | 0.363800 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.3499; delta=+0.0139; source column=idiolink_ft_MRR@10 |
| option_a_results__Linq-Embed-Mistral__gemini/direct_moral/fable+summary | Linq-Embed-Mistral | IdioLink | fable_summary | gemini/direct_moral/fable+summary | direct_moral | 0.364000 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.3502; delta=+0.0138; source column=idiolink_ft_MRR@10 |
| option_a_results__Linq-Embed-Mistral__gemma4-31B/conceptual_abstract/fable+summary | Linq-Embed-Mistral | IdioLink | fable_summary | gemma4-31B/conceptual_abstract/fable+summary | conceptual_abstract | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.3015; delta=; source column=idiolink_ft_MRR@10 |
| option_a_results__Linq-Embed-Mistral__gemma4-31B/conceptual_abstract/summary_only | Linq-Embed-Mistral | IdioLink | summary_only | gemma4-31B/conceptual_abstract/summary_only | conceptual_abstract | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2619; delta=; source column=idiolink_ft_MRR@10 |
| option_a_results__Linq-Embed-Mistral__gemma4-E2B/direct_moral/fable+summary | Linq-Embed-Mistral | IdioLink | fable_summary | gemma4-E2B/direct_moral/fable+summary | direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2212; delta=; source column=idiolink_ft_MRR@10 |
| option_a_results__Linq-Embed-Mistral__gemma4-E2B/direct_moral/summary_only | Linq-Embed-Mistral | IdioLink | summary_only | gemma4-E2B/direct_moral/summary_only | direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.1411; delta=; source column=idiolink_ft_MRR@10 |
| option_a_results__Linq-Embed-Mistral__gemma4-E4B/thinking_direct_moral/fable+summary | Linq-Embed-Mistral | IdioLink | fable_summary | gemma4-E4B/thinking_direct_moral/fable+summary | thinking_direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2502; delta=; source column=idiolink_ft_MRR@10 |
| option_a_results__Linq-Embed-Mistral__gemma4-E4B/thinking_direct_moral/summary_only | Linq-Embed-Mistral | IdioLink | summary_only | gemma4-E4B/thinking_direct_moral/summary_only | thinking_direct_moral | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2306; delta=; source column=idiolink_ft_MRR@10 |
| option_a_results__Linq-Embed-Mistral__raw | Linq-Embed-Mistral | IdioLink | raw | raw | - | 0.215900 | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | zero_shot_MRR=0.2107; delta=+0.0052; source column=idiolink_ft_MRR@10 |
| skipped_192_na_placeholder_rows | - | IdioLink | - | - | - | - | model/ranking artifact incomplete; BGE-M3 local model exists, Linq rows are CSV only | Eval only / Yes | The option_a CSVs contain 192 rows where fine-tuned metrics are n/a and only duplicate zero-shot baselines; those pla... |

| Option | Run | Training data | Eval doc mode | Status | Rerun required? | Note |
| --- | --- | --- | --- | --- | --- | --- |
| IdioLink sequential transfer | option_b_sequential | IdioLink then MORABLES | various | not started | Yes | README/scripts exist; no result files found. |
| IdioLink mixing transfer | option_c_mixing | IdioLink+MORABLES mixed | various | not started | Yes | README/scripts exist; no result files found. |
| IdioLink hard-negative transfer | option_d_hard_neg | IdioLink hard negatives | various | not started | Yes | README/scripts exist; no result files found. |

### Synthetic Data Fine-Tuning
| Run | Model candidates | Training data | Eval doc mode | Config | Status | Rerun required? | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_fables_ft | Linq-Embed-Mistral / BGE-large | Synthetic fables from MORABLES morals | raw/fable_plus_summary | planned: generate synthetic fables per MORABLES moral; train on synthetic; evaluate real fables | not done | Yes | Run after clustered data is final; consider Linq and Qwen3-Embedding-8B as primary candidates. |

## Final Clustered Run Plan After Data Freeze

| Step | What to run | Why | Inputs needed |
| --- | --- | --- | --- |
| 1 | Rescore all Exp02 full-ranking predictions. | Fast zero-shot baseline across many raw-fable embedding models. | Existing `results/runs/2026-03-15_combined_v2/predictions/*.json`. |
| 2 | Reconstruct/rescore Exp11 cached embeddings for BGE-en-ICL, Nomic-v2-MoE, Qwen3-Embedding-8B. | Avoid re-encoding expensive models where full embeddings already exist. | Existing `experiments/11_embedding_baselines/cache/embeddings/*`. |
| 3 | Rerun Exp12 shortlisted full scoring: Linq, Qwen3-8B, BGE-en-ICL, Nomic-v2-MoE; all 6 corpus configs x default/generic first. | This is the main zero-shot summary/fable matrix. | Clustered qrels and summary fields. |
| 4 | Rerun/pull Exp13 summary artifacts, then clustered eval for Linq and Qwen3-8B. | Keeps Gemini/Gemma generated-summary comparisons fair. | Generated summaries for Gemini/Gemma prompt variants. |
| 5 | Re-evaluate FT07 external-transfer models; prioritize Linq s500 and Qwen3 sfull/s500. | External transfer avoids MORABLES split leakage. | Model weights or cached embeddings. |
| 6 | Retrain official MORABLES-supervised CV with cluster-aware folds. | Old FT02/FT03/FT09 folds are not official under clustered labels. | Final clustered data; choose Linq and maybe Qwen3-8B. |
| 7 | Run synthetic-data fine-tuning. | New contribution candidate, not yet done. | Synthetic generation prompt and budget approval. |

