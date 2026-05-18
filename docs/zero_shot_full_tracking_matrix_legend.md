# Zero-Shot Full Tracking Matrix Legend

This file explains `zero_shot_full_tracking_matrix.csv`. The CSV is a blank tracking grid for zero-shot retrieval experiments. It intentionally contains every planned model/config combination, not only experiments already run.

## Scope

- One row = one retrieval model + instruction variant + corpus configuration.
- Metric columns are intentionally empty.
- `exp_id` is intentionally empty for now. Fill it later when a row is matched to an existing run or rerun.
- Rows are grouped by `model_alias` so all rows for the same model stay together.
- CSV templates use escaped `\n` text, not physical line breaks inside cells.

## Instructions

| instruction_variant | Exact meaning |
|---|---|
| `no_instr` | Empty query instruction. Query text is embedded as-is. |
| `general` | `Instruct: Given a text, retrieve the most relevant passage that answers the query\nQuery: ` |
| `moral_specific` | `Instruct: Given a moral statement, retrieve the fable that best conveys this moral.\nQuery: ` |
| `default` | Added only for model families with a materially different native/default prefix, prompt name, or wrapper. See the table below. |

## Default Instruction Families

| model_alias | default behavior |
|---|---|
| `BGE-en-ICL` | `Given a moral statement, retrieve the fable that best conveys this moral.\n` |
| `BGE-large-en-v1.5`, `BGE-base-en-v1.5` | Query prefix: `Represent this sentence for searching relevant passages: ` |
| `E5-large-v2`, `E5-base-v2`, `Multilingual-E5-large` | Query prefix: `query: `; document prefix: `passage: ` |
| `Instructor-xl`, `Instructor-base` | Query pair prompt: `Represent the moral statement for retrieving relevant fables: ` |
| `Nomic-Embed-v2-MoE` | Query prefix: `search_query: `; document prefix: `search_document: ` |
| `Stella-1.5B` | SentenceTransformers query prompt name: `s2p_query` |
| `DRAMA-1B` | SentenceTransformers query prompt name: `query` |
| `TART-dual-contriever` | Query format: `Retrieve a fable that illustrates the following moral [SEP] {query}` |
| `GritLM-7B` | Query wrapper: `<|user|>\nGiven a moral statement, retrieve the fable that best conveys this moral.\n<|embed|>\n` |

## Summary Generators

| summary_generator | summary_generator_model_id |
|---|---|
| `gemini` | `gemini-3-flash-preview` |
| `gemma4-E2B` | `google/gemma-4-E2B-it` |
| `gemma4-E4B` | `google/gemma-4-E4B-it` |
| `gemma4-26B-A4B` | `google/gemma-4-26B-A4B-it` |
| `gemma4-31B` | `google/gemma-4-31B-it` |

## Generator Prompt Variants

| generator_prompt_variant | Prompt intent |
|---|---|
| `direct_moral` | Extract the fable's moral lesson as a single concise sentence. |
| `cot_proverb` | Reason step by step, then output only the final proverb/maxim. |
| `conceptual_abstract` | Reason step by step, then output only the abstract moral principle as one sentence. |
| `proverb` | State the moral as a concise proverb or maxim. |

## Corpus Configurations

| corpus_config | corpus_template | Meaning |
|---|---|---|
| `raw` | `{fable}` | Raw fable text only; no summary generator. |
| `fable_direct_moral` | `{fable}\n\nMoral summary: {direct_moral}` | Fable first, then generated direct moral summary. |
| `direct_moral_only` | `{direct_moral}` | Generated direct moral summary only. |
| `direct_moral_fable` | `Moral summary: {direct_moral}\n\n{fable}` | Generated direct moral summary first, then fable. |
| `fable_cot_proverb` | `{fable}\n\nMoral summary: {cot_proverb}` | Fable first, then generated CoT proverb summary. |
| `cot_proverb_only` | `{cot_proverb}` | Generated CoT proverb summary only. |
| `cot_proverb_fable` | `Moral summary: {cot_proverb}\n\n{fable}` | Generated CoT proverb summary first, then fable. |
| `fable_conceptual_abstract` | `{fable}\n\nMoral summary: {conceptual_abstract}` | Fable first, then generated conceptual abstract summary. |
| `conceptual_abstract_only` | `{conceptual_abstract}` | Generated conceptual abstract summary only. |
| `conceptual_abstract_fable` | `Moral summary: {conceptual_abstract}\n\n{fable}` | Generated conceptual abstract summary first, then fable. |
| `fable_proverb` | `{fable}\n\nMoral summary: {proverb}` | Fable first, then generated proverb summary. |
| `proverb_only` | `{proverb}` | Generated proverb summary only. |
| `proverb_fable` | `Moral summary: {proverb}\n\n{fable}` | Generated proverb summary first, then fable. |

## Metrics To Fill Later

The CSV includes these empty result fields: `MAP@10`, `MRR@10`, `NDCG@10`, `Recall@5`, `Recall@10`, `Recall@15`, `Recall@50`, `Recall@100`, `Recall@200`, `Recall@300`, `Hit@1`, `Hit@5`, `Hit@10`, `Hit@100`.

When reconciling existing artifacts, fill only rows whose full config matches: model, instruction, corpus template, summary generator, and generator prompt variant.
