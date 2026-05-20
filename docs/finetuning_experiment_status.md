# Fine-Tuning Experiment Status

Generated: 2026-05-19

Last updated: 2026-05-19, after checking the active GPU queues and appending follow-up work.

Status meanings:
- `Done`: result files exist for the stated scope.
- `In progress`: currently running on the GPU server.
- `Queued`: included in the current queue, but not reached yet.
- `Partial`: some sizes/configs completed, but the planned scope is not complete.
- `Not started`: no current clustered result for that scope.

## FT11: Clustered MORABLES 5-Fold CV

FT11 trains on clustered MORABLES folds. For each trained fold model, saved-model eval can later evaluate the same trained model on extra document layouts. Current active queue on GPU 0 is training `raw/none` for remaining encoder models, then evaluating those raw-trained models on all Gemini document configs.

| Model | Training config | Saved-model eval config | Status | Notes |
|---|---|---|---|---|
| `sfr` | `raw/none`, 5 folds | Not queued in current saved-eval queue | Done | Also has an old `direct_moral_only/gemini` fold-0 sanity result. |
| `linq` | `raw/none`, 5 folds | Not queued in current saved-eval queue | Done | Strong baseline already available. |
| `qwen3` | `raw/none`, 5 folds | Not queued in current saved-eval queue | Done | Qwen3-Embedding-8B raw 5FCV completed. |
| `qwen3_4b` | `raw/none`, 5 folds | `all` doc configs with `gemini` | Queued | Added to GPU 0 follow-up queue after the current FT11 encoder run. |
| `qwen3_0_6b` | `raw/none`, 5 folds | `all` doc configs with `gemini` | Queued | Added to GPU 0 follow-up queue after the current FT11 encoder run. |
| `bge` | `raw/none`, 5 folds | `all` doc configs with `gemini` | Done / Queued eval | Raw 5FCV completed; saved-model eval will run after GPU 0 training queue. |
| `bge_large` | `raw/none`, 5 folds | `all` doc configs with `gemini` | Done / Queued eval | Raw 5FCV completed since the previous update; saved-model eval will run after GPU 0 training queue. |
| `bge_m3` | `raw/none`, 5 folds | `all` doc configs with `gemini` | In progress | Currently running on GPU 0; latest observed state was fold 3/5. |
| `e5_base` | `raw/none`, 5 folds | `all` doc configs with `gemini` | In progress | Current GPU 0 queue; latest observed active model. |
| `e5_large` | `raw/none`, 5 folds | `all` doc configs with `gemini` | Queued | Current GPU 0 queue. |
| `multilingual_e5_large` | `raw/none`, 5 folds | `all` doc configs with `gemini` | Queued | Current GPU 0 queue. |
| `multilingual_e5_large_instruct` | `raw/none`, 5 folds | `all` doc configs with `gemini` | Queued | Current GPU 0 queue. |
| `all_mpnet` | `raw/none`, 5 folds | `all` doc configs with `gemini` | Queued | Current GPU 0 queue. |
| `all_minilm` | `raw/none`, 5 folds | `all` doc configs with `gemini` | Queued | Current GPU 0 queue. |
| `contriever` | `raw/none`, 5 folds | `all` doc configs with `gemini` | Queued | Current GPU 0 queue. |
| `embeddinggemma` | `raw/none`, 5 folds | `all` doc configs with `gemini` | Queued | Current GPU 0 queue; may need attention if loader/auth issues appear. |
| `stella` | `raw/none`, 5 folds | `all` doc configs with `gemini` | Queued | Current GPU 0 queue; historically more fragile. |

## FT12: STORAL Transfer on Clustered MORABLES Eval

FT12 trains on STORAL moral-story pairs and evaluates on clustered MORABLES. Standard sizes are `s200`, `s500`, `s1000`, `sfull`. Existing broad evals cover raw/Gemini/Gemma summary layouts for many models; the current GPU 3 queue is specifically completing all Qwen STORAL models with all sizes and all Gemini layouts.

| Model | Training sizes | Eval configs covered/planned | Status | Notes |
|---|---|---|---|---|
| `linq` | `s200`, `s500`, `s1000`, `sfull` | Raw, Gemini layouts, Gemma `31B/E4B/E2B` layouts | Done | Best current STORAL family so far. |
| `qwen3` | `s200`, `s500`, `s1000` done; `sfull` running | All 13 Gemini layouts planned; Gemma `31B/E4B/E2B` eval queued after Qwen family finishes | In progress | GPU 3 currently running `qwen3 sfull`; `qwen3 s1000` completed since last update. |
| `qwen3_4b` | `s200`, `s500`, `s1000`, `sfull` | All 13 Gemini layouts planned | Queued | Runs after `qwen3`. |
| `qwen3_0_6b` | `s200`, `s500`, `s1000`, `sfull` | All 13 Gemini layouts planned | Queued | Runs after `qwen3_4b`. |
| `sfr` | `s200`, `s500`, `s1000`, `sfull` | All 13 Gemini layouts planned | Queued | Added to GPU 3 follow-up queue after Qwen Gemma evals. |
| `bge` | `s200`, `s500`, `s1000`, `sfull` | Raw plus Gemma layouts; some Gemini evals | Done | BGE-base STORAL completed. |
| `bge_large` | `s200`, `s500`, `s1000`, `sfull` | Raw, Gemini top layouts, Gemma layouts | Done | Broad eval coverage exists. |
| `bge_m3` | `s200`, `s500`, `s1000`, `sfull` | Raw, Gemini top layouts, Gemma layouts | Done | Broad eval coverage exists. |
| `e5_base` | `s200`, `s500`, `s1000`, `sfull` | Raw, Gemini top layouts, Gemma layouts | Done | Broad eval coverage exists. |
| `e5_large` | `s200`, `s500`, `s1000`; `sfull` missing/errored | Raw, Gemini top layouts, Gemma layouts for completed sizes | Partial | `sfull` checkpoint/eval missing. |
| `multilingual_e5_large` | `s200`, `s500`, `s1000`, `sfull` | Raw, Gemini top layouts, Gemma layouts | Done | Broad eval coverage exists. |
| `multilingual_e5_large_instruct` | `s200`, `s500`, `s1000`, `sfull` | Raw, Gemini top layouts, Gemma layouts | Done | Broad eval coverage exists. |
| `all_mpnet` | `s200`, `s500`, `s1000`; `sfull` missing/errored | Raw, Gemini top layouts, Gemma layouts for completed sizes | Partial | `sfull` checkpoint/eval missing. |
| `all_minilm` | `s200`, `s500`, `s1000`, `sfull` | Raw, Gemini layouts, Gemma layouts | Done | Full small-model STORAL coverage. |
| `contriever` | `s200`, `s500`, `s1000`, `sfull` | Raw, Gemini top layouts, Gemma layouts | Done | Broad eval coverage exists. |
| `embeddinggemma` | Intended `s200`, `s500`, `s1000`, `sfull` | Prior skip-train evals failed due missing checkpoints | Not started / missing checkpoints | Needs actual STORAL training if desired. |
| `stella` | Intended `s200`, `s500`, `s1000`, `sfull` | Prior skip-train evals failed due missing checkpoints | Not started / missing checkpoints | Needs actual STORAL training if desired. |

## Immediate Work Still Running

| GPU | Session | Work |
|---|---|---|
| GPU 0 | `parabelink_gpu0_ft11_more` | FT11 raw 5FCV for remaining encoder models, then saved-model eval on all Gemini layouts. |
| GPU 3 | `parabelink_gpu3_ft12_qwen` | FT12 STORAL training for `qwen3`, `qwen3_4b`, `qwen3_0_6b` across all sizes, evaluating all Gemini layouts. |
| GPU 0 | `parabelink_gpu0_ft11_followup` | Waits for GPU 0 current queue, then evaluates saved `sfr`/`linq`/`qwen3` raw models on all Gemini layouts, trains `qwen3_0_6b` and `qwen3_4b` raw 5FCV, then evaluates them on all Gemini layouts. |
| GPU 3 | `parabelink_gpu3_ft12_followup` | Waits for GPU 3 Qwen STORAL queue, then evaluates Qwen STORAL checkpoints with Gemma `31B/E4B/E2B`, then trains SFR STORAL all sizes with all Gemini layouts. |

Latest observed active states:
- GPU 0: running `ft_11_clustered`, model `e5_base`, `raw/none`.
- GPU 3: running `ft_12_storal_transfer_clustered`, model `qwen3`, size `sfull`.
- Recent completed FT11 results include `bge_large raw/none all_folds`; `bge_m3` has advanced past the previous status.
- Recent completed FT12 results include `qwen3 s1000` with all Gemini eval layouts.

## Main Gaps Versus Zero-Shot

Zero-shot has 31 embedding models, 13 corpus layouts, several instruction variants, and multiple generators. The current FT coverage is intentionally narrower.

Remaining meaningful FT gaps:
- FT11 5FCV: smaller Qwen variants are not started; many encoder models are queued but not done yet.
- FT11 eval: saved-model eval on all layouts is only queued for the current encoder batch, not yet for `sfr`, `linq`, or `qwen3`.
- FT12 STORAL: Qwen family is running now; `sfr`, `embeddinggemma`, and `stella` still need real STORAL training if we want them.
- FT12 STORAL: `e5_large sfull` and `all_mpnet sfull` are partial/missing.
- FT experiments do not currently sweep zero-shot instruction variants; they use the model-specific training instruction from config.
