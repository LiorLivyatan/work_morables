# ft_11_clustered

Fine-tuning on the clustered MORABLES benchmark.

This experiment uses:

- Queries: unique moral queries from `data/clustered/morals_unique_corpus.json`
- Corpus: all 709 fables from `data/clustered/fables_corpus.json`
- Labels: multi-positive qrels from `data/clustered/qrels_moral_to_fable_clustered.json`
- Splits: 5 folds over moral query clusters, generated once into `data/folds.json`

The loss is the shared `InfoNCELoss`, with clustered labels passed to the dataset.
In-batch fables and moral anchors from the same moral cluster are masked out of the
negative denominator, so they are not treated as false negatives.

The evaluator saves both metrics and full 709-fable rankings for every test query.
Those rankings are enough to recalculate future metrics without rerunning the model.

## First run

```bash
./run.sh finetuning/ft_11_clustered/prepare_data.py
./run.sh finetuning/ft_11_clustered/train.py --model sfr --doc_config direct_moral_only --summary_generator gemini --fold 0 --remote --gpu 0
```

## Remaining Saved-Model Evals

The raw 5FCV models can be evaluated later on the remaining non-Gemini summary
layouts with the preset wrapper:

```bash
./run.sh finetuning/ft_11_clustered/eval_remaining_non_gemini.py --remote --gpu 0
```

This evaluates the 15 working models, excluding `embeddinggemma` and `stella`,
against all 12 non-raw summary layouts and the three Gemma 4 summary generators:
`gemma4-E2B`, `gemma4-E4B`, and `gemma4-31B`.

Totals: 540 aggregate configs, 2700 fold-level evals. The wrapper delegates to
`eval_saved_models.py`, keeps `--continue_on_error` enabled by default, and saves
the same metrics JSON plus full 709-fable rankings per test query.

Preview without evaluating:

```bash
./run.sh finetuning/ft_11_clustered/eval_remaining_non_gemini.py --dry_run
```
