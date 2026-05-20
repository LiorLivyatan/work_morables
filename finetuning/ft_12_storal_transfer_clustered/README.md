# ft_12 - STORAL Transfer on Clustered MORABLES

This experiment fine-tunes embedding models on STORAL and evaluates transfer to
the clustered MORABLES moral-to-fable retrieval benchmark.

## Data

- Training source: `data/external/storal/processed/storal_pairs.json`
- Training pairs: clean STORAL rows where `is_duplicate != true`
- Training text: STORAL moral query -> raw STORAL story
- Target evaluation:
  - queries: `data/clustered/morals_unique_corpus.json`
  - corpus: `data/clustered/fables_corpus.json`
  - qrels: `data/clustered/qrels_moral_to_fable_clustered.json`

STORAL rows are not deleted for repeated morals. Exact normalized moral-text
matches receive the same label, so their stories are masked as false negatives
inside InfoNCE when they land in the same batch.

## Evaluation

Every trained model is evaluated against clustered MORABLES with all 709 fables
in the index. The script saves:

- metrics JSON per model/size
- full ranked fable IDs per clustered moral query
- saved model weights under `/data/lior/ft_12_storal_transfer_clustered/models`

Early stopping uses a held-out STORAL validation split, not MORABLES, so target
benchmark metrics are not used to choose the epoch.

## Commands

Smoke test:

```bash
./run.sh finetuning/ft_12_storal_transfer_clustered/train.py --model bge --size s200 --eval_doc_configs raw --no-wandb
```

Remote run:

```bash
./run.sh finetuning/ft_12_storal_transfer_clustered/train.py --models linq qwen3 sfr --sizes s200 s500 s1000 sfull --continue_on_error --remote --gpu 0
```

Evaluate one saved model across the default MORABLES document configs:

```bash
./run.sh finetuning/ft_12_storal_transfer_clustered/train.py --model linq --size s500 --skip_train --remote --gpu 0
```
