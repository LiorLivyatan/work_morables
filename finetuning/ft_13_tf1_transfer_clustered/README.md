# ft_13_tf1_transfer_clustered

TF1-synthetic transfer experiment evaluated on clustered MORABLES.

## What it does

Fine-tunes embedding models on TF1-synthetic moral->fable pairs from the
low-IoU+clean corpus at `data/external/tf1_synthetic_low_iou/`. Trains
with `InfoNCELoss` + multi-positive masking (label = moral_id, exact
clusters only). Evaluates each trained model on clustered MORABLES
(`data/clustered/`) across multiple doc_configs, persists full per-query
rankings, and appends a comprehensive CSV.

This is the TF1 analog of ft_12_storal_transfer_clustered. The two
experiments share the same eval pipeline, ranking format, and CSV
schema (modulo three TF1-specific columns) so cross-experiment
analysis is straightforward.

## Sizes

- s200: 20 morals x 10 fables = 200 rows
- s500: 50 morals x 10 fables = 500 rows
- sfull: 100 morals x 10 fables = 1000 rows
- sfull_n20: 100 morals x 20 fables = 2000 rows (isolates per-moral redundancy)

## Reproduction

Smoke test:
    ./run.sh finetuning/ft_13_tf1_transfer_clustered/train.py \
        --model bge --size s200 --eval_doc_configs raw

Wave 1:
    ./run.sh finetuning/ft_13_tf1_transfer_clustered/train.py \
        --models bge all_minilm all_mpnet qwen3_0_6b qwen3_4b linq \
        --sizes s200 s500 sfull sfull_n20 \
        --eval_doc_configs raw fable_cot_proverb fable_direct_moral conceptual_abstract_fable \
        --summary_generator gemini --remote --gpu 2

## Outputs

- Per-run JSON:    results/<timestamp>_<model>_<size>.json
- Rankings:        results/rankings/<model>__<size>__<doc_config>__<generator>.json
- Comprehensive:   results/ft13_comprehensive_results.csv
- Trained models:  /data/lior/ft_13_tf1_transfer_clustered/models/<model>/<size>
