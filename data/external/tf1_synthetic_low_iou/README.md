# TF1-EN-3M synthetic — MORABLES-shaped derivative

Source: https://huggingface.co/datasets/klusai/ds-tf1-en-3m (MIT)
Paper: https://arxiv.org/abs/2504.20605

Built from experiments/11_tf1_diagnostic/results/runs/20260520_173450/samples.jsonl
via experiments/11_tf1_diagnostic/check_iou.py (--n 50000 --chunks 10).

## Snapshot (this build)

- Selection strategy: low_iou_clean
- N per moral: 20
- Total morals: 100
- Total fables: 2000
- Built: 2026-05-20T20:03:17

## Build commands

./run.sh experiments/11_tf1_diagnostic/build_tf1_corpus.py --selection low_iou_clean --n 20
./run.sh experiments/11_tf1_diagnostic/cluster_tf1_morals.py --mode exact --in /Users/roeirahamim/Documents/MSC/Thesis-Intellexus/work_morables/data/external/tf1_synthetic_low_iou

See experiments/11_tf1_diagnostic/REPORT.md for the analysis that motivated this derivative.


## Clustering — mode=exact (this run)

- Mode: exact
- Threshold: n/a (trivial 1-cluster-per-moral)
- Model: n/a (no embedding required)
- Clusters: 100  (singleton=0, near=0, exact=100)
- Inspection dumps: n/a (skipped in exact mode)
