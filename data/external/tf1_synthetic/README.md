# TF1-EN-3M synthetic — MORABLES-shaped derivative

Source: https://huggingface.co/datasets/klusai/ds-tf1-en-3m (MIT)
Paper: https://arxiv.org/abs/2504.20605

Built from experiments/11_tf1_diagnostic/results/runs/20260502_184314/samples.jsonl
via experiments/11_tf1_diagnostic/check_iou.py (--n 50000 --chunks 10).

## Build commands

./run.sh experiments/11_tf1_diagnostic/build_tf1_corpus.py --n <N> --seed <S>
./run.sh experiments/11_tf1_diagnostic/cluster_tf1_morals.py --threshold <T>

## Snapshot (this build)

- N per moral: 10
- Total morals: 100
- Total fables: 1000
- Seed: 42
- Built: 2026-05-20T00:12:33

See experiments/11_tf1_diagnostic/REPORT.md for the analysis that motivated this derivative.
