# ft_04 — STORAL-Augmented Fine-Tuning

Training data augmentation using STORAL (Guan et al., NAACL 2022) to address
the core bottleneck in ft_03: only 570 training pairs per fold causes the model
to peak at epoch 2 and overfit quickly.

## Hypothesis

Adding STORAL's English (moral, story) pairs alongside MORABLES pairs gives the
model more signal per epoch, allowing longer training without overfitting and
producing a better embedding space.

## Steps

1. **Explore** — `explore_storal.py`: understand STORAL format, compatibility,
   story lengths vs. MORABLES fables
2. **Preprocess** — `preprocess_storal.py`: convert STORAL → `data/external/storal/processed/storal_pairs.json`
3. **Train** — `train.py`: combines STORAL pairs + MORABLES pairs, uses same
   InfoNCE loss as ft_03

## How to Run

```bash
# 1. Download STORAL (one-time)
git clone https://github.com/thu-coai/MoralStory /tmp/storal_repo
cp -r /tmp/storal_repo/data/ data/external/storal/raw/

# 2. Explore the data
./run.sh finetuning/ft_04_storal_augment/explore_storal.py

# 3. Preprocess
./run.sh finetuning/ft_04_storal_augment/preprocess_storal.py

# 4. Train (all folds)
./run.sh finetuning/ft_04_storal_augment/train.py --remote --gpu 3
```

## Key Questions to Answer First (explore_storal.py)

- How many English (moral, story) pairs are there?
- How long are STORAL stories vs. MORABLES fables?
- Are the morals similar in style/abstraction level?
- Any overlap with MORABLES content?
