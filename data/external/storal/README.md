# STORAL — External Dataset

**Source**: "A Corpus for Understanding and Generating Moral Stories" (Guan et al., NAACL 2022)
**Paper**: https://aclanthology.org/2022.naacl-main.374/
**GitHub**: https://github.com/thu-coai/MoralStory

## What it is

Bilingual (Chinese + English) dataset pairing short stories with explicit moral sentences.
Four tasks: moCpt, moPref, ST2MO (story→moral), MO2ST (moral→story).

We use it as **training data augmentation** for moral-to-fable retrieval:
each (story, moral) pair becomes an additional (moral, fable) training triplet.

## Download

```bash
git clone https://github.com/thu-coai/MoralStory /tmp/storal_repo
cp -r /tmp/storal_repo/data/ data/external/storal/raw/
```

## Files

```
raw/      # original STORAL files (downloaded, gitignored)
processed/
  storal_pairs.json   # converted to our (moral, story) format
                      # created by finetuning/ft_04_storal_augment/preprocess_storal.py
```
