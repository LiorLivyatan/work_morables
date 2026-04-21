"""
Generate 5-fold CV splits for ft_01_5fold_cv.

Splits are saved to cache/splits/folds.json and committed to git — this ensures
every team member uses identical folds, making results directly comparable.

Run once, then commit the output:
    python finetuning/ft_01_5fold_cv/prepare_data.py
    git add finetuning/ft_01_5fold_cv/cache/splits/folds.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from lib.data import load_moral_to_fable_retrieval_data

SPLITS_PATH = EXP_DIR / "cache" / "splits" / "folds.json"
N_SPLITS = 5


def generate_folds() -> list[dict]:
    _, moral_texts, _ = load_moral_to_fable_retrieval_data()
    # GroupKFold ensures all instances of identical moral text land in the same
    # fold — prevents a repeated moral from appearing in both train and test,
    # which would let the model "recognise" a test query it saw during training.
    groups = np.array([hash(t) for t in moral_texts])
    gkf = GroupKFold(n_splits=N_SPLITS)
    return [
        {"fold": i, "train": train.tolist(), "test": test.tolist()}
        for i, (train, test) in enumerate(gkf.split(range(len(moral_texts)), groups=groups))
    ]


if __name__ == "__main__":
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    folds = generate_folds()
    with open(SPLITS_PATH, "w") as f:
        json.dump(folds, f, indent=2)
    print(f"Saved {len(folds)} folds → {SPLITS_PATH}")
    for fold in folds:
        print(f"  Fold {fold['fold']}: {len(fold['train'])} train, {len(fold['test'])} test")
