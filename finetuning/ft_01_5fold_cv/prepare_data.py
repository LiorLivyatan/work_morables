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

from sklearn.model_selection import KFold

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from lib.data import load_moral_to_fable_retrieval_data

SPLITS_PATH = EXP_DIR / "cache" / "splits" / "folds.json"
N_SPLITS = 5
SEED = 42


def generate_folds() -> list[dict]:
    _, moral_texts, _ = load_moral_to_fable_retrieval_data()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return [
        {"fold": i, "train": train.tolist(), "test": test.tolist()}
        for i, (train, test) in enumerate(kf.split(range(len(moral_texts))))
    ]


if __name__ == "__main__":
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    folds = generate_folds()
    with open(SPLITS_PATH, "w") as f:
        json.dump(folds, f, indent=2)
    print(f"Saved {len(folds)} folds → {SPLITS_PATH}")
    for fold in folds:
        print(f"  Fold {fold['fold']}: {len(fold['train'])} train, {len(fold['test'])} test")
