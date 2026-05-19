"""
ft_03 — Hard negative mining (required for --hard_neg mode in train.py).

For each fable and each of its 4 MCQA distractors, finds the fable whose
true moral is most similar to the distractor using zero-shot Linq.
That fable becomes the hard negative for training.

Run this once before running train.py with --hard_neg.

Usage:
    ./run.sh finetuning/ft_03_hard_neg/mine_negatives.py
    ./run.sh finetuning/ft_03_hard_neg/mine_negatives.py --force
"""
import argparse
import json
import sys
from pathlib import Path

import torch

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from lib.data import load_fables, load_morals, load_qrels_moral_to_fable

MCQA_PATH = ROOT / "data/raw/mcqa.json"
OUT_PATH = EXP_DIR / "data/hard_negatives.json"
MODEL_NAME = "Linq-AI-Research/Linq-Embed-Mistral"
DISTRACTOR_TYPES = ["similar_characters", "based_on_adjectives", "injected_adjectives", "partial_story"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-mine even if output exists")
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.force:
        print(f"[cache hit] Hard negatives already mined → {OUT_PATH}")
        print("  Use --force to re-mine.")
        return

    notify.send(f"🔍 ft_03: mining hard negatives\nmodel: {MODEL_NAME}")

    fables = load_fables()
    morals = load_morals()
    qrels  = load_qrels_moral_to_fable()

    # Build moral_idx → fable_idx (contiguous, 0-based, matching load_pairs ordering)
    moral_indices = sorted(qrels.keys())
    moral_idx_to_fable_idx = {i: qrels[idx] for i, idx in enumerate(moral_indices)}
    moral_texts = [morals[i]["text"] for i in moral_indices]

    # fable alias → moral_idx (to look up which moral index a fable corresponds to)
    fable_alias_to_moral_idx = {
        fables[fable_idx]["alias"]: moral_idx
        for moral_idx, fable_idx in moral_idx_to_fable_idx.items()
    }

    with open(MCQA_PATH) as f:
        mcqa = json.load(f)

    from sentence_transformers import SentenceTransformer
    print(f"[1/3] Encoding {len(moral_texts)} true morals with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, model_kwargs={"torch_dtype": "auto"})
    moral_embs = model.encode(moral_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    moral_embs = torch.tensor(moral_embs)  # (709, D)

    print("[2/3] Mining hard negatives for each distractor...")
    records = []
    skipped = 0

    for entry in mcqa:
        alias   = entry["alias"]
        classes = entry["classes"]   # ["ground_truth", "similar_characters", ...]
        choices = entry["choices"]   # [true_moral, dist1, dist2, dist3, dist4]

        moral_idx = fable_alias_to_moral_idx.get(alias)
        if moral_idx is None:
            skipped += 1
            continue

        fable_idx = moral_idx_to_fable_idx[moral_idx]

        for dtype in DISTRACTOR_TYPES:
            if dtype not in classes:
                continue
            distractor_text = choices[classes.index(dtype)]

            dist_emb = torch.tensor(model.encode([distractor_text], normalize_embeddings=True))
            sims = (moral_embs @ dist_emb.T).squeeze(1)  # (709,)
            sims[moral_idx] = -1.0  # exclude the fable itself

            hard_neg_moral_idx = int(sims.argmax())
            hard_neg_fable_idx = moral_idx_to_fable_idx[hard_neg_moral_idx]

            records.append({
                "fable_alias":       alias,
                "moral_idx":         moral_idx,
                "moral":             choices[0],
                "distractor_type":   dtype,
                "distractor_text":   distractor_text,
                "hard_neg_fable_idx": hard_neg_fable_idx,
                "hard_neg_alias":    fables[hard_neg_fable_idx]["alias"],
                "hard_neg_moral":    moral_texts[hard_neg_moral_idx],
                "similarity":        float(sims[hard_neg_moral_idx]),
            })

    print(f"[3/3] Saving {len(records)} records (skipped {skipped} without moral mapping)...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  → {OUT_PATH}")

    notify.send(
        f"✅ ft_03: hard negatives mined\n"
        f"{len(records)} records → {OUT_PATH.name}"
    )


if __name__ == "__main__":
    main()
