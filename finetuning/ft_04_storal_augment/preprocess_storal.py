"""
ft_04 — Step 2: Preprocess STORAL into (moral, story) pairs for training.

Loads English STORAL pairs, flags near-duplicates of MORABLES fables
(same Aesop's source), and saves clean pairs for training augmentation.

Output: data/external/storal/processed/storal_pairs.json
  Each record: {"id", "story", "moral", "is_duplicate"}

Run:
    ./run.sh finetuning/ft_04_storal_augment/preprocess_storal.py
"""
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

STORAL_RAW = ROOT / "data/external/storal/raw/storal_en_labeled.json"
MORABLES_FABLES = ROOT / "data/raw/fables.json"
OUT_PATH = ROOT / "data/external/storal/processed/storal_pairs.json"

# Prefix length for duplicate detection: Aesop duplicates share identical
# opening sentences, so comparing 400 chars catches them while being ~5x
# faster than full-text comparison.
PREFIX_LEN = 400
DUP_THRESHOLD = 0.6


def is_near_duplicate(story: str, morables_prefixes: list[str]) -> bool:
    prefix = story[:PREFIX_LEN].lower()
    return any(
        SequenceMatcher(None, prefix, mp).ratio() >= DUP_THRESHOLD
        for mp in morables_prefixes
    )


def main() -> None:
    if not STORAL_RAW.exists():
        print(f"STORAL raw data not found at {STORAL_RAW}")
        print("Download with:")
        print("  git clone https://github.com/thu-coai/MoralStory /tmp/storal_repo")
        print(f"  cp -r /tmp/storal_repo/data/ data/external/storal/raw/")
        sys.exit(1)

    storal = json.loads(STORAL_RAW.read_text())
    fables = json.loads(MORABLES_FABLES.read_text())

    morables_prefixes = [f["story"][:PREFIX_LEN].lower() for f in fables if f.get("story")]

    print(f"Loaded {len(storal)} STORAL pairs, {len(fables)} MORABLES fables")
    print(f"Duplicate detection: SequenceMatcher prefix={PREFIX_LEN} chars, threshold={DUP_THRESHOLD}")
    print("Scanning for duplicates...")

    pairs = []
    for rec in storal:
        dup = is_near_duplicate(rec["story"], morables_prefixes)
        pairs.append({
            "id": rec["id"],
            "story": rec["story"],
            "moral": rec["moral"],
            "is_duplicate": dup,
        })

    n_dup = sum(p["is_duplicate"] for p in pairs)
    n_clean = len(pairs) - n_dup

    print(f"\nResults:")
    print(f"  Total pairs  : {len(pairs)}")
    print(f"  Duplicates   : {n_dup}  ({100*n_dup/len(pairs):.1f}%)")
    print(f"  Clean pairs  : {n_clean}  ({100*n_clean/len(pairs):.1f}%)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(pairs, indent=2, ensure_ascii=False))
    print(f"\nSaved → {OUT_PATH.relative_to(ROOT)}")

    # Sanity check: show a few flagged duplicates
    print("\nSample duplicates (first 3):")
    shown = 0
    for p in pairs:
        if p["is_duplicate"]:
            print(f"  [{p['id']}] {p['story'][:80].strip()}...")
            shown += 1
            if shown >= 3:
                break


if __name__ == "__main__":
    main()
