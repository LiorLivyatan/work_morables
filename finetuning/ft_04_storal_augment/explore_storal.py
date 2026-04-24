"""
ft_04 — Step 1: Explore STORAL dataset structure and compatibility.

Answers the key questions before we invest in preprocessing:
  - How many English (moral, story) pairs are there?
  - How long are STORAL stories vs. MORABLES fables?
  - Are morals similar in style/abstraction level?
  - Is there any content overlap with MORABLES?

Run after downloading STORAL:
    git clone https://github.com/thu-coai/MoralStory /tmp/storal_repo
    cp -r /tmp/storal_repo/data/ data/external/storal/raw/

Then:
    ./run.sh finetuning/ft_04_storal_augment/explore_storal.py
"""
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

STORAL_RAW = ROOT / "data/external/storal/raw"
MORABLES_FABLES = ROOT / "data/raw/fables.json"

def load_storal_english() -> list[dict]:
    """Load all English STORAL entries. Returns list of raw records."""
    candidates = list(STORAL_RAW.rglob("*.json")) + list(STORAL_RAW.rglob("*.jsonl"))
    print(f"Found {len(candidates)} files in {STORAL_RAW}:")
    for p in candidates:
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KB)")

    records = []
    for path in candidates:
        try:
            text = path.read_text()
            if path.suffix == ".jsonl":
                data = [json.loads(l) for l in text.strip().splitlines()]
            else:
                data = json.loads(text)
                if isinstance(data, dict):
                    data = list(data.values()) if all(isinstance(v, list) for v in data.values()) else [data]
                    data = [item for sublist in data for item in (sublist if isinstance(sublist, list) else [sublist])]
            records.extend(data)
        except Exception as e:
            print(f"  [skip] {path.name}: {e}")

    return records


def main() -> None:
    if not STORAL_RAW.exists() or not any(STORAL_RAW.iterdir()):
        print("STORAL raw data not found. Please download first:")
        print("  git clone https://github.com/thu-coai/MoralStory /tmp/storal_repo")
        print(f"  cp -r /tmp/storal_repo/data/ {STORAL_RAW}/")
        sys.exit(1)

    print("=" * 60)
    print("STORAL Dataset Exploration")
    print("=" * 60)

    records = load_storal_english()
    print(f"\nTotal records loaded: {len(records)}")

    if records:
        print(f"\nSample record keys: {list(records[0].keys())}")
        print(f"Sample record:\n{json.dumps(records[0], indent=2, ensure_ascii=False)[:500]}")

    # ── Field analysis ────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("Field value samples (first 3 records):")
    for r in records[:3]:
        print(json.dumps(r, indent=2, ensure_ascii=False)[:300])
        print()

    # ── Story length analysis ─────────────────────────────────────────────
    story_fields = [k for k in (records[0].keys() if records else [])
                    if any(w in k.lower() for w in ["story", "text", "content", "body"])]
    moral_fields = [k for k in (records[0].keys() if records else [])
                    if any(w in k.lower() for w in ["moral", "label", "title", "lesson"])]
    print(f"Likely story fields: {story_fields}")
    print(f"Likely moral fields: {moral_fields}")

    # ── Compare with MORABLES ─────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("MORABLES fable length distribution:")
    with open(MORABLES_FABLES) as f:
        fables = json.load(f)
    morables_lengths = [len(f.get("story", "").split()) for f in fables]
    print(f"  count={len(morables_lengths)}")
    print(f"  min={min(morables_lengths)}  median={sorted(morables_lengths)[len(morables_lengths)//2]}  max={max(morables_lengths)} words")


if __name__ == "__main__":
    main()
