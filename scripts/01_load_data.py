"""
Load MORABLES dataset from HuggingFace and save locally as JSON for easy access.
"""
import json
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Load fables_only (709 fable-moral pairs) ---
print("Loading fables_only...")
fables = load_dataset("cardiffnlp/Morables", "fables_only", split="morables")
fables_list = [dict(row) for row in fables]
with open(DATA_DIR / "fables.json", "w") as f:
    json.dump(fables_list, f, indent=2)
print(f"  Saved {len(fables_list)} fables to data/fables.json")

# --- Load mcqa (core benchmark with distractors) ---
print("Loading mcqa...")
mcqa = load_dataset("cardiffnlp/Morables", "mcqa", split="mcqa_not_shuffled")
mcqa_list = [dict(row) for row in mcqa]
with open(DATA_DIR / "mcqa.json", "w") as f:
    json.dump(mcqa_list, f, indent=2)
print(f"  Saved {len(mcqa_list)} MCQA entries to data/mcqa.json")

# --- Load adversarial variant ---
print("Loading adversarial...")
try:
    adv = load_dataset("cardiffnlp/Morables", "adversarial")
    for split_name in adv:
        adv_list = [dict(row) for row in adv[split_name]]
        with open(DATA_DIR / f"adversarial_{split_name}.json", "w") as f:
            json.dump(adv_list, f, indent=2)
        print(f"  Saved {len(adv_list)} adversarial entries ({split_name})")
except Exception as e:
    print(f"  Warning: Could not load adversarial config: {e}")

# --- Quick summary ---
print("\n=== Dataset Summary ===")
print(f"Total fable-moral pairs: {len(fables_list)}")
print(f"MCQA entries: {len(mcqa_list)}")
print(f"Distractor classes: {mcqa_list[0]['classes']}")

# Show a sample
print(f"\n=== Sample Entry ===")
sample = fables_list[0]
print(f"Title: {sample['title']}")
print(f"Story: {sample['story'][:200]}...")
print(f"Moral: {sample['moral']}")
