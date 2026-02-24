"""
Prepare MORABLES data for retrieval evaluation.

Creates:
1. Corpus files (fables corpus, morals corpus)
2. Query-relevance mappings for both directions
3. Distractor-augmented moral corpus (morals + 4 distractors per fable)

Output goes to data/processed/.
Reads from data/raw/.
"""
import json
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(RAW_DIR / "fables.json") as f:
    fables = json.load(f)
with open(RAW_DIR / "mcqa.json") as f:
    mcqa = json.load(f)

# ══════════════════════════════════════════════════════════════
# 1. Clean corpus: 709 fables, 709 morals (1-to-1 mapping)
# ══════════════════════════════════════════════════════════════

fables_corpus = []
morals_corpus = []
qrels_fable_to_moral = []
qrels_moral_to_fable = []

for i, fable in enumerate(fables):
    fable_id = f"fable_{i:04d}"
    moral_id = f"moral_{i:04d}"

    fables_corpus.append({
        "doc_id": fable_id,
        "title": fable["title"],
        "text": fable["story"],
        "alias": fable["alias"],
    })

    morals_corpus.append({
        "doc_id": moral_id,
        "text": fable["moral"],
        "fable_id": fable_id,
    })

    qrels_fable_to_moral.append({
        "query_id": fable_id,
        "doc_id": moral_id,
        "relevance": 1,
    })

    qrels_moral_to_fable.append({
        "query_id": moral_id,
        "doc_id": fable_id,
        "relevance": 1,
    })

# ══════════════════════════════════════════════════════════════
# 2. Distractor-augmented moral corpus
#
# Two-pass approach to fix the deduplication bug:
#   Pass 1: Collect ALL roles each unique moral plays across all fables.
#   Pass 2: Build the corpus and qrels using the complete information.
#
# A moral is marked is_correct=true if it serves as ground_truth
# for ANY fable (even if it also appears as a distractor elsewhere).
# ══════════════════════════════════════════════════════════════

# --- Pass 1: collect all roles per unique moral text ---
# moral_key -> { "text": original_case_text, "roles": [(fable_id, class), ...] }
moral_roles = {}

for i, entry in enumerate(mcqa):
    fable_id = f"fable_{i:04d}"
    for choice, cls in zip(entry["choices"], entry["classes"]):
        moral_key = choice.strip().lower()
        if moral_key not in moral_roles:
            moral_roles[moral_key] = {
                "text": choice,  # keep original casing from first occurrence
                "roles": [],
            }
        moral_roles[moral_key]["roles"].append((fable_id, cls))

# --- Pass 2: build augmented corpus ---
morals_augmented = []
moral_key_to_aug_id = {}  # moral_key -> doc_id in augmented corpus

for moral_key, info in moral_roles.items():
    roles = info["roles"]
    # A moral is "correct" if it's ground_truth for at least one fable
    is_correct = any(cls == "ground_truth" for _, cls in roles)
    # Collect all classes this moral has played
    all_classes = list(set(cls for _, cls in roles))
    # Collect all fables this moral is the correct answer for
    correct_for_fables = [fid for fid, cls in roles if cls == "ground_truth"]

    doc_id = f"moral_aug_{len(morals_augmented):04d}"
    moral_key_to_aug_id[moral_key] = doc_id

    morals_augmented.append({
        "doc_id": doc_id,
        "text": info["text"],
        "is_correct": is_correct,
        "classes": all_classes,
        "correct_for_fables": correct_for_fables,
    })

# --- Build augmented qrels: every fable maps to its correct moral ---
qrels_augmented = []
for i, entry in enumerate(mcqa):
    fable_id = f"fable_{i:04d}"
    correct_moral_key = entry["choices"][0].strip().lower()
    aug_doc_id = moral_key_to_aug_id.get(correct_moral_key)
    if aug_doc_id is not None:
        qrels_augmented.append({
            "query_id": fable_id,
            "doc_id": aug_doc_id,
            "relevance": 1,
        })

# ══════════════════════════════════════════════════════════════
# Save everything
# ══════════════════════════════════════════════════════════════

outputs = {
    "fables_corpus.json": fables_corpus,
    "morals_corpus.json": morals_corpus,
    "morals_augmented_corpus.json": morals_augmented,
    "qrels_fable_to_moral.json": qrels_fable_to_moral,
    "qrels_moral_to_fable.json": qrels_moral_to_fable,
    "qrels_fable_to_moral_augmented.json": qrels_augmented,
}

for filename, data in outputs.items():
    with open(OUT_DIR / filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {filename}: {len(data)} entries")

# ── Summary ──
correct_count = sum(1 for m in morals_augmented if m["is_correct"])
distractor_count = sum(1 for m in morals_augmented if not m["is_correct"])
print(f"\nRetrieval setup summary:")
print(f"  Fables corpus: {len(fables_corpus)}")
print(f"  Morals corpus (clean): {len(morals_corpus)}")
print(f"  Morals corpus (augmented): {len(morals_augmented)} ({correct_count} correct + {distractor_count} distractors)")
print(f"  Fable→Moral qrels: {len(qrels_fable_to_moral)}")
print(f"  Moral→Fable qrels: {len(qrels_moral_to_fable)}")
print(f"  Fable→Moral (augmented) qrels: {len(qrels_augmented)}")
