"""
Exploratory Data Analysis of the MORABLES dataset.
Produces summary statistics and visualizations.
"""
import json
import re
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Load data ──
with open(DATA_DIR / "fables.json") as f:
    fables = json.load(f)
with open(DATA_DIR / "mcqa.json") as f:
    mcqa = json.load(f)

print(f"Total fable-moral pairs: {len(fables)}")

# ── Basic statistics ──
story_lengths = [len(f["story"].split()) for f in fables]
moral_lengths = [len(f["moral"].split()) for f in fables]
story_sentences = [len(re.split(r'[.!?]+', f["story"])) for f in fables]

print(f"\nStory lengths (words): mean={np.mean(story_lengths):.1f}, "
      f"median={np.median(story_lengths):.1f}, "
      f"min={np.min(story_lengths)}, max={np.max(story_lengths)}")
print(f"Moral lengths (words): mean={np.mean(moral_lengths):.1f}, "
      f"median={np.median(moral_lengths):.1f}, "
      f"min={np.min(moral_lengths)}, max={np.max(moral_lengths)}")
print(f"Story sentences: mean={np.mean(story_sentences):.1f}, "
      f"median={np.median(story_sentences):.1f}")

# ── Source distribution (from alias prefix) ──
sources = []
for f in fables:
    alias = f["alias"]
    # Extract source prefix (e.g., "aesop", "la_fontaine", etc.)
    parts = alias.split("_")
    # Heuristic: source name ends before "section" or a number pattern
    source_parts = []
    for p in parts:
        if p in ("section", "fable", "book", "story", "part") or p.isdigit():
            break
        source_parts.append(p)
    source = "_".join(source_parts) if source_parts else parts[0]
    sources.append(source)

source_counts = Counter(sources)
print(f"\n=== Source Distribution ===")
for src, count in source_counts.most_common():
    print(f"  {src}: {count} ({count/len(fables)*100:.1f}%)")

# ── Lexical overlap between fables and morals ──
def word_set(text):
    return set(re.findall(r'\b\w+\b', text.lower()))

STOP_WORDS = {'the', 'a', 'an', 'is', 'was', 'were', 'are', 'be', 'been',
              'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
              'would', 'could', 'should', 'may', 'might', 'shall', 'can',
              'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
              'as', 'into', 'through', 'during', 'before', 'after', 'and',
              'but', 'or', 'nor', 'not', 'so', 'if', 'than', 'that',
              'this', 'it', 'its', 'his', 'her', 'their', 'who', 'which',
              'what', 'when', 'where', 'how', 'all', 'each', 'every',
              'both', 'few', 'more', 'most', 'other', 'some', 'such',
              'no', 'only', 'own', 'same', 'he', 'she', 'they', 'them',
              'him', 'we', 'you', 'i', 'me', 'my', 'your', 'our'}

ious = []
ious_no_stop = []
for f in fables:
    story_words = word_set(f["story"])
    moral_words = word_set(f["moral"])
    if story_words | moral_words:
        iou = len(story_words & moral_words) / len(story_words | moral_words)
        ious.append(iou)
    story_content = story_words - STOP_WORDS
    moral_content = moral_words - STOP_WORDS
    if story_content | moral_content:
        iou_ns = len(story_content & moral_content) / len(story_content | moral_content)
        ious_no_stop.append(iou_ns)

print(f"\n=== Lexical Overlap (Story ↔ Moral) ===")
print(f"Word IoU (all words): mean={np.mean(ious):.3f}, median={np.median(ious):.3f}")
print(f"Word IoU (no stopwords): mean={np.mean(ious_no_stop):.3f}, median={np.median(ious_no_stop):.3f}")

# ── Moral uniqueness ──
morals_lower = [f["moral"].strip().lower() for f in fables]
unique_morals = set(morals_lower)
print(f"\n=== Moral Uniqueness ===")
print(f"Total morals: {len(morals_lower)}")
print(f"Unique morals (exact match): {len(unique_morals)}")
moral_counts = Counter(morals_lower)
duplicates = {m: c for m, c in moral_counts.items() if c > 1}
if duplicates:
    print(f"Morals shared by multiple fables ({len(duplicates)}):")
    for m, c in sorted(duplicates.items(), key=lambda x: -x[1])[:10]:
        print(f"  [{c}x] \"{m}\"")

# ── Distractor analysis ──
print(f"\n=== Distractor Classes ===")
print(f"Classes: {mcqa[0]['classes']}")
# Check how similar distractors are to the correct moral (simple word overlap)
for class_idx, class_name in enumerate(mcqa[0]['classes']):
    if class_idx == 0:
        continue  # skip ground_truth
    overlaps = []
    for entry in mcqa:
        correct = word_set(entry['choices'][0]) - STOP_WORDS  # ground truth is always idx 0 in not_shuffled
        distractor = word_set(entry['choices'][class_idx]) - STOP_WORDS
        if correct | distractor:
            overlaps.append(len(correct & distractor) / len(correct | distractor))
    print(f"  {class_name}: mean IoU with correct = {np.mean(overlaps):.3f}")

# ══════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("MORABLES Dataset — Exploratory Analysis", fontsize=16, fontweight="bold")

# 1. Source distribution pie chart
ax = axes[0, 0]
top_n = 8
top_sources = source_counts.most_common(top_n)
labels = [s[0].replace("_", " ").title() for s in top_sources]
sizes = [s[1] for s in top_sources]
other = len(fables) - sum(sizes)
if other > 0:
    labels.append("Other")
    sizes.append(other)
colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                   colors=colors, startangle=90, pctdistance=0.85)
for t in texts:
    t.set_fontsize(8)
for t in autotexts:
    t.set_fontsize(7)
ax.set_title("Source Distribution")

# 2. Story and moral length distributions
ax = axes[0, 1]
ax.hist(story_lengths, bins=40, alpha=0.7, label="Stories", color="#2196F3")
ax.axvline(np.mean(story_lengths), color="#1565C0", linestyle="--", label=f"Mean: {np.mean(story_lengths):.0f}")
ax.set_xlabel("Length (words)")
ax.set_ylabel("Count")
ax.set_title("Story Length Distribution")
ax.legend()

ax2 = axes[1, 0]
ax2.hist(moral_lengths, bins=30, alpha=0.7, label="Morals", color="#FF9800")
ax2.axvline(np.mean(moral_lengths), color="#E65100", linestyle="--", label=f"Mean: {np.mean(moral_lengths):.0f}")
ax2.set_xlabel("Length (words)")
ax2.set_ylabel("Count")
ax2.set_title("Moral Length Distribution")
ax2.legend()

# 3. Lexical overlap histogram
ax = axes[1, 1]
ax.hist(ious_no_stop, bins=30, alpha=0.7, color="#4CAF50")
ax.axvline(np.mean(ious_no_stop), color="#1B5E20", linestyle="--",
           label=f"Mean: {np.mean(ious_no_stop):.3f}")
ax.set_xlabel("Word IoU (content words)")
ax.set_ylabel("Count")
ax.set_title("Lexical Overlap: Story ↔ Moral")
ax.legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "eda_overview.png", dpi=150, bbox_inches="tight")
print(f"\nSaved visualization to results/eda_overview.png")

# ── Distractor confusion chart ──
fig2, ax = plt.subplots(figsize=(8, 5))
class_names = mcqa[0]['classes'][1:]  # skip ground_truth
class_overlaps = []
for class_idx in range(1, 5):
    overlaps = []
    for entry in mcqa:
        correct = word_set(entry['choices'][0]) - STOP_WORDS
        distractor = word_set(entry['choices'][class_idx]) - STOP_WORDS
        if correct | distractor:
            overlaps.append(len(correct & distractor) / len(correct | distractor))
    class_overlaps.append(np.mean(overlaps))

bars = ax.bar(class_names, class_overlaps, color=["#E91E63", "#9C27B0", "#3F51B5", "#009688"])
ax.set_ylabel("Mean Word IoU with Correct Moral")
ax.set_title("Distractor Similarity to Correct Moral (lexical)")
for bar, val in zip(bars, class_overlaps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "distractor_similarity.png", dpi=150, bbox_inches="tight")
print(f"Saved distractor chart to results/distractor_similarity.png")
