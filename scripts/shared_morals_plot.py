"""
Visualise morals that serve as ground truth for more than one fable.
Outputs: results/shared_morals.png
"""
import json
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR    = Path(__file__).parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

with open(DATA_DIR / "fables.json") as f:
    fables = json.load(f)

moral_counts = Counter(entry["moral"].strip().lower() for entry in fables)
shared       = {m: c for m, c in moral_counts.items() if c > 1}
all_counts   = sorted(shared.values(), reverse=True)   # e.g. [4, 2, 2, 2, ...]

# ── pretty label: truncate at 55 chars ─────────────────────────────────────
def label(moral, max_len=55):
    m = moral.strip().rstrip(".")
    return (m[:max_len] + "…") if len(m) > max_len else m

shared_sorted = sorted(shared.items(), key=lambda x: -x[1])   # most frequent first

labels = [label(m) for m, _ in shared_sorted]
counts = [c          for _, c  in shared_sorted]

fig, axes = plt.subplots(
    1, 2,
    figsize=(18, 8),
    gridspec_kw={"width_ratios": [2.5, 1]}
)
fig.suptitle("Morals Shared Across Multiple Fables", fontsize=15, fontweight="bold")

# ── LEFT: horizontal bar chart (one bar per repeated moral) ─────────────────
ax = axes[0]
y   = np.arange(len(labels))
colors = ["#E53935" if c >= 4 else "#FB8C00" if c == 3 else "#1E88E5"
          for c in counts]
bars = ax.barh(y, counts, color=colors, edgecolor="white", height=0.7)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Number of fables sharing this moral", fontsize=11)
ax.set_title(f"All {len(shared)} repeated morals  (covering {sum(counts)} fables)", fontsize=11)
ax.set_xlim(0, max(counts) + 1)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.axvline(1, color="gray", linewidth=0.5, linestyle="--")

# value labels on bars
for bar, c in zip(bars, counts):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            str(c), va="center", ha="left", fontsize=9, fontweight="bold")

# legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#E53935", label="4 fables"),
    Patch(facecolor="#FB8C00", label="3 fables"),
    Patch(facecolor="#1E88E5", label="2 fables"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

# ── RIGHT: histogram of repetition counts ───────────────────────────────────
ax2 = axes[1]
freq = Counter(counts)   # e.g. {2: 24, 3: 2, 4: 1}
xs   = sorted(freq.keys())
ys   = [freq[x] for x in xs]
bar_colors = ["#E53935" if x >= 4 else "#FB8C00" if x == 3 else "#1E88E5"
              for x in xs]
bars2 = ax2.bar(xs, ys, color=bar_colors, edgecolor="white", width=0.6)

ax2.set_xlabel("Fables sharing the same moral", fontsize=11)
ax2.set_ylabel("Number of distinct morals", fontsize=11)
ax2.set_title("Distribution of repetition counts", fontsize=11)
ax2.set_xticks(xs)
ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

for bar, v in zip(bars2, ys):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
             str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")

# summary box
total_affected = sum(counts)
unique_repeated = len(shared)
textstr = (
    f"Total fables: 709\n"
    f"Unique morals: {len(moral_counts)}\n"
    f"Repeated morals: {unique_repeated}\n"
    f"Fables affected: {total_affected}"
)
ax2.text(0.97, 0.97, textstr, transform=ax2.transAxes,
         fontsize=9, va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
out = RESULTS_DIR / "shared_morals.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
