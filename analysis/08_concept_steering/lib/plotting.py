"""Headline specificity figure: columns (targets + placebo) × rows (layers).

Each subplot shows S(C, L, α) on y-axis vs alpha on x-axis, with paired-bootstrap
95% CI shaded and the shuffled-tag null envelope as a grey band.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt


def plot_specificity_summary(
    *,
    summary: dict[str, Any],
    save_path: Path,
) -> None:
    concepts = summary["concept_order"]
    layers   = summary["layers"]
    alphas   = summary["alphas"]

    fig, axes = plt.subplots(len(layers), len(concepts),
                              figsize=(3.4 * max(1, len(concepts)),
                                       2.4 * max(1, len(layers))),
                              sharex=True, sharey=True, squeeze=False)

    for ci, concept in enumerate(concepts):
        for li, layer in enumerate(layers):
            ax = axes[li, ci]
            cell = summary["cells"][concept][str(layer)]
            s_mid = cell["S_median"]
            s_lo  = cell["S_ci_lo"]
            s_hi  = cell["S_ci_hi"]
            null_lo = cell.get("null_lo", [None] * len(alphas))
            null_hi = cell.get("null_hi", [None] * len(alphas))

            if null_lo and null_lo[0] is not None:
                ax.fill_between(alphas, null_lo, null_hi, color="0.85", label="null")
            ax.fill_between(alphas, s_lo, s_hi, color="C0", alpha=0.3, label="95% CI")
            ax.plot(alphas, s_mid, color="C0", marker=".", label="S = ΔMRR_t − ΔMRR_c")
            ax.axhline(0, color="black", lw=0.5)
            ax.axvline(0, color="black", lw=0.5)
            if li == 0:
                ax.set_title(f"{concept}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"layer {layer}", fontsize=9)
            if li == len(layers) - 1:
                ax.set_xlabel("α (suppress →)", fontsize=9)

    fig.suptitle("Specificity gap by (concept, layer, α)", fontsize=12)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
