"""
analysis/lib/plotting.py — shared matplotlib style for all paper figures.

Import setup_style() at the top of every analysis script to get consistent
fonts, colours, and DPI across all outputs.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

# Colour palette — consistent across all analysis plots
COLOURS = {
    "primary":    "#2563EB",   # blue  — main series
    "secondary":  "#DC2626",   # red   — contrast / wrong fable
    "gt":         "#16A34A",   # green — ground truth
    "neutral":    "#6B7280",   # grey  — background / reference
    "highlight":  "#D97706",   # amber — important outliers
}

# Ordered palette for multi-series plots (zero-shot vs ft, multiple models, etc.)
PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2"]


def setup_style(font_size: int = 12, fig_dpi: int = 150) -> None:
    """Apply consistent style for all paper figures. Call once per script."""
    mpl.rcParams.update({
        "figure.dpi":        fig_dpi,
        "savefig.dpi":       300,          # high-res for paper
        "savefig.bbox":      "tight",
        "font.size":         font_size,
        "axes.titlesize":    font_size + 2,
        "axes.labelsize":    font_size,
        "xtick.labelsize":   font_size - 1,
        "ytick.labelsize":   font_size - 1,
        "legend.fontsize":   font_size - 1,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "figure.figsize":    (8, 5),
    })


def save_fig(path: str, fig: plt.Figure | None = None) -> None:
    """Save figure to path. Creates parent dirs if needed."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    (fig or plt).savefig(path)
    plt.close("all")
    print(f"  [saved] {path}")
