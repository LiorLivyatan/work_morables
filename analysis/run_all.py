"""
analysis/run_all.py — run all analyses for one or more experiments.

Usage
-----
# Best model (ft07 linq_s500 fable+summary):
./run.sh analysis/run_all.py --remote --gpu 0

# Custom:
./run.sh analysis/run_all.py \\
    --moral_embs finetuning/ft_07_storal_transfer/cache/embeddings/linq_s500/fable_plus_summary/moral_embs.npy \\
    --doc_embs   finetuning/ft_07_storal_transfer/cache/embeddings/linq_s500/fable_plus_summary/doc_embs.npy \\
    --label      "ft07-linq-s500-fable+summary" \\
    --skip       06
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).parent.parent
SCRIPTS = [
    ("01", "01_rank_distribution/analyze.py"),
    ("02", "02_nearest_neighbor_confusion/analyze.py"),
    ("03", "03_score_gap_distribution/analyze.py"),
    ("04", "04_thematic_overlap/analyze.py"),
    ("05", "05_length_richness_bias/analyze.py"),
    # 06 skipped by default — requires umap-learn
]

# Default: best model from ft_07
DEFAULTS = {
    "moral_embs": "finetuning/ft_07_storal_transfer/cache/embeddings/linq_s500/fable_plus_summary/moral_embs.npy",
    "doc_embs":   "finetuning/ft_07_storal_transfer/cache/embeddings/linq_s500/fable_plus_summary/doc_embs.npy",
    "label":      "ft07-linq-s500-fable+summary",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--moral_embs", default=DEFAULTS["moral_embs"])
    p.add_argument("--doc_embs",   default=DEFAULTS["doc_embs"])
    p.add_argument("--label",      default=DEFAULTS["label"])
    p.add_argument("--skip",       nargs="*", default=["06"],
                   help="Analysis numbers to skip (default: 06)")
    return p.parse_args()


def main():
    args   = parse_args()
    skip   = set(args.skip or [])
    ana_dir = ROOT / "analysis"

    print(f"\n{'='*60}")
    print(f"  Running all analyses for: {args.label}")
    print(f"{'='*60}\n")

    for num, script in SCRIPTS:
        if num in skip:
            print(f"  [skipped] {script}")
            continue

        script_path = ana_dir / script
        out_dir     = ana_dir / script.split("/")[0] / "results" / args.label.replace("/", "_")

        print(f"\n{'─'*60}")
        print(f"  [{num}] {script}")

        cmd = [
            sys.executable, str(script_path),
            "--moral_embs", args.moral_embs,
            "--doc_embs",   args.doc_embs,
            "--label",      args.label,
            "--output_dir", str(out_dir),
        ]
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"  ✗ Analysis {num} failed (exit {result.returncode})")
        else:
            print(f"  ✓ Analysis {num} done → {out_dir}")

    print(f"\n{'='*60}")
    print(f"  All analyses complete. Results in analysis/*/results/{args.label}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
