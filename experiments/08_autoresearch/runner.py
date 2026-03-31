#!/usr/bin/env python3
"""
runner.py — Retrieval autoresearch loop orchestrator.
READ-ONLY: Never modified by the agent.

Called by Claude Code after editing retrieval_pipeline.py:
    python runner.py "description of this experiment"
    python runner.py --baseline    (first run: logs without committing)

Flow:
  1. Commit retrieval_pipeline.py with the description
  2. Run it (10-min timeout)
  3. Parse MRR from stdout
  4. MRR > best → keep commit, record "keep"
     MRR ≤ best → git reset, record "discard"
     crash/timeout → git reset, record "crash"
  5. Append row to results.tsv
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_TSV = EXPERIMENT_DIR / "results.tsv"
PIPELINE_FILE = EXPERIMENT_DIR / "retrieval_pipeline.py"
REPO_ROOT = EXPERIMENT_DIR.parent.parent
TIMEOUT_SECONDS = 600
TSV_HEADER = ["commit", "MRR", "R@1", "R@5", "R@10", "status", "description"]


# ── Utility functions (unit-tested) ─────────────────────────────────────────

def parse_metric(stdout: str, key: str) -> float | None:
    """Parse 'key: 0.XXXX' from stdout. Returns None if not found."""
    match = re.search(rf"^{re.escape(key)}:\s*([0-9.]+)", stdout, re.MULTILINE)
    return float(match.group(1)) if match else None


def get_best_mrr() -> float:
    """Return highest MRR from kept+baseline rows in results.tsv. 0.0 if no file."""
    if not RESULTS_TSV.exists():
        return 0.0
    with open(RESULTS_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        mrrs = [
            float(row["MRR"])
            for row in reader
            if row.get("status") in ("keep", "baseline")
        ]
    return max(mrrs, default=0.0)


def append_result(
    commit: str, mrr: float, r1: float, r5: float, r10: float,
    status: str, description: str,
) -> None:
    """Append one row to results.tsv, writing header on first call."""
    write_header = not RESULTS_TSV.exists()
    with open(RESULTS_TSV, "a") as f:
        if write_header:
            f.write("\t".join(TSV_HEADER) + "\n")
        f.write(
            f"{commit}\t{mrr:.4f}\t{r1:.4f}\t{r5:.4f}\t{r10:.4f}"
            f"\t{status}\t{description}\n"
        )


# ── Git helpers ──────────────────────────────────────────────────────────────

def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )


def git_commit(description: str) -> str:
    _git("add", str(PIPELINE_FILE))
    _git("commit", "-m", f"exp08: {description}")
    return _git("rev-parse", "--short", "HEAD").stdout.strip()


def git_reset() -> None:
    _git("reset", "--hard", "HEAD~1")


def get_current_commit() -> str:
    return _git("rev-parse", "--short", "HEAD").stdout.strip()


# ── Pipeline execution ───────────────────────────────────────────────────────

def run_pipeline() -> tuple[float, float, float, float]:
    """Run retrieval_pipeline.py. Returns (mrr, r1, r5, r10). Raises on failure."""
    result = subprocess.run(
        [sys.executable, str(PIPELINE_FILE)],
        capture_output=True, text=True,
        timeout=TIMEOUT_SECONDS,
        cwd=EXPERIMENT_DIR,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline exited {result.returncode}:\n{result.stderr[-2000:]}")
    stdout = result.stdout
    mrr = parse_metric(stdout, "mrr")
    if mrr is None:
        raise RuntimeError(f"'mrr: X.XXXX' not found in stdout:\n{stdout[-1000:]}")
    r1  = parse_metric(stdout, "r@1") or 0.0
    r5  = parse_metric(stdout, "r@5") or 0.0
    r10 = parse_metric(stdout, "r@10") or 0.0
    return mrr, r1, r5, r10


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    baseline_mode = "--baseline" in sys.argv
    description_parts = [a for a in sys.argv[1:] if not a.startswith("--")]
    description = " ".join(description_parts) if description_parts else "no description"

    best_mrr = get_best_mrr()
    print(f"\n{'='*60}")
    print(f"Experiment: {description}")
    print(f"Best so far: {best_mrr:.4f} MRR")
    print(f"{'='*60}\n")

    if baseline_mode:
        commit = get_current_commit()
        print(f"Baseline mode — using current commit: {commit}")
    else:
        commit = git_commit(description)
        print(f"Committed: {commit}")

    try:
        mrr, r1, r5, r10 = run_pipeline()
        print(f"\nResult: MRR={mrr:.4f}  R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}")

        if baseline_mode:
            status = "baseline"
            print("BASELINE — logged without keep/discard decision")
        elif mrr > best_mrr:
            status = "keep"
            print(f"KEEP  (+{mrr - best_mrr:.4f} MRR)")
        else:
            status = "discard"
            git_reset()
            print(f"DISCARD  ({mrr:.4f} ≤ {best_mrr:.4f})")

    except (subprocess.TimeoutExpired, RuntimeError) as e:
        mrr = r1 = r5 = r10 = 0.0
        status = "crash"
        if not baseline_mode:
            git_reset()
        print(f"CRASH: {e}")

    append_result(commit, mrr, r1, r5, r10, status, description)
    print(f"\nLogged → results.tsv  [{status}]  {commit}  MRR={mrr:.4f}")


if __name__ == "__main__":
    main()
