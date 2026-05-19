"""
Cluster review tool for MORABLES.

Modes
-----
    # Show inter-cluster similarity (find clusters that might need merging)
    ./run.sh analysis/cluster_review.py --similarity [--threshold 0.85]

    # Interactive review of non-singleton clusters (default)
    ./run.sh analysis/cluster_review.py [--include-singletons]

    # Apply decisions → write updated clusters_full.json
    ./run.sh analysis/cluster_review.py --apply

Commands during interactive review
-----------------------------------
    approve / a           — keep cluster as-is
    e <text>              — change the canonical text
    reject / r            — reject cluster (fables should NOT share a moral)
    skip / s              — defer this cluster
    quit / q              — save progress and exit
"""

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data/processed"
ANALYSIS    = Path(__file__).parent

CLUSTERS_PATH   = ANALYSIS / "clusters_full.json"
MORALS_PATH     = DATA_DIR / "morals_corpus.json"
FABLES_PATH     = DATA_DIR / "fables_corpus.json"
SIM_MATRIX_PATH = DATA_DIR / "moral_sim_matrix.npy"
DECISIONS_PATH  = ANALYSIS / "cluster_decisions.json"

W = 100  # display width


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    with open(CLUSTERS_PATH) as f:
        clusters = json.load(f)
    with open(MORALS_PATH) as f:
        morals_list = json.load(f)
    with open(FABLES_PATH) as f:
        fables_list = json.load(f)
    sim = np.load(SIM_MATRIX_PATH)

    # fable_id → moral text + moral index
    fable_to_moral = {m["fable_id"]: m["text"] for m in morals_list}
    fable_to_idx   = {m["fable_id"]: int(m["doc_id"].split("_")[1]) for m in morals_list}
    fable_to_title = {fb["doc_id"]: fb["title"] for fb in fables_list}

    return clusters, fable_to_moral, fable_to_idx, fable_to_title, sim


def load_decisions() -> dict:
    if DECISIONS_PATH.exists():
        return json.loads(DECISIONS_PATH.read_text())
    return {}


def save_decisions(decisions: dict):
    DECISIONS_PATH.write_text(json.dumps(decisions, indent=2, ensure_ascii=False))


# ── Inter-cluster similarity ───────────────────────────────────────────────────

def inter_cluster_sim(c1, c2, fable_to_idx, sim) -> float:
    """Max cosine similarity between any member pair across two clusters."""
    idxs1 = [fable_to_idx[f] for f in c1["fables"] if f in fable_to_idx]
    idxs2 = [fable_to_idx[f] for f in c2["fables"] if f in fable_to_idx]
    if not idxs1 or not idxs2:
        return 0.0
    sub = sim[np.ix_(idxs1, idxs2)]
    return float(sub.max())


def cmd_similarity(args):
    threshold = args.threshold
    clusters, fable_to_moral, fable_to_idx, fable_to_title, sim = load_data()

    # Only non-singleton clusters are interesting for cross-cluster overlap
    non_single = [c for c in clusters if c["type"] != "singleton"]
    print(f"\n{'═'*W}")
    print(f"  INTER-CLUSTER SIMILARITY  (threshold > {threshold})")
    print(f"  Comparing {len(non_single)} non-singleton clusters pairwise")
    print(f"{'═'*W}\n")

    pairs = []
    for i in range(len(non_single)):
        for j in range(i + 1, len(non_single)):
            c1, c2 = non_single[i], non_single[j]
            # Skip same-type pairs that share no fables (already handled)
            s = inter_cluster_sim(c1, c2, fable_to_idx, sim)
            if s >= threshold:
                pairs.append((s, c1, c2))

    pairs.sort(key=lambda x: -x[0])

    if not pairs:
        print(f"  No cluster pairs found above threshold {threshold}.")
        return

    print(f"  Found {len(pairs)} cluster pair(s) with max-sim >= {threshold}:\n")
    for rank, (s, c1, c2) in enumerate(pairs, 1):
        print(f"  [{rank}] Sim: {s:.3f}")
        print(f"      {c1['cluster_id']}  →  \"{c1['canonical']}\"")
        print(f"      {c2['cluster_id']}  →  \"{c2['canonical']}\"")
        print(f"      Fables: {c1['fables']}  |  {c2['fables']}")
        print()

    print(f"{'─'*W}")
    print(f"  Summary: {len(pairs)} pairs flagged at threshold {threshold}")
    print(f"  Consider merging cluster pairs with sim > 0.90\n")


# ── Display helpers ────────────────────────────────────────────────────────────

def show_cluster(cluster, fable_to_moral, fable_to_title, idx, total, done_count):
    print()
    print("═" * W)
    kind = cluster["type"].upper()
    print(f"  [{idx}/{total}]  {kind}  —  {cluster['cluster_id']}   ({done_count} decisions saved)")
    print(f"  Canonical: \"{cluster['canonical']}\"")
    print("─" * W)

    for fable_id in cluster["fables"]:
        moral_text = fable_to_moral.get(fable_id, "?")
        title      = fable_to_title.get(fable_id, "?")
        print(f"\n  {fable_id}  ({title})")
        wrapped = textwrap.fill(f"  → \"{moral_text}\"", width=W - 2,
                                initial_indent="    ", subsequent_indent="       ")
        print(wrapped)

    print()
    print("─" * W)
    print("  approve/a  |  e <new canonical>  |  reject/r  |  skip/s  |  quit/q")
    print("─" * W)


# ── Interactive review ─────────────────────────────────────────────────────────

def cmd_review(args):
    clusters, fable_to_moral, fable_to_idx, fable_to_title, sim = load_data()
    decisions = load_decisions()

    # Filter which clusters to review
    if args.include_singletons:
        to_review = clusters
    else:
        to_review = [c for c in clusters if c["type"] != "singleton"]

    remaining = [c for c in to_review if c["cluster_id"] not in decisions]
    total     = len(to_review)
    done_base = total - len(remaining)

    print(f"\n{'═'*W}")
    print(f"  CLUSTER REVIEW")
    print(f"  {done_base}/{total} clusters already decided — {len(remaining)} remaining")
    print(f"{'═'*W}")

    if not remaining:
        print("\n  All clusters reviewed. Run with --apply to apply decisions.")
        return

    for i, cluster in enumerate(remaining, start=done_base + 1):
        show_cluster(cluster, fable_to_moral, fable_to_title, i, total, len(decisions))

        while True:
            try:
                raw = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Saving and exiting.")
                save_decisions(decisions)
                return

            if not raw:
                continue

            low = raw.lower()

            if low in ("q", "quit"):
                save_decisions(decisions)
                print("  Progress saved.")
                return

            if low in ("s", "skip"):
                decisions[cluster["cluster_id"]] = {"action": "skip"}
                print("  → Skipped.")
                break

            if low in ("a", "approve"):
                decisions[cluster["cluster_id"]] = {
                    "action": "approve",
                    "canonical": cluster["canonical"],
                }
                print("  → Approved.")
                break

            if low in ("r", "reject"):
                decisions[cluster["cluster_id"]] = {"action": "reject"}
                print("  → Rejected.")
                break

            if low.startswith("e ") and len(raw) > 2:
                new_canon = raw[2:].strip()
                decisions[cluster["cluster_id"]] = {
                    "action": "approve",
                    "canonical": new_canon,
                }
                print(f"  → Approved with new canonical: \"{new_canon}\"")
                break

            print("  Unknown command. Use: approve/a  |  e <text>  |  reject/r  |  skip/s  |  quit/q")

        save_decisions(decisions)

    print(f"\n  Session complete. {len(decisions)} decisions saved.")
    _print_summary(decisions)
    print("  Run with --apply to apply decisions.")


# ── Apply decisions ────────────────────────────────────────────────────────────

def cmd_apply(args):
    with open(CLUSTERS_PATH) as f:
        clusters = json.load(f)
    decisions = load_decisions()

    approved = {k: v for k, v in decisions.items() if v["action"] == "approve"}
    rejected = {k for k, v in decisions.items() if v["action"] == "reject"}

    updated = []
    for c in clusters:
        cid = c["cluster_id"]
        if cid in rejected:
            # Explode cluster back to singletons
            for fable_id in c["fables"]:
                updated.append({
                    "cluster_id": f"singleton_{fable_id.split('_')[1]}",
                    "canonical": None,
                    "type": "singleton",
                    "fables": [fable_id],
                })
        elif cid in approved:
            # Update canonical if edited
            updated.append({**c, "canonical": approved[cid]["canonical"]})
        else:
            updated.append(c)

    out_path = ANALYSIS / "clusters_reviewed.json"
    out_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False))
    print(f"\n  Applied {len(approved)} approvals, {len(rejected)} rejections.")
    print(f"  Written: {out_path}")
    _print_summary(decisions)


# ── Summary ────────────────────────────────────────────────────────────────────

def _print_summary(decisions):
    from collections import Counter
    counts = Counter(v["action"] for v in decisions.values())
    print(f"\n  Decisions: {counts.get('approve',0)} approved  "
          f"{counts.get('reject',0)} rejected  "
          f"{counts.get('skip',0)} skipped")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cluster review tool")
    parser.add_argument("--similarity", action="store_true",
                        help="Show inter-cluster similarity report")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Similarity threshold for --similarity (default: 0.85)")
    parser.add_argument("--include-singletons", action="store_true",
                        help="Also review singleton clusters in interactive mode")
    parser.add_argument("--apply", action="store_true",
                        help="Apply saved decisions → write clusters_reviewed.json")
    args, _ = parser.parse_known_args()

    if args.similarity:
        cmd_similarity(args)
    elif args.apply:
        cmd_apply(args)
    else:
        cmd_review(args)


if __name__ == "__main__":
    main()
