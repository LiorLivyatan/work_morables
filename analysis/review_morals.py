"""
Interactive moral-pair review tool.

Shows near-duplicate moral pairs side by side with their fables.
Pairs with sim = 1.0 (identical text) are skipped — they are handled
automatically as multi-label during --apply.

Usage
-----
    # Review mode (interactive):
    ./run.sh analysis/review_morals.py

    # Apply saved decisions → write corrected dataset:
    ./run.sh analysis/review_morals.py --apply

Commands during review
----------------------
    A          — merge: keep Moral A's text as canonical for both fables
    B          — merge: keep Moral B's text as canonical for both fables
    <text>     — merge: use a custom canonical text for both fables
    keep       — keep both as-is (accept the ambiguity)
    skip       — defer this pair (come back later)
    quit       — save progress and exit

Output
------
    analysis/review_decisions.json              — saved decisions (resume-safe)
    data/processed/morals_corpus_reviewed.json  — fixed morals corpus
    data/processed/qrels_moral_to_fable_reviewed.json  — fixed qrels (multi-label)
"""

import argparse
import json
import textwrap
from pathlib import Path

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data/processed"
DECISIONS   = Path(__file__).parent / "review_decisions.json"
FLAGGED_CSV = Path(__file__).parent / "moral_similarity_flagged.csv"

FABLES_PATH = DATA_DIR / "fables_corpus.json"
MORALS_PATH = DATA_DIR / "morals_corpus.json"
QRELS_PATH  = DATA_DIR / "qrels_moral_to_fable.json"

OUT_MORALS = DATA_DIR / "morals_corpus_reviewed.json"
OUT_QRELS  = DATA_DIR / "qrels_moral_to_fable_reviewed.json"

W = 100


# ── Data loading ───────────────────────────────────────────────────────────────

def load_pairs(threshold: float) -> list[dict]:
    import csv
    seen, pairs = set(), []
    for row in csv.DictReader(open(FLAGGED_CSV)):
        sim = float(row["similarity"])
        if sim >= 1.0 or sim <= threshold:
            continue
        key = tuple(sorted([row["moral_a"], row["moral_b"]]))
        if key not in seen:
            seen.add(key)
            pairs.append(row)
    return pairs


def load_all_pairs_counts() -> dict[float, int]:
    import csv
    seen, all_unique = set(), []
    for row in csv.DictReader(open(FLAGGED_CSV)):
        sim = float(row["similarity"])
        if sim >= 1.0:
            continue
        key = tuple(sorted([row["moral_a"], row["moral_b"]]))
        if key not in seen:
            seen.add(key)
            all_unique.append(sim)
    return {t: sum(1 for s in all_unique if s > t) for t in (0.95, 0.90, 0.85, 0.80)}


def load_corpus():
    with open(FABLES_PATH) as f:
        fables = {fb["doc_id"]: fb for fb in json.load(f)}
    with open(MORALS_PATH) as f:
        morals = json.load(f)
    with open(QRELS_PATH) as f:
        qrels = json.load(f)
    return fables, morals, qrels


def load_decisions() -> dict:
    if DECISIONS.exists():
        return json.loads(DECISIONS.read_text())
    return {}


def save_decisions(decisions: dict):
    DECISIONS.write_text(json.dumps(decisions, indent=2, ensure_ascii=False))


# ── Threshold picker ───────────────────────────────────────────────────────────

def pick_threshold() -> float:
    counts    = load_all_pairs_counts()
    decisions = load_decisions()

    options = [
        (0.95, "> 0.95  — near-identical wording"),
        (0.90, "> 0.90  — very similar meaning"),
        (0.85, "> 0.85  — similar meaning"),
        (0.80, "> 0.80  — somewhat similar  (long session)"),
    ]

    print()
    print("═" * W)
    print("  MORAL REVIEW TOOL")
    print("═" * W)
    print()
    print("  Pairs with sim = 1.0 (identical text) are skipped — handled automatically.")
    print()
    print(f"  {'#':<4}  {'Threshold':<30}  {'Pairs':<8}  {'Already decided'}")
    print("  " + "─" * 60)
    for i, (thresh, label) in enumerate(options, 1):
        total = counts[thresh]
        pairs = load_pairs(thresh)
        done  = sum(1 for p in pairs
                    if str(tuple(sorted([p["moral_a"], p["moral_b"]]))) in decisions)
        print(f"  {i:<4}  {label:<30}  {total:<8}  {done}/{total}")

    print()
    while True:
        try:
            raw = input("  Choose threshold [1-4]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            raise SystemExit(0)
        if raw in ("1", "2", "3", "4"):
            return options[int(raw) - 1][0]
        print("  Please enter 1, 2, 3, or 4.")


# ── Display ────────────────────────────────────────────────────────────────────

def show_pair(pair: dict, fables: dict, idx: int, total: int, done_count: int):
    sim  = float(pair["similarity"])

    print()
    print("═" * W)
    print(f"  [{idx}/{total}]  Sim: {sim:.3f}   ({done_count} decisions saved so far)")
    print("─" * W)

    for side, mk, fk, tk in [
        ("A", "moral_a", "fable_a_id", "fable_a_title"),
        ("B", "moral_b", "fable_b_id", "fable_b_title"),
    ]:
        fable_text = fables.get(pair[fk], {}).get("text", "?")
        snippet    = fable_text[:320].replace("\n", " ") + ("…" if len(fable_text) > 320 else "")
        wrapped    = textwrap.fill(snippet, width=W - 6, initial_indent=" " * 6,
                                   subsequent_indent=" " * 6)
        print(f"\n  [{side}] {pair[mk]}")
        print(f"      → {pair[tk]} ({pair[fk]})")
        print(wrapped)

    print()
    print("─" * W)
    print("  A  B  <custom text>  keep  skip  quit")
    print("─" * W)


# ── Review loop ────────────────────────────────────────────────────────────────

def review():
    threshold = pick_threshold()
    fables, morals, qrels = load_corpus()
    pairs     = load_pairs(threshold)
    decisions = load_decisions()

    remaining = [p for p in pairs
                 if str(tuple(sorted([p["moral_a"], p["moral_b"]]))) not in decisions]
    total     = len(pairs)

    if not remaining:
        print(f"\n  All {total} pairs at this threshold already reviewed.")
        print("  Run with --apply to write the corrected dataset.")
        return

    print(f"\n  {total - len(remaining)}/{total} pairs already decided — {len(remaining)} remaining.")

    for i, pair in enumerate(remaining, start=(total - len(remaining)) + 1):
        show_pair(pair, fables, i, total, len(decisions))
        key = str(tuple(sorted([pair["moral_a"], pair["moral_b"]])))

        while True:
            try:
                raw = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Saving and exiting.")
                save_decisions(decisions)
                return

            if not raw:
                continue

            if raw.lower() == "quit":
                save_decisions(decisions)
                print("  Progress saved.")
                return

            if raw.lower() == "skip":
                decisions[key] = {"action": "skip",
                                  "moral_a": pair["moral_a"], "moral_b": pair["moral_b"]}
                print("  → Skipped.")
                break

            if raw.lower() == "keep":
                decisions[key] = {"action": "keep",
                                  "moral_a": pair["moral_a"], "moral_b": pair["moral_b"]}
                print("  → Kept as-is.")
                break

            if raw.upper() == "A":
                canonical = pair["moral_a"]
            elif raw.upper() == "B":
                canonical = pair["moral_b"]
            else:
                canonical = raw  # custom text

            decisions[key] = {"action": "merge", "canonical": canonical,
                               "moral_a": pair["moral_a"], "moral_b": pair["moral_b"]}
            print(f"  → Merged: \"{canonical}\"")
            break

        save_decisions(decisions)

    print(f"\n  Session complete.")
    _summary(decisions)
    print("  Run with --apply to write the corrected dataset.")


# ── Apply decisions ────────────────────────────────────────────────────────────

def apply_decisions():
    fables, morals, qrels = load_corpus()
    decisions = load_decisions()
    merges    = {k: v for k, v in decisions.items() if v["action"] == "merge"}

    # Build rewrite map: old_text -> canonical
    rewrite: dict[str, str] = {}
    for d in merges.values():
        canon = d["canonical"]
        for side in ("moral_a", "moral_b"):
            if d[side] != canon:
                rewrite[d[side]] = canon

    # Also auto-handle sim=1.0 exact duplicates (no rewrite needed, just multi-label)
    print(f"  Merge decisions: {len(merges)}  ({len(rewrite)} text rewrites)")

    # Rewrite morals corpus
    new_morals = []
    changed = 0
    for m in morals:
        if m["text"] in rewrite:
            m = dict(m, text=rewrite[m["text"]])
            changed += 1
        new_morals.append(m)
    print(f"  Moral entries rewritten: {changed}")

    # Build multi-label qrels: collect all fable_ids per canonical text
    moral_id_to_text = {m["doc_id"]: m["text"] for m in new_morals}
    text_to_fables: dict[str, set] = {}
    for mid, fid in qrels.items():
        text = moral_id_to_text.get(mid, "")
        text_to_fables.setdefault(text, set()).add(fid)

    new_qrels: dict[str, object] = {}
    for mid, fid in qrels.items():
        text  = moral_id_to_text.get(mid, "")
        valid = sorted(text_to_fables.get(text, {fid}))
        new_qrels[mid] = valid if len(valid) > 1 else fid

    multi = sum(1 for v in new_qrels.values() if isinstance(v, list))
    print(f"  Morals with multiple valid fables (multi-label): {multi}")

    OUT_MORALS.write_text(json.dumps(new_morals, indent=2, ensure_ascii=False))
    OUT_QRELS.write_text(json.dumps(new_qrels, indent=2, ensure_ascii=False))
    print(f"\n  Written:")
    print(f"    {OUT_MORALS}")
    print(f"    {OUT_QRELS}")
    _summary(decisions)


def _summary(decisions: dict):
    from collections import Counter
    counts = Counter(v["action"] for v in decisions.values())
    print(f"\n  Decisions: {counts.get('merge',0)} merges  "
          f"{counts.get('keep',0)} keeps  {counts.get('skip',0)} skips")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Apply saved decisions and write corrected dataset")
    args, _ = parser.parse_known_args()  # ignore empty string run.sh passes when no args given

    if args.apply:
        apply_decisions()
    else:
        review()


if __name__ == "__main__":
    main()
