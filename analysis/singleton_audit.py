"""
Singleton audit — finds formatting issues and cluster-fit candidates.
"""
import json
import numpy as np
from pathlib import Path

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data/processed"
ANALYSIS  = Path(__file__).parent

with open(ANALYSIS / "clusters_full.json") as f:
    clusters = json.load(f)
with open(DATA_DIR / "morals_corpus.json") as f:
    morals_list = json.load(f)
with open(DATA_DIR / "fables_corpus.json") as f:
    fables_list = json.load(f)

morals       = {m["fable_id"]: m["text"] for m in morals_list}
fable_to_idx = {m["fable_id"]: int(m["doc_id"].split("_")[1]) for m in morals_list}
fable_to_title = {fb["doc_id"]: fb["title"] for fb in fables_list}

sim = np.load(DATA_DIR / "moral_sim_matrix.npy")

singletons     = [c for c in clusters if c["type"] == "singleton"]
non_singletons = [c for c in clusters if c["type"] != "singleton"]

W = 110

# ── 1. Missing period ──────────────────────────────────────────────────────────
print(f"\n{'═'*W}")
print("  1. SINGLETONS MISSING TERMINAL PERIOD")
print(f"{'═'*W}")
no_period = []
for c in singletons:
    fid  = c["fables"][0]
    text = morals.get(fid, "")
    if text and not text.rstrip().endswith("."):
        no_period.append((fid, fable_to_title.get(fid, "?"), text))
print(f"  Found: {len(no_period)}")
for fid, title, text in no_period:
    print(f"  {fid} ({title})")
    print(f"    \"{text}\"")

# ── 2. Very short morals (≤ 3 words) ──────────────────────────────────────────
print(f"\n{'═'*W}")
print("  2. VERY SHORT MORALS (≤ 3 words) — possible truncation")
print(f"{'═'*W}")
for c in singletons:
    fid  = c["fables"][0]
    text = morals.get(fid, "")
    if 0 < len(text.split()) <= 3:
        print(f"  {fid} ({fable_to_title.get(fid,'?')}): \"{text}\"")

# ── 3. Singleton–cluster sim ≥ 0.82 ───────────────────────────────────────────
print(f"\n{'═'*W}")
print("  3. SINGLETONS THAT MAY BELONG TO AN EXISTING CLUSTER (sim ≥ 0.82)")
print(f"{'═'*W}")

threshold = 0.82
candidates = []

for sc in singletons:
    fid = sc["fables"][0]
    if fid not in fable_to_idx:
        continue
    sidx = fable_to_idx[fid]

    best_sim     = 0.0
    best_cluster = None
    for nc in non_singletons:
        nc_idxs = [fable_to_idx[f] for f in nc["fables"] if f in fable_to_idx]
        if not nc_idxs:
            continue
        max_s = float(sim[sidx, nc_idxs].max())
        if max_s > best_sim:
            best_sim     = max_s
            best_cluster = nc

    if best_sim >= threshold and best_cluster:
        ms    = best_cluster.get("moral_set", [])
        canon = ms[0][:75] if ms else "?"
        candidates.append((best_sim, fid, fable_to_title.get(fid, "?"),
                           morals.get(fid, "?"), best_cluster["cluster_id"], canon))

candidates.sort(key=lambda x: -x[0])
print(f"  Found: {len(candidates)}\n")
for s, fid, title, moral, cid, canon in candidates:
    print(f"  {s:.3f}  {fid}  [{title}]")
    print(f"    Singleton moral: \"{moral}\"")
    print(f"    Cluster:         {cid}")
    print(f"    Cluster canon:   \"{canon}\"")
    print()

# ── 4. Singleton–singleton pairs sim ≥ 0.85 ───────────────────────────────────
print(f"\n{'═'*W}")
print("  4. SINGLETON PAIRS THAT COULD FORM A NEW CLUSTER (sim ≥ 0.85)")
print(f"{'═'*W}")

threshold2 = 0.85
pairs = []
fids_with_idx = [(c["fables"][0], fable_to_idx[c["fables"][0]])
                 for c in singletons if c["fables"][0] in fable_to_idx]

for i in range(len(fids_with_idx)):
    for j in range(i + 1, len(fids_with_idx)):
        fa, ia = fids_with_idx[i]
        fb, ib = fids_with_idx[j]
        s = float(sim[ia, ib])
        if s >= threshold2:
            pairs.append((s, fa, fb))

pairs.sort(key=lambda x: -x[0])
print(f"  Found: {len(pairs)}\n")
for s, fa, fb in pairs:
    print(f"  {s:.3f}  {fa} [{fable_to_title.get(fa,'?')}]")
    print(f"         \"{morals.get(fa,'?')}\"")
    print(f"       + {fb} [{fable_to_title.get(fb,'?')}]")
    print(f"         \"{morals.get(fb,'?')}\"")
    print()
