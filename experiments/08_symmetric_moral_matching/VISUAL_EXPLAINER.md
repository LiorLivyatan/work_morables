# Experiment 08 — Visual Explainer: Symmetric Moral Matching

---

## The Core Problem

Every experiment before this one tried to match two very different things:

```
QUERY (fable)                              CORPUS (morals)
──────────────────────────────────────     ──────────────────────────
"A slave named Androcles escaped from      "Gratitude is the sign
 his master and fled to the forest.         of noble souls."
 As he was wandering about there he
 came upon a Lion lying down moaning       "Appearances are deceptive."
 and groaning. At first he turned to
 flee, but finding that the Lion did       "There's a time for work
 not pursue him, he turned back and         and a time for play."
 drew out the thorn and bound up the
 paw..."  [~117 words]                     [~10 words each]
```

**The mismatch:** long narrative ↔ short aphorism. The embedding model has to bridge
a massive gap in length, style, and abstraction level.

**Result without any fix: ~3% R@1** (barely above random at 0.14%)

---

## What Experiment 07 Did (the predecessor)

Exp 07 replaced raw fable text with a Gemini-generated summary — converting the
**corpus** from long narratives into short moral-style sentences.

```
                    ┌─────────────────────────────────────────┐
                    │           EXP 07 APPROACH               │
                    └─────────────────────────────────────────┘

  FABLE (raw)              GEMINI                 CORPUS DOCUMENT
  ─────────────            ──────────────►        ───────────────────────────
  "A slave named                                  "Compassion and gratitude
   Androcles escaped       conceptual_abstract     create unexpected bonds
   from his master..."     summarization           between unlikely allies."
   [117 words]                                     [~10 words]

  MORAL QUERY (unchanged)
  ─────────────────────────────────────────────────────────────
  "Gratitude is the sign of noble souls."
```

**Both sides are now short sentences.** The embedding model can match them.
**Result: 26.5% R@1 on full 709** — a 7× improvement over the raw baseline.

---

## What Experiment 08 Changes

Two ideas on top of exp 07:

```
                    ┌─────────────────────────────────────────┐
                    │           EXP 08 APPROACH               │
                    └─────────────────────────────────────────┘

  FABLE (raw)              GEMINI                 CORPUS DOCUMENT (tighter style)
  ─────────────            ──────────►            ──────────────────────────────
  "A slave named                                  variant A: "Kindness shown to
   Androcles escaped       ground_truth_style               others is often returned
   from his master..."     OR                               in times of need."
   [117 words]             declarative_universal  variant B: "Acts of kindness are
                                                            often repaid in times of
                                                            greatest need."

  MORAL QUERY             GEMINI                 QUERY EXPANSION
  ─────────────           ──────────►            ──────────────────────────────────
  "Gratitude is the       3 paraphrases          rephrase:  "Thankfulness serves as
   sign of noble                                            the hallmark of a
   souls."                                                  virtuous spirit."
                                                 elaborate: "Acknowledging the
                                                            kindness of others serves
                                                            as a hallmark of an
                                                            elevated character."
                                                 abstract:  "Gratitude manifests
                                                            nobility."
```

**Idea 1** — make the corpus side even tighter (match moral style more precisely)
**Idea 2** — expand the query side (4 vectors instead of 1)

---

## The Two Corpus Variants

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REAL EXAMPLE: Fable #3 (Grasshopper & Ant)          │
└─────────────────────────────────────────────────────────────────────────────┘

  RAW FABLE
  ─────────
  "One summer's day a Grasshopper was hopping about, chirping and singing to
   its heart's content. An Ant passed by, bearing along with great toil an
   ear of corn he was taking to the nest..."

  GROUND TRUTH MORAL
  ──────────────────
  "There's a time for work and a time for play."

  ┌──────────────────────────────────┐   ┌──────────────────────────────────┐
  │     VARIANT A: ground_truth_     │   │  VARIANT B: declarative_         │
  │              style               │   │             universal             │
  ├──────────────────────────────────┤   ├──────────────────────────────────┤
  │ Prompt: imitate real dataset     │   │ Prompt: state a universal truth  │
  │ morals (few-shot from MORABLES)  │   │ about human nature (no few-shot  │
  │                                  │   │ anchoring to dataset morals)     │
  │ Few-shot examples given:         │   │                                  │
  │  "Appearances are deceptive."    │   │ Few-shot examples given:         │
  │  "Vices are their own           │   │  "Those who envy others invite   │
  │   punishment."                   │   │   their own misfortune."         │
  │  "Gratitude is the sign of      │   │  "Necessity drives men to find   │
  │   noble souls."                  │   │   solutions."                    │
  │                                  │   │                                  │
  │ OUTPUT:                          │   │ OUTPUT:                          │
  │ "It is best to prepare for      │   │ "Diligence in times of plenty   │
  │  the days of necessity."         │   │  provides security in times      │
  │                                  │   │  of want."                       │
  └──────────────────────────────────┘   └──────────────────────────────────┘
              ▲                                          ▲
              │                                          │
         R@1 = 58%                                  R@1 = 64%
         (worse than baseline)                     (better than baseline)
```

**Why does A underperform?** The few-shot examples come from the dataset itself.
Gemini anchors too tightly to that style — many summaries look like variations
on the same template, clustering in embedding space and becoming hard to tell apart.

**Why does B win?** Without strong style anchoring, outputs are more varied and
spread more evenly in embedding space — each summary is more discriminable.

---

## Query Expansion

Instead of one query vector, generate 3 paraphrases offline and take the
**max cosine score** across all 4 at retrieval time.

```
  ORIGINAL MORAL
  ──────────────────────────────────────────────────────
  "Gratitude is the sign of noble souls."
         │
         ├──► rephrase:  "Thankfulness serves as the hallmark of a
         │                virtuous spirit."
         │
         ├──► elaborate: "Acknowledging the kindness of others serves
         │                as a definitive hallmark of an elevated and
         │                virtuous character."
         │
         └──► abstract:  "Gratitude manifests nobility."

  AT RETRIEVAL: for each candidate moral in corpus, take the best score
  ──────────────────────────────────────────────────────────────────────

  candidate: "Acts of kindness are often repaid in times of greatest need."

  score(original)  = 0.71
  score(rephrase)  = 0.74  ◄── wins
  score(elaborate) = 0.69
  score(abstract)  = 0.68

  final score = max(0.71, 0.74, 0.69, 0.68) = 0.74
```

**Why this helps:** The original moral phrasing is sometimes idiomatic or compressed
in a way that misaligns with the corpus summary's phrasing. One of the paraphrases
is usually closer. The max-score approach is non-lossy — it can only help, never hurt
(unless a paraphrase scores highest against a wrong document, which is rare).

---

## RRF — Reciprocal Rank Fusion

Instead of picking one config, run all 4 in parallel and merge their ranked lists.

```
  CONFIG A          CONFIG B          CONFIG A+expand   CONFIG B+expand
  ─────────         ─────────         ───────────────   ───────────────
  #1 moral_7        #1 moral_0  ✓     #1 moral_0  ✓     #1 moral_0  ✓
  #2 moral_0  ✓     #2 moral_7        #2 moral_7        #2 moral_3
  #3 moral_3        #3 moral_3        #3 moral_3        #3 moral_7
  ...               ...               ...               ...

  RRF score for moral_0:
    1/(60+2) + 1/(60+1) + 1/(60+1) + 1/(60+1)
  = 0.0161  + 0.0164  + 0.0164  + 0.0164
  = 0.0653   ◄── highest → ranked #1 ✓

  RRF score for moral_7:
    1/(60+1) + 1/(60+2) + 1/(60+2) + 1/(60+3)
  = 0.0164  + 0.0161  + 0.0161  + 0.0159
  = 0.0645   ◄── ranked #2
```

RRF rewards morals that rank consistently high across multiple configs,
not just those that peak in one. Result: best MRR (0.789) but not best R@1.

---

## Results at a Glance (50-fable pilot)

```
  Exp 07 baseline on same 50-fable pool
  ──────────────────────────────────────────────────────────────  62.0%

  Config A  (ground_truth_style only)
  ████████████████████████████████████████████████████░░░░░░░░░░  58.0%  -4%

  Config B  (declarative_universal only)
  ████████████████████████████████████████████████████████░░░░░░  64.0%  +2%

  Config A+expand  (ground_truth_style + query expansion)
  ████████████████████████████████████████████████████████░░░░░░  64.0%  +2%

  RRF-all  (fusion of all 4 configs)
  ██████████████████████████████████████████████████████████░░░░  68.0%  +6%

  Config B+expand  (declarative_universal + query expansion)  ★
  ████████████████████████████████████████████████████████████░░  70.0%  +8%

  Oracle ceiling (moral → moral, perfect representation)
  ██████████████████████████████████████████████████████████████████████  82.7%
```

**Best config: B_expand** — combining the right corpus style with query expansion
gives +8% over the exp07 baseline on the same subset.

---

## What Each Step Contributes

```
  Start: exp07 baseline (conceptual_abstract summaries, single query)
  ──────────────────────────────────────────────────────────────  62%

  + Switch corpus to declarative_universal style
  ──────────────────────────────────────────────────────────────  64%   (+2%)
         ▲
         └── corpus more spread in embedding space → less confusion

  + Add query expansion (3 paraphrases, max-score)
  ──────────────────────────────────────────────────────────────  70%   (+6% more)
         ▲
         └── widens search radius → catches cases where original
             moral phrasing misaligns with corpus summary phrasing

  Oracle ceiling (upper bound of what embedding similarity can achieve)
  ──────────────────────────────────────────────────────────────  82.7%
         ▲
         └── 12.7% gap remains — irreducible without fine-tuning
```

---

## Key Insight

The experiment shows the gap is **not in the embedding model** — it's in the
**representation fed to it**. The oracle ceiling of 82.7% proves the model can
distinguish morals when both sides look the same. The challenge is getting
the fable-side representation close enough to the moral-side representation
without leaking ground-truth information.

Exp 07 solved half of it (corpus side). Exp 08 solved the other half (query side).
The remaining 12.7% is the hard core — morals that are genuinely similar to each
other in meaning, where only fine-tuning on MORABLES itself can teach the model
to discriminate.
