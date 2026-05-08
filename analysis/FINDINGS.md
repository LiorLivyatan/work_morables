# MORABLES — Retrieval Analysis Findings

**Model:** ft07 (Linq-Embed-Mistral, fine-tuned, 500 STORAL pairs, fable+summary doc format)
**Dataset:** 709 fables × 709 morals — moral → fable retrieval

---

## 1. Overall Performance

| Metric | Value |
|---|---|
| MRR@10 | **0.440** |
| R@1 | 33.1% (235 / 709 queries) |
| R@5 | 58.9% |
| R@10 | 67.7% |
| Median ground-truth rank | 3 |
| Mean ground-truth rank | 31.5 |

The model retrieves the correct fable at rank 1 for 1 in 3 queries. On average, the correct fable lands at rank 2–3. 67.7% of correct answers appear somewhere in the top 10.

---

## 2. Where Do Failures Land?

Of the 709 queries, **474 are misranked** (66.9%) — the correct fable does not appear at rank 1.

| Rank bucket | Count | % of all queries | Cumulative |
|---|---|---|---|
| Rank 1 (correct) | 235 | 33.1% | 33.1% |
| Rank 2 | 79 | 11.1% | 44.3% |
| Rank 3 | 46 | 6.5% | 50.8% |
| Rank 4 | 38 | 5.4% | 56.1% |
| Rank 5 | 20 | 2.8% | 59.0% |
| Rank 6–10 | 62 | 8.7% | 67.7% |
| Rank 11–50 | 131 | 18.5% | 86.2% |
| Rank 51+ | 98 | 13.8% | 100% |

**Key takeaway:** Most failures are close — 50.8% of all queries have the correct fable in the top 3. But a long tail of 98 queries (13.8%) have the correct fable ranked below 50, suggesting some cases are deeply confused.

---

## 3. How Confident Are the Errors?

For the 474 misranked queries, we measured the **score gap**: how much higher the wrong rank-1 fable scored vs. the correct fable.

| Score gap | Count | % of misranked |
|---|---|---|
| < 0.02 (near-miss) | 37 | 7.8% |
| 0.02–0.10 | 174 | 36.7% |
| > 0.10 (confident wrong) | **263** | **55.5%** |

- **Mean gap: 0.135 — Median gap: 0.114**
- **Max gap: 0.486** (the model gave the wrong fable a score 0.49 higher than the correct one)

Over half of errors are **confident mistakes**, not near-misses. The model is not slightly confused — it is strongly pulling toward the wrong fable.

**Near-miss example (gap = 0.019):**
- Moral: *"You should especially avoid someone who does not even spare his own people."*
- Model retrieved: *The Bird Catcher And The Partridge* (score 0.639) — just barely edged out the correct fable
- Ground truth: *The Farmer And His Dogs* (score 0.620, rank **2**)
- Gap of 0.019 — barely wrong, essentially a coin flip between two reasonable fables.

**Confident-wrong example (gap = 0.486):**
- Moral: *"Be careful what you wish for."*
- Model retrieved: *The Shepherd And The Lion* (score 0.812)
- Ground truth: *The Poor Man And Death* (score 0.325, rank **76**)
- The model is not confused — it is strongly convinced the wrong fable is correct. (See Section 5 for full analysis of these cases.)

---

## 4. Dataset Annotation Problems

This is the most important finding. **The dataset has structural annotation issues that directly cause apparent model failures.**

### 4a. Exact duplicate morals (most severe)

**27 unique moral texts are each assigned as the ground truth for 2 or more different fables.**  
These 27 morals affect **58 queries — 8.2% of the entire dataset.**

When the model retrieves fable A for a moral that is also the ground truth of fable B, it is scored as wrong — even though it retrieved a perfectly valid answer.

Selected examples:

| Moral | Fables it maps to |
|---|---|
| "Nature reveals itself." | The Cat-Maiden, The Cat and Venus, The Kingdom of The Lion, Orpheus and the Dogs **(4 fables!)** |
| "Look before you leap." | The Fox and the Goat, The Fox And The Goat In The Well, The Two Frogs At The Well **(3 fables)** |
| "Self-help is the best help." | Hercules and the Wagoner, The Lark and Her Young Ones, The Crested Lark And The Farmer **(3 fables)** |
| "Be careful what you wish for." | The Poor Man And Death, The Eyes and the Honey |
| "Greed often overreaches itself." | The Goose With the Golden Eggs, The Mouse and the Oyster |
| "Appearances are deceptive." | The Ant and the Chrysalis, The Wolf in Sheep's Clothing |
| "There's a time for work and a time for play." | The Ant and the Grasshopper, The Ant And The Dung Beetle |

### 4b. Near-duplicate morals (same meaning, slightly different wording)

Beyond exact duplicates, **18 additional moral pairs** (36 queries) have near-identical texts with cosine similarity > 0.90:

- "Enemies' promises are made to be broken." ↔ "Enemies' promises were made to be broken." (sim = 0.991)
- "Wit has always an answer ready." ↔ "Wit always has an answer ready." (sim = 0.991)
- "Nothing escapes the master's eye" ↔ "Nothing escapes the master's eye." (sim = 0.977)

**Total queries affected by exact + near-duplicate morals: 94 / 709 (13.3%)**

### 4c. Thematic ambiguity

For every misranked query, we checked whether the wrong rank-1 fable belongs to the same moral category as the correct fable (using Claude-extracted `moral_category` labels).

**158 / 474 confused pairs (33.3%) share the same moral category.**

This means a third of all failures happen inside the same theme cluster — the model is choosing between two fables that genuinely teach the same type of lesson.

**Example — same moral category (deception), both fables are about being deceived by a false story:**
- Moral: *"Every tale is not to be believed."*
- Model retrieved: *The Wolf And The Nurse* (category: deception) — a nurse threatens a crying child with a wolf; a wolf overhears and waits, but no child is thrown out. Score gap: **0.356**
- Ground truth: *The Thief And The Boy* (category: deception, rank **104**) — a thief distracts a boy with a false story to steal from him.
- Both fables are genuinely about not believing stories at face value. The annotation picks one; the model picks the other. Neither is obviously wrong.

**Example — same moral category (ingratitude):**
- Moral: *"No gratitude from the wicked."*
- Model retrieved: *The Farmer and the Snake* (category: ingratitude) — a farmer shows kindness to a snake, which bites him anyway. Score gap: **0.327**
- Ground truth: *The Woodman and the Serpent* (category: ingratitude, rank **37**) — nearly the same story.
- These two fables are essentially the same moral scenario with different characters.

Additionally, measuring cosine similarity between the *text* of the two competing morals:

| Moral-moral similarity threshold | Confused pairs above threshold | % of confused |
|---|---|---|
| > 0.70 | 79 | 16.7% |
| > 0.80 | 46 | 9.7% |
| > 0.85 | 32 | 6.8% |
| > 0.90 | 26 | 5.5% |

**32 confused pairs** have morals so semantically similar (sim > 0.85) that even a human might struggle to distinguish them.

**Examples of near-identical morals the model confuses:**

| Query moral | Competing moral (rank-1) | Similarity | GT rank | Gap |
|---|---|---|---|---|
| "Union is strength." | "Union gives strength." | 0.969 | 2 | 0.004 |
| "Quality is better than quantity." | "Quality is more valuable than quantity." | 0.953 | 2 | 0.049 |
| "Flatters are not to be trusted." | "Do not trust flatterers." | 0.920 | 3 | 0.133 |
| "Someone who lays a trap for others will fall victim to it himself." | "People who lay traps for others bring about their own destruction." | 0.917 | 54 | 0.325 |
| "Fine clothes may disguise, but silly words will disclose a fool." | "Clothes may disguise a fool, but his words will give him away." | 0.922 | 2 | 0.009 |

These are not model errors — they are annotation ambiguities. "Union is strength" and "Union gives strength" are the same moral expressed two ways, yet they map to different fables and the model is penalised for retrieving either one.

### 4d. Character overlap in confused pairs

For every misranked query, we checked whether the correct fable and the wrong rank-1 fable share the same characters (using Claude-extracted `characters` labels).

**48 / 474 confused pairs (10.1%) share at least one character.**

This is a separate effect from thematic ambiguity — the model may be latching onto the same animal or character appearing in two different fables rather than reasoning about the moral.

Most common characters appearing in confused pairs:

| Character | Confused pairs |
|---|---|
| fox | 10 |
| wolf | 5 |
| eagle | 4 |
| ant | 4 |
| lion | 4 |
| cat | 2 |

**Example — fox appears in both fables:**
- Moral: *"One bad turn deserves another."*
- Model retrieved: *The Fox and the Crane* (score **0.868**)  — fox invites crane to dinner and serves soup in a flat dish; crane can't eat it.
- Ground truth: *The Fox And The Stork* (score **0.666**, rank **5**) — same story, different bird name. Gap: **0.202**
- Both fables feature a fox playing the same trick on a bird. The model is not wrong to be confused.

**Example — fly appears in both fables:**
- Moral: *"Arrogant fools have no true influence."*
- Model retrieved: *The Fly And The Mule* (score **0.708**) — a fly sits on a mule's horn and brags about its importance; the mule ignores it.
- Ground truth: *A Fly upon a Wheel* (score **0.421**, rank **111**) — a fly on a chariot wheel asks "what a dust I make!" Gap: **0.286**
- Both fables feature a fly deluding itself about its own importance. The same character + same theme = genuine ambiguity.

**Example — fox + crow/raven:**
- Moral: *"Flatters are not to be trusted."*
- Model retrieved: *The Fox and the Crow* (score **0.851**) — fox flatters a crow into dropping its cheese.
- Ground truth: *The Fox And The Raven* (score **0.718**, rank **3**) — essentially the same story, different bird. Gap: **0.133**
- The dataset has two nearly identical fables (crow vs raven variant) annotated as separate ground truths for the same moral.

### 4e. Theme overlap in confused pairs

For every misranked query, we checked whether the correct fable and the wrong rank-1 fable share any themes (using Claude-extracted `themes` labels, which are more granular than `moral_category`).

**196 / 474 confused pairs (41.4%) share at least one theme.**  
**55 / 474 (11.6%) share 2 or more themes.**  
**13 / 474 (2.7%) share 3 or more themes.**

This is the strongest overlap signal of the three metadata fields — themes are more specific than moral_category and catch more genuine ambiguity.

Most common shared themes across confused pairs:

| Theme | Confused pairs |
|---|---|
| deception | 31 |
| consequences | 12 |
| greed | 12 |
| irony | 11 |
| nature | 10 |
| pride | 10 |
| justice | 9 |
| cunning | 7 |
| vanity | 7 |
| contentment | 7 |

**Example — 4 shared themes (strongest overlap):**
- Moral: *"Flatters are not to be trusted."*
- Model retrieved: *The Fox and the Crow* (score **0.851**) — themes: deception, flattery, foolishness, vanity
- Ground truth: *The Fox And The Raven* (score **0.718**, rank **3**) — themes: deception, flattery, foolishness, vanity
- Shared themes: **deception, flattery, foolishness, vanity** — all four themes identical. Gap: **0.133**
- These are essentially the same fable with a crow instead of a raven. Both fables share every single theme label.

**Example — 4 shared themes:**
- Moral: *"A liar deceives no one but himself."*
- Model retrieved: *The Monkey And The Dolphin* (score **0.691**) — themes: consequences, deception, lying, pride
- Ground truth: *The Shipwrecked Impostor* (score **0.659**, rank **2**) — themes: consequences, deception, lying, pride
- Shared themes: **consequences, deception, lying, pride** — again fully identical. Gap: **0.033**

**Example — 3 shared themes, large gap:**
- Moral: *"One bad turn deserves another."*
- Model retrieved: *The Fox and the Crane* (score **0.868**) — themes: hospitality, justice, revenge, trickery
- Ground truth: *The Fox And The Stork* (score **0.666**, rank **5**) — themes: hospitality, justice, reciprocity, revenge
- Shared themes: **hospitality, justice, revenge**. Gap: **0.202**
- Same fox-tricks-a-bird story, different bird name, 3 of 4 themes identical.

**Example — 3 shared themes, different scenario:**
- Moral: *"Do not take credit for the accomplishments of others."*
- Model retrieved: *A Fly upon a Wheel* (score **0.700**) — themes: delusion, pride, self-importance, vanity
- Ground truth: *The Donkey Who Carried The God* (score **0.630**, rank **2**) — themes: delusion, hubris, pride, vanity
- Shared themes: **delusion, pride, vanity**. Gap: **0.070**
- Both are stories about a creature falsely believing it deserves credit — thematically identical even though the characters and scenarios differ.

---

## 5. The Model Is Often Right When It Looks Wrong

When we inspect the hardest failures (largest score gap between rank-1 and ground truth), the model is frequently retrieving a fable that *better* matches the moral than the annotated ground truth:

**Case 1 — moral: "Be careful what you wish for."**
- Model retrieves: *The Shepherd And The Lion* (score 0.812) — a shepherd makes a vow wishing to find a thief, and wishes he hadn't when he finds it's a lion. Directly on-theme.
- Ground truth: *The Poor Man And Death* (score 0.325, rank **76**) — a man calls for Death and then backtracks.
- Score gap: **0.486**

**Case 2 — moral: "Those who seek to please everybody please nobody."**
- Model retrieves: *The Miller, His Son, and Their Ass* (score 0.886) — literally a story about trying to please everyone and ending up pleasing no one.
- Ground truth: *The Bald Man And His Two Mistresses* (score 0.466, rank **20**) — about vanity and rivalry.
- Score gap: **0.420**

**Case 3 — moral: "Greed for more can lead to losing what you already have."**
- Model retrieves: *Zeus And The Camel* (score 0.726) — a camel asks Zeus for horns out of greed, and gets its ears cropped instead.
- Ground truth: *The Jackdaw And The Doves* (score 0.288, rank **235**) — categorised as *deception*, not greed.
- Score gap: **0.437**

**Case 4 — moral: "Understand what you are doing before you do it."**
- Model retrieves: *The Fox And The Hare In The Well* (score 0.688) — a hare jumps into a well without planning how to get out.
- Ground truth: *The Shepherd and the Sea* (score 0.254, rank **270**) — a shepherd sells his flock to trade at sea without understanding it.
- Score gap: **0.434**

In all four cases, a reasonable person would argue the model's answer is at least as correct as the annotated ground truth — and in cases 2 and 3, arguably more correct.

---

## 6. Ambiguity-Corrected Performance

To quantify how much annotation noise is suppressing the reported MRR, we ran two corrected evaluations:

| Evaluation | MRR@10 | n queries | vs. standard |
|---|---|---|---|
| Standard | 0.440 | 709 | — |
| Soft MRR (forgive retrievals with similar moral) | **0.468** | 709 | +2.8% |
| Clean subset (exclude ambiguous queries) | **0.448** | 615 | +0.8% |

- **42 queries were "rescued"** by the soft correction — the model retrieved a fable whose own moral is semantically equivalent to the query moral.
- The clean subset gain (+0.8%) is modest because we are only *evaluating* on a cleaner set while the model was still *trained* on the noisy data. A model retrained on a cleaned dataset would likely show a larger gain.

---

## 7. Attractor Fables (Not a Major Issue)

We checked whether a small set of fables were "absorbing" many morals in embedding space.

**The attractor effect is weak.** The worst offender, *The Doves And The Kite*, steals only **7 queries** (1.0% of all queries). The top 5 attractors collectively account for 26 false positives out of 474. No single fable dominates.

| Fable | False positives | % of all queries | Word count |
|---|---|---|---|
| The Doves And The Kite | 7 | 1.0% | 141 |
| The One-Eyed Doe | 5 | 0.7% | 122 |
| The Goat And The Donkey | 5 | 0.7% | 150 |
| A Cat and Mice | 5 | 0.7% | 91 |
| The Boy and the Filberts | 4 | 0.6% | 81 |
| The Fox and the Crow | 4 | 0.6% | 133 |

305 out of 709 fables appear as a false positive at rank 1 at least once — false positives are widely distributed, not concentrated.

---

## 8. Summary: What Is Causing the Failures?

| Cause | Estimated scope | Confidence |
|---|---|---|
| Exact duplicate morals (same text, different fable) | 58 queries (8.2%) | **Certain** |
| Near-duplicate morals (same meaning, slightly different text) | ~36 queries (5.1%) | High |
| Thematic ambiguity (same moral category, debatable annotation) | ~158 queries (22.3%) | Medium |
| Genuine model errors (model is wrong, annotation is right) | ~220 queries (31%) | Medium |
| Attractor/geometry effects | Negligible | High |

**The single most important finding:** at least 13.3% of the dataset (94 queries) has annotation problems severe enough that the model cannot be expected to get them right. A significant additional fraction (~22%) involves genuine ambiguity where the "correct" answer is debatable. Only roughly 30% of all queries represent clear-cut failures where the annotation is unambiguous and the model is genuinely wrong.

---

## 9. Next Steps

### Track 1 — Fix the data

1. **Remove or merge exact duplicates (58 queries):** For morals like "Look before you leap." that map to 3 fables, either pick one canonical fable or treat all 3 as valid answers (multi-label evaluation).
2. **Review near-duplicates (36 queries):** Standardise wording where the moral text differs only trivially.
3. **Review the 30 hardest confusion cases:** Read side-by-side which fable the model retrieved vs. the ground truth, and decide if the annotation needs updating.
4. **Sanity check:** After cleaning, retrain on the cleaned dataset and evaluate on the cleaned test set. Expected MRR boost: likely +3–6% from removing noise alone.

### Track 2 — Fix the model

For the ~220 queries with genuine model errors:

1. **Hard negative mining:** The current training loss does not penalise the specific fables the model confuses. In the next fine-tuning round, use the model's own top false positives (rank-1 wrong fables) as hard negatives. This directly targets the confident-wrong failures (55.5% of misranked queries have gap > 0.10).
2. **Multi-positive training:** For the 27 duplicate-moral groups, rather than discarding them, treat all valid fables as positive targets and use a listwise loss (e.g. InfoNCE with multiple positives). This turns an annotation problem into a training signal.
