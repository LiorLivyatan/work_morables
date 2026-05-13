# Cluster Review Progress

## Overview

This document tracks all decisions made during the manual review of the MORABLES fable cluster dataset.
The dataset contains 709 fables, each with a one-sentence moral. Fables are grouped into clusters
where multiple fables share the same or semantically equivalent moral — enabling a retrieval model
to learn that one moral can be the "answer" to many different fable stories.

**Cluster types:**
- `exact` — all fables share identical moral text
- `near` — fables share the same underlying lesson with different phrasing
- `singleton` — a fable whose moral is unique in the corpus

**Final state after full review (Sections 1–4):**
- 563 total clusters: 58 near · 14 exact · 491 singletons
- Started from: 593 clusters (39 near · 16 exact · 538 singletons)
- 25 new near clusters created across all sections
- 2 existing near clusters split into sub-clusters
- 2 existing near clusters reclassified exact → near
- 23 morals corpus fixes applied

---

## Section 1 — Near Cluster Review (Session 1)

All 30 original near clusters were reviewed. Changes made where noted.

| # | Cluster ID | Changes Made |
|---|-----------|--------------|
| 1 | `appearances_deceptive` | Clean — no changes |
| 2 | `flatters_not_trusted` | Fixed "Flatters"→"Flatterers" (fable_0043, fable_0620); removed "and deceivers" from fable_0597 |
| 3 | `content_with_lot` | Removed fable_0260 → singleton; fixed fable_0318 moral: "Be content with your station" → "Be content with your lot" |
| 4 | `careful_wish` | Fixed fable_0431: removed redundant clause "; your wish may be granted" |
| 5 | `enemies_promises` | Fixed fable_0082: "were made to be broken" → "are made to be broken" |
| 6 | `union_strength` | Added missing period to fable_0487 |
| 7 | `nothing_escapes_master` | Added missing period to fable_0055 |
| 8 | `lay_trap` | Clean — no changes |
| 9 | `greed_overreaches` | Removed fable_0565 → singleton; moved fable_0305 → greed_misery |
| 10 | `grasp_for_more` | Clean — later restructured in Section 3B |
| 11 | `bird_in_hand` | Clean — no changes |
| 12 | `dont_trust_wicked` | Rewrote fable_0121 moral to be more faithful to the fable; moved fable_0473 → new cluster support_becomes_enemy |
| 13 | `ungrateful_wicked` | Clean — no changes |
| 14 | `please_nobody` | Clean — no changes |
| 15 | `wit_answer` | Fixed fable_0009: "Wit has always an answer ready" → "Wit always has an answer ready" |
| 16 | `be_yourself` | Added missing period to fable_0530 |
| 17 | `innate_nature_unchangeable` | Clean — later expanded in Section 4B |
| 18 | `same_worse_off` | Clean — no changes |
| 19 | `true_friends_hard` | Clean — no changes |
| 20 | `no_wealth_liberty` | Clean — no changes |
| 21 | `old_friends` | Split: kept fable_0049+0595 in old_friends; formed new cluster friends_vs_foes (fable_0312, fable_0194) |
| 22 | `deeds_not_words` | Clean — no changes |
| 23 | `not_judged_appearance` | Clean — no changes |
| 24 | `obscurity_safety` | Resolved in Section 3A: fable_0358 removed → singleton (moral too verbose); approved with fable_0105 + fable_0549 |
| 25 | `what_you_do` | Clean — no changes |
| 26 | `vices_punishment` | Resolved in Section 3A: approved as-is |
| 27 | `promises_quick` | Result of earlier split; clean |
| 28 | `promises_cant_deliver` | Clean — no changes |
| 29 | `dont_count_chickens` | Resolved in Section 3A: approved as-is |
| 30 | `poor_life_safer` | Restructured in Section 3B: see below |

### New Clusters Created During Section 1 Review

**`friends_vs_foes`** — Split from old_friends
- fable_0312 *The Wolves, The Sheep And The Dogs* — "Do not give up friends for foes."
- fable_0194 *The Bees And The Beekeeper* — "Don't take your friends for foes."

**`support_becomes_enemy`** — fable_0473 moved from dont_trust_wicked; moral rewritten
- fable_0465 *The Thief And The Lamp* — "Sometimes, those you support become your worst enemy."
- fable_0473 *The Shepherd And The Wolf Cub* — "Sometimes, those you support become your worst enemy."

**`promises_cant_deliver`** — Split from promises_quick; fable_0662 added from singleton audit
- fable_0480 *The Witch On Trial* — "People often make extravagant promises which they are completely unable to carry out."
- fable_0413 *The Mountain In Labour* — "Don't promise big things if you can't deliver."
- fable_0662 *A Hunts-man and a Currier* — "Don't promise more than you can deliver."

**`quality_over_quantity`** — Created during singleton audit
- fable_0111 *The Vixen and the Lioness* — "Quality is better than quantity."
- fable_0377 *The Sow And The Lioness* — "Quality is more valuable than quantity."

**`dont_attempt_impossible`** — Created during singleton audit
- fable_0183 *The Frog And The Ox* — "Do not attempt the impossible."
- fable_0292 *The Wolves And The Hides* — "Do not try to do impossible things."
- fable_0264 *The Donkey And The Cricket* — "You must not act unnaturally, trying to achieve some impossible thing."

**`disguise_reveals_fool`** — Created during singleton audit
- fable_0008 *The Ass in the Lion's Skin* — "Fine clothes may disguise, but silly words will disclose a fool."
- fable_0316 *The Fox, The Donkey And The Lion Skin* — "Clothes may disguise a fool, but his words will give him away."

**`kindness_never_wasted`** — Created during singleton audit
- fable_0430 *The Lion And The Mouse* — "No act of kindness, no matter how small, is ever wasted."
- fable_0560 *The Magic Colours* — "Kindness is never wasted."

**`avoid_lesser_fall_greater`** — Created during singleton audit
- fable_0281 *The Deer And The Lion* — "In avoiding a lesser danger, one may inadvertently fall into a greater peril."
- fable_0640 *The Fishes and the Frying-Pan* — "When we are avoiding present dangers, we should not fall into even worse peril."
- fable_0552 *The Frogs asking for a King* — "Better to endure a small hardship than risk a greater one."

**`overconfidence_danger`** — Created during singleton audit
- fable_0330 *The Lion, The Rooster And The Donkey* — "False confidence often leads into danger."
- fable_0504 *The Rooster And The Cats* — "Overconfidence often leads people into danger."

**`dog_in_manger`** — Created during singleton audit
- fable_0030 *The Dog in the Manger* — "People often grudge others what they cannot enjoy themselves."
- fable_0483 *The Dog In The Manger* — "Some begrudge others what they cannot enjoy themselves."

---

## Section 2 — Morals Corpus Fixes

All fixes include the original text preserved in an `original_text` field for traceability.

| Fable | Title | Original | Fixed | Reason |
|-------|-------|----------|-------|--------|
| fable_0043 | *The Fox And The Raven* | "Flatters are not to be trusted." | "Flatterers are not to be trusted." | Typo |
| fable_0620 | *The Fox and the Ground-Bird* | "Flatters are not to be trusted." | "Flatterers are not to be trusted." | Typo |
| fable_0597 | *The Wolf, the Fox and the Gift* | "Flatterers and deceivers are not to be trusted." | "Flatterers are not to be trusted." | Cluster alignment |
| fable_0082 | *The Nurse and the Wolf* | "Enemies' promises were made to be broken." | "Enemies' promises are made to be broken." | Tense |
| fable_0431 | *The Shepherd And The Lion* | "Be careful what you wish for; your wish may be granted." | "Be careful what you wish for." | Redundant clause |
| fable_0318 | *The Donkey, The Horse And The War* | "Be content with your station." | "Be content with your lot." | Cluster alignment |
| fable_0487 | *The Dogs And Their Commander* | "...slavish thing" (no period) | "...slavish thing." | Missing period |
| fable_0055 | *The Hart in the Ox-Stall* | "Nothing escapes the master's eye" (no period) | "Nothing escapes the master's eye." | Missing period |
| fable_0305 | *The Dog And The Treasure* | "Greed can lead to ruin." | "A life of greed and hoarding wealth leads to misery." | Cluster alignment |
| fable_0121 | *The Abbot And The Flea* | "Do not trust the wicked." | "Do not ever trust your enemy. Always protect yourself from him, even if he comes to you humble and supplicating." | More faithful to fable |
| fable_0473 | *The Shepherd And The Wolf Cub* | "False men cannot be trusted." | "Sometimes, those you support become your worst enemy." | Moved to new cluster; moral rewritten |
| fable_0009 | *The Ass's Brains* | "Wit has always an answer ready." | "Wit always has an answer ready." | Word order |
| fable_0530 | *The Farmers, The Donkey And The Lion Skin* | "...not" (no period) | "...not." | Missing period |
| fable_0016 | *The Belly and the Members* | "...common good" (no period) | "...common good." | Missing period |
| fable_0064 | *The Labourer and the Nightingale* | "...lost forever" (no period) | "...lost forever." | Missing period |
| fable_0089 | *The Scorpion and the Frog* | "...own interests" (no period) | "...own interests." | Missing period |
| fable_0164 | *The Fox And The Cat* | "...useless tricks" (no period) | "...useless tricks." | Missing period |
| fable_0203 | *The Decoys And The Doves* | "...over kinship" (no period) | "...over kinship." | Missing period |
| fable_0303 | *The Dog And The Shellfish* | "...unsuspected danger" (no period) | "...unsuspected danger." | Missing period |
| fable_0430 | *The Lion And The Mouse* | "...ever wasted" (no period) | "...ever wasted." | Missing period |
| fable_0472 | *The Sheep, The Shepherd And His Cloak* | "...surroundings" (no period) | "...surroundings." | Missing period |
| fable_0489 | *Aesop And His Ugly Mistress* | "...be dangerous" (no period) | "...be dangerous." | Missing period |
| fable_0567 | *The boy and the tripe* | "Avoid overindulgence" (no period) | "Avoid overindulgence." | Missing period |

---

## Section 3 — Team Review Decisions (Session 2)

### 3A — Pending Near Clusters (resolved)

Three clusters were deferred during Session 1.

**`obscurity_safety`** — fable_0358 removed (moral too verbose); cluster approved with 2 fables:
- fable_0105 *The Tree and the Reed* — "Obscurity often brings safety."
- fable_0549 *The Battle of the Mice and the Weasels* — "Humble commonalty easily finds safety in obscurity."
- fable_0358 *The Fisherman And The Fish* — removed → singleton: "To be small is a way to stay safe and avoid problems, whereas you rarely see a man with a big reputation who is able to keep out of danger."

**`vices_punishment`** — approved as-is:
- fable_0010 *Avaricious and Envious* — "Vices are their own punishment."
- fable_0116 *The Wolf in Sheep's Clothing* — "Seek to harm and harm shall find you."
- fable_0197 *Zeus And The Bee* — "Evil wishes, like chickens, come home to roost."

**`dont_count_chickens`** — approved as-is:
- fable_0077 *The Milk-Woman and Her Pail* — "Do not count your chickens before they are hatched."
- fable_0334 *The Piece Of Driftwood* — "Our mere anticipations of life outrun its realities."
- fable_0185 *The Bald Men And The Comb* — "Expectations can lead to disappointment when reality does not match our desires."

---

### 3B — Singleton → Existing Cluster (26 candidates)

Each singleton was reviewed against a proposed target cluster. Similarity scores are cosine similarity on moral embeddings.

#### Added to existing clusters

**✅ fable_0405** *The Hen And The Eggs* → `dont_trust_wicked` (sim 0.895)
Moral: "We should never put our trust in a wicked man, even if he seems to be completely innocuous."
Reason: Direct fit

**✅ fable_0344** *The Farmer, The Wolf And The Plow* → `dont_trust_wicked` (sim 0.885)
Moral: "People with a history of wrongdoing are often not trusted."
Reason: Same lesson, consequence angle

**✅ fable_0420** *The Lion And The Unicorn* → `dont_trust_wicked` (sim 0.872)
Moral: "Do not ever trust your enemy. Always protect yourself from him, even if he comes to you humble and supplicating."
Reason: Identical moral text to fable_0121

**✅ fable_0261** *The Crane And The Crow* → `dont_trust_wicked` (sim 0.861)
Moral: "Don't be seduced by deceptive words, they will cause nothing but trouble."
Reason: Deceptive words = wicked people deceiving you

**✅ fable_0255** *The Cobbler And The King* → `dont_trust_wicked` (sim 0.861)
Moral: "Beware of swindlers, as they often exploit the gullibility of others."
Reason: Swindlers = wicked people; beware = don't trust

**✅ fable_0297** *The Mother Dog And Her Puppies* → `dont_trust_wicked` (sim 0.859)
Moral: "There is danger lurking in the words of a wicked person."
Reason: Danger in wicked person's words

**✅ fable_0229** *The Lion, The Bull And His Horns* → `dont_trust_wicked` (sim 0.857)
Moral: "Believing your enemies makes you fall victim to their tricks and gets you into trouble."
Reason: Fits better here than proposed support_becomes_enemy

**✅ fable_0571** *The Jackdaw and The Fox* → `be_yourself` (sim 0.863)
Moral: "Don't deceive yourself."
Reason: Self-deception is part of the authenticity theme

**✅ fable_0582** *The Wolf and the Sheep in the Sheepfold* → `avoid_lesser_fall_greater` (sim 0.867)
Moral: "Better to face a known danger than to trust a deceitful promise of safety."
Reason: Fits better here than proposed overconfidence_danger

#### Kept as singletons

**❌ fable_0486** *The Wolf And The Nurse* (proposed → `dont_trust_wicked`, sim 0.893)
Moral: "Do not believe everything you hear."
Reason: Too general — requires no wicked actor; different from the cluster's specific warning about wicked people

**❌ fable_0243** *The Fox And The Stork* (proposed → `one_good_turn`, sim 0.921)
Moral: "One bad turn deserves another."
Reason: Inverse moral — reciprocating harm ≠ reciprocating kindness; different lesson

**❌ fable_0101** *The Tortoise and the Birds* (proposed → `friends_vs_foes`, sim 0.895)
Moral: "Never put yourself in your enemy's clutches."
Reason: Caution lesson, not about friend/foe distinction

**❌ fable_0664** *A Husband and Wife twice Married* (proposed → `friends_vs_foes`, sim 0.871)
Moral: "Do not fight with those who are perfectly capable of fighting back."
Reason: Power-dynamics lesson — later formed its own cluster dont_fight_stronger

**❌ fable_0168** *The Ax And The Trees* (proposed → `friends_vs_foes`, sim 0.857)
Moral: "You should think twice before offering anything to your enemies."
Reason: Different nuance — about giving to enemies, not about friend/foe distinction

**❌ fable_0260** *The Crab On Dry Land* (proposed → `content_with_lot`, sim 0.871)
Moral: "Contentment with our lot is an element of happiness."
Reason: Previously removed from this cluster; kept out

**❌ fable_0482** *The Ants And The Pigs* (proposed → `greed_misery`, sim 0.876)
Moral: "Even those who work hard may find their material wealth undone by the greed of others."
Reason: Being victimized by others' greed ≠ your own greed causing your own misery

**❌ fable_0661** *A Horse and a Hog* (proposed → `poor_life_safer`, sim 0.876)
Moral: "It is more honorable to die having achieved noble deeds than to live a life of vice."
Reason: Honorable death > vicious life is a different lesson from poverty being physically safer

**❌ fable_0258** *The Coward And The Lion Of Gold* (proposed → `poor_life_safer`, sim 0.855)
Moral: "Wealth is of little value if one is too afraid to use or enjoy it."
Reason: About cowardice with wealth — different angle from the cluster

**❌ fable_0559** *The Gardener and His Master* (proposed → `grasp_for_more`, sim 0.870)
Moral: "Don't go to extremes."
Reason: General moderation ≠ greed/overreach

**❌ fable_0601** *Truth and The Traveler* (proposed → `support_becomes_enemy`, sim 0.869)
Moral: "Falsehood is with all men."
Reason: False match — pervasive lying has nothing to do with allies becoming enemies

**❌ fable_0539** *Augustus And The Murder* (proposed → `every_tale`, sim 0.861)
Moral: "It is dangerous to believe a story, and dangerous not to believe it."
Reason: A paradox — distinct from the cluster's simple "don't believe every tale"

**❌ fable_0282** *The Deer And The Vine* (proposed → `dont_trust_wicked`, sim 0.857)
Moral: "Do not harm your benefactors."
Reason: Wrong direction — about gratitude, not about trusting wicked people

**❌ fable_0622** *The River Fish and The Sea Fish* (proposed → `not_judged_appearance`, sim 0.853)
Moral: "Value is what others think of you, not what you think of yourself."
Reason: Self-perception vs external perception ≠ actions vs appearance

**❌ fable_0262** *The Crane And The Peacock* (proposed → `quality_over_quantity`, sim 0.851)
Moral: "The useful is of much more importance and value, than the ornamental."
Reason: Useful vs ornamental ≠ quality vs quantity

**❌ fable_0307** *The Wolf And The Dog In Pursuit* (proposed → `not_judged_appearance`, sim 0.850)
Moral: "Do not take pride in the good qualities that actually belong to someone else."
Reason: False self-credit ≠ being judged by appearance

**❌ fable_0526** *The Goatherd And The Goat* (proposed → `dont_attempt_impossible`, sim 0.862)
Moral: "Do not attempt to hide things which cannot be hid."
Reason: Hiding the unhideable is too specific; different nuance from general impossibility

#### Cluster restructuring done during 3B

**`poor_life_safer` split:**
- Removed fable_0220 *The Fir Tree And The Bramble Bush* — "Better poverty without care, than riches with."
- Removed fable_0252 *The City Mouse And The Country Mouse* — "It is better to live in self-sufficient poverty than to be tormented by the worries of wealth."
- → Formed new cluster **`poverty_beats_wealth`** (wealth brings anxiety; poverty brings peace of mind)
- Removed fable_0532 *The Fox And The Lion Hunting* — "It is better to serve in safety than to rule in peril." → singleton
- Kept in poor_life_safer: fable_0447, fable_0592, fable_0631 (physical safety of humble life)

**`grasp_for_more` split:**
- New cluster **`lose_what_you_have`** (7 fables) — greed makes you lose what you already had
- `grasp_for_more` kept with 3 fables (fable_0019, 0350, 0508) — don't overreach your capacity
- fable_0515 *Mercury And The Two Women* — "Do not ask for more than you deserve." → singleton (entitlement angle, different from both)

**`lose_what_you_have`** — the 7 fables:
- fable_0181 *The Man And The Golden Eggs* — "People often grasp for more than they need and thus lose the little they have."
- fable_0235 *Zeus And The Camel* — "People who grasp for more than they need are deprived of what they have."
- fable_0279 *The Lion, The Hare And The Deer* — "Those who are not content with what they have may end up losing it while pursuing greater desires."
- fable_0332 *The Jackdaw And The Doves* — "Greed for more can lead to losing what you already have."
- fable_0404 *The Widow And Her Hen* — "People who grasp at more than they need lose the little that they held in their hands."
- fable_0298 *The Dog, The Meat And The Reflection* — "Do not grasp at more than you need."
- fable_0347 *The Kite And The Partridges* — "Do not try to grasp too much at once."

**Type corrections:** `neither_one_thing` and `greed_misery` reclassified `exact` → `near` (each gained a second distinct moral phrasing after earlier additions).

---

### 3C — New Cluster Candidates from Singleton Pairs (24 candidates)

Singleton pairs with cosine similarity ≥ 0.85 were reviewed. 11 clusters created, 13 kept as singletons.

#### Clusters created

**✅ `repay_kindness`** (sim 0.898) — All about reciprocating kindness/favours
- fable_0250 *The Eagle And The Farmer* — "Those who do you a favour you must repay kind."
- fable_0367 *The Man, The Mare And The Foal* — "We should do favours for someone who can do us a good deed in return."
- fable_0247 *The Two Men, The Eagle And The Fox* — "Favor those who do you kindness."
- fable_0146 *The Shepherd And The Lion* — "Return favors and favors will be bestowed on you."

**✅ `think_before_speak`** (sim 0.888) — Near-identical lesson
- fable_0382 *The Fox And The Partridge* — "Stay vigilant and think before you speak."
- fable_0626 *A Country-man and a Hawk* — "It's wise to think before we speak."

**✅ `familiarity_reduces_fear`** (sim 0.884) — Familiarity reduces terror of threatening things
- fable_0234 *The Camel And The People* — "Familiarity mollifies even the most terrifying things."
- fable_0385 *The Fox And The Lion* — "Familiarity makes it easy to confront even frightening situations."
- fable_0467 *The Rich Man And The Tanner* — "Familiarity can alleviate seemingly intractable problems."

**✅ `dont_fight_stronger`** (sim 0.883) — Don't pick fights with stronger opponents
- fable_0664 *A Husband and Wife twice Married* — "Do not fight with those who are perfectly capable of fighting back."
- fable_0688 *A Bull and a Ram* — "We should not pick fights with people stronger than we are."

**✅ `know_your_enemies`** (sim 0.883) — Know / pick / watch your enemies
- fable_0562 *The Cat and The Birds* — "Know your enemies."
- fable_0619 *The Crow and The Sheep* — "Pick your enemies."
- fable_0590 *The Sun and the Frogs* — "Watch the actions of your enemy."

**✅ `dont_rival_superiors`** (sim 0.870) — Competing with superiors destroys you
- fable_0290 *The Dog And The Lion* — "It is a foolish man who wants to rival his superiors."
- fable_0342 *The Earthworm And The Snake* — "He who competes with his superiors destroys himself before he can equal them."

**✅ `prepare_for_war`** (sim 0.865) — Preparedness guarantees peace
- fable_0214 *The Fox, The Boar And His Tusks* — "Preparedness for war is the best guarantee of peace."
- fable_0678 *A Wolf and a Porcupine* — "No one can be safe in peace unless they are always ready to face an enemy in case of war."

**✅ `virtue_by_necessity`** (sim 0.864) — People abandon vices only when forced to
- fable_0663 *A Hermit and a Soldier* — "Many people renounce wicked activities only because they are prevented from conducting them any longer."
- fable_0683 *An Old Man resolv'd to give over Whoring* — "Many people give up their vices not out of a love for virtue, but because they can no longer continue indulging in them."

**✅ `weak_can_avenge`** (sim 0.860) — Powerful are unwise to provoke the weak
- fable_0163 *The Eagle And The Fox* — "Even a high and mighty person should beware of his inferiors; their ingenuity can find a way to take revenge."
- fable_0690 *An Eagle and Rabbets* — "It is very unwise, even for the greatest of men, to provoke the least, as they may find a way to enact revenge."

**✅ `false_friend_worst`** (sim 0.857) — Hidden enemy worse than open enemy
- fable_0687 *A Sheep-Biter Hang'd* — "An enemy in disguise is much more unforgivable than an open enemy."
- fable_0695 *A Fox Praising Hare's Flesh* — "A false friend is the worst kind of enemy."

**✅ `company_you_keep`** (sim 0.851) — You are judged by the company you keep
- fable_0327 *The Man And The New Donkey* — "A man is known by the company he keeps."
- fable_0525 *The Stork And The Cranes* — "You are judged by the company you keep."

#### Kept as singletons

**❌ proposed `dont_blame_others`** (sim 0.924)
- fable_0311 *The Hunting Dog And The Watch Dog* — "Do not blame others for the circumstances of their upbringing."
- fable_0468 *The Rivers And The Sea* — "Do not blame others."
Reason: "Don't blame upbringing" ≠ "don't blame others" — different scope

**❌ proposed `dont_boast`** (sim 0.907)
- fable_0271 *The Crow, The Eagle And The Feathers* — "Do not boast to have something you do not."
- fable_0598 *The Lamp* — "Do not boast lest you be taken down."
Reason: Different nuances of boasting — false boasting vs prideful boasting

**❌ proposed `friends_in_adversity`** (sim 0.900)
- fable_0528 *The Kite And His Mother* — "We must make friends in prosperity if we would have their help in adversity."
- fable_0702 *A Lion and a Hog* — "Seek the friendship of those who do not withdraw from offering help even in a time of adversity."
Reason: Different angles on friendship in adversity

**❌ proposed `tyrant_excuses`** (sim 0.894)
- fable_0238 *The Cat And The Rooster* — "Tyrants need no excuse."
- fable_0545 *The Wolf and the Lamb* — "Any excuse will serve a tyrant."
Reason: Close but kept distinct

**❌ proposed `liars`** (sim 0.885)
- fable_0094 *The Shepherd's Boy and the Wolf* — "There is no believing a liar, even when he speaks the truth."
- fable_0095 *The Shipwrecked Impostor* — "A liar deceives no one but himself."
Reason: Listener losing trust ≠ liar self-deceiving — different lessons

**❌ proposed `vanity_downfall`** (sim 0.876)
- fable_0267 *The Birds, The Peacock And His Feathers* — "Vanity can lead to self-destruction."
- fable_0406 *Hermes And The Statues* — "Vanity can blind one to their true worth in the eyes of others."
Reason: Too varied across proposed members

**❌ proposed `beauty_subjective`** (sim 0.869)
- fable_0221 *The Children And The Mirror* — "Inner beauty is better than outer beauty."
- fable_0500 *The Beauty Contest Of The Animals* — "Beauty is in the eyes of the beholder."
Reason: Inner beauty > outer beauty ≠ beauty is in the eye of the beholder

**❌ proposed `enemy_offers_help`** (sim 0.866)
- fable_0399 *The Halcyon And The Sea* — "Sometimes, the greatest threats come from those you trust."
- fable_0517 *The Sow And The Wolf* — "An enemy is most dangerous when offering you help."
Reason: Different angles — misplaced trust vs enemy posing as helper

**❌ proposed `deceivers_retaliate`** (sim 0.863)
- fable_0380 *The Fox, The Monkey And His Ancestors* — "Deceivers often exploit the inability of others to challenge their falsehoods."
- fable_0429 *The Monkeys And The Two Men* — "Those who thrive on deception often retaliate against those who speak the truth."
Reason: Exploiting inability to challenge ≠ retaliating against truth-tellers

**❌ proposed `gratitude_rewarded`** (sim 0.860)
- fable_0139 *The Nut Tree And The People* — "Gratitude should be shown through kindness, not harm."
- fable_0533 *The Snake, The Eagle And The Farmer* — "The man who treats others well is rewarded by gratitude."
Reason: Opposite directions — ingratitude causing harm vs gratitude being rewarded

**❌ proposed `mind_own_business`** (sim 0.859)
- fable_0100 *The Three Tradesmen* — "Every man for himself."
- fable_0423 *The Seagull And The Kite* — "Every man should be content to mind his own business."
Reason: Self-interest ≠ don't meddle — different lessons

**❌ proposed `unequal_alliance`** (sim 0.859)
- fable_0257 *The Lion, The Cow, The She-Goat And The Sheep* — "An alliance made with the high and mighty can never be trusted."
- fable_0641 *A League of Beasts and Fishes* — "An alliance with the powerless is futile."
Reason: Powerful partner betraying you ≠ weak partner being useless — opposite angles

**❌ proposed `superior_insulted_by_inferiors`** (sim 0.857)
- fable_0132 *Aesop And The Shipbuilders* — "People are asking for trouble if they make fun of someone who is better than they are."
- fable_0543 *The Horse And The Goats* — "People with excellent qualities are often insulted by their inferiors."
Reason: Insulter's POV ≠ insulted's POV — different perspectives

---

## Section 4 — Manual Reading Candidates (Session 2)

Found by reading all singleton morals directly, without similarity scores.

### 4A — New Clusters

Seven proposed; three created.

**✅ `learn_from_misfortunes`** — Tight theme, three clear phrasings of the same lesson
- fable_0068 *The Lion, the Fox, and the Ass* — "Happy is the man who learns from the misfortunes of others."
- fable_0248 *The Cicada And The Fox* — "A discerning person is made wise by the misfortunes of his neighbours."
- fable_0379 *The Fox, The Lion And The Footprints* — "Take warning from the misfortunes of others."

**✅ `slow_steady_wins`** — Classic pair — same lesson
- fable_0027 *The Crow and the Pitcher* — "Little by little does the trick."
- fable_0371 *The Tortoise And The Hare* — "Slow and steady wins the race."

**✅ `pretenders_found_out`** — 5 fables, all about pretenders and hypocrites being exposed
- fable_0174 *The Monkey And The Dolphin* — "Those who pretend to be what they are not, sooner or later, find themselves in deep water."
- fable_0196 *The Bees, The Drones And The Wasps* — "Pretenders will be found out."
- fable_0273 *The Travellers And The Crow* — "Those who assume a character which does not belong to them, only make themselves ridiculous."
- fable_0422 *The She-Goat, The Kid And The Wolf* — "A hypocrite can usually be found out."
- fable_0424 *The Sheep And The Injured Wolf* — "Hypocritical speeches are easily seen through."

**❌ proposed `equals_best_friends`** — Not approved — kept as singletons
- fable_0107 *The Two Pots* — "Equals make the best friends."
- fable_0108 *The Two Pots* — "The strong and the weak cannot keep company."

**❌ proposed `brave_in_deeds_not_words`** — Not approved — kept as singletons
- fable_0061 *The Hunter and the Woodman* — "The hero is brave in deeds as well as words."
- fable_0469 *The Two Soldiers And The Robber* — "Courage in words means nothing if you flee in danger."
- fable_0523 *The Sheep, The Goat And The Sow* — "It is easy to be brave when there is no danger."

**❌ proposed `know_thyself_first`** — Not approved — kept as singletons
- fable_0087 *The Quack Frog* — "Those who would mend others, should first mend themselves."
- fable_0550 *Of the Vices of Men* — "We are not able to see our own faults: but as soon as others make a slip, we are ready to censure."
- fable_0596 *The Wolf and The Fox* — "Know yourself and your limits."

**❌ proposed `honesty_best_policy`** — Not approved — kept as singletons
- fable_0171 *The Snake And The Crab* — "Be frank and open in your dealings."
- fable_0408 *The Man, Hermes And The Axes* — "Honesty is the best policy."

### 4B — Expansions: Singletons Added to Existing Clusters

#### `innate_nature_unchangeable` (was 2 fables, now 4)

**✅ Added** fable_0028 *The Dancing Monkeys* — "Men often revert to their natural instincts."
Reason: Monkeys trained to dance revert to animal behaviour when nuts are thrown — nature beats nurture

**✅ Added** fable_0088 *The Raven and the Swan* — "Change of habit cannot alter nature."
Reason: Raven bathes constantly trying to turn white; his colour never changes

**❌ Singleton** fable_0090 *The Scorpion and the Ladybug* — "Regardless of our wishes, or even our intent, it is to our Nature alone that we will be faithful."
Reason: Team felt this was not clearly related to the cluster

**❌ New cluster** fable_0147 *Zeus And Prometheus* — "Nature and instincts can persist despite changes in form or appearance."
Reason: Zeus-transformation theme — paired with fable_0152 to form `nature_survives_transformation`

**❌ New cluster** fable_0152 *Zeus And The Ant* — "When someone with a wicked nature changes his appearance, his behaviour remains the same."
Reason: Zeus-transformation theme — paired with fable_0147 to form `nature_survives_transformation`

**❌ Singleton** fable_0165 *The Shepherd, The Wolf Cub And The Wolf* — "A wicked nature does not produce a good character."
Reason: Wolf raised as dog still reverts — nature-vs-nurture story, but kept as singleton

**❌ Singleton** fable_0224 *The Bull And The Calf* — "Some people just won't change."
Reason: Stubbornness is a choice; different from nature being inherently immutable

**New cluster formed from this review:**

**✅ `nature_survives_transformation`** — both fables involve Zeus physically transforming a creature; nature survives the divine transformation
- fable_0147 *Zeus And Prometheus* — "Nature and instincts can persist despite changes in form or appearance."
- fable_0152 *Zeus And The Ant* — "When someone with a wicked nature changes his appearance, his behaviour remains the same."

#### `deeds_not_words` (unchanged — all 3 candidates kept as singletons)

**❌ Singleton** fable_0177 *The Boastful Athlete* — "Talking is a waste of time when you can simply provide a demonstration."
Reason: About demonstrating skill specifically — not about judging character by deeds

**❌ Singleton** fable_0280 *The Deer And Her Friends* — "Good will is worth nothing unless it is accompanied by good acts."
Reason: Intentions vs acts is a subtly different lesson

**❌ Singleton** fable_0612 *The Jar goes to Court* — "Action is more valuable than arguments."
Reason: Kept as singleton

#### `company_you_keep` (was 2 fables, now 3)

**✅ Added** fable_0035 *The Farmer and the Stork* — "Birds of a feather flock together."
Reason: The story is literally about being punished by association — a stork caught in a net with thieving cranes is sentenced with them.
