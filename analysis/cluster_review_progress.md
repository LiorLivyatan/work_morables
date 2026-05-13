# Cluster Review Progress

## Overview

This document tracks all decisions made during the manual review of the MORABLES fable cluster dataset.
The dataset contains 709 fables, each with a one-sentence moral. Fables are grouped into clusters
where multiple fables share the same or semantically equivalent moral — enabling a retrieval model
to learn that one moral can be the "answer" to many different fable stories.

**Cluster types:**
- `exact` — all fables in the cluster share identical moral text
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

All 30 original near clusters were reviewed. Decisions below.

### Approved ✅

| # | Cluster ID | Changes Made |
|---|-----------|--------------|
| 1 | `appearances_deceptive` | Clean — no changes |
| 2 | `flatters_not_trusted` | Fixed "Flatters"→"Flatterers" typo (fable_0043, 0620); removed "and deceivers" from fable_0597 |
| 3 | `content_with_lot` | Removed fable_0260 → singleton; fixed fable_0318 "station"→"lot" |
| 4 | `careful_wish` | Fixed fable_0431 (removed "; your wish may be granted") |
| 5 | `enemies_promises` | Fixed fable_0082 "were"→"are" |
| 6 | `union_strength` | Added missing period to fable_0487 |
| 7 | `nothing_escapes_master` | Added missing period to fable_0055 |
| 8 | `lay_trap` | Clean — no changes |
| 9 | `greed_overreaches` | Removed fable_0565 → singleton; moved fable_0305 → `greed_misery` |
| 10 | `grasp_for_more` | Clean — later restructured in Section 3B (see below) |
| 11 | `bird_in_hand` | Clean — no changes |
| 12 | `dont_trust_wicked` | Changed fable_0121 moral; moved fable_0473 → new cluster `support_becomes_enemy` |
| 13 | `ungrateful_wicked` | Clean — no changes |
| 14 | `please_nobody` | Clean — no changes |
| 15 | `wit_answer` | Fixed fable_0009 word order ("has always"→"always has") |
| 16 | `be_yourself` | Added missing period to fable_0530 |
| 17 | `innate_nature_unchangeable` | Clean — no changes |
| 18 | `same_worse_off` | Clean — no changes |
| 19 | `true_friends_hard` | Clean — no changes |
| 20 | `no_wealth_liberty` | Clean — no changes |
| 21 | `old_friends` | Split into `old_friends` (fable_0049, 0595) + `friends_vs_foes` (fable_0312, 0194) |
| 22 | `deeds_not_words` | Clean — no changes |
| 23 | `not_judged_appearance` | Clean — no changes |
| 24 | `obscurity_safety` | **Resolved in Section 3A:** fable_0358 removed → singleton; approved with fable_0105 + fable_0549 |
| 25 | `what_you_do` | Clean — no changes |
| 26 | `vices_punishment` | **Resolved in Section 3A:** approved as-is (all 3 fables) |
| 27 | `promises_quick` | Result of earlier split; clean (fable_0320, 0521) |
| 28 | `promises_cant_deliver` | Clean — no changes |
| 29 | `dont_count_chickens` | **Resolved in Section 3A:** approved as-is (all 3 fables) |
| 30 | `poor_life_safer` | **Restructured in Section 3B:** see below |

### New Clusters Created During Section 1 Review

| Cluster ID | Fables | Rationale |
|-----------|--------|-----------|
| `friends_vs_foes` | fable_0312, fable_0194 | Split from `old_friends` — different lesson |
| `support_becomes_enemy` | fable_0465, fable_0473 | fable_0473 moved from `dont_trust_wicked` |
| `promises_cant_deliver` | fable_0480, fable_0413, fable_0662 | Split from `promises_quick`; fable_0662 added from singleton audit |
| `quality_over_quantity` | fable_0111, fable_0377 | Created during singleton audit |
| `dont_attempt_impossible` | fable_0183, fable_0292, fable_0264 | Created during singleton audit |
| `disguise_reveals_fool` | fable_0008, fable_0316 | Created during singleton audit |
| `kindness_never_wasted` | fable_0430, fable_0560 | Created during singleton audit |
| `avoid_lesser_fall_greater` | fable_0281, fable_0640, fable_0552 | Created during singleton audit |
| `overconfidence_danger` | fable_0330, fable_0504 | Created during singleton audit |
| `dog_in_manger` | fable_0030, fable_0483 | Created during singleton audit |

---

## Section 2 — Morals Corpus Fixes

All fixes include the original text preserved in an `original_text` field for traceability.

| Fable | Original | Fixed | Reason |
|-------|----------|-------|--------|
| fable_0043 | "Flatters are not to be trusted." | "Flatterers are not to be trusted." | Typo |
| fable_0620 | "Flatters are not to be trusted." | "Flatterers are not to be trusted." | Typo |
| fable_0597 | "Flatterers and deceivers are not to be trusted." | "Flatterers are not to be trusted." | Cluster alignment |
| fable_0082 | "Enemies' promises were made to be broken." | "Enemies' promises are made to be broken." | Tense fix |
| fable_0431 | "Be careful what you wish for; your wish may be granted." | "Be careful what you wish for." | Redundant clause |
| fable_0318 | "Be content with your station." | "Be content with your lot." | Cluster alignment |
| fable_0487 | "...slavish thing" | "...slavish thing." | Missing period |
| fable_0055 | "Nothing escapes the master's eye" | "Nothing escapes the master's eye." | Missing period |
| fable_0305 | "Greed can lead to ruin." | "A life of greed and hoarding wealth leads to misery." | Cluster alignment |
| fable_0121 | "Do not trust the wicked." | "Do not ever trust your enemy. Always protect yourself from him, even if he comes to you humble and supplicating." | More faithful to fable |
| fable_0473 | "False men cannot be trusted." | "Sometimes, those you support become your worst enemy." | Moved to new cluster; moral rewritten |
| fable_0009 | "Wit has always an answer ready." | "Wit always has an answer ready." | Word order |
| fable_0530 | "...not" | "...not." | Missing period |
| fable_0016 | "...common good" | "...common good." | Missing period |
| fable_0064 | "...lost forever" | "...lost forever." | Missing period |
| fable_0089 | "...own interests" | "...own interests." | Missing period |
| fable_0164 | "...useless tricks" | "...useless tricks." | Missing period |
| fable_0203 | "...over kinship" | "...over kinship." | Missing period |
| fable_0303 | "...unsuspected danger" | "...unsuspected danger." | Missing period |
| fable_0430 | "...ever wasted" | "...ever wasted." | Missing period |
| fable_0472 | "...surroundings" | "...surroundings." | Missing period |
| fable_0489 | "...be dangerous" | "...be dangerous." | Missing period |
| fable_0567 | "Avoid overindulgence" | "Avoid overindulgence." | Missing period |

---

## Section 3 — Team Review Decisions (Session 2)

### 3A — Pending Near Clusters (resolved)

Three clusters were deferred during Session 1. Resolved as follows:

| Cluster | Issue | Decision |
|---------|-------|----------|
| `obscurity_safety` | fable_0358 moral too verbose ("To be small is a way to stay safe...") | Removed fable_0358 → singleton. Cluster approved with fable_0105 + fable_0549 |
| `vices_punishment` | fable_0197 (Zeus & Bee): bee sought a weapon, moral calls it an "evil wish" | Approved — bee sought power to harm; fits the "vice punishes itself" pattern |
| `dont_count_chickens` | fable_0185 (Bald Men & Comb): lesson is about irony, not premature counting | Kept — "expectations lead to disappointment" is close enough to the cluster theme |

### 3B — Singleton → Existing Cluster (26 candidates)

Each singleton was reviewed for fit in a proposed existing cluster. Similarity scores from cosine similarity on moral embeddings.

| Decision | Fable | Moral | Target Cluster | Sim | Reasoning |
|----------|-------|-------|---------------|-----|-----------|
| ✅ Added | fable_0405 | "We should never put our trust in a wicked man, even if he seems completely innocuous." | `dont_trust_wicked` | 0.895 | Direct fit |
| ✅ Added | fable_0344 | "People with a history of wrongdoing are often not trusted." | `dont_trust_wicked` | 0.885 | Same lesson, consequence angle |
| ✅ Added | fable_0420 | "Do not ever trust your enemy. Always protect yourself from him, even if he comes to you humble and supplicating." | `dont_trust_wicked` | 0.872 | Identical moral text to fable_0121 |
| ✅ Added | fable_0261 | "Don't be seduced by deceptive words, they will cause nothing but trouble." | `dont_trust_wicked` | 0.861 | Fits — deceptive words = wicked people |
| ✅ Added | fable_0255 | "Beware of swindlers, as they often exploit the gullibility of others." | `dont_trust_wicked` | 0.861 | Fits — swindlers = wicked people |
| ✅ Added | fable_0297 | "There is danger lurking in the words of a wicked person." | `dont_trust_wicked` | 0.859 | Fits — danger in wicked words |
| ✅ Added | fable_0229 | "Believing your enemies makes you fall victim to their tricks." | `dont_trust_wicked` | 0.857 | Fits better here than proposed `support_becomes_enemy` |
| ✅ Added | fable_0571 | "Don't deceive yourself." | `be_yourself` | 0.863 | Self-deception is part of the authenticity theme |
| ✅ Added | fable_0582 | "Better to face a known danger than to trust a deceitful promise of safety." | `avoid_lesser_fall_greater` | 0.867 | Fits better here than proposed `overconfidence_danger` |
| ❌ Singleton | fable_0486 | "Do not believe everything you hear." | (`dont_trust_wicked`) | 0.893 | Too general — no wicked actor required |
| ❌ Singleton | fable_0243 | "One bad turn deserves another." | (`one_good_turn`) | 0.921 | Inverse moral — different lesson |
| ❌ Singleton | fable_0101 | "Never put yourself in your enemy's clutches." | (`friends_vs_foes`) | 0.895 | Caution lesson, not about friend/foe distinction |
| ❌ Singleton | fable_0664 | "Do not fight with those who are perfectly capable of fighting back." | (`friends_vs_foes`) | 0.871 | Power dynamics lesson — later formed its own cluster |
| ❌ Singleton | fable_0168 | "You should think twice before offering anything to your enemies." | (`friends_vs_foes`) | 0.857 | Different nuance |
| ❌ Singleton | fable_0260 | "Contentment with our lot is an element of happiness." | (`content_with_lot`) | 0.871 | Previously removed; kept out |
| ❌ Singleton | fable_0482 | "Even those who work hard may find their material wealth undone by the greed of others." | (`greed_misery`) | 0.876 | Victim of greed ≠ own greed causing misery |
| ❌ Singleton | fable_0661 | "It is more honorable to die having achieved noble deeds than to live a life of vice." | (`poor_life_safer`) | 0.876 | Different lesson entirely |
| ❌ Singleton | fable_0258 | "Wealth is of little value if one is too afraid to use or enjoy it." | (`poor_life_safer`) | 0.855 | Cowardice + wealth ≠ poverty being safer |
| ❌ Singleton | fable_0559 | "Don't go to extremes." | (`grasp_for_more`) | 0.870 | General moderation ≠ greed/overreach |
| ❌ Singleton | fable_0601 | "Falsehood is with all men." | (`support_becomes_enemy`) | 0.869 | False match |
| ❌ Singleton | fable_0539 | "It is dangerous to believe a story, and dangerous not to believe it." | (`every_tale`) | 0.861 | Paradox — distinct from simple skepticism |
| ❌ Singleton | fable_0282 | "Do not harm your benefactors." | (`dont_trust_wicked`) | 0.857 | Wrong direction — gratitude lesson, not trust |
| ❌ Singleton | fable_0622 | "Value is what others think of you, not what you think of yourself." | (`not_judged_appearance`) | 0.853 | Self-perception vs external ≠ actions vs appearance |
| ❌ Singleton | fable_0262 | "The useful is of much more importance and value, than the ornamental." | (`quality_over_quantity`) | 0.851 | Useful vs ornamental ≠ quality vs quantity |
| ❌ Singleton | fable_0307 | "Do not take pride in good qualities that belong to someone else." | (`not_judged_appearance`) | 0.850 | False self-credit ≠ judged by appearance |

**Cluster restructuring done during 3B:**

| Action | Detail |
|--------|--------|
| `poor_life_safer` split | Removed fable_0220 + fable_0252 → new cluster `poverty_beats_wealth` ("Better poverty without care" / "better self-sufficient poverty than anxious wealth"). fable_0532 → singleton. Kept: fable_0447 + 0592 + 0631 |
| `grasp_for_more` split | New cluster `lose_what_you_have` (7 fables: 0181, 0235, 0279, 0332, 0404, 0298, 0347) — greed makes you lose what you already had. `grasp_for_more` kept with 3 fables (0019, 0350, 0508) — don't overreach your capacity. fable_0515 → singleton |
| Type corrections | `neither_one_thing` and `greed_misery` reclassified `exact` → `near` (each has 2 distinct moral phrasings after earlier additions) |

### 3C — New Cluster Candidates from Singleton Pairs (24 candidates)

Singleton pairs with cosine similarity ≥ 0.85 were reviewed for potential new clusters.

| Decision | Cluster ID | Fables | Sim | Reasoning |
|----------|-----------|--------|-----|-----------|
| ✅ Created | `repay_kindness` | fable_0250, 0367, 0247, 0146 | 0.898 | All about reciprocating kindness/favours |
| ✅ Created | `think_before_speak` | fable_0382, 0626 | 0.888 | Near-identical lesson |
| ✅ Created | `familiarity_reduces_fear` | fable_0234, 0385, 0467 | 0.884 | Familiarity reduces terror of threatening things |
| ✅ Created | `dont_fight_stronger` | fable_0664, 0688 | 0.883 | Don't pick fights with stronger opponents |
| ✅ Created | `know_your_enemies` | fable_0562, 0619, 0590 | 0.883 | Know / pick / watch your enemies |
| ✅ Created | `dont_rival_superiors` | fable_0290, 0342 | 0.870 | Competing with superiors destroys you |
| ✅ Created | `prepare_for_war` | fable_0214, 0678 | 0.865 | Preparedness guarantees peace |
| ✅ Created | `virtue_by_necessity` | fable_0663, 0683 | 0.864 | People abandon vices only when forced to |
| ✅ Created | `weak_can_avenge` | fable_0163, 0690 | 0.860 | Powerful are unwise to provoke the weak |
| ✅ Created | `false_friend_worst` | fable_0687, 0695 | 0.857 | Hidden enemy worse than open enemy |
| ✅ Created | `company_you_keep` | fable_0327, 0525 | 0.851 | You are judged by the company you keep |
| ❌ Singletons | `dont_blame_others` | fable_0311, 0468 | 0.924 | "Don't blame upbringing" ≠ "don't blame others" — different scope |
| ❌ Singletons | `dont_boast` | fable_0271, 0598 | 0.907 | Different nuances of boasting |
| ❌ Singletons | `friends_in_adversity` | fable_0528, 0702 | 0.900 | Different angles on friendship |
| ❌ Singletons | `tyrant_excuses` | fable_0238, 0545 | 0.894 | Close but kept distinct |
| ❌ Singletons | `liars` | fable_0094, 0095 | 0.885 | Listener losing trust ≠ liar self-deceiving |
| ❌ Singletons | `vanity_downfall` | fable_0267, 0406 | 0.876 | Too varied across proposed members |
| ❌ Singletons | `beauty_subjective` | fable_0221, 0500 | 0.869 | Inner beauty ≠ beauty in the eye of the beholder |
| ❌ Singletons | `enemy_offers_help` | fable_0399, 0517 | 0.866 | Different angles — misplaced trust vs enemy posing as helper |
| ❌ Singletons | `deceivers_retaliate` | fable_0380, 0429 | 0.863 | Exploiting inability to challenge ≠ retaliating against truth |
| ❌ Singletons | `gratitude_rewarded` | fable_0139, 0533 | 0.860 | Opposite directions — ingratitude vs gratitude rewarded |
| ❌ Singletons | `mind_own_business` | fable_0100, 0423 | 0.859 | Self-interest ≠ don't meddle |
| ❌ Singletons | `unequal_alliance` | fable_0257, 0641 | 0.859 | Powerful partner betraying ≠ weak partner being useless |
| ❌ Singletons | `superior_insulted_by_inferiors` | fable_0132, 0543 | 0.857 | Insulter's POV ≠ insulted's POV |

---

## Section 4 — Manual Reading Candidates (Session 2)

Found by reading all 491 remaining singleton morals directly, without similarity scores.
Each candidate was reviewed individually. All decisions applied to `clusters_full.json`.

### 4A — New Clusters

Seven new clusters were proposed. Three were created; four were kept as singletons.

| Decision | Cluster ID | Fables | Morals | Reasoning |
|----------|-----------|--------|--------|-----------|
| ✅ Created | `learn_from_misfortunes` | fable_0068, 0248, 0379 | "Happy is the man who learns from the misfortunes of others." / "A discerning person is made wise by the misfortunes of his neighbours." / "Take warning from the misfortunes of others." | Tight theme, 3 clear phrasings |
| ✅ Created | `slow_steady_wins` | fable_0027, 0371 | "Little by little does the trick." / "Slow and steady wins the race." | Classic pair, same lesson |
| ✅ Created | `pretenders_found_out` | fable_0174, 0196, 0273, 0422, 0424 | "Those who pretend to be what they are not, sooner or later, find themselves in deep water." / "Pretenders will be found out." / "Those who assume a character which does not belong to them, only make themselves ridiculous." / "A hypocrite can usually be found out." / "Hypocritical speeches are easily seen through." | 5 fables, all about pretenders being exposed |
| ❌ Singletons | `equals_best_friends` | fable_0107, 0108 | "Equals make the best friends." / "The strong and the weak cannot keep company." | Not approved — kept as singletons |
| ❌ Singletons | `brave_in_deeds_not_words` | fable_0061, 0469, 0523 | "The hero is brave in deeds as well as words." / "Courage in words means nothing if you flee in danger." / "It is easy to be brave when there is no danger." | Not approved — kept as singletons |
| ❌ Singletons | `know_thyself_first` | fable_0087, 0550, 0596 | "Those who would mend others, should first mend themselves." / "We are not able to see our own faults..." / "Know yourself and your limits." | Not approved — kept as singletons |
| ❌ Singletons | `honesty_best_policy` | fable_0171, 0408 | "Be frank and open in your dealings." / "Honesty is the best policy." | Not approved — kept as singletons |

### 4B — Expansions: Singletons Added to Existing Clusters

Each proposed singleton was reviewed individually for fit in its target cluster.

**`innate_nature_unchangeable`** (was 2 fables, now 4)

| Decision | Fable | Moral | Reasoning |
|----------|-------|-------|-----------|
| ✅ Added | fable_0028 [The Dancing Monkeys] | "Men often revert to their natural instincts." | Monkeys trained to dance revert to animal behaviour when nuts are thrown — nature over nurture |
| ✅ Added | fable_0088 [The Raven and the Swan] | "Change of habit cannot alter nature." | Raven bathes constantly trying to turn white — his colour never changes |
| ❌ Singleton | fable_0090 [The Scorpion and the Ladybug] | "Regardless of our wishes, it is to our Nature alone that we will be faithful." | Team felt this was not clearly related to the cluster |
| ❌ New cluster | fable_0147 [Zeus And Prometheus] | "Nature and instincts can persist despite changes in form or appearance." | Zeus-transformation theme — paired with fable_0152 instead (see below) |
| ❌ New cluster | fable_0152 [Zeus And The Ant] | "When someone with a wicked nature changes his appearance, his behaviour remains the same." | Zeus-transformation theme — paired with fable_0147 to form `nature_survives_transformation` |
| ❌ Singleton | fable_0165 [The Shepherd, The Wolf Cub And The Wolf] | "A wicked nature does not produce a good character." | Wolf raised as dog still reverts — nature-vs-nurture, but kept as singleton |
| ❌ Singleton | fable_0224 [The Bull And The Calf] | "Some people just won't change." | Stubbornness is a choice; different from nature being inherently immutable |

**New cluster created from this review:**

| Cluster ID | Fables | Morals | Reasoning |
|-----------|--------|--------|-----------|
| `nature_survives_transformation` | fable_0147, fable_0152 | "Nature and instincts can persist despite changes in form or appearance." / "When someone with a wicked nature changes his appearance, his behaviour remains the same." | Both involve Zeus physically transforming a creature — their nature survives the divine transformation. More specific than `innate_nature_unchangeable` |

**`deeds_not_words`** (unchanged — all 3 candidates kept as singletons)

| Decision | Fable | Moral | Reasoning |
|----------|-------|-------|-----------|
| ❌ Singleton | fable_0177 [The Boastful Athlete] | "Talking is a waste of time when you can simply provide a demonstration." | About demonstrating skill, not about judging character by deeds |
| ❌ Singleton | fable_0280 [The Deer And Her Friends] | "Good will is worth nothing unless it is accompanied by good acts." | About intentions vs acts — subtly different lesson |
| ❌ Singleton | fable_0612 [The Jar goes to Court] | "Action is more valuable than arguments." | Kept as singleton |

**`company_you_keep`** (was 2 fables, now 3)

| Decision | Fable | Moral | Reasoning |
|----------|-------|-------|-----------|
| ✅ Added | fable_0035 [The Farmer and the Stork] | "Birds of a feather flock together." | Story is literally about being judged by association — stork caught with thieves is punished with thieves |
