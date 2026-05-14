# Singleton Cluster Proposals

Review of all 489 remaining singletons for potential new clusters or expansions to existing ones.
Each entry shows the proposed cluster name, the shared theme, and every candidate fable with its
title and full moral text.

**For each fable, respond:**
- ✅ In cluster
- ❌ Keep as singleton
- ↔ Move to a different cluster (specify which)

**For each proposed cluster as a whole:**
- ✅ Create it
- ❌ Keep all as singletons
- ✂ Split (specify how)

---

## STRONG PROPOSALS (5+ fables)

---

### 1. `friends_in_adversity`
**Theme:** Real friends reveal themselves in adversity — fair-weather friends are worthless.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0014 | The Bear and the Two Travelers | Misfortune tests the sincerity of friends. |
| fable_0227 | The Dolphin And The Lion | When we form friendships with one another, we must choose allies who can come to our aid in moments of peril. |
| fable_0464 | The Raven, The Swallow And The Seasons | Fair weather friends are not worth much. |
| fable_0528 | The Kite And His Mother | We must make friends in prosperity if we would have their help in adversity. |
| fable_0702 | A Lion and a Hog | Seek the friendship of those who do not withdraw from offering help even in a time of adversity. |

**Decision:** ✂ Split into two clusters.
- ✅ Cluster `adversity_reveals_friends` (2): fable_0014, fable_0464 — diagnostic: hardship is the test that exposes sincere vs fair-weather friends.
- ✅ Cluster `choose_friends_who_help_in_adversity` (3): fable_0227, fable_0528, fable_0702 — prescriptive: actively select / cultivate allies who will stand by you in peril.

---

### 2. `boasting_downfall`
**Theme:** Pride and boasting lead to destruction or humiliation.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0145 | The Foxes At The River | People get themselves into trouble because of their boasting. |
| fable_0154 | The Olive Tree And The Fig Tree | Brag and nature may bring you down. |
| fable_0267 | The Birds, The Peacock And His Feathers | Vanity can lead to self-destruction. |
| fable_0338 | The Two Roosters And The Eagle | Pride goes before destruction. |
| fable_0439 | The Mole And The Frankincense | Boast of one thing and you will be found lacking in that and a few other things as well. |
| fable_0580 | The Gnat and The Lion | No matter how you brag, you can be undone. |
| fable_0598 | The Lamp | Do not boast lest you be taken down. |
| fable_0629 | A Cuckow and a Hawk | Pride and vanity leads to our downfall. |

**Note:** fable_0321 (*The Donkey And The Lion Go Hunting* — "The loud-mouthed boaster does not impress nor frighten those who know him.") is adjacent but slightly different — about the boaster being unimpressive rather than being destroyed.

**Decision:** ✂ Split.
- ✅ Cluster `pride_before_destruction` (6): fable_0154, fable_0267, fable_0338, fable_0580, fable_0598, fable_0629 — all share the schema *pride/vanity/brag → severe, irreversible downfall* (destruction / self-destruction / downfall / undone / taken down / bring down).
- ❌ fable_0145 — keep singleton. "Trouble" is too generic/mild to belong with destruction-framed morals.
- ❌ fable_0439 — keep singleton. Different causal schema: the boast *exposes* a pre-existing deficiency rather than *causing* destruction. Candidate seed for a future `boasters_exposed` cluster.

---

### 3. `value_unrecognized`
**Theme:** People fail to recognize or appreciate what is truly valuable.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0026 | The Cock and the Pearl | Precious things are for those that can prize them. |
| fable_0054 | The Hart and the Hunter | We often despise what is most useful to us. |
| fable_0153 | The Gods And Their Trees | True value comes from usefulness. |
| fable_0167 | The Rooster And The Pearl | True value is wasted on those who cannot appreciate it. |
| fable_0172 | The Travellers And The Plane Tree | Some men underrate their best blessings. |

**Note:** fable_0026 and fable_0167 are both about a rooster and a pearl — check if they are near-duplicates that should be in an exact/near cluster instead. **Confirmed:** same story (cock/rooster finds pearl in straw/manure, prefers a grain of food). Recommend marking as a `near` cluster, not just thematic.

**Decision:** ✂ Split.
- ✅ Cluster `value_wasted_on_unappreciative` (2): fable_0026, fable_0167 — same story, paraphrased morals. Mark as `near` cluster.
- ❌ fable_0054 — keep singleton. Different proposition (agent's own blindness to what serves them).
- ❌ fable_0153 — keep singleton. Definitional claim (usefulness = true value), not about misrecognition.
- ❌ fable_0172 — keep singleton. Same proposition as 0054 (agent's own blindness), but no third companion to form a cluster.

---

### 4. `inner_worth_over_appearance`
**Theme:** Inner worth matters more than outer show or beauty.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0110 | The Vain Jackdaw | It is not only fine feathers that make fine birds. |
| fable_0221 | The Children And The Mirror | Inner beauty is better than outer beauty. |
| fable_0262 | The Crane And The Peacock | The useful is of much more importance and value, than the ornamental. |
| fable_0370 | The Fox And The Mask | Outside show is a poor substitute for inner worth. |

**Note:** Thematically adjacent to `value_unrecognized` (#3) but distinct — these specifically contrast outer appearance vs. inner quality, rather than about appreciation of value.

**Decision:** ✂ Split.
- ✅ Cluster `inner_worth_over_outer_show` (3): fable_0110, fable_0221, fable_0370 — inner trait/character/worth > outer adornment.
- ❌ fable_0262 — keep singleton. Different axis (useful capability > decorative trait, both visible); doesn't fit the inner-vs-outer schema.

---

### 5. `blame_yourself_first`
**Theme:** People criticize others for the very flaws they ignore in themselves.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0087 | The Quack Frog | Those who would mend others, should first mend themselves. |
| fable_0115 | The Wolf and the Shepherds | Men often condemn others for what they see no wrong in doing themselves. |
| fable_0340 | The Hare And The Sparrow | It is a foolish thing to give advice to others while not looking out for oneself. |
| fable_0550 | Of the Vices of Men | We are not able to see our own faults: but as soon as others make a slip, we are ready to censure. |

**Decision:** ✂ Split into two pairs along the prescriptive/descriptive axis.
- ✅ Cluster `fix_yourself_before_helping_others` (2): fable_0087, fable_0340 — prescriptive: don't mend/advise others while neglecting your own issues.
- ✅ Cluster `blind_to_own_faults` (2): fable_0115, fable_0550 — descriptive: we condemn/censure others for what we ignore in ourselves.

---

### 6. `think_before_acting`
**Theme:** Acting without forethought leads to disaster — look before you leap.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0046 | The Fox and the Hedgehog | Consider carefully before changing your situation. |
| fable_0093 | The Shepherd and the Sea | Understand what you are doing before you do it. |
| fable_0303 | The Dog And The Shellfish | They who act without sufficient thought, will often fall into unsuspected danger. |
| fable_0337 | The Hares And The Foxes | Count the cost before you commit yourselves. |
| fable_0375 | The Fox And The Hare In The Well | People who act impulsively can end up in regrettable situations. |

**Decision:** ✂ Split into two clusters along *deliberation vs impulse-failure*.
- ✅ Cluster `deliberate_before_committing` (3): fable_0046, fable_0093, fable_0337 — active evaluation (consider / understand / count the cost) before a significant choice or commitment.
- ✅ Cluster `do_not_act_impulsively` (2): fable_0303, fable_0375 — generic impulse failure → unsuspected danger / regret.

---

### 7. `contentment_with_lot`
**Theme:** Accept your situation with equanimity — contentment is a source of happiness.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0260 | The Crab On Dry Land | Contentment with our lot is an element of happiness. |
| fable_0369 | The Horse And The Miller | Accept the changes in life. |
| fable_0558 | The Chariot Horse sold for the Mill | Whatever happens, we must bear it with equanimity. |
| fable_0667 | A Husbandman turn'd Soldier and Merchant | Everyone should be satisfied with their own lot in life, given that disaster awaits us on every side. |

**Note:** fable_0357 (*The Father, The Son And The Lion* — "We had better bear our troubles bravely than try to escape them.") could also fit here — about enduring rather than fleeing difficulty. **Decided against** — escape-backfires schema is distinct from passive acceptance.

**Decision:** ✂ One pair, rest singletons.
- ✅ Cluster `bear_change_with_equanimity` (2): fable_0369, fable_0558 — accept the changes circumstance imposes; complaining is futile. **Flag for story-level near-duplicate check** (both are former-glorious-horse-demoted-to-mill stories).
- ❌ fable_0260 — keep singleton. "Don't leave your given station" framing; different from passive acceptance of imposed change.
- ❌ fable_0667 — keep singleton. Same "don't seek to change your station" framing as 0260, but kept singleton to match this decision.
- ❌ fable_0357 — keep singleton. Escape-backfires schema (active avoidance causes the fate); not passive equanimity.

---

## MEDIUM PROPOSALS (3–4 fables)

---

### 8. `necessity`
**Theme:** Necessity is the force that overrides rules, drives creativity, and cannot be denied.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0017 | The Birdcatcher, the Partridge, and the Cock | Necessity knows no law. |
| fable_0051 | The Hare and the Hound | Necessity is our strongest weapon. |
| fable_0127 | The Crow And The Water Jar | Necessity is the mother of invention. |

**Decision:** ❌ Do not create cluster. All 3 kept as singletons.
- Surface-word "necessity" matches across all three, but the propositions are entirely distinct: (0017) necessity overrides moral/social rules; (0051) desperation intensifies effort; (0127) need drives creativity. Textbook *false-cognate* clustering — would be surface-matching, not semantic-matching.

---

### 9. `timing_matters`
**Theme:** The right action done at the wrong time is wasted or harmful.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0037 | The Fisherman Piping | To do the right thing at the right season is a great art. |
| fable_0218 | The Farmer'S Boy And The Snails | Anything which is done at the wrong time is liable to be ridiculed. |
| fable_0645 | An Old Fellow and a Young Wench | All things are to be done in their own time. |

**Decision:** ❌ Do not create cluster. All 3 kept as singletons.

---

### 10. `deceivers_deceived`
**Theme:** Those who deceive or betray are repaid in kind.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0041 | The Fox, the Cock and the Dog | Cunning often outwits itself. |
| fable_0299 | The Fox, The Rooster And The Dog | Those who try to deceive may expect to be paid in their own coin. |
| fable_0315 | The Fox, The Donkey And The Lion | Traitors may expect treachery. |

**Decision:** ❌ Do not create cluster. All 3 kept as singletons. (Flag: fable_0041 and fable_0299 are story-level near-duplicates — fox tries to trick cock/rooster but bird uses dog against fox — worth re-checking the near-duplicate detection pipeline.)

---

### 11. `unequal_company`
**Theme:** Partnerships between unequals are doomed — the strong and weak cannot keep company.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0107 | The Two Pots | Equals make the best friends. |
| fable_0108 | The Two Pots | The strong and the weak cannot keep company. |
| fable_0665 | A Lion and a Mouse | Unequal matches bring misery. |

**Note:** fable_0107 and fable_0108 appear to be two versions of the same fable (same title). Check whether they should be an exact cluster. **Confirmed same story (earthen + brass pot in river); converse paraphrased morals. Should be merged as a `near`-cluster pair in the existing typology.**

**Decision:** ❌ Do not create thematic cluster. All 3 kept as singletons. (Flag: fable_0107 + fable_0108 should still be merged as a `near`-cluster pair via the near-duplicate pipeline.)

---

### 12. `self_interest_true_motive`
**Theme:** Self-interest, not virtue or goodwill, is the true driver of most human action.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0086 | The Peasant and the Apple-Tree | Self-interest alone moves some men. |
| fable_0287 | The Old Woman And Her Doctor | Few things are done except for profit. |
| fable_0492 | The Statue Of Hermes And The Treasure | People adjust their beliefs based on what is profitable for them. |

**Decision:** ❌ Do not create cluster. All 3 kept as singletons.

---

### 13. `pleasant_things_are_traps`
**Theme:** Easy gains, pleasant invitations, and pleasurable things often conceal danger.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0208 | The Cat And The Birds | Rushing into pleasant invitations can lead to dire consequences. |
| fable_0394 | Hermes And The Earth | Easy gains often come with hidden costs and hardships. |
| fable_0518 | The Young Man And The Prostitute | The things which bring us pleasure can often be hazardous as well. |

**Decision:** ❌ Do not create cluster. All 3 kept as singletons.
- Proposed theme is a post-hoc generalization; each fable makes a distinct claim. 0208 = *deceptive lures from others* (deceiver present); 0394 = *no free lunch* (intrinsic cost, no deceiver); 0518 = *pleasure is hazardous* (self-knowing indulgence). Three different cognitive structures — clustering would be surface-grouping by the word "pleasant."

---

### 14. `wisdom_beats_strength`
**Theme:** Cleverness and wisdom prevail where brute force cannot.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0275 | The Eagle And The Crow | Tricksters can outsmart the strong. |
| fable_0512 | The Lizard And The Snake | Where force is not enough, cleverness must be used instead. |
| fable_0569 | Zeus and Apollo, A Contest in Archery | Wit outshines skill. |
| fable_0682 | A Fox and a Hare | Wisdom is far superior to strength. |

**Decision:** ✂ Pair + 2 singletons. Split by *applied situation* (offensive vs defensive cunning).
- ✅ Cluster `cunning_escapes_strength` (2): fable_0512, fable_0682 — defensive cunning: weaker prey uses cleverness to escape stronger predator.
- ❌ fable_0275 — keep singleton. Offensive cunning: trickster steals spoils from stronger party (different transaction, not survival).
- ❌ fable_0569 — keep singleton. Peer wit-vs-*skill* contest (not weak-vs-strong); no survival/threat dynamic.

---

### 15. `perspective_is_relative`
**Theme:** The same thing means different things to different people — there is no universal value.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0020 | The Boys and the Frogs | One man's pleasure may be another's pain. |
| fable_0195 | The Bees And The Beetles | What is valued in one group may be dismissed by another. |
| fable_0289 | The Donkey, The Dog And The Letter | Different people are interested in different things. |

**Decision:** ❌ Do not create cluster. All 3 kept as singletons.

---

### 16. `false_courage`
**Theme:** Bravery displayed when there is no real danger, or in words alone, is not true courage.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0061 | The Hunter and the Woodman | The hero is brave in deeds as well as words. |
| fable_0469 | The Two Soldiers And The Robber | Courage in words means nothing if you flee in danger. |
| fable_0523 | The Sheep, The Goat And The Sow | It is easy to be brave when there is no danger. |

**Note:** Could alternatively expand the existing `deeds_not_words` cluster (currently fable_0499 + fable_0534) rather than creating a new cluster. **Decided against** — existing cluster is a tight `near` paraphrase of general "deeds > words"; mixing in bravery-specific morals would loosen it.

**Decision:** ✂ Pair + singleton. Split by *timing/structure of false-courage display*.
- ✅ Cluster `bravery_in_safety_is_cheap` (2): fable_0469, fable_0523 — performing/showing courage when there's no personal threat (post-danger boast / untested calm). Story-level emphasis of 0469 (sword-drawing after robber leaves) maps directly onto 0523's schema.
- ❌ fable_0061 — keep singleton. Different timing: *bravado before testing, cowardice when tested* (no post-danger safe-performance phase).

---

## PAIRS

---

### 17. `change_of_place`
**Theme:** Moving somewhere new does not change who you are inside.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0141 | The Magpie And Her Tail | Changing your location does not change your state of mind. |
| fable_0463 | The Raven, The Stork And His Beak | A change of place does not make you a saint. |

**Note:** Could absorb into existing `innate_nature_unchangeable` cluster instead. **Decided against.**

**Decision:** ❌ Do not create cluster. Both kept as singletons.

---

### 18. `false_fears`
**Theme:** People fear the wrong things — imaginary dangers are the most paralyzing.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0388 | The Lion And The Frog | Imaginary fears are the worst. |
| fable_0652 | The Birds and Beetles | Many people fear danger where there is none, and feel safe where there is danger: put things in perspective. |

**Decision:** ✅ Create cluster `imaginary_fears` (2): fable_0388, fable_0652 — fear of non-existent / harmless dangers; misjudging the real threat level. Both stories share the schema *creature paralyzed by fear of something that turns out to be harmless*.

---

### 19. `complaining_is_weakness`
**Theme:** Those who complain the most are usually those who suffer the least.

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0502 | The Bulls And The Wagon | They complain most who suffer least. |
| fable_0693 | A Creaking Wheel | Complaining is the privilege of the weak. |

**Decision:** ❌ Do not create cluster. Both kept as singletons. (Flag: stories are structurally near-duplicates — creaking part complains while harder-working silent parts bear the load. Worth a story-level near-duplicate check.)

---

## EXISTING CLUSTER EXPANSIONS

---

### A. Expand `learn_from_misfortunes`
Current cluster (learning from *others'* misfortunes): fable_0068, fable_0248, fable_0379.

These 3 candidates are about learning from *your own* experience — decide whether to absorb them into the existing cluster or create a separate `learn_from_experience` cluster:

| Fable | Title | Moral |
|-------|-------|-------|
| fable_0191 | The Bat, The Booby And The Bramble Bush | Learn from past misfortunes. |
| fable_0232 | The Butcher And The Dog | Experience teaches valuable lessons and encourages caution. |
| fable_0700 | A Lad Robbing an Orchard | Those who won't learn from advice must be taught by experience. |

**Decision:** ❌ Keep existing `learn_from_misfortunes` cluster as-is. All 3 candidates kept as singletons.
- Existing cluster is tightly about *vicarious* learning (observing others' misfortunes). Candidates are about *direct/own* experience — a different proposition. Merging would dilute the existing `near` cluster's tightness.

---

*Total candidates: ~90 fables across 19 proposed clusters + 1 expansion.*
*Prepared by Lior & Claude — 2026-05-13*
