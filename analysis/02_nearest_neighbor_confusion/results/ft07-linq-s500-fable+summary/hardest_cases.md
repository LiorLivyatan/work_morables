# Nearest-Neighbor Confusion — ft07-linq-s500-fable+summary

Top 30 hardest misranked cases (largest score gap between rank-1 and ground truth).

---
## Case 1 — Query 278 | gt_rank=76 | gap=0.4864

**Moral (query):**
> Be careful what you wish for.

**Rank-1 (wrong) — `The Shepherd And The Lion` — score 0.8118:**
> A herdsman tending his flock in a forest lost a Bull-calf from the fold. After a long and fruitless search, he made a vow that, if he could only discover the thief who had stolen the Calf, he would offer a lamb in sacrifice to Hermes, Pan, and the Guardian Deities of the forest. Not long afterwards, as he ascended a small hillock, he saw at its foot a Lion feeding on the Calf. Terrified at the sight, he lifted …

**Ground truth (rank 76) — `The Poor Man And Death` — score 0.3255:**
> A poor man was carrying a load of wood on his shoulders. After a while he was feeling faint, so he sat down by the side of the road. Putting aside his burden, he bitterly called out to Death, summoning Death with the words 'O Death!' Death immediately showed up and said to the man, 'Why have you summoned me?' The man said, 'Oh, just to have you help me pick this burden up off the ground!'

**Claude Analysis:** *[taxonomy: MODEL_FOUND_BETTER_MATCH]* The shepherd literally makes a wish (a sacrificial vow) that gets granted in the worst possible way — he finds the thief, but the thief is a lion who will kill him. He escalates his vow again to escape. This IS "be careful what you wish for" enacted precisely. The model's retrieval is semantically stronger than the labeled ground truth. The Poor Man & Death story illustrates the same moral through a different mechanism (calling for Death literally appearing), but the narrative alignment is weaker. The score gap of 0.49 — the largest in the dataset — reflects genuine model confidence, not confusion.

---
## Case 2 — Query 617 | gt_rank=58 | gap=0.4761

**Moral (query):**
> Be careful what you wish for.

**Rank-1 (wrong) — `The Shepherd And The Lion` — score 0.8118:**
> A herdsman tending his flock in a forest lost a Bull-calf from the fold. After a long and fruitless search, he made a vow that, if he could only discover the thief who had stolen the Calf, he would offer a lamb in sacrifice to Hermes, Pan, and the Guardian Deities of the forest. Not long afterwards, as he ascended a small hillock, he saw at its foot a Lion feeding on the Calf. Terrified at the sight, he lifted …

**Ground truth (rank 58) — `The Eyes and the Honey` — score 0.3357:**
> Aesop said that the eyes were distressed because they considered themselves the most worthy part of the body, yet they saw the mouth enjoying every possible pleasure, especially honey, the sweetest substance of all. Filled with resentment, the eyes complained to the man. But when the man put honey in his eyes, they felt a stinging pain and burst into tears, since they thought that the honey was harsh and unpleasant.

**Claude Analysis:** *[taxonomy: MODEL_FOUND_BETTER_MATCH + ATTRACTOR_FABLE]* A second query with the identical moral "Be careful what you wish for" that also retrieves The Shepherd And The Lion at rank 1 with the same score (0.8118). This is the same attractor fable appearing for two different ground truth fables that share the same moral text. This case simultaneously illustrates dataset ambiguity (two fables sharing an identical moral) and genuine model confidence in a strong thematic match. The Eyes and Honey story is also a valid illustration — the eyes wished for honey, got it, were harmed — but the embedding space places The Shepherd And The Lion much closer to the moral's phrasing.

---
## Case 3 — Query 332 | gt_rank=235 | gap=0.4374

**Moral (query):**
> Greed for more can lead to losing what you already have.

**Rank-1 (wrong) — `Zeus And The Camel` — score 0.7257:**
> When the camel saw another animal's horns, she begged Zeus to give her horns too. Zeus was angry at the camel's greediness, so he cropped her ears instead.

**Ground truth (rank 235) — `The Jackdaw And The Doves` — score 0.2882:**
> A jackdaw saw that the doves in a dovecote were very well fed, so he dyed himself white and went to join them, expecting to share in their food. So long as the jackdaw kept quiet, the doves thought he was another dove and accepted him, but when he forgot to keep quiet and let out a squawk, the pigeons then recognized who he was and they pecked at him until he went away. Unable to feed with the doves, …

**Claude Analysis:** *[taxonomy: MODEL_FOUND_BETTER_MATCH]* The Zeus/Camel story is a textbook illustration of this exact moral — the camel greedily wanted more (horns she didn't have), and as punishment lost what she already had (her ears). The narrative-moral alignment is near-perfect. The Jackdaw story also fits — the jackdaw tried to gain more (dove food) and ended up with nothing — but the Zeus/Camel story maps more directly to the precise wording "losing what you already have." The score gap of 0.44 and catastrophic gt_rank of 235 suggest the embedding space strongly preferred the more literal match.

---
## Case 4 — Query 93 | gt_rank=270 | gap=0.4344

**Moral (query):**
> Understand what you are doing before you do it.

**Rank-1 (wrong) — `The Fox And The Hare In The Well` — score 0.6882:**
> A thirsty hare had gone done into a well to drink the water. He took a good long drink, and when he wanted to get back out again, he found himself trapped with no means of escape. It was a very discouraging situation. A fox then arrived on the scene and when she found the hare she said to him, 'You have made a very serious mistake indeed: you should have first decided on a way to get out and …

**Ground truth (rank 270) — `The Shepherd and the Sea` — score 0.2538:**
> A Shepherd, keeping watch over his sheep near the shore, saw the Sea very calm and smooth, and longed to make a voyage with a view to commerce. He sold all his flock, invested it in a cargo of dates, and set sail. But a very great tempest came on, and the ship being in danger of sinking, he threw all his merchandise overboard, and barely escaped with his life in the empty ship. Not long afterwards when someone passed …

**Claude Analysis:** *[taxonomy: MODEL_FOUND_BETTER_MATCH + VERBATIM_MATCH]* The Fox and the Hare story contains a near-verbatim expression of the moral: the fox literally says "you should have first decided on a way to get out." This is not just thematic overlap — the moral's exact meaning is stated as dialogue within the fable. This is the strongest possible embedding signal. The Shepherd/Sea story also illustrates acting without forethought (selling everything on impulse), but does so implicitly through narrative rather than explicit statement. The model found the verbatim match; the label preferred the implicit one. gt_rank=270 suggests the labeled fable is genuinely weak in embedding space.

---
## Case 5 — Query 187 | gt_rank=20 | gap=0.4199

**Moral (query):**
> Those who seek to please everybody please nobody.

**Rank-1 (wrong) — `The Miller, His Son, and Their Ass` — score 0.8861:**
> A Miller and his son were driving their Ass to a neighboring fair to sell him. They had not gone far when they met with a troop of women collected round a well, talking and laughing. Look there, cried one of them, did you ever see such fellows, to be trudging along the road on foot when they might ride?' The old man hearing this, quickly made his son mount the Ass, and continued to walk along merrily by his …

**Ground truth (rank 20) — `The Bald Man And His Two Mistresses` — score 0.4662:**
> There was a woman who had a middle-aged man as her lover and although she was no spring chicken herself, she concealed her age with exquisite grace. There was also a beautiful young girl who had caught the man's fancy. Both women wanted to seem a suitable partner for him, so they began plucking out his hair in turn. The man imagined that his looks were being improved by their attentions but in the end he went bald, since the …

**Claude Analysis:** *[taxonomy: LIKELY_LABELING_ERROR]* The Miller, His Son, and Their Ass is the canonical Aesop fable for this moral. The story's entire structure is built around a man who tries to satisfy every passerby's opinion (ride the ass, don't ride, both ride, carry the ass), fails at each attempt, and the moral is stated directly at the end. This is one of Aesop's most famous fables precisely because it illustrates "please everybody, please nobody" so perfectly. The Bald Man story is a secondary illustration — a man caught between two mistresses each grooming him ends up bald. The model found the canonical fable; the dataset labeled a less-famous one as ground truth. The score of 0.886 (highest in the top-30) reflects the model's near-certainty.

---
## Case 6 — Query 185 | gt_rank=79 | gap=0.4107

**Moral (query):**
> Expectations can lead to disappointment when reality does not match our desires.

**Rank-1 (wrong) — `The Fishermen And The Stone` — score 0.6361:**
> Some fishermen were hauling in their net. It was quite heavy, so the fishermen made merry and danced for joy, thinking that they had landed a very big catch. Yet when they finally dragged it in, they found that the net contained only a few fish, together with a very large stone. The fishermen now grew extremely despondent, not so much because of the lack of fish but because they had been expecting just the opposite. Then one of the …

**Ground truth (rank 79) — `The Bald Men And The Comb` — score 0.2254:**
> A bald man happened to find a comb lying in the street. Another man who also had no hair on his head accosted him and said, 'Hey, you must share whatever you've found!' The first man showed him the loot and said, 'The will of the gods is on our side, but fate must have a grudge against us: as the saying goes, we've found lumps of coal, not gold!'

**Claude Analysis:** *[taxonomy: MODEL_FOUND_BETTER_MATCH]* The Fishermen and the Stone story explicitly narrates expectations leading to disappointment — the fishermen dance with joy at a heavy net (high expectations), find mostly rocks (reality fails), and are "extremely despondent, not so much because of the lack of fish but because they had been expecting just the opposite." This last phrase directly encodes the moral's language. The Bald Men & Comb story is about finding something useless — a glancing illustration of the same idea. The model correctly identified the stronger, more explicit match.

---
## Case 7 — Query 593 | gt_rank=373 | gap=0.4074

**Moral (query):**
> Nature reveals itself.

**Rank-1 (wrong) — `Zeus And Prometheus` — score 0.6552:**
> Following Zeus's orders, Prometheus fashioned humans and animals. When Zeus saw that the animals far outnumbered the humans, he ordered Prometheus to reduce the number of the animals by turning them into people. Prometheus did as he was told, and as a result those people who were originally animals have a human body but the soul of an animal.

**Ground truth (rank 373) — `The Kingdom of The Lion` — score 0.2479:**
> The beasts of the field and forest had a Lion as their king. He was neither wrathful, cruel, nor tyrannical, but just and gentle as a king could be. During his reign he made a royal proclamation for a general assembly of all the birds and beasts, and drew up conditions for a universal league, in which the Wolf and the Lamb, the Panther and the Kid, the Tiger and the Stag, the Dog and the Hare, should live together …

**Claude Analysis:** *[taxonomy: MODEL_FOUND_BETTER_MATCH]* The Zeus/Prometheus story's entire point is that animal nature persists even when form changes — people who were originally animals "have a human body but the soul of an animal." The conclusion IS the moral: nature reveals itself regardless of outward transformation. The Kingdom of the Lion story is about a just lion king; the connection to "nature reveals itself" is indirect at best. A gt_rank of 373 confirms the labeled fable is a very weak semantic match. This is a case where the annotation may have relied on thematic proximity rather than precise moral alignment.

---
## Case 8 — Query 568 | gt_rank=174 | gap=0.4009

**Moral (query):**
> Everyone is more or less master of his own fate.

**Rank-1 (wrong) — `The Man And The Oracle` — score 0.7460:**
> A wicked man had gone to visit Apollo in Delphi, wanting to test the god. He took a sparrow in one hand, concealing it with his cloak, and then stood by the oracle and inquired of the god, 'Apollo, the thing that I am carrying in my hand: is it living, or is it dead?' The man planned to show the sparrow alive if the god said 'dead,' and if the god said 'living,' he would strangle the sparrow immediately …

**Ground truth (rank 174) — `The Ploughman and Fortune` — score 0.3450:**
> As a Countryman was one day turning up the ground with his plough, he came across a great store of treasure. Transported with joy, he fell upon the earth and thanked her for her kindness and liberality. Fortune appeared, and said to him, 'You thank the ground thus warmly, and never think of me. If, instead of finding this treasure, you had lost it, I should have been the first you would have blamed.'

**Claude Analysis:** *[taxonomy: PARTIAL_MATCH]* The Man and the Oracle story is about a man who tries to outwit Apollo — attempting to be master of the situation through cleverness. But the oracle story's moral is more about divine omniscience than human agency over fate. The Ploughman and Fortune story makes a much more direct argument: Fortune tells the ploughman that he controls less of his fate than he thinks, and credits or blames Fortune arbitrarily. The labeled gt is actually the stronger semantic match here; the model latched onto the "agency/control" framing in the Oracle story. A genuine model confusion.

---
## Case 9 — Query 576 | gt_rank=27 | gap=0.3791

**Moral (query):**
> The grass is always greener on the other side of the fence.

**Rank-1 (wrong) — `A Husbandman turn'd Soldier and Merchant` — score 0.8239:**
> 'Oh, the endless misery of the life I lead!' cries the hard-working farmer, 'spending all my days plowing, sowing, digging, and fertilizing, and in the end, gaining nothing from it! Now, in a soldier's life, there's honor to be earned, and one lucky strike can set a man up forever. Indeed, I'll sell my stock, get a horse and arms, and try my luck in war'. Off he goes, makes his attempt, faces the battle, and ends up leaving a …

**Ground truth (rank 27) — `A Wild Ass and A Tame` — score 0.4448:**
> As a tame ass was airing himself in a pleasant meadow, with a coat and carcass in very good plight, up comes a wild one to him from the next wood, with this short greeting. Brother (says he) I envy your happiness; and so he left him. It was his hap some short time after this encounter, to see his tame brother groaning under a unmerciful pack, and a fellow at his heels goading him forward. He rounds him in …

**Claude Analysis:** *[taxonomy: NEAR_MISS]* Both fables fit the moral well, with different framing. The Husbandman story is about a farmer who envies other occupations (soldier, merchant), tries each, fails at each, and eventually returns to farming — a multi-step "grass is greener" illustration. The Wild Ass story is more compact and precise: wild ass envies the tame ass's comfort, then encounters the tame ass burdened under a heavy load and realizes the trade-off. The gt is the canonical two-sided comparison (envy → reality check); the model found the more elaborated narrative. Both are valid; the model's confidence (0.82) is justified.

---
## Case 10 — Query 183 | gt_rank=21 | gap=0.3746

**Moral (query):**
> Do not attempt the impossible.

**Rank-1 (wrong) — `The Wolves And The Hides` — score 0.8362:**
> Some wolves saw some cowhides in the river. They wanted to take the hides, but the depth of the river in flood prevented them from doing so. The wolves therefore decided to drink up all the water so that they could then reach the hides. A man said to the wolves, 'If you try to drink up all that water, you will immediately burst into pieces and die on the spot!'

**Ground truth (rank 21) — `The Frog And The Ox` — score 0.4616:**
> There was once a frog who noticed an ox standing in the meadow. The frog was seized by a jealous desire to equal the ox in size so she puffed herself up, inflating her wrinkled skin. She then asked her children if she was now bigger than the ox. They said that she was not. Once again she filled herself full of air, straining even harder than before, and asked her children which of the two of them was bigger. …

**Claude Analysis:** *[taxonomy: DATASET_AMBIGUITY]* Both fables illustrate "do not attempt the impossible" through nearly identical narrative structures: a creature attempts something physically impossible (drinking a river / inflating to ox size), is warned of fatal consequences, and the moral is clear. This is dataset ambiguity — two fables teach the same lesson. The wolves story may even be the more literal illustration (drinking an entire river IS impossible; the frog's inflation IS impossible and kills her). Neither fable is "more correct" — the dataset arbitrarily labeled one as ground truth.

---
## Case 11 — Query 629 | gt_rank=97 | gap=0.3659

**Moral (query):**
> Pride and vanity leads to our downfall.

**Rank-1 (wrong) — `The Wolf and his Shadow` — score 0.7181:**
> Roaming by the mountainside at sundown, a Wolf saw his own shadow become greatly extended and magnified, and he said to himself, 'Why should I, being of such an immense size and extending nearly an acre in length, be afraid of the Lion? Ought I not to be acknowledged as King of all the collected beasts?' While he was indulging in these proud thoughts, a Lion fell upon him and killed him. He exclaimed with a too late repentance, 'Wretched …

**Ground truth (rank 97) — `A Cuckow and a Hawk` — score 0.3523:**
> By the beak and claws of a Cuckoo, one might mistake her for a kind of Hawk, except one eats worms and the other eats meat. One day, a Hawk mocked a Cuckoo for her simple diet. 'If you want to look like a Hawk, why don't you live like one?' The Cuckoo felt a bit offended, but later, as she passed a pigeon house, she saw the skin of that very Hawk on a pole atop the dove house. …

**Claude Analysis:** *[taxonomy: MODEL_FOUND_BETTER_MATCH]* The Wolf and his Shadow is a textbook "pride leads to downfall" fable — wolf is deceived by his own inflated shadow, lets pride override reason, ignores a real threat, and dies. The story ends with explicit recognition of the error. The Cuckow and Hawk story illustrates humility vs pride more obliquely: the hawk's pride in its predatory lifestyle leads to its death, while the cuckoo's humility saves it. The wolf story has a more direct causal chain (pride → fatal miscalculation). The model found the more explicit match; the labeled gt encodes the same moral through a subtler irony.

---
## Case 12 — Query 621 | gt_rank=104 | gap=0.3559

**Moral (query):**
> Every tale is not to be believed.

**Rank-1 (wrong) — `The Wolf And The Nurse` — score 0.7172:**
> A nurse was annoyed because her infant charge kept crying. When he refused to be quiet, she said that if he didn't stop bawling, she would throw him to the wolf. A wolf just happened to be passing by and he waited for the nurse to carry out her threat. But the child fell asleep, so the wolf went away bereft of the gain he could have gotten on his own, as well as the gain he had hoped to …

**Ground truth (rank 104) — `The Thief and The Boy` — score 0.3613:**
> A Boy sat weeping upon the side of a well. A Thief happening to come by just at the same time, asked him why he wept. The Boy, sighing and sobbing, showed a bit of cord, and said that a silver tankard had come off from it, and was now at the bottom of the well. The Thief pulled off his clothes and went down into the well, meaning to keep the tankard for himself. Having groped about for some …

**Claude Analysis:** *[taxonomy: MODEL_FOUND_BETTER_MATCH]* The Wolf and the Nurse story is a classic illustration of taking words too literally — the wolf believed the nurse's idle threat as a literal promise and acted on it, losing both potential gains. The moral "every tale is not to be believed" is the direct lesson: don't take every utterance as truth. The Thief and the Boy story is the inverse — the boy fabricates a tale (silver tankard) to deceive the thief. Both illustrate credulity and deception but from opposite angles. The wolf story illustrates the cost of believing an idle tale; the thief story illustrates how a false tale is used as a weapon. The model found the more direct victim-of-credulity story.

---
## Case 13 — Query 590 | gt_rank=622 | gap=0.3535

**Moral (query):**
> Watch the actions of your enemy.

**Rank-1 (wrong) — `The Cicada And The Fox` — score 0.5679:**
> A cicada was singing on top of a tall tree. The fox wanted to eat the cicada, so she came up with a trick. She stood in front of the tree and marvelled at the cicada's beautiful song. The fox then asked the cicada to come down and show himself, since the fox wanted to see how such a tiny creature could be endowed with such a sonorous voice. But the cicada saw through the fox's trick. He tore a …

**Ground truth (rank 622) — `The Sun and the Frogs` — score 0.2143:**
> Once upon a time, the Sun wanted to get married but the frogs raised a cry of protest up to the heavens. Jupiter, disturbed by their shouting, asked the reason for their complaint, and one of the swamp's inhabitants explained, 'Already one Sun is enough to burn up all the ponds, condemning us to a miserable death in our parched abode. What is going to happen to us when he will have sons of his own?'

**Claude Analysis:** *[taxonomy: CATASTROPHIC_LABEL_MISMATCH]* This is one of the most striking failures in the dataset. The Cicada and Fox story is a perfect illustration of "watch the actions of your enemy" — the cicada *literally* watches the fox's actions, sees through the trick, and outsmarts the predator. The story is about vigilance against an enemy. The labeled ground truth, The Sun and the Frogs (gt_rank=622, the second-worst in the dataset), is about frogs lobbying against the Sun getting married — the connection to "watch the actions of your enemy" is essentially absent. This is almost certainly a labeling error or a very distant metaphorical interpretation.

---
## Case 14 — Query 473 | gt_rank=323 | gap=0.3533

**Moral (query):**
> False men cannot be trusted.

**Rank-1 (wrong) — `The Crane And The Crow` — score 0.6686:**
> A crane and a crow had made a mutual pledge of assistance, agreeing that the crane was to defend the crow from other birds, while the crow would use her powers of prophecy to warn the crane about future events. These two birds often went to the field of a certain man and ate the crops that he had sowed there, tearing them up by the roots. When the farmer saw what was happening to his field, he was upset, …

**Ground truth (rank 323) — `The Shepherd And The Wolf Cub` — score 0.3153:**
> A shepherd found a little wolf cub and raised it. Then, when the cub was bigger he taught it to steal from his neighbours' flocks. Once he had learned how to do this, the wolf said to the shepherd, 'Now that you have shown me how to steal, take care that many of your own sheep don't go missing!'

**Claude Analysis:** *[taxonomy: NEAR_MISS]* Both fables involve betrayal and deceitful characters. The Crane and Crow story is about two birds who form a pact while secretly stealing — false cooperation masking false intent. The Shepherd and Wolf Cub story is about a wolf raised by a shepherd who turns treacherous — false loyalty eventually revealed. The Wolf Cub story more directly maps to "false men cannot be trusted" (the wolf appears domesticated/trustworthy but cannot be reformed). The Crane story is more about mutual deceit and theft. The model found a valid but less precise match.

---
## Case 15 — Query 626 | gt_rank=484 | gap=0.3526

**Moral (query):**
> It's wise to think before we speak.

**Rank-1 (wrong) — `The Fox And The Hare In The Well` — score 0.5595:**
> A thirsty hare had gone done into a well to drink the water. He took a good long drink, and when he wanted to get back out again, he found himself trapped with no means of escape. It was a very discouraging situation. A fox then arrived on the scene and when she found the hare she said to him, 'You have made a very serious mistake indeed: you should have first decided on a way to get out and …

**Ground truth (rank 484) — `A Country-man and a Hawk` — score 0.2069:**
> A Country Fellow had the luck to catch a Hawk chasing a Pigeon. The Hawk begged for her life, saying she never harmed the Country-man, so she hoped he wouldn't harm her. The Country-man replied, 'And what harm did the Pigeon ever do to you?' By your own reasoning, you should expect to be treated the same way you would have treated this Pigeon.

**Claude Analysis:** *[taxonomy: ATTRACTOR_FABLE + MORAL_SHIFT]* The Fox/Hare story appears again as an attractor (also top-1 in Case 4). However, the moral match is weaker here — the story is about thinking before you *act*, not thinking before you *speak*. The model conflated "think before acting" with "think before speaking." The Country-man and Hawk story actually illustrates verbal reasoning — the hawk speaks rashly (appeals to not harming the country-man) and is caught by her own argument turned back on her. The labeled gt is a better fit; the model over-matched on "think before you..." without distinguishing act vs speak.

---
## Case 16 — Query 105 | gt_rank=38 | gap=0.3467

**Moral (query):**
> Obscurity often brings safety.

**Rank-1 (wrong) — `The Fisherman And The Fish` — score 0.8260:**
> A fisherman was pulling in the net which he had just cast and, as luck would have it, the net was filled with all kinds of sea creatures. The little fish escaped to the bottom of the net and slipped out through its many holes, but the big fish was caught and lay stretched out flat aboard the boat.

**Ground truth (rank 38) — `The Tree and the Reed` — score 0.4794:**
> 'Well, little one', said a Tree to a Reed that was growing at its foot, 'why do you not plant your feet deeply in the ground, and raise your head boldly in the air as I do?' 'I am contented with my lot', said the Reed. 'I may not be so grand, but I think I am safer.' 'Safe!' sneered the Tree. 'Who shall pluck me up by the roots or bow my head to the ground?' But it soon …

**Claude Analysis:** *[taxonomy: DATASET_AMBIGUITY]* Both fables illustrate "obscurity brings safety" through parallel narrative logic. The Fisherman/Fish story: small fish (obscure, unremarkable) slips through net holes; big fish (prominent) is caught. The Tree/Reed story: tall tree (visible, prominent) is eventually uprooted by a storm; humble reed (small, obscure) survives by bending. Both are canonical Aesop fables for this moral. The model found the more concise, direct version; the dataset labeled the dialogic version as ground truth. This is genuine dataset ambiguity — two valid fables for the same moral.

---
## Case 17 — Query 515 | gt_rank=462 | gap=0.3455

**Moral (query):**
> Do not ask for more than you deserve.

**Rank-1 (wrong) — `Zeus And The Camel` — score 0.6682:**
> When the camel saw another animal's horns, she begged Zeus to give her horns too. Zeus was angry at the camel's greediness, so he cropped her ears instead.

**Ground truth (rank 462) — `Mercury And The Two Women` — score 0.3227:**
> Mercury was once the guest of two women who treated him in a cheap and tawdry manner. One of these women was the mother of an infant still in his cradle, while the other woman was a prostitute. In order to return the women's hospitality as they deserved, Mercury paused on the threshold of their door as he was leaving and said, 'You are gazing upon a god: I am prepared to give you right now whatever it is you …

**Claude Analysis:** *[taxonomy: ATTRACTOR_FABLE + SEMANTICALLY_ADJACENT_MORALS]* Zeus and the Camel appears again (also rank-1 in Case 3 for "Greed for more leads to losing what you have"). Here, the camel's moral is adjacent but slightly different — asking for more than you deserve vs greed leading to loss. The camel story fits both morals, making it a multi-moral attractor. The Mercury/Two Women story is about getting exactly what you deserve (Mercury gives each woman what she asked for, which turns out to be ironic punishment) — a "reap what you sow" variant of desert. The model found the more explicit greed-punishment story; the labeled gt encodes the moral through divine irony.

---
## Case 18 — Query 602 | gt_rank=640 | gap=0.3397

**Moral (query):**
> Consider all before you judge.

**Rank-1 (wrong) — `The Philosopher, The Ants, and Mercury` — score 0.5928:**
> A philosopher witnessed from the shore the shipwreck of a vessel, of which the crew and passengers were all drowned. He inveighed against the injustice of Providence, which would for the sake of one criminal perchance sailing in the ship allow so many innocent persons to perish. As he was indulging in these reflections, he found himself surrounded by a whole army of Ants, near whose nest he was standing. One of them climbed up and stung him, and he …

**Ground truth (rank 640) — `The Sheep and The Dog` — score 0.2531:**
> The Sheep one day complained to the Shepherd that while they were shorn of their fleece, and their young ones often taken and killed for food, they received nothing in return but the green herbage of the earth, which grew of itself, and cost him no pains to procure. 'On the other hand, your Dog,' said they, 'which gives no wool, and is of no use for food, is petted and fed with as good meat as his master.' 'Peace, …

**Claude Analysis:** *[taxonomy: MODEL_FOUND_BETTER_MATCH + CATASTROPHIC_LABEL_MISMATCH]* The Philosopher/Ants story is a precise illustration of "consider all before you judge" — the philosopher judges Providence harshly for the shipwreck, then immediately kills ants without considering the same logic he just applied. He is caught in his own failure to consider all perspectives before judging. The lesson is explicitly meta: the philosopher judged without considering. The labeled gt, The Sheep and the Dog (gt_rank=640, second from last), is about sheep complaining that the dog gets better treatment — which could be read as incomplete perspective-taking, but the connection is very weak. gt_rank=640 out of 709 is a near-complete failure of the ground truth label.

---
## Case 19 — Query 466 | gt_rank=562 | gap=0.3356

**Moral (query):**
> Not all advice is wise.

**Rank-1 (wrong) — `A Man that would not take a Clyster` — score 0.6303:**
> A certain man, German by birth, and exceedingly rich, was ailing. A large number of doctors had come to cure him (flies do indeed come swarming in crowds to honey), and one of them said among other things that the man needed an enema if he wanted to get well. When the man, who was unfamiliar with this medical procedure, heard what the doctor said, he flew into a rage and ordered that the doctors be thrown out of his …

**Ground truth (rank 562) — `The Rich Man And The Tanner` — score 0.2947:**
> A tanner was about to move in next door to a rich man but the rich man tried to get rid of him on account of the foul smell. The tanner said to him, 'It will bother you for a little while but then you will get used to it, and afterwards you will not even notice the smell.' The rich man said, 'We will not kill our sense of smell simply on account of your profession!'

**Claude Analysis:** *[taxonomy: MORAL_AMBIGUITY]* The match here is genuinely ambiguous. The Clyster story is about rejected medical advice — a man refuses a doctor's advice out of ignorance/pride. This fits "not all advice is wise" only if we read it as the man's rejection being justified, but the story's tone suggests the man was foolish to refuse. The Rich Man/Tanner story is about advice from the tanner ("you'll get used to the smell") that turns out to be false or irrelevant — clearer example of unwise advice being offered. The moral's interpretation (unwise advice given vs unwise rejection of advice) creates ambiguity about which story fits. The model may have matched on "advice refused" rather than "advice that is unwise."

---
## Case 20 — Query 297 | gt_rank=457 | gap=0.3353

**Moral (query):**
> There is danger lurking in the words of a wicked person.

**Rank-1 (wrong) — `The Wolf And The Goat On The Cliff` — score 0.6383:**
> There was a goat grazing up high on a cliff. At the bottom of the cliff there was a wolf who wanted to catch the goat and eat her. Since it was impossible for the wolf to climb up the cliff, he stood down below and said to the goat, 'You poor creature! Why have you left the level plains and meadows in order to graze upon the cliff? Are you trying to tempt death from that height?' The goat …

**Ground truth (rank 457) — `The Mother Dog And Her Puppies` — score 0.3030:**
> A dog who was about to give birth to puppies asked another dog if she could deliver the litter in her kennel. The owner of the kennel agreed. Later on, when the owner asked for her house back, the mother dog begged her to let her stay just a little while longer, until her puppies were strong enough to follow her. When this new deadline had passed, the owner of the kennel began to assert her claim more forcefully, but …

**Claude Analysis:** *[taxonomy: MODEL_FOUND_BETTER_MATCH]* The Wolf and Goat story is almost definitionally about "danger lurking in the words of a wicked person" — the wolf (wicked) uses deceptive, seemingly caring words ("you poor creature!") to try to lure the goat down to her death. The goat recognizes the danger in the wolf's words. This is a perfect embodiment of the moral. The Mother Dog story is about a dog using progressively more persuasive language to delay eviction — a softer form of manipulation. The wolf story has a predator using words as a weapon; the model found the more explicit and dangerous version. gt_rank=457 confirms the labeled fable is weak in embedding space.

---
## Case 21 — Query 118 | gt_rank=37 | gap=0.3274

**Moral (query):**
> No gratitude from the wicked.

**Rank-1 (wrong) — `The Farmer and the Snake` — score 0.8052:**
> One Winter a Farmer found a Snake stiff and frozen with cold. He had compassion on it, and taking it up, placed it in his bosom. The Snake was quickly revived by the warmth, and resuming its natural instincts, bit its benefactor, inflicting on him a mortal wound. 'Oh', cried the Farmer with his last breath, 'I am rightly served for pitying a scoundrel.'

**Ground truth (rank 37) — `The Woodman and the Serpent` — score 0.4778:**
> One wintry day a Woodman was tramping home from his work when he saw something black lying on the snow. When he came closer he saw it was a Serpent to all appearance dead. But he took it up and put it in his bosom to warm while he hurried home. As soon as he got indoors he put the Serpent down on the hearth before the fire. The children watched it and saw it slowly come to life again. …

**Claude Analysis:** *[taxonomy: DATASET_DUPLICATE]* This is the clearest case of a near-duplicate fable pair in the dataset. Both stories have identical structure: a human (farmer / woodman) finds a frozen snake/serpent in winter, warms it in his bosom out of compassion, and is fatally bitten in return. Both teach "no gratitude from the wicked" as their direct moral. The only differences are minor details (farmer vs woodman, snake vs serpent, bosom vs fireside). The model found "The Farmer and the Snake" at rank 1 with high confidence (0.805); the nearly-identical "The Woodman and the Serpent" is labeled as ground truth at rank 37. This is a dataset annotation issue, not a model failure. Both are equally valid answers.

---
## Case 22 — Query 280 | gt_rank=266 | gap=0.3264

**Moral (query):**
> Good will is worth nothing unless it is accompanied by good acts.

**Rank-1 (wrong) — `The Two Dung Beetles` — score 0.5945:**
> There was a bull who was pastured on a little island. Two dung beetles lived there too, feeding on the bull's manure. Winter was approaching, so one of the dung beetles said to the other, 'I want to go to the mainland and I will live there by myself during the winter. If I happen to find a good feeding ground over there, I bring back something for you too.' The beetle then moved to the mainland and found a …

**Ground truth (rank 266) — `The Deer And Her Friends` — score 0.2681:**
> A Stag had fallen sick. He had just strength enough to gather some food and find a quiet clearing in the woods, where he lay down to wait until his strength should return. The Animals heard about the Stag's illness and came to ask after his health. Of course, they were all hungry, and helped themselves freely to the Stag's food; and as you would expect, the Stag soon starved to death.

**Claude Analysis:** *[taxonomy: NEAR_MISS]* Both fables illustrate the gap between good intentions and good deeds. The Two Dung Beetles story: one beetle promises to bring back food (good will) but never does (no good act). The Deer and Her Friends story: animals come with good will (to wish the deer well) but good acts (not eating his food) are absent — and the deer dies. The deer story is the more direct illustration: good intentions are not enough when actions harm the beneficiary. The beetle story shows empty promises. The model found a valid but weaker match; the labeled gt encodes the moral more completely.

---
## Case 23 — Query 445 | gt_rank=21 | gap=0.3262

**Moral (query):**
> Appearances can be deceiving.

**Rank-1 (wrong) — `A Cat and Mice` — score 0.8062:**
> As a group of mice were peeking out of their holes to see what was around, they spotted a cat on a shelf. The cat looked so calm and harmless, as if she had no life or spirit in her. 'Well,' said one of the mice, 'that's a kind creature, I'm sure of it. You can see it in her face, and I really want to get to know her.' No sooner said than done; but as soon as the …

**Ground truth (rank 21) — `The Mice And The Weasel` — score 0.4800:**
> A weasel, enfeebled by old age and senility, was no longer able to pursue the swift-footed mice, so she decided to coat herself with flour and lie down nonchalantly in a dark corner of the house. One of the mice thought that she must be something good to eat, but as soon as he pounced, the weasel caught him and consigned him to oblivion; another mouse did the same, and a third mouse likewise met his doom. A few mice …

**Claude Analysis:** *[taxonomy: DATASET_AMBIGUITY]* Both fables tell nearly the same story from the same victim's perspective with the same moral. In both cases, a predator (cat / weasel) uses a deceptive appearance (calm/harmless / coated in flour) to fool mice, who approach and are killed. The narrative structure is essentially identical; only the predator species and the deception mechanism differ. Both fables teach "appearances can be deceiving" directly and explicitly. The model found one version; the dataset labeled the other as ground truth. This is clear dataset ambiguity — two near-equivalent fables for the same moral.

---
## Case 24 — Query 204 | gt_rank=54 | gap=0.3248

**Moral (query):**
> Someone who lays a trap for others will fall victim to it himself.

**Rank-1 (wrong) — `The Goat And The Donkey` — score 0.8066:**
> There was a man who kept a goat and a donkey. The goat was jealous of the donkey because he was given more to eat, so she made a deceptive proposal to the donkey, under the guise of giving him advice. 'Look,' said the goat, 'you are always being punished, constantly having to turn the millstone or carry burdens on your back. Why don't you pretend to have a seizure and throw yourself into a ditch?' The donkey trusted the …

**Ground truth (rank 54) — `The Bird Catcher And The Partridge` — score 0.4818:**
> A bird catcher had captured a partridge and was ready to strangle her right there on the spot. The partridge wanted to save her life so she pleaded with the bird catcher and said, 'If you release me from this snare, I will lure many partridges here and bring them to you.' The bird catcher was made even more angry by this and he killed the partridge immediately.

**Claude Analysis:** *[taxonomy: MODEL_FOUND_BETTER_MATCH]* The Goat and Donkey story is a direct enactment of this moral — the goat lays a trap for the donkey (advises him to fake a seizure and fall into a ditch), the donkey does so, is injured, and the owner slaughters the goat for medicine. The trap-layer falls victim to their own scheme. This is a textbook illustration. The Bird Catcher/Partridge story is thematically related (a trap has been laid, the partridge offers to lay a trap for others) but the moral is more about the wickedness of betraying one's kind than the trapper being caught. The model found the more precise fit.

---
## Case 25 — Query 57 | gt_rank=329 | gap=0.3242

**Moral (query):**
> He laughs best that laughs last.

**Rank-1 (wrong) — `Mercury And The Two Women` — score 0.6194:**
> Mercury was once the guest of two women who treated him in a cheap and tawdry manner. One of these women was the mother of an infant still in his cradle, while the other woman was a prostitute. In order to return the women's hospitality as they deserved, Mercury paused on the threshold of their door as he was leaving and said, 'You are gazing upon a god: I am prepared to give you right now whatever it is you …

**Ground truth (rank 329) — `The Heifer and the Ox` — score 0.2952:**
> A Heifer saw an Ox hard at work harnessed to a plow, and tormented him with reflections on his unhappy fate in being compelled to labor. Shortly afterwards, at the harvest festival, the owner released the Ox from his yoke, but bound the Heifer with cords and led him away to the altar to be slain in honor of the occasion. The Ox saw what was being done, and said with a smile to the Heifer: For this you were …

**Claude Analysis:** *[taxonomy: ATTRACTOR_FABLE]* Mercury and the Two Women appears as an attractor across multiple queries (see also Case 28). The Mercury story is about divine retribution — Mercury gives each woman the opposite of what she actually wanted, so the "last laugh" belongs to Mercury/justice. The Heifer/Ox story is a more direct enactment: the heifer mocked the working ox, but the ox gets the last laugh when the heifer is led to slaughter. The Heifer story has a clear temporal structure (mock → reversal → "he who laughs last") that the Mercury story only approximates through divine irony. The model matched on poetic-justice framing rather than the specific "laughs last" reversal.

---
## Case 26 — Query 691 | gt_rank=87 | gap=0.3210

**Moral (query):**
> Be content with your lot.

**Rank-1 (wrong) — `The Author` — score 0.7692:**
> If Nature had formed the human race according to my notions, it would have been far better endowed: for she would have given us every good quality that indulgent Fortune has bestowed on any animal: the strength of the Elephant, and the impetuous force of the Lion, the age of the Crow, the majestic port of the fierce Bull, the gentle tractableness of the fleet Horse; and Man should still have had the ingenuity that is peculiarly his own. Jupiter …

**Ground truth (rank 87) — `A Pike sets up for Sovereignty` — score 0.4482:**
> There was a Master-Pike, who for his size, beauty, and strength, was seen as the Prince of the River. But being the ruler of the freshwater wasn't enough for him; he wanted to rule the sea too. With this ambitious plan, he ventured into the ocean and claimed it as his own. However, a mighty Dolphin took offense at this intrusion and chased the Pike back to the edge of his own stream, barely allowing him to escape. From then …

**Claude Analysis:** *[taxonomy: NEAR_MISS]* Both fables illustrate discontent with one's natural station. The Author story: a person imagines being given every animal's best quality and is corrected by Jupiter — nature gave each creature exactly what it needs. The Pike story: a pike rules freshwater supremely but tries to rule the sea, is humiliated, and must return to its own domain. Both carry the "be content" message. The Author story is more philosophical (discontent with being human); the Pike story is more narrative (overreach, failure, return). The model found the reflective/philosophical version; the labeled gt used the action-based narrative.

---
## Case 27 — Query 440 | gt_rank=140 | gap=0.3209

**Moral (query):**
> A sinful mind can even change a person's nature, causing it to be impaired.

**Rank-1 (wrong) — `The Cat-Maiden` — score 0.6591:**
> The gods were once disputing whether it was possible for a living being to change its nature. Jupiter said Yes, but Venus said No. So, to try the question, Jupiter turned a Cat into a Maiden, and gave her to a young man for a wife. The wedding was duly performed and the young couple sat down to the wedding-feast. See, said Jupiter, to Venus, how becomingly she behaves. Who could tell that yesterday she was but a Cat? Surely …

**Ground truth (rank 140) — `The Mole And His Mother` — score 0.3382:**
> The mole is a handicapped animal: he is blind. There was once a mole who wanted to kiss his mother, but instead of pressing up against her mouth, he pressed against her private parts. His brothers realized what he was doing and one of them remarked, 'It serves you right! You had great expectations, but you have gone and lost even your sense of smell.'

**Claude Analysis:** *[taxonomy: PARTIAL_MATCH + MORAL_INVERSION]* The Cat-Maiden story argues the opposite of the query moral — it concludes that nature *cannot* be changed (the cat-maiden reverts to chasing mice). But the query moral says a sinful mind *can* change nature, causing impairment. The model matched on "nature / change / impairment" language without catching the inversion. The Mole story maps more directly: the mole's impaired nature (blindness, but also implied moral failing) leads to further impairment (loss of smell). This is a case where the model matched surface-level thematic keywords while missing the logical direction of the moral.

---
## Case 28 — Query 498 | gt_rank=296 | gap=0.3188

**Moral (query):**
> He laughs best that laughs last.

**Rank-1 (wrong) — `Mercury And The Two Women` — score 0.6194:**
> Mercury was once the guest of two women who treated him in a cheap and tawdry manner. One of these women was the mother of an infant still in his cradle, while the other woman was a prostitute. In order to return the women's hospitality as they deserved, Mercury paused on the threshold of their door as he was leaving and said, 'You are gazing upon a god: I am prepared to give you right now whatever it is you …

**Ground truth (rank 296) — `The Bull And The Bullock` — score 0.3006:**
> There was a bullock who had been turned loose in the fields without ever having borne the burden of the yoke. When he saw a hard-working bull who was pulling a plow, the bullock said to him, 'You poor thing! What a lot of hard work you have to endure!' The bull made no reply and continued pulling the plow. Later on, when the people were about to make a sacrifice to the gods, the old bull was unyoked and …

**Claude Analysis:** *[taxonomy: ATTRACTOR_FABLE + DATASET_AMBIGUITY]* Mercury and the Two Women appears for the third time across the top-30 cases (also Cases 25 and 57), making it one of the strongest attractor fables for "last laugh"-type morals. The Bull and Bullock story is a clean "he laughs last" narrative — the bullock mocked the working bull, but when sacrifice time came, the old bull was freed while the young bullock was slaughtered. The bull gets the last laugh. This is also an instance of dataset ambiguity: the moral "He laughs best that laughs last" appears for multiple queries (57 and 498), pointing to multiple labeled fables for the same moral.

---
## Case 29 — Query 267 | gt_rank=211 | gap=0.3061

**Moral (query):**
> Vanity can lead to self-destruction.

**Rank-1 (wrong) — `The Frog And The Ox` — score 0.6042:**
> There was once a frog who noticed an ox standing in the meadow. The frog was seized by a jealous desire to equal the ox in size so she puffed herself up, inflating her wrinkled skin. She then asked her children if she was now bigger than the ox. They said that she was not. Once again she filled herself full of air, straining even harder than before, and asked her children which of the two of them was bigger. …

**Ground truth (rank 211) — `The Birds, The Peacock And His Feathers` — score 0.2981:**
> The peacock was a remarkable bird both because of the beauty of his feathers with their various colours and also because he was gentle and courteous. On his way to the assembly of the birds, the peacock ran into the raven. The raven asked the peacock if he would give him two of his feathers. The peacock said, 'What will you do for me in return?' The raven replied, 'I will squawk your praises throughout the courts in the presence …

**Claude Analysis:** *[taxonomy: MODEL_FOUND_BETTER_MATCH]* The Frog and the Ox story is a direct enactment of the moral — the frog's vanity (wanting to match the ox's size) drives her to inflate herself until she bursts, destroying herself. This is a literal "vanity leading to self-destruction." The Peacock story is less clear: the peacock gives away feathers in exchange for flattery (the raven's praises), which might impoverish the peacock — but the story is more about vanity exploited by flattery than vanity causing active self-destruction. The model found the more literal narrative match. The labeled gt is a subtler, indirect illustration of the same moral.

---
## Case 30 — Query 664 | gt_rank=104 | gap=0.3057

**Moral (query):**
> Do not fight with those who are perfectly capable of fighting back.

**Rank-1 (wrong) — `The Viper And The File` — score 0.7083:**
> A viper entered a blacksmith's workshop and bit the file, testing it to see if this was something she could eat. The file protested fiercely, 'You fool! Why are you trying to wound me with your teeth, when I am able to gnaw through every sort of iron?'

**Ground truth (rank 104) — `A Husband and Wife twice Married` — score 0.4026:**
> A certain man, after the death of his wife whom he had greatly loved, married another woman who was herself a widow. She continuously prattled to him about the virtues and great deeds of her late husband. The man, in order to give tit for tat, also talked all the time about the excellent character and remarkable good wisdom of his late wife. One day, however, the woman grew angry and gave to a poor man who was begging alms …

**Claude Analysis:** *[taxonomy: MODEL_FOUND_BETTER_MATCH]* The Viper and the File story is a near-perfect, minimalist illustration of this moral — the viper attacks something (the file) that is specifically impervious and capable of fighting back at a structural level. The file's response is essentially the moral stated directly: "why fight me when I can gnaw through iron?" The Husband/Wife story requires several inferential steps to reach the same moral (couple fight verbally → both capable of fighting back → standoff). The model found the more direct and explicit version. The labeled gt encodes the moral through a social/human conflict; the model matched on the more literal physical confrontation.
