# Rejected approaches

Read this file before proposing an approach.
Everything here was considered and ruled out, with the reason and the date.

Rejections are not permanent laws.
If the reason no longer holds, say which reason changed and propose it again.
Do not propose one of these without addressing why it was rejected.

---

## R1. The s1m0n38 481M battle dataset

**Rejected 2026-08-21.**
92 GB covering September 2022 to December 2023, which spans the evolution era.

Its schema is `datetime, gamemode, tag, trophies, crowns, card1..card8`.
It carries **no card level**, so deck design and account investment cannot be separated in it at all.
That separation is the entire research question.

Eighteen times larger than Season 18 and missing the one control that matters.

**Reconsider if** the question narrows to something that does not need investment controls.

## R2. Win condition labels as archetype definitions

**Rejected 2026-08-24.**
`Wincons.csv` labels 24 of 102 cards as win conditions.

**57 percent of decks contain two or more win conditions**, and only 38.6 percent contain exactly one.
The label does not merely lose nuance, it fails to apply to most decks.

Hog Cycle, Hog Exenado and Hog EQ share a win condition and play nothing alike.

Win conditions are now used for **validation** instead: the outcome-blind space recovers them at 71.8 percent against a 21.2 percent baseline, having never been shown them.

## R3. A per-player skill embedding on single-day data

**Rejected 2026-08-20.**
Ablation: the model scored 59.31 percent without it against 59.24 percent with it.

One day gives about 2.3 battles per player, so the parameter fits noise.

**No longer applies to the full season**, which has 2,070,597 players with 10 or more battles.
See `DECISIONS.md` D14.

## R4. Canonical player tag ordering as the final label orientation

**Rejected 2026-08-22.**
Assigning side A to the lexicographically smaller tag looked deterministic and clean.

The smaller tag wins **50.61 percent** of battles, n = 1.1M, z = +12.80, consistently across all ten day files.
Supercell issues tags roughly in sequence, so tag order proxies account age, which proxies card level, which is the quantity under study.

Canonical ordering is still used **in storage** for deduplication.
Orientation is randomised at load with a fixed seed.

## R5. Accuracy against a 50 percent baseline as the headline metric

**Rejected 2026-08-20.**
This is the flaw the whole project exists to critique.

Accuracy is a whether metric.
It says a signal exists, not how large it is or what it is made of.

With n above a million, any effect above about 0.2 points clears p < 0.001, so significance carries no information.

Use effect sizes, a nested baseline ladder, and proper scoring rules.

## R6. A linear model alone for card identity

**Rejected 2026-08-20.**
Logistic regression on card identity scores 51.80 percent.
Gradient boosting on the same features scores 55.05 percent.

Linear models cannot represent synergy, so they understate deck design by roughly three points.
Reporting the linear figure alone would have produced an overclaim in the post.

Any comparison of deck against investment must give deck a model that can represent interactions.

## R7. Dropping rows with a null king tower hit point value

**Rejected 2026-08-22.**
Standard null handling deleted 26.8 percent of battles.

Null occurs **if and only if** the winner took three crowns, verified on 107,283 rows with perfect correspondence both ways.
It means the tower was destroyed.

Dropping them removed every decisive win, which is exactly the non-random deletion this project criticises the original study for.

Read null as zero.

## R8. Measuring anything on a 2D or 3D projection

**Rejected 2026-08-26.**
Deck-level win condition recovery is 71.8 percent in the full 24-dimensional space.
It falls to 36.9 percent in the best projection and 30.6 percent in the worst.

Every projection loses about half the signal.

Projections are for looking at.
Every reported number comes from the full space.

## R9. Defining archetypes from the matchup model's card embeddings

**Rejected 2026-08-24.**
Those embeddings are trained on win and loss.
Grouping decks with them and then reporting that a group is strong is circular, and a reviewer would see it immediately.

The archetype space is built from co-occurrence only.

## R10. Cheap previews and exploratory side tests

**Rejected 2026-08-24 by user instruction.**
Running a fast approximation as a stand-in for the real thing is not acceptable in this project.

Recorded in `../CLAUDE.md`.

## R11. Skip-gram instead of PMI plus SVD for the card space

**Not rejected on merit, deferred 2026-08-26.**
Levy and Goldberg (2014) proved skip-gram with negative sampling implicitly factorises a shifted PMI matrix, so the two are approximately the same algorithm.

At 102 items, skip-gram's advantages do not apply, and SVD is deterministic with one hyperparameter.

The PMI baseline passes validation.
A deeper model must beat 71.8 percent on the same test to displace it.
