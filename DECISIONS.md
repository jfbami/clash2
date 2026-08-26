# Decisions log

Append-only.
Add new entries at the bottom with the date they were made.
Never edit or delete an earlier entry.
When a decision is reversed, add a new entry that says so and names the entry it supersedes.

A decision here is current only if no later entry supersedes it.
Being in this file is not the same as being in force.

---

## 2026-08-19

**D1. Poll a fixed cohort rather than crawling outward.**
The Clash Royale API returns only a player's last 30 battles, covering about a day.
Discovering new players adds breadth; revisiting the same players adds depth.
Depth is what allows a deck effect to be measured inside one player, holding that player's skill and spending constant.

**D2. Deduplicate on battle time plus the sorted pair of player tags.**
The API exposes no battle ID, and every battle appears in both participants' logs.
See `crdata.api.battle_key`.

**D3. Flag rows rather than dropping them during collection.**
Duels, boat battles and 2v2 are parsed and marked with `is_clean_1v1`.
Filtering is a modelling decision and belongs downstream.

## 2026-08-20

**D4. Report effect sizes, not accuracy against a 50 percent baseline.**
Accuracy answers whether a signal exists, not how large it is.
The research question is an effect size question.

**D5. Use a nested baseline ladder.**
The increment from one rung to the next is what each information source is worth.
Recorded in `RESULTS_BASELINE_LADDER.md`.

**D6. Drop the player skill embedding from the single-day model.**
Ablation showed the model improves without it.
One day gives about 2.3 battles per player, so a per-player parameter fits noise.
Superseded in scope by D14, which makes skill estimable on the full season.

## 2026-08-21

**D7. Use the bwandowando Season 18 dataset over the s1m0n38 92 GB dataset.**
Season 18 carries per-card levels; the larger dataset does not.
Without card level, deck design and account investment cannot be separated at all.
See `REJECTED.md` R1.

**D8. Hybrid data strategy.**
Season 18 for the core model, the live collector for the current evolution era.
Neither source alone covers both scale and the modern card pool.

## 2026-08-22

**D9. Exclude 4 January 2021 from all modelling.**
Assumption A1 was tested and failed.
The final day of the season runs +0.250 mean card level and +186 mean trophies against mid-season, because casual players have stopped playing.
It is also 99.99 percent one game mode, where the season overall is mixed.

**D10. Randomise orientation at load with a fixed seed.**
Supersedes the canonical tag ordering used at storage time.
The lexicographically smaller tag wins 50.61 percent of battles, n = 1.1M, z = +12.80, because Supercell issues tags roughly in sequence and older accounts have more leveled cards.
Storage stays canonical for deterministic deduplication; the label is randomised downstream.
See `REJECTED.md` R4.

**D11. Read a null king tower hit point value as zero, not as a missing row.**
Null occurs if and only if the winner took three crowns, confirmed on 107,283 rows with perfect correspondence in both directions.
Dropping those rows deleted 26.8 percent of battles, all of them the most decisive wins.

**D12. Keep all three ladder modes, exclude 2v2.**
Ladder, Crown Rush and Gold Rush all carry `valid_ladder_mode: true`, real card levels and player-chosen decks.
2v2 has two players per side and was parsed incorrectly by an ETL built for 1v1.

## 2026-08-24

**D13. The archetype space must be outcome-blind.**
Defining archetypes with win and loss data and then measuring archetype strength is circular.
`crdata/embedding.py` uses deck co-occurrence only.

**D14. Keep player skill inside the research question.**
The full season gives 2,070,597 players with 10 or more battles, so skill is estimable.
Report both the effect conditional on matchmaking and the population-level effect, since matchmaking compresses the first by design.

**D15. Count deck instances, not unique decks, when building card co-occurrence.**
Archetypes should reflect the meta as actually played.

**D16. Keep negative PMI rather than clamping to zero.**
The 102 by 102 matrix rests on 4.13 billion pair observations, so it is dense and precisely estimated, unlike the sparse text matrices the clamping convention was built for.
A negative value means two cards compete for the same deck slot, which is real information.
64.2 percent of card pairs are negative, so clamping would discard most of the signal.

**D17. Card space dimension is 24, chosen by validation rather than by the spectrum.**
The singular value spectrum has no elbow, so it does not support any particular choice.
Win condition recovery improves monotonically to k=48; 24 is where the gain begins flattening.
This overrode a user instruction to choose from the spectrum and was flagged as such.

## 2026-08-26

**D18. Park the 3D projection and the masked card model.**
Neither is on the path to answering the research question.
The PMI baseline already passes its validation, and 3D is presentation only.
Revisit after the deck strength measurement exists.
