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

## 2026-09-04

**D19. Build the first sequential-model slice as a next-switch example from live data.**
Ten clean, chronological battles form the input and the eleventh battle supplies only a binary deck-switch target.
The data contract keeps deck ids and card levels as separate 10 by 8 arrays and stores result, crown difference, prior switch, Jaccard switch magnitude, log time gap in hours, and trophies as battle features.
The first row's transition features are zero because the battle before the chosen window is outside the model input.
The first implementation uses RoyaleAPI data, then a Season 18 adapter must produce the same contract before historical pre-training is compared.

**D20. Map card ids to deterministic vocabulary indices before embedding lookup.**
The vocabulary sorts the outcome-blind reference card ids, assigns known cards indices starting at one, and reserves index zero for an unseen card.
Vocabulary indices carry no numerical similarity and exist only to address rows in a future embedding table.

## 2026-09-05

**D21. Start the sequential card encoder with a trainable, normally initialized embedding table.**
`crdata/card_encoder.py` maps vocabulary indices to vectors initialized independently with mean zero and standard deviation `1 / sqrt(embedding_dim)`.
The initialization gives each row an expected squared Euclidean norm of one.
The table is intended to learn jointly with the next-switch model, separate from the outcome-blind archetype space.
Index zero represents an unknown card, not padding, and remains trainable when used.
Embedding dimension is an explicit constructor argument; 24 remains an example rather than a validated choice.
Card-level fusion, deck pooling, and GRU construction remain subsequent steps.

**D22. Convert rarity-relative API levels before model standardization.**
`crdata/card_levels.py` adds the documented rarity offset to every raw API level, producing the displayed level scale while preserving actual level differences.
The collector retains raw API values, and conversion occurs only while preparing model inputs.
An unknown card rarity is rejected because its correct offset cannot be inferred from the stored level alone.
Training-set mean and standard deviation will be fitted on the converted levels in the next encoder step.

**D23. Standardize displayed card levels with training-set statistics before concatenation.**
`crdata/card_levels.py` fits the population mean and standard deviation using only converted training levels.
Validation, test, and inference inputs reuse those fixed statistics.
`crdata/card_encoder.py` appends the standardized level as one scalar to each card embedding, preserving the association between card identity and level.
The concrete statistics remain unset until the player-level data split exists.
Card pooling remains a subsequent decision.

**D24. Use a two-dense-layer shared MLP for the first per-card transformation.**
`crdata/card_encoder.py` maps each card's embedding-plus-level vector through `Linear(input_dim, hidden_dim)`, GELU, and `Linear(hidden_dim, hidden_dim)`.
The same `CardMLP` parameters process every card in every battle.
Xavier uniform initializes both dense weight matrices, and both bias vectors start at zero.
The initial candidate dimensions are 25 inputs and 48 outputs when the embedding dimension is 24, but those dimensions remain validation hyperparameters.
The next-switch loss will supervise this representation only after the full history model and training pipeline exist.
