# Data inventory

Every dataset in the project, what each column holds, and which questions each source can answer.
Schemas below were read off the files, not from memory.

## Where everything lives

| dataset | rows | location |
|---|---|---|
| Season 18 battles | 36,865,856 across 9 day files | scratchpad, `season18_parquet/` |
| Player table | 11,007,296 | `data/players.parquet` |
| Live collector battles | 3,687 so far | `data/battles/date=*/` |
| Card reference | 122 | `data/reference/cards.parquet` |
| Clan pool | 1,416 | `data/state/clan_pool.parquet` |
| Cohort | 240, being replaced | `data/state/cohort.parquet` |

Season 18 sits in a session scratchpad rather than the repo, because it is 1.7 GB and `data/` is gitignored.
It took hours to build from 22 GB of CSV and is not reproducible without re-downloading the source.

## Season 18 battles

One row per battle, 46 columns.
The two players are side `a` and side `b`.
**Side A is not the winner.**
Storage puts the alphabetically smaller player tag on side A, and orientation is randomised at load per D10.

**Who played**
`a_tag`, `b_tag` are player identifiers.
`a_clan`, `b_clan` are clan tags, empty string when the player has no clan.

**When and where**
`battle_time` is the timestamp.
`game_mode` and `arena` are numeric ids.

**The decks**
`a_card1` to `a_card8` and `b_card1` to `b_card8` hold card ids.
`a_level1` to `a_level8` and `b_level1` to `b_level8` hold each card's level.
Levels run 1 to 13 where 13 means fully upgraded, so a maxed deck sums to 104.
Rarities floor at different points: Common 1, Rare 3, Epic 6, Legendary 9.

**The result**
`a_won` is 1 when side A won.
`a_crowns`, `b_crowns` give the margin, 0 to 3.
`a_king_hp`, `b_king_hp` give remaining king tower health, where 0 means the tower fell.

**Skill proxy**
`a_trophies`, `b_trophies` are trophies at the start of the battle.
Matchmaking pairs players within 50 trophies in 99.86 percent of battles, so this barely varies within a battle.

**What it does not have**
Nothing about what happened during the battle.
No evolutions, no Tower Troops, no elixir management.
Those did not exist in December 2020 or were not recorded.

## Player table

One row per player, 18 columns, built by `scripts/build_player_table.py` from the Season 18 battles.
Every player is included with no minimum battle count, so any threshold applies downstream per `CONVENTIONS.md`.

**Volume**
`battles` is how many battles the player fought.
`distinct_decks` is how many different decks they used.

**Consistency**
`top_deck_share` is the fraction of battles on their most-used deck.

**Switching**
`switch_rate` is the fraction of consecutive battles where the deck changed.
`switch_after_loss` and `switch_after_win` split that by the previous result.
`loss_reactivity` is `switch_after_loss` minus `switch_after_win`.
`switch_drift` is how much the deck changed when it changed, as a Jaccard distance from 0 to 1.

**Investment**
`level_mean`, `level_first`, `level_last`, `level_max` track the deck level total over the season.
`level_gain` is `level_last` minus `level_first`.
`level_gain` moves both when a player upgrades a card and when they switch to a deck of different levels, so it is not a clean upgrade measure.

**Outcome**
`win_rate` is wins over battles.
`trophies_first`, `trophies_last`, `trophy_gain` track ladder movement.

## Live collector battles

One row per battle, 47 columns, written by `crdata/parse.py`.
Same side convention as Season 18.

Everything Season 18 has, plus:

**In-match behaviour**
`a_elixir_leaked`, `b_elixir_leaked` measure wasted elixir.
This is the only direct measure of how a player actually played, and it was present on 3,687 of 3,687 battles in the test run.

**Modern investment**
`a_card_evo`, `b_card_evo` give evolution level per card.
`a_support_ids`, `b_support_ids` hold Tower Troops, a ninth deck slot.

**Better outcome detail**
`a_princess_hp`, `b_princess_hp` give both side towers, not just the king.
`a_trophy_change`, `b_trophy_change` give the ladder movement from that battle.
`a_global_rank`, `b_global_rank` give leaderboard position where it applies.

**Battle context**
`type` separates pathOfLegend, riverRacePvP, trail, boatBattle, friendly and tournament.
`deck_selection`, `league_number`, `event_tag`, `is_ladder_tournament`, `is_hosted_match` describe the match setup.
`is_clean_1v1` flags rows usable without extra assumptions, true for 93 percent of the test run.
`battle_key`, `collected_at`, `run_id` support deduplication and provenance.

Decks are stored as list columns here (`a_card_ids`, `a_card_levels`) rather than eight numbered columns, because duels carry 16 or 24 cards.

## Reference and state

`data/reference/cards.parquet` holds `id`, `name`, `elixir`, `rarity` for 122 cards.
Only 102 existed in Season 18.
Mirror has no elixir cost, because it costs whatever it copies plus one.

`data/state/clan_pool.parquet` holds `clan_tag`, `clan_name`, `clan_score`, `members` for 1,416 clans found by name search, scores from 16,322 to 139,991.

`data/state/cohort.parquet` holds the players being polled, with `tag`, `name`, `clan_tag`, `added_at`, `last_polled`, `n_polls`, `n_battles`, `n_fail`.
The 240 rows there are a war-clan test seed and are being replaced by a trophy-balanced cohort.

## Counts and aggregates

`data/card_level_counts.npz` holds `battles` and `wins`, each an 80 by 81 grid of trophy band against card level difference.
This is what `figures/card_level_effect.png` is drawn from.

`data/profile_level_gap.npz` holds `level_histogram` (per card, per level), `conditioning_cells`, `trophy_difference`, and `card_ids`.
This is what proved the card level scale is unified at 1 to 13.

## What each source can answer

**Season 18 answers scale questions.**
36.9M battles, 768,410 players with 20 or more battles, 5.6M deck switches.
It cannot say anything about how a battle was played, about evolutions, or about the current game.

**The live collector answers mechanism questions.**
`elixir_leaked` measures piloting quality directly, which is the only way to test whether a post-switch dip is real skill degradation rather than a leftover.
It cannot reach Season 18's scale, and it can only collect forward, since the API keeps just the last 30 battles per player.

The two are kept separate and compared rather than merged into one model.
Card levels cap at 13 in Season 18 and 16 now, so any cross-era level statement uses distance from max, never the raw number.
