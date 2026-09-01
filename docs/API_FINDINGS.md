# Clash Royale API measured behaviour

Facts measured against the live Clash Royale API on 2026-08-19.
Supercell does not document most of the values below.

## Access

The API key used by this project is IP-locked to `45.79.218.79`, the RoyaleAPI proxy.
All calls must therefore use the base URL `https://proxy.royaleapi.dev/v1`.
The key tier is `developer/silver`.

## Endpoints that differ from the published Swagger

`/locations/global/rankings/players` returns zero items because the trophy ladder was replaced by Path of Legends.

`/locations/global/pathoflegend/players` is the live player leaderboard.

`/locations/global/rankings/clans` and `/locations/global/rankings/clanwars` both work.
`/locations/global/rankings/clanwars` is the best seed source for war battles.

`/clans?name=...` works as an alternate seed source.

The location ID `57000000` is Europe, not global.
The global pseudo-location is the literal string `global`.

## Battlelog behaviour

The `/players/{tag}/battlelog` endpoint returns 30 battles, not the 25 commonly cited.

Battlelog retention is about one day for an active player.
Short retention is the binding constraint on this project.
Within-player deck variation can only be built by repeated scheduled polling.

The Clash Royale API exposes no battle ID.
A battle appears in the battlelog of both participants, so a naive crawl collects that battle twice with the sides swapped.
`crdata.api.battle_key` deduplicates on battle time plus sorted participant tags.

Measured sustained throughput is about 2.3 requests per second with no 429 responses.
The RoyaleAPI proxy exposes no rate-limit headers.

## Fields available beyond card identity

`elixirLeaked` is a per-match skill measurement.
Observed values ranged from 2.14 to 9.76 in one sample.

`kingTowerHitPoints` and `princessTowersHitPoints` give a graded outcome instead of a binary win or loss.

`supportCards` holds the Tower Troop, effectively a ninth deck slot.
A cards-only encoding omits `supportCards` entirely.

`globalRank`, `startingTrophies`, and `trophyChange` are skill proxies.

`evolutionLevel` appears on a card object only when that card is evolved.

The `riverRaceDuel` battle type carries 16 or 24 cards plus a `rounds` field because a duel is best of three.
Parse `riverRaceDuel` separately or exclude it.

## Card levels by battle type

Sample: 4,970 decks from members of the top 12 clan-war clans.

| battle type | decks | max level | mean level | p10 | p90 | percent with evolution |
|---|---|---|---|---|---|---|
| pathOfLegend | 3106 | 16 | 12.74 | 8 | 16 | 99.8 |
| trail | 1278 | 16 | 9.13 | 3 | 14 | 61.7 |
| riverRacePvP | 244 | 16 | 12.26 | 8 | 16 | 98.0 |
| riverRaceDuel | 88 | 16 | 12.51 | 8 | 16 | 100.0 |
| boatBattle | 102 | 16 | 11.79 | 7 | 16 | 97.1 |
| friendly | 120 | 11 | 7.31 | 3 | 11 | 30.0 |
| tournament | 6 | 11 | 7.12 | 3 | 11 | 100.0 |

The `friendly` and `tournament` battle types cap at level 11, which is tournament standard.
The level 11 cap proves the Clash Royale API reports normalised levels wherever normalisation applies.

The `riverRacePvP` battle type, which is War Day, reaches level 16 with a p10 of 8.
War Day battles are therefore not played at tournament-standard card levels.
War Day decks carry raw account card levels spanning eight levels of account investment, and 98 percent of War Day decks contain an evolution.

The comparison above is internally controlled.
The same API, parser, and sample produce a level 11 cap in the modes where normalisation is expected and no cap in War Day.
