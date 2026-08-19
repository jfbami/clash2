# Clash Royale API — measured facts (probed 2026-08-19)

Access: key is IP-locked to `45.79.218.79`, so **all calls must go through
`https://proxy.royaleapi.dev/v1`**. Tier `developer/silver`.

## Endpoint reality vs. the published Swagger

| Endpoint | Status |
|---|---|
| `/locations/global/rankings/players` | **returns 0 items** — trophy ladder is dead |
| `/locations/global/pathoflegend/players` | works — this is the live player leaderboard |
| `/locations/global/rankings/clans` | works |
| `/locations/global/rankings/clanwars` | works — best seed for war battles |
| `/clans?name=…` | works — alternate seed source |

`57000000` is **Europe**, not global. Global is the literal string `global`.

## Battlelog

- Page size: **30 battles** (not 25).
- Retention: **~1 day** for an active player. This is the binding constraint:
  within-player deck variation can only be built by **repeated scheduled polling**.
- **No battle ID exists.** A battle appears in both participants' logs, so it is
  collected twice with team/opponent swapped. Dedup on
  `(battleTime, sorted(participant tags))` — see `crdata.api.battle_key`.
- Sustained throughput measured: **~2.3 req/s** with no 429. No rate-limit
  headers are exposed by the proxy.

## Fields the original analysis did not use

- `elixirLeaked` — per-match skill measurement (observed 2.14 vs 9.76).
- `kingTowerHitPoints`, `princessTowersHitPoints` — graded outcome, far more
  informative than binary win/loss.
- `supportCards` — Tower Troop, effectively a 9th deck slot. Omitted entirely
  from a cards-only encoding.
- `globalRank`, `startingTrophies`, `trophyChange` — skill proxies.
- `evolutionLevel` on cards — distinguishes an evolved card from its base.
- `riverRaceDuel` carries 16 cards + a `rounds` field (best-of-3, multiple decks).
  Must be parsed separately or excluded.

## Card levels by battle type — the key result

n = 4,970 decks sampled from members of the top 12 clan-war clans.

| battle type | n decks | max lvl | mean lvl | p10 | p90 | % decks w/ evo |
|---|---|---|---|---|---|---|
| pathOfLegend | 3106 | 16 | 12.74 | 8 | 16 | 99.8% |
| trail | 1278 | 16 | 9.13 | 3 | 14 | 61.7% |
| **riverRacePvP (War Day)** | **244** | **16** | **12.26** | **8** | **16** | **98.0%** |
| riverRaceDuel | 88 | 16 | 12.51 | 8 | 16 | 100.0% |
| boatBattle | 102 | 16 | 11.79 | 7 | 16 | 97.1% |
| friendly | 120 | **11** | 7.31 | 3 | 11 | 30.0% |
| tournament | 6 | **11** | 7.12 | 3 | 11 | 100.0% |

**Interpretation.** `friendly` and `tournament` cap at level 11 — that is
tournament standard, and it proves the API reports *normalized* levels when
normalization applies. `riverRacePvP` reaches level 16 with a p10 of 8.

Therefore **War Day battles are NOT played at tournament-standard card levels.**
The premise that War Day "isolates deck effects from skill confounds" does not
hold in the data. War decks carry raw account card levels spanning 8 levels of
account investment, and 98% of them contain an evolution.

This is an internally controlled result: same API, same parser, same sample —
the modes where normalization *is* expected show the cap, and War Day does not.
