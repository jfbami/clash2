# Clash Royale deck-effect study — data collection

## Pipeline

```
scripts/seed_cohort.py   # build a FIXED player cohort (run once, extend later)
scripts/collect.py       # one polling pass  (run repeatedly, on a schedule)
scripts/check_key.py     # verify API access, measure live limits
```

```
crdata/api.py     # HTTP client: proxy, 429 backoff, 403 diagnosis, dedup key
crdata/parse.py   # battle JSON -> one flat row, canonical side ordering
crdata/store.py   # Parquet append, persistent dedup, cohort state
```

Data lands in `data/battles/date=YYYY-MM-DD/<run_id>.parquet`; dedup hashes in
`data/state/seen_keys.parquet`; cohort in `data/state/cohort.parquet`.

## Design decisions

**Fixed cohort, polled repeatedly** — not a BFS crawl. Repeat observation of the
same players is what produces within-player deck variation, which is what allows
deck effects to be estimated free of player skill and account investment.

**Canonical side ordering** (`a` = lexicographically smaller tag) — removes the
player-ordering artifact deterministically at parse time rather than papering
over it with random swaps. Verified: `a` wins 51.2% (n=3,441, z=+1.38, n.s.).

**Persistent dedup** on `(battleTime, sorted participant tags)` — the API exposes
no battle ID, so a battle collected from both participants appears twice with
the sides swapped.

**Nothing is filtered at collection time.** Duels (16/24 cards), boat battles and
2v2 are parsed and flagged via `is_clean_1v1`. Filtering is a modelling decision.

## Verified against live data (2026-08-19, n=3,687 battles)

- Canonical ordering holds; label balance is unbiased.
- Evolutions per deck: 63% run 3, 25% run 2, 11% run 0.
- `elixir_leaked` present on 100% of rows — a per-match skill measure.
- Polled players: median 30 battles, 4 distinct decks, 85.7% use >1 deck.

## Limitations — read before drawing conclusions

1. **Opponents are singletons.** Only polled players are observed repeatedly
   (median 30 battles / 4 decks). Their opponents appear once (median 1 battle,
   0.5% with >1 deck). Within-player identification therefore rests entirely on
   the cohort, not on the raw player count. Cohort size is the real sample size.

2. **The cohort is not a random sample.** Seeded from top clan-war clans, it
   skews hard toward high-engagement, high-level, clan-active players. This is
   the same selection bias present in the work this study critiques. It is not
   solved here — only made explicit and controllable. Stratify by adding
   mid/low-ranked clans and multiple locations before claiming generality.

3. **Retention forces a polling cadence.** ~30 battles, ~1 day of history. A
   player who plays more than 30 battles between polls silently loses the
   overflow. Coverage is therefore activity-dependent and currently non-adaptive.

4. **Time-of-day sampling bias.** Polling at a fixed hour oversamples whichever
   timezones are active then. Vary the schedule.

5. **Deck choice is endogenous.** `deck_selection == "collection"` on all
   pathOfLegend and riverRacePvP battles: players picked their own decks. Even
   within-player, switching is not random — people switch to decks they favour,
   or after losing. Within-player estimates remove the *between*-player
   confound, not this one.

6. **Draft modes are the exception, and they are small.** `draft` /
   `draftCompetitive` appear in `trail` (153 of 3,687 battles). Decks there are
   quasi-randomly assigned — much stronger identification — but `trail` has a
   different level distribution (mean 9.13 vs 12.5) and different rules, so
   findings may not transfer to ladder.

7. **Card levels are not normalized** in any competitive mode (see
   `API_FINDINGS.md`). Deck strength is confounded with account investment
   unless level is modelled explicitly. Levels are stored per card so this is
   possible, but it is a real modelling burden, not a solved problem.

8. **Matchmaking is not random.** We observe only realized matchups, which are
   skill-matched. Deck-vs-deck pairings are therefore selected, not random.

9. **Ties** carry `label_a_win = NaN`. They are retained and flagged; the
   modelling stage must decide. Tower HP allows a graded outcome instead.

10. **Meta drift.** Balance patches land regularly. Any dataset spanning weeks
    spans patches; record patch boundaries and fit per-patch.

11. **The API changes.** Trophy-ladder rankings already returned 0 items.
    Re-run `check_key.py` periodically to catch schema and endpoint drift.

12. **Supercell API terms** govern redistribution of collected data. Check
    before publishing a dataset alongside the post.
