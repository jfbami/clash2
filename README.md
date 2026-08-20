# clash2

Collects Clash Royale battle data to measure how much match outcome comes from deck design, player skill, and account investment.

## Prerequisites

1. Python 3.13 with the packages in `requirements.txt`.
2. A Clash Royale API key from developer.clashroyale.com.
3. The API key must allowlist the IP `45.79.218.79`, the RoyaleAPI proxy.
4. A `.env` file created from `.env.example`, holding `CR_API_TOKEN` and `CR_API_BASE`.

## Verify API access

Run `python scripts/check_key.py`.

`scripts/check_key.py` confirms the token works and measures the limits Supercell does not document: battlelog page size, retention window, and sustained request rate.

Expected failure if the key allowlist is wrong:

```
403 from API. Most likely the key's IP allowlist.
```

Fix that error by editing the key at developer.clashroyale.com to allowlist `45.79.218.79`.

## Collect data

Run `python scripts/seed_cohort.py --clans 200 --per-clan 40` once to build the player cohort.

Run `python scripts/collect.py` repeatedly on a schedule to poll that cohort.

`scripts/collect.py` is safe to re-run because deduplication persists across runs.

## Modules

`crdata/api.py` is the HTTP client for the Clash Royale API.
`crdata/api.py` handles the proxy base URL, 429 backoff, 403 diagnosis, and the synthetic battle key.

`crdata/parse.py` converts one battle JSON object into one flat row.

`crdata/store.py` writes Parquet, maintains the persistent deduplication set, and tracks cohort state.

## Data layout

Battles land in `data/battles/date=YYYY-MM-DD/<run_id>.parquet`.
Deduplication hashes live in `data/state/seen_keys.parquet`.
The cohort lives in `data/state/cohort.parquet`.

## Design decisions

**Fixed cohort, polled repeatedly.**
Repeated observation of the same players produces within-player deck variation.
Within-player deck variation is what allows deck effects to be estimated free of player skill and account investment.
A single breadth-first crawl produces a wide shallow snapshot instead.

**Canonical side ordering.**
`crdata/parse.py` assigns side `a` to the lexicographically smaller player tag.
Canonical ordering removes the player-ordering artifact at parse time rather than randomising it away.
Verified on 3,441 battles: side `a` wins 51.2 percent, z = +1.38, not significant.

**Persistent deduplication.**
The Clash Royale API exposes no battle ID.
A battle collected from both participants appears twice with the sides swapped.
`crdata.api.battle_key` keys on battle time plus the sorted participant tags.

**No filtering during collection.**
`crdata/parse.py` parses and flags duels, boat battles, and 2v2 rather than dropping them.
The `is_clean_1v1` column marks rows usable without extra assumptions.
Filtering is a modelling decision and belongs downstream.

## Related documents

`API_FINDINGS.md` records measured Clash Royale API behaviour.
`ASSUMPTIONS.md` records identifying assumptions and acknowledged biases.
