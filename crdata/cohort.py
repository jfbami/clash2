"""Build a player cohort that spans the ladder instead of sitting on top of it.

Seeding from the clan-war leaderboard produces a cohort with no variation in
the quantity this project measures. Measured on the live API: the top war clans
return 48 of 50 members pinned at the 14,000 trophy ceiling, so card levels are
uniformly maxed and the level cost of anything is unmeasurable within them.

`discover_clans` therefore searches by name rather than by ranking, because
`minScore` always returns the strongest clans above its floor and `maxScore` is
ignored by the API. Name search surfaces clans across the whole score range.

`stratify` then buckets the *players*, not the clans. Clan score is only a
proxy for member strength and clans are internally wide: one clan measured at
score 95,302 held members from 5,300 to 10,887 trophies. Bucketing on player
trophies targets the spread directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TROPHY_CEILING = 14_000
DEFAULT_BUCKETS = 5
MIN_QUERY_LENGTH = 3  # the API rejects shorter name searches with a 400

# Common substrings in clan names. Breadth matters more than cleverness: each
# query returns a different slice of the clan population, and the union is what
# fills the weaker buckets that ranking endpoints never reach.
DISCOVERY_QUERIES = (
    "clan", "royale", "king", "team", "war", "the", "club", "elite", "pro",
    "boys", "gaming", "squad", "army", "lords", "empire", "legion", "force",
    "wolf", "dragon", "knight", "star", "fire", "ice", "storm", "shadow",
    "brasil", "india", "espana", "france", "deutsch", "italia", "polska",
    "turk", "arab", "china", "korea", "japan", "mexico", "chile", "peru",
    "friends", "family", "casual", "chill", "fun", "noobs", "rookies", "new",
)


@dataclass(frozen=True)
class BucketReport:
    edges: np.ndarray
    available: np.ndarray
    taken: np.ndarray

    @property
    def shortfall(self) -> np.ndarray:
        return self.taken.max() - self.taken if len(self.taken) else np.array([])

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "bucket": [f"{int(self.edges[i]):,}-{int(self.edges[i + 1]):,}"
                       for i in range(len(self.edges) - 1)],
            "available": self.available,
            "taken": self.taken,
        })


MEMBER_COLUMNS = ("tag", "name", "clan_tag", "clan_score", "trophies")
MAX_FAILURE_RATE = 0.5


def discover_clans(client, queries=DISCOVERY_QUERIES, min_members: int = 10,
                   per_query: int = 30) -> pd.DataFrame:
    """Search clans by name and return the distinct union.

    Raises ValueError when a query is too short for the API to accept, and
    RuntimeError when more than half the queries fail, which means the sweep
    hit a network or access problem rather than a few empty searches.
    """
    found: dict[str, dict] = {}
    failures = 0
    for query in queries:
        if len(query) < MIN_QUERY_LENGTH:
            raise ValueError(f"query {query!r} is shorter than {MIN_QUERY_LENGTH} characters")
        try:
            items = client.get("/clans", name=query, minMembers=min_members,
                               limit=per_query).get("items", [])
        except Exception as exc:  # one bad query must not abandon the sweep
            failures += 1
            print(f"    query {query!r} failed: {type(exc).__name__}")
            continue
        for clan in items:
            found[clan["tag"]] = {
                "clan_tag": clan["tag"],
                "clan_name": clan.get("name"),
                "clan_score": clan.get("clanScore", 0),
                "members": clan.get("members", 0),
            }

    if failures > MAX_FAILURE_RATE * len(queries):
        raise RuntimeError(
            f"{failures} of {len(queries)} clan searches failed; the API or network "
            f"is unhealthy, so the discovered pool would be biased")
    if failures:
        print(f"    {failures} of {len(queries)} queries failed")
    return pd.DataFrame(found.values())


def fetch_members(client, clans: pd.DataFrame, per_clan: int) -> pd.DataFrame:
    """Pull member lists, one request per clan, keeping each member's trophies.

    Raises RuntimeError when more than half the clans fail. A silent empty
    result here previously turned a network outage into a confusing error much
    further downstream.
    """
    rows: list[dict] = []
    failures = 0
    for position, clan in enumerate(clans.itertuples(), start=1):
        try:
            members = client.clan_members(clan.clan_tag)
        except Exception as exc:
            failures += 1
            if failures <= 3:
                print(f"    {clan.clan_tag} failed: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        for member in members[:per_clan]:
            rows.append({
                "tag": member["tag"],
                "name": member.get("name"),
                "clan_tag": clan.clan_tag,
                "clan_score": clan.clan_score,
                "trophies": member.get("trophies", 0),
            })
        if position % 100 == 0:
            print(f"    {position}/{len(clans)} clans, {len(rows):,} players, "
                  f"{failures} failed", flush=True)

    if failures > MAX_FAILURE_RATE * len(clans):
        raise RuntimeError(
            f"{failures} of {len(clans)} member fetches failed; refusing to build a "
            f"cohort from what survived")
    print(f"    done: {len(rows):,} players, {failures} clans failed")
    return pd.DataFrame(rows, columns=list(MEMBER_COLUMNS)).drop_duplicates("tag")


def bucket_edges(buckets: int = DEFAULT_BUCKETS) -> np.ndarray:
    """Equal-width trophy bands from zero to the ceiling."""
    return np.linspace(0, TROPHY_CEILING, buckets + 1)


def stratify(players: pd.DataFrame, target: int,
             buckets: int = DEFAULT_BUCKETS,
             seed: int = 0) -> tuple[pd.DataFrame, BucketReport]:
    """Take an equal number of players from each trophy band.

    Balanced rather than population-weighted, so every band carries the same
    weight in the level-cost estimate. The real population is heavily
    concentrated, and matching it would leave the weakest bands too thin to
    estimate anything within.

    A band with fewer players than the quota contributes everything it has, and
    the returned report names the shortfall rather than hiding it.
    """
    edges = bucket_edges(buckets)
    band = np.clip(np.digitize(players["trophies"], edges[1:-1]), 0, buckets - 1)
    working = players.assign(band=band)

    quota = target // buckets
    generator = np.random.default_rng(seed)
    chosen, available, taken = [], [], []

    for index in range(buckets):
        pool = working[working.band == index]
        available.append(len(pool))
        if pool.empty:
            taken.append(0)
            continue
        size = min(quota, len(pool))
        chosen.append(pool.sample(n=size, random_state=generator.integers(2**32)))
        taken.append(size)

    cohort = (pd.concat(chosen, ignore_index=True) if chosen
              else working.iloc[:0].copy())
    report = BucketReport(edges=edges, available=np.array(available),
                          taken=np.array(taken))
    return cohort.drop(columns="band"), report
