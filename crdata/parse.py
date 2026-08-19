"""Battle JSON -> one flat row per battle.

Two decisions worth knowing about:

1. Sides are stored in CANONICAL order (a = lexicographically smaller player
   tag). This removes the player-ordering artifact at the source, so no
   random-swap preprocessing is needed and the dedup key is stable.
2. Nothing is dropped here. Duels (16/24 cards), boat battles and any 2v2 are
   parsed and FLAGGED, not filtered. Filtering is a modelling decision and
   belongs downstream where it can be justified and varied.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from crdata.api import battle_key

TS_FMT = "%Y%m%dT%H%M%S.%fZ"


def parse_time(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, TS_FMT).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _side(players: list[dict]) -> dict[str, Any]:
    """Flatten one side. Multi-player sides (2v2) are concatenated and the
    caller records team_size so they can be excluded downstream."""
    cards, levels, evos, support = [], [], [], []
    for p in players:
        for cd in p.get("cards", []):
            cards.append(cd.get("id"))
            levels.append(cd.get("level"))
            # evolutionLevel is absent for un-evolved cards
            evos.append(int(cd.get("evolutionLevel") or 0))
        for cd in p.get("supportCards", []):
            support.append(cd.get("id"))
    p0 = players[0] if players else {}
    return {
        "tag": p0.get("tag"),
        "name": p0.get("name"),
        "clan": (p0.get("clan") or {}).get("tag"),
        "crowns": p0.get("crowns"),
        "king_hp": p0.get("kingTowerHitPoints"),
        "princess_hp": p0.get("princessTowersHitPoints") or [],
        "trophies": p0.get("startingTrophies"),
        "trophy_change": p0.get("trophyChange"),
        "global_rank": p0.get("globalRank"),
        "elixir_leaked": p0.get("elixirLeaked"),
        "card_ids": cards,
        "card_levels": levels,
        "card_evo": evos,
        "support_ids": support,
        "team_size": len(players),
        "n_cards": len(cards),
    }


def parse_battle(b: dict, collected_at: datetime, run_id: str) -> dict | None:
    team, opp = b.get("team") or [], b.get("opponent") or []
    if not team or not opp:
        return None

    A, B = _side(team), _side(opp)
    if not A["tag"] or not B["tag"]:
        return None
    # canonical orientation - kills the ordering artifact deterministically
    if A["tag"] > B["tag"]:
        A, B = B, A

    ca, cb = A["crowns"], B["crowns"]
    if ca is None or cb is None:
        winner, label = None, None
    elif ca > cb:
        winner, label = "a", 1
    elif cb > ca:
        winner, label = "b", 0
    else:
        winner, label = "tie", None

    row: dict[str, Any] = {
        "battle_key": battle_key(b),
        "battle_time": parse_time(b.get("battleTime", "")),
        "collected_at": collected_at,
        "run_id": run_id,
        "type": b.get("type"),
        "game_mode": (b.get("gameMode") or {}).get("name"),
        "arena": (b.get("arena") or {}).get("name"),
        "deck_selection": b.get("deckSelection"),
        "is_ladder_tournament": b.get("isLadderTournament"),
        "is_hosted_match": b.get("isHostedMatch"),
        "league_number": b.get("leagueNumber"),
        "event_tag": b.get("eventTag"),
        "winner": winner,
        "label_a_win": label,
    }
    for pre, S in (("a", A), ("b", B)):
        for k, v in S.items():
            row[f"{pre}_{k}"] = v

    # clean == the subset the model can actually use without extra assumptions
    row["is_clean_1v1"] = bool(
        A["team_size"] == 1 and B["team_size"] == 1
        and A["n_cards"] == 8 and B["n_cards"] == 8
        and winner in ("a", "b")
    )
    return row
