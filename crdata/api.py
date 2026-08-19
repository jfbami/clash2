"""Thin client for the official Clash Royale API.

Handles the two things that actually bite: IP-bound API keys (403) and the
per-second rate limit (429). Tags contain '#' and must be percent-encoded.
"""
from __future__ import annotations

import os
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import requests

GLOBAL = "global"  # literal string; 57000000 is EUROPE, not global
PROXY_BASE = "https://proxy.royaleapi.dev/v1"
DIRECT_BASE = "https://api.clashroyale.com/v1"
PROXY_ALLOWLIST_IP = "45.79.218.79"


class CRAuthError(RuntimeError):
    """403 - almost always the key's IP allowlist, not the token itself."""


class CRNotFound(RuntimeError):
    pass


def encode_tag(tag: str) -> str:
    """'#ABC123' or 'ABC123' -> '%23ABC123'. Tags are case-insensitive."""
    tag = tag.strip().upper()
    if not tag.startswith("#"):
        tag = "#" + tag
    return urllib.parse.quote(tag, safe="")


@dataclass
class CRClient:
    token: str = field(default_factory=lambda: os.environ.get("CR_API_TOKEN", ""))
    base: str = field(default_factory=lambda: os.environ.get("CR_API_BASE", PROXY_BASE))
    timeout: int = 20
    max_retries: int = 5

    def __post_init__(self) -> None:
        if not self.token:
            raise RuntimeError("CR_API_TOKEN is not set (put it in .env)")
        self.base = self.base.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        )
        # Populated from response headers so we can report the real limit.
        self.last_headers: dict[str, str] = {}

    def get(self, path: str, **params: Any) -> Any:
        url = f"{self.base}/{path.lstrip('/')}"
        backoff = 1.0
        for attempt in range(self.max_retries):
            r = self.session.get(url, params=params or None, timeout=self.timeout)
            self.last_headers = dict(r.headers)

            if r.status_code == 200:
                return r.json()
            if r.status_code == 403:
                raise CRAuthError(self._explain_403(r))
            if r.status_code == 404:
                raise CRNotFound(f"404 not found: {url}")
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == self.max_retries - 1:
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            r.raise_for_status()
        raise RuntimeError(f"giving up on {url} after {self.max_retries} tries")

    def _explain_403(self, r: requests.Response) -> str:
        detail = ""
        try:
            detail = r.json().get("message", "") or r.json().get("reason", "")
        except Exception:
            detail = r.text[:200]
        hint = (
            f"using proxy base {self.base}: allowlist {PROXY_ALLOWLIST_IP} on your key"
            if "royaleapi" in self.base
            else f"using direct base {self.base}: allowlist your CURRENT public IP on the key"
        )
        return f"403 from API ({detail!r}). Most likely the key's IP allowlist. You are {hint}."

    # --- endpoints -------------------------------------------------------
    def cards(self) -> list[dict]:
        return self.get("/cards")["items"]

    def player(self, tag: str) -> dict:
        return self.get(f"/players/{encode_tag(tag)}")

    def battlelog(self, tag: str) -> list[dict]:
        """Recent battles only - the API keeps a short rolling window."""
        return self.get(f"/players/{encode_tag(tag)}/battlelog")

    def clan(self, tag: str) -> dict:
        return self.get(f"/clans/{encode_tag(tag)}")

    def clan_members(self, tag: str) -> list[dict]:
        return self.get(f"/clans/{encode_tag(tag)}/members")["items"]

    def top_players(self, limit: int = 50, location: str = GLOBAL) -> list[dict]:
        """Path of Legends leaderboard. NOTE: the old trophy-based
        /rankings/players returns an empty list now - ladder was replaced."""
        return self.get(f"/locations/{location}/pathoflegend/players", limit=limit)["items"]

    def top_clans(self, limit: int = 50, location: str = GLOBAL) -> list[dict]:
        return self.get(f"/locations/{location}/rankings/clans", limit=limit)["items"]

    def top_war_clans(self, limit: int = 50, location: str = GLOBAL) -> list[dict]:
        """Clan-war leaderboard - the best seed source for War Day battles."""
        return self.get(f"/locations/{location}/rankings/clanwars", limit=limit)["items"]

    def rate_limit_info(self) -> dict[str, str]:
        return {
            k: v for k, v in self.last_headers.items()
            if "ratelimit" in k.lower() or k.lower() == "retry-after"
        }


def battle_key(battle: dict) -> str:
    """Stable dedup key. The API exposes NO battle id, so a battle collected
    from both participants' logs appears twice with team/opponent swapped.
    Key on time + the sorted set of participant tags to collapse the mirror."""
    tags = sorted(
        p.get("tag", "") for side in ("team", "opponent") for p in battle.get(side, [])
    )
    return f"{battle.get('battleTime', '')}|{'|'.join(tags)}"
