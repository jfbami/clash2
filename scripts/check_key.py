"""Verify API access and empirically measure the limits the docs don't state.

Run:  python scripts/check_key.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from dotenv import load_dotenv

from crdata.api import CRAuthError, CRClient, battle_key

load_dotenv()


def parse_battle_time(s: str) -> datetime:
    # API format: 20260819T153900.000Z
    return datetime.strptime(s, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)


def show_public_ip() -> None:
    for url in ("https://api.ipify.org", "https://ifconfig.me/ip"):
        try:
            ip = requests.get(url, timeout=8).text.strip()
            print(f"  your current public IP : {ip}")
            return
        except Exception:
            continue
    print("  your current public IP : (could not determine)")


def main() -> int:
    print("=" * 68)
    print("CLASH ROYALE API ACCESS CHECK")
    print("=" * 68)

    token = os.environ.get("CR_API_TOKEN", "")
    base = os.environ.get("CR_API_BASE", "(default)")
    print(f"\n[1] Config")
    print(f"  base URL              : {base}")
    print(f"  token present         : {'yes (%d chars)' % len(token) if token else 'NO'}")
    show_public_ip()
    if not token or token == "paste_your_jwt_here":
        print("\n  -> CR_API_TOKEN is not set. Create .env from .env.example first.")
        return 1

    client = CRClient()

    print(f"\n[2] Auth check  (GET /cards)")
    try:
        cards = client.cards()
    except CRAuthError as e:
        print(f"  FAILED: {e}")
        return 1
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return 1

    evo = [c for c in cards if "evolution" in str(c.get("rarity", "")).lower()
           or c.get("maxEvolutionLevel")]
    print(f"  OK - {len(cards)} cards returned")
    print(f"  cards with an evolution: {len(evo)}")
    rl = client.rate_limit_info()
    print(f"  rate-limit headers    : {rl if rl else '(none exposed)'}")

    print(f"\n[3] Battlelog probe  (top global player)")
    try:
        top = client.top_players(limit=5)
        tag = top[0]["tag"]
        print(f"  probing {tag} ({top[0].get('name','?')})")
        log = client.battlelog(tag)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return 1

    print(f"  battles returned      : {len(log)}   <- the real page size")
    if log:
        times = sorted(parse_battle_time(b["battleTime"]) for b in log)
        span = times[-1] - times[0]
        age = datetime.now(timezone.utc) - times[0]
        print(f"  oldest battle         : {times[0].isoformat()}")
        print(f"  newest battle         : {times[-1].isoformat()}")
        print(f"  span covered          : {span}   <- the retention window")
        print(f"  oldest is this old    : {age}")

        modes = Counter(b.get("gameMode", {}).get("name", "?") for b in log)
        types = Counter(b.get("type", "?") for b in log)
        print(f"  battle types          : {dict(types)}")
        print(f"  game modes            : {dict(modes.most_common(6))}")

        b0 = log[0]
        t0 = b0.get("team", [{}])[0]
        print(f"\n[4] Field availability on a sample battle")
        print(f"  has battleTime        : {'battleTime' in b0}")
        print(f"  has unique battle id  : {any('id' == k.lower() for k in b0)}  <- expect False")
        print(f"  dedup key             : {battle_key(b0)}")
        print(f"  team[0] keys          : {sorted(t0.keys())}")
        print(f"  startingTrophies      : {t0.get('startingTrophies')}")
        print(f"  trophyChange          : {t0.get('trophyChange')}")
        print(f"  crowns                : {t0.get('crowns')}")
        print(f"  cards in deck         : {len(t0.get('cards', []))}")
        names = [c.get("name") for c in t0.get("cards", [])]
        print(f"  deck                  : {names}")
        lv = [c.get("level") for c in t0.get("cards", [])]
        print(f"  card levels           : {lv}  <- check if normalized in war/tournament")
        if "supportCards" in t0:
            print(f"  supportCards          : {[c.get('name') for c in t0['supportCards']]}")

    print("\n" + "=" * 68)
    print("ACCESS OK - ready to collect.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
