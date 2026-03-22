from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _normalize_team_name(name: str) -> str:
    if not name:
        return ""

    value = str(name).strip()
    value = re.sub(r"\s+", " ", value)

    replacements = {
        "Brighton & Hove Albion": "Brighton and Hove Albion",
        "Spurs": "Tottenham Hotspur",
        "Wolves": "Wolverhampton Wanderers",
        "Man United": "Manchester United",
        "Man City": "Manchester City",
    }
    value = replacements.get(value, value)

    value = re.sub(r"\s+FC$", "", value)
    value = re.sub(r"\s+AFC$", "", value)
    value = value.strip()

    return value


def _request_json(path: str, params: dict | None = None) -> dict:
    api_key = os.getenv("FOOTBALL_API_KEY", "").strip()
    base_url = os.getenv("FOOTBALL_API_BASE_URL", "https://api.football-data.org/v4").rstrip("/")

    if not api_key:
        raise RuntimeError("missing_football_api_key")

    query = f"?{urlencode(params)}" if params else ""
    url = f"{base_url}{path}{query}"

    request = Request(url)
    request.add_header("X-Auth-Token", api_key)
    request.add_header("Accept", "application/json")

    with urlopen(request, timeout=10) as response:
        payload = response.read().decode("utf-8")
        return json.loads(payload)


def fetch_live_standings(limit: int = 20) -> tuple[list[dict], str] | None:
    competition = os.getenv("FOOTBALL_COMPETITION_CODE", "PL")
    data = _request_json(f"/competitions/{competition}/standings")

    standings_groups = data.get("standings") or []
    if not standings_groups:
        return None

    table = standings_groups[0].get("table") or []
    if not table:
        return None

    standings = []
    for row in table[:limit]:
        team_name = _normalize_team_name((row.get("team") or {}).get("name", ""))
        standings.append(
            {
                "position": int(row.get("position", 0)),
                "name": team_name,
                "played": int(row.get("playedGames", 0)),
                "points": int(row.get("points", 0)),
                "change": "same",
                "changeLabel": "-",
            }
        )

    return standings, "external-football-data"


def fetch_live_fixtures(limit: int = 3) -> tuple[list[dict], str] | None:
    competition = os.getenv("FOOTBALL_COMPETITION_CODE", "PL")
    now = datetime.now(timezone.utc)
    date_from = (now.date() + timedelta(days=1)).isoformat()
    date_to = (now.date() + timedelta(days=14)).isoformat()

    data = _request_json(
        f"/competitions/{competition}/matches",
        params={
            "status": "SCHEDULED",
            "dateFrom": date_from,
            "dateTo": date_to,
        },
    )

    matches = data.get("matches") or []
    if not matches:
        return None

    fixtures = []
    for idx, match in enumerate(matches[:limit], start=1):
        home = _normalize_team_name((match.get("homeTeam") or {}).get("name", ""))
        away = _normalize_team_name((match.get("awayTeam") or {}).get("name", ""))
        utc_date = str(match.get("utcDate", ""))
        venue = str(match.get("venue") or (match.get("competition") or {}).get("name") or "Premier League")

        kickoff = utc_date.replace("T", " ").replace("Z", "")[:16]
        fixtures.append(
            {
                "id": idx,
                "home": home,
                "away": away,
                "time": kickoff,
                "venue": venue,
            }
        )

    return fixtures, "external-football-data"
