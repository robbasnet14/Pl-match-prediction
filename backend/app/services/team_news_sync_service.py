from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from ..data.repository import load_team_news_frame, save_team_news_frame

logger = logging.getLogger("pl_api")

STATUS_SUSPENDED_KEYWORDS = ("suspend", "banned", "ban")
STATUS_DOUBTFUL_KEYWORDS = ("doubt", "question", "late fitness", "test")
STATUS_AVAILABLE_KEYWORDS = ("fit", "available", "returned")
STATUS_INJURY_KEYWORDS = ("injur", "knock", "hamstring", "ankle", "knee", "muscle", "groin", "calf")
POSITION_IMPORTANCE = {
    "g": 0.72,
    "d": 0.58,
    "m": 0.62,
    "f": 0.68,
}


def _canonical_team_name(raw_name: str) -> str:
    value = re.sub(r"\s+", " ", str(raw_name).strip())
    value = re.sub(r"\s+FC$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+AFC$", "", value, flags=re.IGNORECASE)
    aliases = {
        "Brighton & Hove Albion": "Brighton",
        "Brighton and Hove Albion": "Brighton",
        "Manchester Utd": "Manchester United",
        "Man United": "Manchester United",
        "Man City": "Manchester City",
        "Spurs": "Tottenham Hotspur",
        "Wolves": "Wolverhampton Wanderers",
        "AFC Bournemouth": "Bournemouth",
    }
    return aliases.get(value, value)


def _default_season_start_year(now_utc: datetime | None = None) -> int:
    now = now_utc or datetime.now(timezone.utc)
    # Premier League season convention: starts around August.
    return now.year if now.month >= 7 else now.year - 1


def _normalize_status(reason: str) -> str:
    text = str(reason).strip().lower()
    if any(keyword in text for keyword in STATUS_AVAILABLE_KEYWORDS):
        return "available"
    if any(keyword in text for keyword in STATUS_SUSPENDED_KEYWORDS):
        return "suspended"
    if any(keyword in text for keyword in STATUS_DOUBTFUL_KEYWORDS):
        return "doubtful"
    return "injured"


def _normalize_position(raw: str) -> str:
    value = str(raw).strip().upper()
    if not value:
        return ""
    if value in {"G", "GK", "GOALKEEPER"}:
        return "GK"
    if value in {"D", "DEF", "DEFENDER", "CB", "LB", "RB", "LWB", "RWB"}:
        return "DF"
    if value in {"M", "MID", "MIDFIELDER", "CM", "DM", "AM", "LM", "RM"}:
        return "MF"
    if value in {"F", "FW", "ATT", "ATTACKER", "ST", "CF", "LW", "RW"}:
        return "FW"
    return value


def _importance_from_position(position: str) -> float:
    key = str(position).strip().lower()[:1]
    if key in POSITION_IMPORTANCE:
        return POSITION_IMPORTANCE[key]
    return 0.6


def _request_json(base_url: str, api_key: str, path: str, params: dict[str, str]) -> dict:
    query = urlencode(params)
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}?{query}" if query else f"{base_url.rstrip('/')}/{path.lstrip('/')}"

    request = Request(url)
    request.add_header("Accept", "application/json")
    request.add_header("x-apisports-key", api_key)
    rapid_host = os.getenv("TEAM_NEWS_RAPIDAPI_HOST", "").strip()
    rapid_key = os.getenv("TEAM_NEWS_RAPIDAPI_KEY", "").strip()
    if rapid_host:
        request.add_header("x-rapidapi-host", rapid_host)
    if rapid_key:
        request.add_header("x-rapidapi-key", rapid_key)

    with urlopen(request, timeout=20) as response:
        payload = response.read().decode("utf-8")
        return json.loads(payload)


def _fetch_api_football_injuries() -> list[dict]:
    base_url = os.getenv("TEAM_NEWS_API_BASE_URL", "https://v3.football.api-sports.io").strip()
    api_key = os.getenv("TEAM_NEWS_API_KEY", "").strip()
    league_id = os.getenv("TEAM_NEWS_LEAGUE_ID", "39").strip()
    season_start_year = int(os.getenv("TEAM_NEWS_SEASON_START_YEAR", str(_default_season_start_year())).strip())
    source_label = os.getenv("TEAM_NEWS_SOURCE_LABEL", "api-football").strip() or "api-football"

    if not api_key:
        raise RuntimeError("missing_team_news_api_key")

    raw = _request_json(
        base_url=base_url,
        api_key=api_key,
        path="/injuries",
        params={
            "league": str(league_id),
            "season": str(season_start_year),
        },
    )
    errors = raw.get("errors")
    if isinstance(errors, dict):
        messages = [str(value).strip() for value in errors.values() if str(value).strip()]
        if messages:
            raise RuntimeError(f"team_news_provider_error:{'; '.join(messages)}")

    records = raw.get("response", []) or []
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    output: list[dict] = []
    for item in records:
        team_payload = item.get("team") or {}
        player_payload = item.get("player") or {}

        team = _canonical_team_name(team_payload.get("name", ""))
        player_name = str(player_payload.get("name", "")).strip()
        if not team or not player_name:
            continue

        reason = str(item.get("reason") or item.get("type") or "").strip()
        position = _normalize_position(
            player_payload.get("position", "")
            or player_payload.get("type", "")
        )
        output.append(
            {
                "team": team,
                "player": player_name,
                "status": _normalize_status(reason),
                "position": position,
                "importance": _importance_from_position(position),
                "expected_return": "",
                "source": source_label,
                "updated_at": now_iso,
            }
        )

    # Keep latest unique player-team tuple.
    unique: dict[tuple[str, str], dict] = {}
    for row in output:
        unique[(row["team"], row["player"])] = row
    return sorted(unique.values(), key=lambda row: (row["team"], row["player"]))


def _fpl_status_from_code(status_code: str, news_text: str) -> str:
    code = str(status_code).strip().lower()
    text = str(news_text).strip().lower()

    if code == "s":
        return "suspended"
    if code == "i":
        return "injured"
    if code == "d":
        return "doubtful"
    if code in {"a"}:
        return "available"
    if code in {"u", "n"}:
        if any(keyword in text for keyword in STATUS_SUSPENDED_KEYWORDS):
            return "suspended"
        if any(keyword in text for keyword in STATUS_INJURY_KEYWORDS):
            return "injured"
        if any(keyword in text for keyword in STATUS_DOUBTFUL_KEYWORDS):
            return "doubtful"
        return "available"

    if "suspend" in text or "ban" in text:
        return "suspended"
    if any(keyword in text for keyword in STATUS_INJURY_KEYWORDS):
        return "injured"
    if "doubt" in text:
        return "doubtful"
    return "available"


def _fpl_position_from_type(element_type: int) -> str:
    mapping = {
        1: "GK",
        2: "DF",
        3: "MF",
        4: "FW",
    }
    return mapping.get(int(element_type), "")


def _fpl_importance(player: dict, position: str) -> float:
    base = _importance_from_position(position)
    minutes = float(player.get("minutes", 0) or 0)
    minutes_factor = min(1.0, minutes / 1800.0) * 0.2
    starts = float(player.get("starts", 0) or 0)
    starts_factor = min(1.0, starts / 20.0) * 0.1
    influence = float(player.get("influence", 0) or 0)
    creativity = float(player.get("creativity", 0) or 0)
    threat = float(player.get("threat", 0) or 0)
    ict = min(1.0, (influence + creativity + threat) / 450.0) * 0.2
    return max(0.25, min(0.98, base + minutes_factor + starts_factor + ict))


def _fetch_fpl_team_news() -> list[dict]:
    base_url = os.getenv("TEAM_NEWS_FPL_BASE_URL", "https://fantasy.premierleague.com/api").strip().rstrip("/")
    source_label = os.getenv("TEAM_NEWS_SOURCE_LABEL", "fpl").strip() or "fpl"

    request = Request(f"{base_url}/bootstrap-static/")
    request.add_header("Accept", "application/json")
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    teams = payload.get("teams") or []
    elements = payload.get("elements") or []
    team_by_id = {int(team.get("id", 0)): _canonical_team_name(team.get("name", "")) for team in teams}
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    rows: list[dict] = []
    for player in elements:
        team_name = team_by_id.get(int(player.get("team", 0)), "")
        if not team_name:
            continue

        news_text = str(player.get("news", "")).strip()
        status_code = str(player.get("status", "")).strip().lower()
        status = _fpl_status_from_code(status_code, news_text)

        first = str(player.get("first_name", "")).strip()
        second = str(player.get("second_name", "")).strip()
        web_name = str(player.get("web_name", "")).strip()
        full_name = " ".join(part for part in [first, second] if part).strip() or web_name
        if not full_name:
            continue

        position = _fpl_position_from_type(int(player.get("element_type", 0) or 0))
        chance = player.get("chance_of_playing_next_round")
        expected_return = ""
        if chance is not None and str(chance).strip() != "":
            expected_return = f"chance_next_round:{chance}%"
        influence = float(player.get("influence", 0) or 0)
        creativity = float(player.get("creativity", 0) or 0)
        threat = float(player.get("threat", 0) or 0)
        expected_goals = float(player.get("expected_goals", 0) or 0)
        expected_assists = float(player.get("expected_assists", 0) or 0)
        minutes = float(player.get("minutes", 0) or 0)
        starts = float(player.get("starts", 0) or 0)

        rows.append(
            {
                "team": team_name,
                "player": full_name,
                "status": status,
                "position": position,
                "importance": _fpl_importance(player, position),
                "expected_return": expected_return,
                "source": source_label,
                "updated_at": now_iso,
                "minutes": minutes,
                "starts": starts,
                "influence": influence,
                "creativity": creativity,
                "threat": threat,
                "expected_goals": expected_goals,
                "expected_assists": expected_assists,
            }
        )

    unique: dict[tuple[str, str], dict] = {}
    for row in rows:
        unique[(row["team"], row["player"])] = row
    return sorted(unique.values(), key=lambda row: (row["team"], row["player"]))


def fetch_team_news_rows() -> list[dict]:
    provider = os.getenv("TEAM_NEWS_PROVIDER", "api-football").strip().lower()
    if provider in {"api-football", "api_football", "apisports"}:
        return _fetch_api_football_injuries()
    if provider in {"fpl", "fantasy-premier-league", "fantasy_premier_league"}:
        return _fetch_fpl_team_news()
    raise RuntimeError(f"unsupported_team_news_provider:{provider}")


def sync_team_news(*, min_rows: int = 1, dry_run: bool = False) -> dict:
    rows = fetch_team_news_rows()
    min_rows = max(0, int(min_rows))
    if dry_run:
        return {
            "ok": True,
            "updated": len(rows),
            "dryRun": True,
            "minimumRequired": min_rows,
            "minimumSatisfied": len(rows) >= min_rows,
        }

    if len(rows) < min_rows:
        raise RuntimeError(f"team_news_rows_below_minimum:{len(rows)}<{min_rows}")

    save_team_news_frame(pd.DataFrame(rows))
    saved = load_team_news_frame()
    if saved is not None:
        save_team_news_frame(saved.sort_values(["team", "player"]).reset_index(drop=True))

    logger.info("team news sync completed: rows=%s", len(rows))
    return {"ok": True, "updated": len(rows), "dryRun": False}
