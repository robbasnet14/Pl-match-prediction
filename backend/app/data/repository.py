from __future__ import annotations

import pandas as pd

from ..config import MATCHES_PATH, TEAM_AVAILABILITY_PATH, TEAM_NEWS_PATH, UPCOMING_FIXTURES_PATH


REQUIRED_UPCOMING_COLUMNS = {"home", "away", "time", "venue"}
REQUIRED_AVAILABILITY_COLUMNS = {"team", "injured", "suspended", "key_impact"}
REQUIRED_TEAM_NEWS_COLUMNS = {"team", "player", "status", "importance"}
TEAM_NEWS_OPTIONAL_TEXT_COLUMNS = ["position", "expected_return", "source", "updated_at"]
TEAM_NEWS_OPTIONAL_NUMERIC_COLUMNS = [
    "minutes",
    "starts",
    "influence",
    "creativity",
    "threat",
    "expected_goals",
    "expected_assists",
]


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(MATCHES_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    return df.dropna(subset=["date", "season", "team", "result"])


def latest_season_frame(df: pd.DataFrame) -> pd.DataFrame:
    latest_season = int(df["season"].max())
    return df[df["season"] == latest_season].copy()


def load_upcoming_fixtures_frame() -> pd.DataFrame | None:
    if not UPCOMING_FIXTURES_PATH.exists():
        return None

    df = pd.read_csv(UPCOMING_FIXTURES_PATH)
    normalized_columns = {col.strip().lower() for col in df.columns}
    if not REQUIRED_UPCOMING_COLUMNS.issubset(normalized_columns):
        return None

    column_map = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=column_map)
    df = df[["home", "away", "time", "venue"]].dropna()

    if df.empty:
        return None

    return df.reset_index(drop=True)


def load_team_availability_frame() -> pd.DataFrame | None:
    if not TEAM_AVAILABILITY_PATH.exists():
        return None

    df = pd.read_csv(TEAM_AVAILABILITY_PATH)
    normalized_columns = {col.strip().lower() for col in df.columns}
    if not REQUIRED_AVAILABILITY_COLUMNS.issubset(normalized_columns):
        return None

    column_map = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=column_map)
    df = df[["team", "injured", "suspended", "key_impact"]].copy()
    df["team"] = df["team"].astype(str).str.strip()
    df["injured"] = pd.to_numeric(df["injured"], errors="coerce").fillna(0).astype(float)
    df["suspended"] = pd.to_numeric(df["suspended"], errors="coerce").fillna(0).astype(float)
    df["key_impact"] = pd.to_numeric(df["key_impact"], errors="coerce").fillna(0).astype(float)

    if df.empty:
        return None

    return df.reset_index(drop=True)


def save_team_availability_frame(df: pd.DataFrame) -> None:
    output = df[["team", "injured", "suspended", "key_impact"]].copy()
    output.to_csv(TEAM_AVAILABILITY_PATH, index=False)


def load_team_news_frame() -> pd.DataFrame | None:
    if not TEAM_NEWS_PATH.exists():
        return None

    df = pd.read_csv(TEAM_NEWS_PATH)
    normalized_columns = {col.strip().lower() for col in df.columns}
    if not REQUIRED_TEAM_NEWS_COLUMNS.issubset(normalized_columns):
        return None

    column_map = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=column_map)

    for optional in TEAM_NEWS_OPTIONAL_TEXT_COLUMNS:
        if optional not in df.columns:
            df[optional] = ""
    for optional in TEAM_NEWS_OPTIONAL_NUMERIC_COLUMNS:
        if optional not in df.columns:
            df[optional] = 0

    df = df[
        [
            "team",
            "player",
            "status",
            "position",
            "importance",
            "expected_return",
            "source",
            "updated_at",
            "minutes",
            "starts",
            "influence",
            "creativity",
            "threat",
            "expected_goals",
            "expected_assists",
        ]
    ].copy()
    df["team"] = df["team"].astype(str).str.strip()
    df["player"] = df["player"].astype(str).str.strip()
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df["position"] = df["position"].astype(str).str.strip()
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0).clip(lower=0, upper=1)
    df["expected_return"] = df["expected_return"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()
    df["updated_at"] = df["updated_at"].astype(str).str.strip()
    for column in TEAM_NEWS_OPTIONAL_NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).clip(lower=0)

    df = df[(df["team"] != "") & (df["player"] != "") & (df["status"] != "")]
    if df.empty:
        return None

    return df.reset_index(drop=True)


def save_team_news_frame(df: pd.DataFrame) -> None:
    frame = df.copy()
    for optional in TEAM_NEWS_OPTIONAL_TEXT_COLUMNS:
        if optional not in frame.columns:
            frame[optional] = ""
    for optional in TEAM_NEWS_OPTIONAL_NUMERIC_COLUMNS:
        if optional not in frame.columns:
            frame[optional] = 0

    output = frame[
        [
            "team",
            "player",
            "status",
            "position",
            "importance",
            "expected_return",
            "source",
            "updated_at",
            "minutes",
            "starts",
            "influence",
            "creativity",
            "threat",
            "expected_goals",
            "expected_assists",
        ]
    ].copy()
    output.to_csv(TEAM_NEWS_PATH, index=False)
