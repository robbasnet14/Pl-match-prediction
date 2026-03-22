from __future__ import annotations

import pandas as pd

from ..data.repository import load_upcoming_fixtures_frame


def _from_upcoming_file(limit: int) -> list[dict] | None:
    upcoming = load_upcoming_fixtures_frame()
    if upcoming is None:
        return None

    fixtures = []
    for idx, row in enumerate(upcoming.head(limit).itertuples(index=False), start=1):
        fixtures.append(
            {
                "id": idx,
                "home": str(row.home),
                "away": str(row.away),
                "time": str(row.time),
                "venue": str(row.venue),
            }
        )
    return fixtures


def _historical_fallback(season_df: pd.DataFrame, limit: int) -> list[dict]:
    home_rows = season_df[season_df["venue"].astype(str).str.lower() == "home"].copy()
    home_rows = home_rows.sort_values(["date", "time"], ascending=[False, False]).head(limit)

    fixtures = []
    for idx, row in enumerate(home_rows.itertuples(index=False), start=1):
        fixtures.append(
            {
                "id": idx,
                "home": row.team,
                "away": row.opponent,
                "time": str(row.time),
                "venue": str(row.round),
            }
        )

    return fixtures


def get_upcoming_fixtures(season_df: pd.DataFrame, limit: int = 3) -> tuple[list[dict], str]:
    fixtures = _from_upcoming_file(limit)
    if fixtures is not None:
        return fixtures, "upcoming-file"

    return _historical_fallback(season_df, limit), "historical-fallback"
