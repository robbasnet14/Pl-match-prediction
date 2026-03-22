from __future__ import annotations

import logging

import pandas as pd

from .external_data_service import fetch_live_fixtures, fetch_live_standings
from .fixtures_service import get_upcoming_fixtures
from .standings_service import build_standings

logger = logging.getLogger("pl_api")


def get_standings_with_source(season_df: pd.DataFrame, limit: int = 20) -> tuple[list[dict], str]:
    try:
        external = fetch_live_standings(limit=limit)
        if external is not None:
            return external
    except Exception as exc:  # noqa: BLE001
        logger.info("standings external fallback triggered: %s", exc)

    return build_standings(season_df, limit=limit), "historical-standings-fallback"


def get_fixtures_with_source(season_df: pd.DataFrame, limit: int = 3) -> tuple[list[dict], str]:
    try:
        external = fetch_live_fixtures(limit=limit)
        if external is not None:
            return external
    except Exception as exc:  # noqa: BLE001
        logger.info("fixtures external fallback triggered: %s", exc)

    return get_upcoming_fixtures(season_df, limit=limit)
