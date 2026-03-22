from __future__ import annotations

import os
from datetime import datetime, timezone

from ..data.repository import latest_season_frame, load_matches
from .data_status_service import build_data_status
from .live_data_service import get_fixtures_with_source, get_standings_with_source
from .prediction_service import build_predictions, get_model_metrics


def _build_live_data_status(fixtures_source: str, standings_source: str) -> dict:
    api_key_configured = bool(os.getenv("FOOTBALL_API_KEY", "").strip())
    fixtures_using_external = fixtures_source.startswith("external")
    standings_using_external = standings_source.startswith("external")

    if not api_key_configured:
        message = "Live API key missing. Showing fallback data for fixtures and standings."
    elif fixtures_using_external and standings_using_external:
        message = "Live API configured and active for fixtures and standings."
    else:
        message = "Live API configured, but fallback data is active for some dashboard sections."

    return {
        "configured": api_key_configured,
        "fixturesUsingExternal": fixtures_using_external,
        "standingsUsingExternal": standings_using_external,
        "message": message,
    }


def build_dashboard_payload() -> dict:
    all_matches = load_matches()
    season_df = latest_season_frame(all_matches)
    standings, standings_source = get_standings_with_source(season_df)
    fixtures, fixtures_source = get_fixtures_with_source(season_df)
    predictions = build_predictions(all_matches, standings, fixtures)
    model_metrics = get_model_metrics(all_matches)
    data_status = build_data_status(all_matches)
    live_data_status = _build_live_data_status(fixtures_source, standings_source)

    return {
        "summary": {
            "upcomingFixtures": len(fixtures),
            "predictionConfidence": int(
                round(sum(item["confidence"] for item in predictions) / max(len(predictions), 1))
            ),
            "updatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "activeModels": 1,
        },
        "fixtures": fixtures,
        "fixturesSource": fixtures_source,
        "standings": standings,
        "standingsSource": standings_source,
        "predictions": predictions,
        "modelMetrics": model_metrics,
        "dataStatus": data_status,
        "liveDataStatus": live_data_status,
    }
