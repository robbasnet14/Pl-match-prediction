from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
LEGACY_DIR = ROOT_DIR / "MatchPredicting"
DATA_DIR = ROOT_DIR / "backend" / "app" / "data"

MATCHES_PATH = Path(os.getenv("MATCHES_PATH", str(LEGACY_DIR / "matches.csv")))
UPCOMING_FIXTURES_PATH = Path(
    os.getenv("UPCOMING_FIXTURES_PATH", str(DATA_DIR / "upcoming_fixtures.csv"))
)
TEAM_AVAILABILITY_PATH = Path(
    os.getenv("TEAM_AVAILABILITY_PATH", str(DATA_DIR / "team_availability.csv"))
)
TEAM_NEWS_PATH = Path(
    os.getenv("TEAM_NEWS_PATH", str(DATA_DIR / "team_news.csv"))
)

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", os.getenv("PORT", "5000")))

# External live data provider config (optional).
FOOTBALL_API_BASE_URL = os.getenv("FOOTBALL_API_BASE_URL", "https://api.football-data.org/v4")
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY", "")
FOOTBALL_COMPETITION_CODE = os.getenv("FOOTBALL_COMPETITION_CODE", "PL")

MODEL_TRAINING_YEARS = int(os.getenv("MODEL_TRAINING_YEARS", "5"))
MODEL_FORM_WINDOW = int(os.getenv("MODEL_FORM_WINDOW", "5"))
MODEL_H2H_WINDOW = int(os.getenv("MODEL_H2H_WINDOW", "5"))
MODEL_RF_TREES = int(os.getenv("MODEL_RF_TREES", "350"))
MODEL_RF_MIN_SAMPLES_SPLIT = int(os.getenv("MODEL_RF_MIN_SAMPLES_SPLIT", "8"))
MODEL_RF_MAX_DEPTH = int(os.getenv("MODEL_RF_MAX_DEPTH", "0"))
MODEL_ODDS_BLEND_WEIGHT = float(os.getenv("MODEL_ODDS_BLEND_WEIGHT", "0.18"))
MODEL_SCORE_DISPERSION = float(os.getenv("MODEL_SCORE_DISPERSION", "0.2"))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


MODEL_ARTIFACT_ENABLED = _env_bool("MODEL_ARTIFACT_ENABLED", True)
MODEL_ARTIFACT_PATH = Path(
    os.getenv("MODEL_ARTIFACT_PATH", str(DATA_DIR / "model_bundle.pkl"))
)
